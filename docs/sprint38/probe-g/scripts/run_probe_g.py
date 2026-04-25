#!/usr/bin/env python3
"""
S38-PROBE-G: Empirical mechanism capture for the small-N LG+Cosine drift.

Phase 1 — single anchor capture (N=1k seed=42, ST/Cosine):
    Generates the N=1000 anchor.csv with the canonical formula
    (np.random.default_rng(42), 20 features, y = 0.5 X[0] + 0.3 X[1]
    + 0.1 * noise, float32). Runs csv_train_probe_g for 2 iterations
    (the instrumentation arms at iter=1 → iter=2 in the human numbering)
    with --score-function Cosine --grow-policy SymmetricTree --seed 42.
    Drops the per-(feat, bin, partition) leaf records into
    docs/sprint38/probe-g/data/cos_leaf_seed42_depth{0..5}.csv and the
    per-bin cos_accum shadow (fp32 vs fp64) into
    docs/sprint38/probe-g/data/cos_accum_seed42_depth{0..5}.csv.

Phase 2 — scaling sweep (ST/Cosine):
    For N in {500, 1000, 2000, 5000, 10000, 20000, 50000} runs MLX vs
    CPU CatBoost with iter=50, depth=6, bins=128, lr=0.03, l2=3,
    SymmetricTree, Cosine. 5 seeds per N. Computes per-seed drift =
    |MLX_RMSE - CPU_RMSE| / CPU_RMSE * 100. Aggregate drift = mean
    across seeds. Writes data/scaling_sweep.csv.

Phase 3 — diagnostics:
    Loads the cos_leaf csv files, computes per-depth skip rate, MLX-pick
    vs CPU-pick at each depth, and per-bin contribution magnitudes.
    Cross-references PROBE-E's N=50k numbers (hardcoded constants below).
    Writes data/diagnostics.json + data/diagnostics_summary.txt.

Phase 4 — boundary estimation:
    From the scaling sweep, fits a smooth-ish monotone curve through the
    measured drift and reports the smallest N where drift crosses the 2%
    threshold (linear interpolation between consecutive measurements).

Usage:
    python docs/sprint38/probe-g/scripts/run_probe_g.py [--phase {1,2,3,4,all}]
"""
from __future__ import annotations

import argparse
import csv as csv_mod
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
PROBE_G_DIR = REPO_ROOT / "docs" / "sprint38" / "probe-g"
DATA_DIR = PROBE_G_DIR / "data"
BINARY = REPO_ROOT / "csv_train_probe_g"

# Anchor identifiers
ANCHOR_SEED = 42
ANCHOR_N = 1_000
ANCHOR_FEATS = 20
DEPTH = 6
BINS = 128
LR = 0.03
L2 = 3.0
SCORE_FN = "Cosine"
LOSS = "rmse"

# Scaling sweep
SWEEP_NS = [500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000]
SWEEP_SEEDS = [42, 43, 44, 45, 46]
SWEEP_ITERS = 50
SWEEP_GROW = "SymmetricTree"
DRIFT_THRESHOLD_PCT = 2.0

# PROBE-E reference (N=50000, from docs/sprint33/probe-e/FINDING.md)
PROBE_E_SKIP_RATES = {0: 0.0, 1: 2.5, 2: 5.0, 3: 7.6, 4: 10.6, 5: 14.6}


# ---------------------------------------------------------------------------
# Phase 0: data generation
# ---------------------------------------------------------------------------

def make_anchor(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Canonical anchor: 20 features, y = 0.5 X[0] + 0.3 X[1] + 0.1 noise (fp32)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, ANCHOR_FEATS)).astype(np.float32)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(n) * 0.1).astype(np.float32)
    return X, y


def write_csv(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    n_feat = X.shape[1]
    with open(path, "w", newline="") as f:
        w = csv_mod.writer(f)
        w.writerow([f"f{i}" for i in range(n_feat)] + ["target"])
        for i in range(len(y)):
            w.writerow(list(map(float, X[i])) + [float(y[i])])


# ---------------------------------------------------------------------------
# Phase 1: single anchor capture
# ---------------------------------------------------------------------------

def phase1_anchor_capture() -> None:
    if not BINARY.exists():
        print(f"ERROR: {BINARY} not found. Build with:", file=sys.stderr)
        print("  bash docs/sprint38/probe-g/scripts/build_probe_g.sh", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    anchor_csv = DATA_DIR / "anchor_n1000_seed42.csv"

    X, y = make_anchor(ANCHOR_N, ANCHOR_SEED)
    write_csv(anchor_csv, X, y)
    print(f"[phase1] wrote {anchor_csv} ({ANCHOR_N} docs, {ANCHOR_FEATS} features, seed={ANCHOR_SEED})")

    # Anchor capture mirrors PROBE-E reproducibility line:
    #   ./csv_train_probe_e <csv> --iterations 2 --depth 6 --bins 128 --l2 3 \
    #     --lr 0.03 --seed 42 --loss RMSE --score-function Cosine \
    #     --grow-policy SymmetricTree
    # S38-PROBE-Q-PHASE-2 (2026-04-25): the original PROBE-G run did NOT
    # pass --random-strength 0, while the Phase 2 sweep called CPU CatBoost
    # with random_strength=0.0. The asymmetry caused MLX to apply gain
    # perturbation while CPU was deterministic — producing a phantom
    # ~14% drift at N=1k that fully accounted for the "small-N residual"
    # Sprint 37 #113 T3 G3b/G3c flagged. With matched RS=0, MLX trees are
    # bit-identical to CPU's at the F2 N=1k seed=42 anchor (12/12 splits
    # match). Always pass --random-strength 0 in cross-runtime parity tests.
    cmd = [
        str(BINARY), str(anchor_csv),
        "--iterations", "2",
        "--depth", str(DEPTH),
        "--lr", str(LR),
        "--bins", str(BINS),
        "--l2", str(L2),
        "--loss", LOSS,
        "--score-function", SCORE_FN,
        "--grow-policy", "SymmetricTree",
        "--seed", str(ANCHOR_SEED),
        "--random-strength", "0",  # S38-PROBE-Q-PHASE-2: parity with CPU
        "--verbose",
    ]
    env = os.environ.copy()
    env["COSINE_RESIDUAL_OUTDIR"] = str(DATA_DIR)
    env["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/mlx/lib"

    print(f"[phase1] running probe binary (COSINE_RESIDUAL_OUTDIR={DATA_DIR})...")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"[phase1] ERROR: probe exited {result.returncode}", file=sys.stderr)
        print(f"[phase1] STDERR: {result.stderr[:1500]}", file=sys.stderr)
        sys.exit(1)

    # Save stdout/stderr for the record
    (DATA_DIR / "probe_run_stdout.txt").write_text(result.stdout)
    (DATA_DIR / "probe_run_stderr.txt").write_text(result.stderr)
    print(f"[phase1] done in {elapsed:.1f}s; outputs:")
    for d in range(DEPTH):
        leaf_csv = DATA_DIR / f"cos_leaf_seed{ANCHOR_SEED}_depth{d}.csv"
        if leaf_csv.exists():
            n_rows = sum(1 for _ in open(leaf_csv)) - 1
            print(f"  cos_leaf_seed{ANCHOR_SEED}_depth{d}.csv -- {n_rows} rows")
        else:
            print(f"  cos_leaf_seed{ANCHOR_SEED}_depth{d}.csv -- MISSING")


# ---------------------------------------------------------------------------
# Phase 2: scaling sweep
# ---------------------------------------------------------------------------

def parse_final_loss(stdout: str) -> float:
    last_loss = None
    for line in stdout.split("\n"):
        if "loss=" in line and "iter=" in line:
            for tok in line.split():
                if tok.startswith("loss="):
                    try:
                        last_loss = float(tok.split("=", 1)[1])
                    except ValueError:
                        pass
    if last_loss is None:
        raise ValueError(f"no loss in stdout: {stdout[:500]}")
    return last_loss


def run_mlx_sweep_cell(data_path: Path, seed: int) -> tuple[float, float]:
    """Run csv_train_probe_g without instrumentation arming (iter > 1 for >2 iters).
    Note: COSINE_RESIDUAL_INSTRUMENT armed only at PROBE_D_ARM_AT_ITER=1, but
    iter=50 will see arming fire for iter=2 only and re-disarm at end of iter=2.
    Outputs from arming will overwrite themselves but the final RMSE is on iter=49.
    For sweep cells, we send instrumentation to a /tmp directory to avoid
    polluting the canonical anchor capture in DATA_DIR.
    """
    # S38-PROBE-Q-PHASE-2 (2026-04-25): RS=0 on both runtimes for parity.
    # The original PROBE-G sweep used MLX default RS=1.0 vs CPU RS=0.0 —
    # this asymmetry produced the phantom ~14% N=1k drift the sprint was
    # chasing. Sweep numbers in the originally-committed scaling_sweep.csv
    # reflect that asymmetry; re-running with this fix yields ~0% drift.
    cmd = [
        str(BINARY), str(data_path),
        "--iterations", str(SWEEP_ITERS),
        "--depth", str(DEPTH),
        "--lr", str(LR),
        "--bins", str(BINS),
        "--l2", str(L2),
        "--loss", LOSS,
        "--score-function", SCORE_FN,
        "--grow-policy", SWEEP_GROW,
        "--seed", str(seed),
        "--random-strength", "0",  # S38-PROBE-Q-PHASE-2: parity with CPU
        "--verbose",
    ]
    env = os.environ.copy()
    # Sandbox the instrumentation outputs under repo docs/ (env-var policy).
    sweep_outdir = REPO_ROOT / "docs" / "sprint38" / "probe-g" / "data" / "_sweep_tmp"
    sweep_outdir.mkdir(parents=True, exist_ok=True)
    env["COSINE_RESIDUAL_OUTDIR"] = str(sweep_outdir)
    env["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/mlx/lib"
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"  ERROR: sweep cell exited {result.returncode}", file=sys.stderr)
        print(f"  STDERR: {result.stderr[:500]}", file=sys.stderr)
        raise RuntimeError(f"csv_train_probe_g failed for seed={seed}")
    return parse_final_loss(result.stdout), elapsed


def run_cpu_sweep_cell(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(
        iterations=SWEEP_ITERS,
        depth=DEPTH,
        learning_rate=LR,
        loss_function="RMSE",
        grow_policy=SWEEP_GROW,
        score_function=SCORE_FN,
        max_bin=BINS,
        random_seed=seed,
        random_strength=0.0,
        bootstrap_type="No",
        l2_leaf_reg=L2,
        verbose=0,
        thread_count=1,
    )
    m.fit(X, y)
    return float(m.evals_result_["learn"]["RMSE"][-1])


def phase2_scaling_sweep() -> list[dict]:
    if not BINARY.exists():
        print(f"ERROR: {BINARY} not found. Build first.", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    print(f"[phase2] scaling sweep: ST+Cosine, depth={DEPTH}, bins={BINS}, iters={SWEEP_ITERS}")
    print(f"[phase2] N values: {SWEEP_NS}; seeds: {SWEEP_SEEDS}")

    for n in SWEEP_NS:
        per_seed: list[dict] = []
        for seed in SWEEP_SEEDS:
            X, y = make_anchor(n, seed)
            with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tf:
                data_path = Path(tf.name)
            try:
                write_csv(data_path, X, y)
                mlx_rmse, mlx_secs = run_mlx_sweep_cell(data_path, seed)
                cpu_rmse = run_cpu_sweep_cell(X, y, seed)
            finally:
                os.unlink(data_path)
            ratio = mlx_rmse / cpu_rmse
            drift = abs(mlx_rmse - cpu_rmse) / cpu_rmse * 100.0
            print(f"  N={n:>5d} seed={seed} MLX={mlx_rmse:.6f} CPU={cpu_rmse:.6f} "
                  f"ratio={ratio:.5f} drift={drift:6.3f}% wall={mlx_secs:5.1f}s")
            per_seed.append({
                "N": n, "seed": seed,
                "mlx_rmse": mlx_rmse, "cpu_rmse": cpu_rmse,
                "ratio": ratio, "drift_pct": drift,
                "wall_secs": mlx_secs,
            })
            rows.append(per_seed[-1])
        agg = float(np.mean([r["drift_pct"] for r in per_seed]))
        mx = float(np.max([r["drift_pct"] for r in per_seed]))
        mn = float(np.min([r["drift_pct"] for r in per_seed]))
        print(f"  N={n:>5d} aggregate drift = {agg:.4f}% (min={mn:.4f}%, max={mx:.4f}%)")

    # Write CSV
    out_csv = DATA_DIR / "scaling_sweep.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv_mod.DictWriter(
            f, fieldnames=["N", "seed", "mlx_rmse", "cpu_rmse", "ratio", "drift_pct", "wall_secs"]
        )
        w.writeheader()
        w.writerows(rows)
    print(f"[phase2] wrote {out_csv}")
    return rows


# ---------------------------------------------------------------------------
# Phase 3: diagnostics
# ---------------------------------------------------------------------------

def phase3_diagnostics() -> dict:
    """Compute per-depth skip rate, post-fix MLX-actual pick, pre-fix counterfactual.

    IMPORTANT semantics. The instrumentation lives inside FindBestSplit, which
    after DEC-042 uses the per-side mask formula. The captured columns:
      - cpu_termN/D: per-side mask formula (= post-DEC-042 MLX-actual).
      - mlx_termN/D: pre-DEC-042 joint-skip view (zero whenever either side empty).
    So `cpu_termN/D` IS what MLX accumulates today; `mlx_termN/D` is what MLX
    would have accumulated before the fix. Their diff measures DEC-042's
    reshaping of the per-bin argmax. If diff is small or zero, DEC-042 had
    no effect at this regime — and any residual MLX-vs-CPU-CatBoost drift
    comes from a separate mechanism not surfaced by this instrumentation
    (e.g., quantization borders, basePred init, leaf-value computation).

    The "actual_pick" reported below is therefore the **post-fix MLX pick**;
    the "oldmlx_pick" is the pre-fix counterfactual. Neither is the real
    CPU-CatBoost pick — that requires a CPU-runtime trace not present here.
    """
    diag: dict = {"depths": {}, "anchor_n": ANCHOR_N, "anchor_seed": ANCHOR_SEED}

    summary_lines = [
        f"S38-PROBE-G diagnostics (N={ANCHOR_N}, seed={ANCHOR_SEED}, ST+Cosine)",
        f"Reference: PROBE-E at N=50000 (docs/sprint33/probe-e/FINDING.md)",
        "",
        "Columns:",
        "  rows         total per-(feat,bin,partition) records",
        "  skip_n/pct   rows where pre-fix MLX would have joint-skipped (wL<eps or wR<eps)",
        "  PE_skip_pct  PROBE-E reference at N=50k same depth",
        "  postfix_pick (feat,bin):gain — post-DEC-042 MLX-actual argmax (= per-side mask)",
        "  prefix_pick  (feat,bin):gain — pre-DEC-042 counterfactual argmax (= joint-skip)",
        "  gap          postfix_gain - prefix_gain (DEC-042's effect on argmax magnitude)",
        "",
        f"{'depth':<6}{'rows':<8}{'skip_n':<8}{'skip_pct':<10}{'PE_skip_pct':<14}{'postfix_pick':<26}{'prefix_pick':<26}{'gap':<10}",
    ]

    for d in range(DEPTH):
        leaf_csv = DATA_DIR / f"cos_leaf_seed{ANCHOR_SEED}_depth{d}.csv"
        if not leaf_csv.exists():
            print(f"[phase3] missing {leaf_csv}; skipping depth {d}", file=sys.stderr)
            continue
        rows = []
        with open(leaf_csv) as f:
            r = csv_mod.DictReader(f)
            for row in r:
                rows.append({
                    "feat": int(row["featIdx"]),
                    "bin": int(row["bin"]),
                    "p": int(row["partition"]),
                    "skipped": int(row["mlx_skipped"]),
                    "mlxN": float(row["mlx_termNum"]),
                    "mlxD": float(row["mlx_termDen"]),
                    "cpuN": float(row["cpu_termNum"]),
                    "cpuD": float(row["cpu_termDen"]),
                    "wL": float(row["weightLeft"]),
                    "wR": float(row["weightRight"]),
                })
        n_rows = len(rows)
        n_skip = sum(1 for r in rows if r["skipped"])
        skip_pct = 100.0 * n_skip / max(1, n_rows)

        # Aggregate per-(feat, bin) — MLX-actual is the per-side mask formula
        # (i.e. cpu_termNum/Den columns; MLX matches CPU post-DEC-042). The
        # mlx_termNum/Den columns describe the OLD pre-DEC-042 joint-skip view.
        # For PROBE-G the actual MLX gain is the "cpu" formula (post-fix); for
        # diagnostics, we still surface both rules to expose the residual gap.
        from collections import defaultdict
        agg_actual = defaultdict(lambda: [0.0, 0.0])  # actual MLX (= post-fix per-side mask)
        agg_oldmlx = defaultdict(lambda: [0.0, 0.0])  # pre-fix joint-skip view
        for r in rows:
            key = (r["feat"], r["bin"])
            agg_actual[key][0] += r["cpuN"]
            agg_actual[key][1] += r["cpuD"]
            agg_oldmlx[key][0] += r["mlxN"]
            agg_oldmlx[key][1] += r["mlxD"]

        # Compute Cosine gain per (feat, bin) under each rule.
        # Cosine gain = sum(termNum) / sqrt(sum(termDen) + 1e-20). Approximate.
        def gain(N: float, D: float) -> float:
            return N / math.sqrt(D + 1e-20) if D > 0 else 0.0

        actual_gains = {k: gain(*v) for k, v in agg_actual.items()}
        oldmlx_gains = {k: gain(*v) for k, v in agg_oldmlx.items()}

        if not actual_gains:
            continue
        actual_pick = max(actual_gains.items(), key=lambda kv: kv[1])
        oldmlx_pick = max(oldmlx_gains.items(), key=lambda kv: kv[1])
        # In the post-DEC-042 codebase, MLX-actual === actual_pick. So the
        # "MLX pick" we want to compare to CPU pick is actual_pick. We label
        # the OLD pick separately to expose how DEC-042 reshaped the argmax.
        # CPU pick is the actual_pick (since post-fix MLX matches CPU formula
        # on a single bin). The legitimate question at small N: is CPU's
        # *real* CatBoost choosing differently from MLX-actual? That requires
        # a CPU-CatBoost runtime trace, not available from this binary. For
        # now we report the post-fix vs pre-fix divergence as a proxy.

        # Per-bin contribution magnitude (median |cpuN|+|cpuD| across nonzero rows)
        nonzero = [r for r in rows if r["cpuN"] > 0]
        if nonzero:
            mag_med_N = float(np.median([r["cpuN"] for r in nonzero]))
            mag_med_D = float(np.median([r["cpuD"] for r in nonzero]))
        else:
            mag_med_N = mag_med_D = 0.0

        # Difference cardinality: rows where mlx and cpu term differ
        n_diff_n = sum(1 for r in rows if abs(r["mlxN"] - r["cpuN"]) > 1e-12)
        n_diff_d = sum(1 for r in rows if abs(r["mlxD"] - r["cpuD"]) > 1e-12)

        diag["depths"][d] = {
            "rows": n_rows,
            "n_skip": n_skip,
            "skip_pct": skip_pct,
            "probe_e_skip_pct_n50k": PROBE_E_SKIP_RATES.get(d, None),
            "postfix_pick": {  # MLX-actual after DEC-042 (per-side mask)
                "feat": actual_pick[0][0], "bin": actual_pick[0][1],
                "gain": actual_pick[1],
            },
            "prefix_pick": {  # pre-DEC-042 counterfactual (joint-skip)
                "feat": oldmlx_pick[0][0], "bin": oldmlx_pick[0][1],
                "gain": oldmlx_pick[1],
            },
            "gap_postfix_minus_prefix": actual_pick[1] - oldmlx_pick[1],
            "n_rows_termN_diff": n_diff_n,
            "n_rows_termD_diff": n_diff_d,
            "mag_median_termN_nonzero": mag_med_N,
            "mag_median_termD_nonzero": mag_med_D,
        }

        pe_skip_str = (f"{PROBE_E_SKIP_RATES[d]:.2f}"
                       if d in PROBE_E_SKIP_RATES else "n/a")
        post_str = f"({actual_pick[0][0]},{actual_pick[0][1]}):{actual_pick[1]:.4f}"
        pre_str = f"({oldmlx_pick[0][0]},{oldmlx_pick[0][1]}):{oldmlx_pick[1]:.4f}"
        line = (f"{d:<6}{n_rows:<8}{n_skip:<8}{skip_pct:<10.2f}"
                f"{pe_skip_str:<14}{post_str:<26}{pre_str:<26}"
                f"{actual_pick[1] - oldmlx_pick[1]:<10.4f}")
        summary_lines.append(line)

    # Write outputs
    diag_json = DATA_DIR / "diagnostics.json"
    with open(diag_json, "w") as f:
        json.dump(diag, f, indent=2)

    diag_txt = DATA_DIR / "diagnostics_summary.txt"
    diag_txt.write_text("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))
    print(f"\n[phase3] wrote {diag_json}")
    print(f"[phase3] wrote {diag_txt}")
    return diag


# ---------------------------------------------------------------------------
# Phase 4: boundary estimation
# ---------------------------------------------------------------------------

def phase4_boundary() -> dict:
    sweep_csv = DATA_DIR / "scaling_sweep.csv"
    if not sweep_csv.exists():
        print(f"[phase4] {sweep_csv} not found; run phase 2 first", file=sys.stderr)
        return {}
    by_n: dict[int, list[float]] = {}
    with open(sweep_csv) as f:
        for r in csv_mod.DictReader(f):
            by_n.setdefault(int(r["N"]), []).append(float(r["drift_pct"]))
    table = sorted([(n, float(np.mean(v))) for n, v in by_n.items()])
    # Find where drift crosses DRIFT_THRESHOLD_PCT (transitioning from above to below)
    crossing = None
    for i in range(1, len(table)):
        n0, d0 = table[i - 1]
        n1, d1 = table[i]
        if d0 >= DRIFT_THRESHOLD_PCT and d1 < DRIFT_THRESHOLD_PCT:
            # Linear interpolation in (log N, drift)
            if d0 == d1:
                ncross = n1
            else:
                t = (DRIFT_THRESHOLD_PCT - d0) / (d1 - d0)
                ncross = math.exp(math.log(n0) + t * (math.log(n1) - math.log(n0)))
            crossing = {"n_cross": ncross, "from": (n0, d0), "to": (n1, d1)}
            break
    out = {"table": table, "crossing": crossing, "threshold_pct": DRIFT_THRESHOLD_PCT}
    print(f"[phase4] N → aggregate drift table:")
    for n, d in table:
        marker = "  " if d < DRIFT_THRESHOLD_PCT else "**"
        print(f"  {marker} N={n:>5d}  drift={d:7.4f}%")
    if crossing:
        print(f"[phase4] crossing: N* ≈ {crossing['n_cross']:.0f} "
              f"(between N={crossing['from'][0]} drift={crossing['from'][1]:.3f}% "
              f"and N={crossing['to'][0]} drift={crossing['to'][1]:.3f}%)")
    else:
        print(f"[phase4] no crossing of {DRIFT_THRESHOLD_PCT}% threshold detected in sweep range")
    (DATA_DIR / "boundary.json").write_text(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="all", choices=["1", "2", "3", "4", "all"])
    args = p.parse_args()

    if args.phase in ("1", "all"):
        phase1_anchor_capture()
    if args.phase in ("2", "all"):
        phase2_scaling_sweep()
    if args.phase in ("3", "all"):
        phase3_diagnostics()
    if args.phase in ("4", "all"):
        phase4_boundary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
