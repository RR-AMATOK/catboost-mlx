#!/usr/bin/env python3
# HISTORICAL: PROBE_H_INSTRUMENT block removed in S39 (DEC-044 withdrawn). This script preserved as historical record. Re-running requires restoring the source block.
"""
S38-PROBE-H: Per-side cosNum/cosDen capture at iter=1 for formula-divergence localisation.

Single phase — anchor capture:
    Uses the canonical N=1000 seed=42 anchor (np.random.default_rng(42), 20 features,
    y = 0.5*X[0] + 0.3*X[1] + 0.1*noise, float32). Matches PROBE-G/F2 exactly.

    Runs csv_train_probe_h for 2 iterations (instrumentation arms at PROBE_D_ARM_AT_ITER=0
    = iter=0 in code = iter=1 in user-facing 1-indexed terms = first tree from constant
    basePred). Captures per-(feat, bin, partition) records with split per-side accumulators:

      cosNumL, cosDenL  — left-child contribution (sum_L^2/(w_L+lambda), etc.)
      cosNumR, cosDenR  — right-child contribution
      cosNumTotal       = cosNumL + cosNumR  (= cpu_termNum in PROBE-E)
      cosDenTotal       = cosDenL + cosDenR  (= cpu_termDen in PROBE-E)
      gain_mlx          — Cosine gain MLX computed for this (feat,bin): cosNumTotal/sqrt(cosDenTotal)
                          aggregated across all partitions
      picked_by_mlx     — 1 for the (feat,bin) MLX selected as argmax at this depth, else 0

    Output: docs/sprint38/probe-h/data/probe_h_iter1_depth{0..5}.csv (6 files, one per depth).
    Log:    docs/sprint38/probe-h/data/probe_h_run.log

Context for downstream consumer (data-scientist):
    Cross-join probe_h_iter1_depth{d}.csv against F2's cos_leaf_seed42_depth{d}.csv
    (PROBE-G iter=2 leaf records) to compare per-side accumulators across iterations.
    The key question: for the depth where CPU and MLX picks diverge (d=1..5 at iter=1),
    does CPUs CalcScoreOnSide produce a higher gain for the CPU-preferred (feat,bin)
    than MLX computes for its own pick? If so, the formula is correct but the
    accumulation differs; if not, the formula definitions differ.

    iter=1 pick reference (from F2 mlx_model.json trees[0]):
      d=0: feat=0  bin=69   border=0.10254748  (matches CPU — d=0 ULP-identical)
      d=1: feat=1  bin=64   border=0.09566100  (CPU picks feat=1, border=0.43849730)
      d=2: feat=0  bin=29   border=-0.70789814 (CPU picks feat=0, border=-0.81075025)
      d=3: feat=15 bin=28   border=-0.86897051 (CPU picks feat=0, border=1.03529096)
      d=4: feat=1  bin=23   border=-0.89314175 (CPU picks feat=1, border=-0.80015314)
      d=5: feat=0  bin=98   border=0.74944925  (CPU picks feat=0, border=1.74658537)

Usage:
    python docs/sprint38/probe-h/scripts/run_probe_h.py
"""
from __future__ import annotations

import csv as csv_mod
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
PROBE_H_DIR = REPO_ROOT / "docs" / "sprint38" / "probe-h"
DATA_DIR = PROBE_H_DIR / "data"
BINARY = REPO_ROOT / "csv_train_probe_h"

# Canonical anchor — must match PROBE-G/F2 exactly
ANCHOR_SEED = 42
ANCHOR_N = 1_000
ANCHOR_FEATS = 20
DEPTH = 6
BINS = 128
LR = 0.03
L2 = 3.0
SCORE_FN = "Cosine"
LOSS = "rmse"

# iter=1 MLX picks from F2 mlx_model.json trees[0] (0-indexed bin_threshold).
# Used to verify that picked_by_mlx==1 rows in the output match the known splits.
MLX_ITER1_PICKS = {
    0: {"feat": 0,  "bin": 69},
    1: {"feat": 1,  "bin": 64},
    2: {"feat": 0,  "bin": 29},
    3: {"feat": 15, "bin": 28},
    4: {"feat": 1,  "bin": 23},
    5: {"feat": 0,  "bin": 98},
}


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


def run_anchor_capture(anchor_csv: Path) -> None:
    """Run the probe binary and collect per-depth CSV outputs."""
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
        "--verbose",
    ]
    env = os.environ.copy()
    env["COSINE_RESIDUAL_OUTDIR"] = str(DATA_DIR)
    env["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/mlx/lib"

    print(f"[probe-h] running binary (COSINE_RESIDUAL_OUTDIR={DATA_DIR})...")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - t0

    log_path = DATA_DIR / "probe_h_run.log"
    with open(log_path, "w") as lf:
        lf.write(f"=== STDOUT ===\n{result.stdout}\n")
        lf.write(f"=== STDERR ===\n{result.stderr}\n")
        lf.write(f"=== exit_code={result.returncode} elapsed={elapsed:.2f}s ===\n")

    if result.returncode != 0:
        print(f"[probe-h] ERROR: binary exited {result.returncode}", file=sys.stderr)
        print(f"[probe-h] STDERR:\n{result.stderr[:2000]}", file=sys.stderr)
        sys.exit(1)

    print(f"[probe-h] done in {elapsed:.1f}s")
    print(f"[probe-h] log: {log_path}")


def verify_outputs() -> bool:
    """Check all 6 depth CSVs exist with non-zero rows and picked_by_mlx matches F2."""
    ok = True
    print()
    print("=== PROBE-H output verification ===")
    print(f"{'depth':<6}{'rows':<8}{'picked_feat':<14}{'picked_bin':<12}{'expected_feat':<16}{'expected_bin':<14}{'match'}")

    for d in range(DEPTH):
        csv_path = DATA_DIR / f"probe_h_iter1_depth{d}.csv"
        if not csv_path.exists():
            print(f"  depth={d}: MISSING {csv_path}", file=sys.stderr)
            ok = False
            continue

        rows = []
        with open(csv_path) as f:
            for row in csv_mod.DictReader(f):
                rows.append(row)

        if not rows:
            print(f"  depth={d}: EMPTY CSV", file=sys.stderr)
            ok = False
            continue

        # Find picked_by_mlx==1 row(s)
        picked_rows = [r for r in rows if int(r["picked_by_mlx"]) == 1]
        if not picked_rows:
            print(f"  depth={d}: no picked_by_mlx==1 row — instrumentation hooked wrong place",
                  file=sys.stderr)
            ok = False
            continue

        # All picked rows should have same (feat, bin) — one argmax winner
        p_feat = int(picked_rows[0]["feat"])
        p_bin  = int(picked_rows[0]["bin"])
        exp = MLX_ITER1_PICKS.get(d, {})
        exp_feat = exp.get("feat", -1)
        exp_bin  = exp.get("bin", -1)
        match = "YES" if (p_feat == exp_feat and p_bin == exp_bin) else "NO"
        if match == "NO":
            ok = False
        print(f"{d:<6}{len(rows):<8}{p_feat:<14}{p_bin:<12}{exp_feat:<16}{exp_bin:<14}{match}")

    return ok


def print_sample_rows(depth: int = 0, n: int = 3) -> None:
    """Print first n rows of the depth-0 CSV for the report."""
    csv_path = DATA_DIR / f"probe_h_iter1_depth{depth}.csv"
    if not csv_path.exists():
        print(f"[probe-h] cannot print sample: {csv_path} missing", file=sys.stderr)
        return
    print(f"\n=== First {n} rows of probe_h_iter1_depth{depth}.csv ===")
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        header = reader.fieldnames
        print(",".join(header))
        for i, row in enumerate(reader):
            if i >= n:
                break
            print(",".join(row[h] for h in header))


def main() -> int:
    if not BINARY.exists():
        print(f"ERROR: {BINARY} not found. Build with:", file=sys.stderr)
        print("  bash docs/sprint38/probe-h/scripts/build_probe_h.sh", file=sys.stderr)
        return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse anchor CSV from probe-g if present (same anchor); else generate fresh.
    probe_g_anchor = REPO_ROOT / "docs" / "sprint38" / "probe-g" / "data" / "anchor_n1000_seed42.csv"
    anchor_csv = DATA_DIR / "anchor_n1000_seed42.csv"
    if probe_g_anchor.exists() and not anchor_csv.exists():
        import shutil
        shutil.copy(probe_g_anchor, anchor_csv)
        print(f"[probe-h] reused anchor from probe-g: {anchor_csv}")
    elif not anchor_csv.exists():
        X, y = make_anchor(ANCHOR_N, ANCHOR_SEED)
        write_csv(anchor_csv, X, y)
        print(f"[probe-h] wrote anchor: {anchor_csv} ({ANCHOR_N} docs, {ANCHOR_FEATS} features)")
    else:
        print(f"[probe-h] anchor exists: {anchor_csv}")

    run_anchor_capture(anchor_csv)

    ok = verify_outputs()
    print_sample_rows(depth=0, n=3)

    if ok:
        print("\n[probe-h] ALL CHECKS PASSED")
    else:
        print("\n[probe-h] SOME CHECKS FAILED — review output above", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
