#!/usr/bin/env python3
"""
S38-F2: Cross-runtime tree-split comparison harness.

Captures iter=2 (0-indexed: trees[1]) split decisions from both CPU CatBoost
and MLX csv_train at the IDENTICAL anchor as PROBE-G (N=1000, seed=42, ST+Cosine).
Produces a side-by-side comparison CSV for the data-scientist discriminator.

Discriminator routing:
  - All 6 depths match (cpu_feat==mlx_feat AND cpu_bin==mlx_bin):
      DEC-042 closed for d=2 cross-runtime -> next: PROBE-I (precision/leaf-values)
  - Any depth differs:
      DEC-042 per-side formula != CPU CalcScoreOnSide -> next: PROBE-H
      (CPU per-side instrumentation)

Usage:
    python docs/sprint38/f2/scripts/run_f2.py

Outputs (all under docs/sprint38/f2/data/):
    anchor_n1000_seed42.csv    -- canonical anchor data (byte-identical to PROBE-G)
    cpu_model.json             -- CatBoost CPU model (JSON format)
    cpu_iter2_splits.json      -- iter=2 splits extracted from CPU model
    mlx_model.json             -- MLX csv_train model (JSON format)
    mlx_iter2_splits.json      -- iter=2 splits extracted from MLX model
    comparison.csv             -- side-by-side split table with match column
    run.log                    -- combined stdout/stderr log
"""
from __future__ import annotations

import bisect
import csv as csv_mod
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
F2_DIR = REPO_ROOT / "docs" / "sprint38" / "f2"
DATA_DIR = F2_DIR / "data"
PROBE_G_DATA = REPO_ROOT / "docs" / "sprint38" / "probe-g" / "data"

# Binary: prefer csv_train_probe_g (same binary as PROBE-G), fall back to csv_train
BINARY = REPO_ROOT / "csv_train_probe_g"
if not BINARY.exists():
    BINARY = REPO_ROOT / "csv_train"

# ---------------------------------------------------------------------------
# Anchor parameters — MUST match PROBE-G exactly
# ---------------------------------------------------------------------------
ANCHOR_SEED = 42
ANCHOR_N = 1_000
ANCHOR_FEATS = 20
DEPTH = 6
BINS = 128
LR = 0.03
L2 = 3.0
SCORE_FN = "Cosine"
LOSS = "rmse"
ITERS = 2
# iter=2 is the 2nd tree built; in 0-indexed JSON arrays this is trees[1] / oblivious_trees[1]
TARGET_TREE_IDX = 1

# ---------------------------------------------------------------------------
# Logging: tee to both stdout and run.log
# ---------------------------------------------------------------------------
_log_file = None


def log(msg: str) -> None:
    print(msg, flush=True)
    if _log_file is not None:
        _log_file.write(msg + "\n")
        _log_file.flush()


def log_section(title: str) -> None:
    bar = "=" * 70
    log(f"\n{bar}\n{title}\n{bar}")


# ---------------------------------------------------------------------------
# Step 0: Generate anchor data (byte-identical to PROBE-G's make_anchor)
# ---------------------------------------------------------------------------

def make_anchor():
    """Canonical anchor: 20 features, y = 0.5 X[0] + 0.3 X[1] + 0.1 * noise (fp32).
    Exact replica of PROBE-G's make_anchor(n=1000, seed=42)."""
    import numpy as np
    rng = np.random.default_rng(ANCHOR_SEED)
    X = rng.standard_normal((ANCHOR_N, ANCHOR_FEATS)).astype("float32")
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(ANCHOR_N) * 0.1).astype("float32")
    return X, y


def write_anchor_csv(path: Path, X, y) -> None:
    n_feat = X.shape[1]
    with open(path, "w", newline="") as f:
        w = csv_mod.writer(f)
        w.writerow([f"f{i}" for i in range(n_feat)] + ["target"])
        for i in range(len(y)):
            w.writerow(list(map(float, X[i])) + [float(y[i])])


# ---------------------------------------------------------------------------
# Step 1: CPU CatBoost — train, save JSON, extract iter=2 splits
# ---------------------------------------------------------------------------

def run_cpu(X, y) -> dict:
    """Train CatBoost CPU, save model JSON, return extracted iter=2 split dict."""
    log_section("CPU CatBoost run")

    import catboost
    log(f"catboost version: {catboost.__version__}")

    # API note: border_count (not max_bin) controls quantization bin count in
    # CatBoostRegressor.__init__; score_function and grow_policy are accepted as-is
    # in catboost 1.2.x.
    model = catboost.CatBoostRegressor(
        iterations=ITERS,
        depth=DEPTH,
        learning_rate=LR,
        l2_leaf_reg=L2,
        border_count=BINS,           # <-- border_count, not max_bin
        grow_policy="SymmetricTree",
        score_function=SCORE_FN,
        loss_function="RMSE",
        random_seed=ANCHOR_SEED,
        random_strength=0.0,
        bootstrap_type="No",
        verbose=False,
        thread_count=1,
    )

    log("Fitting CPU model...")
    t0 = time.perf_counter()
    import catboost as _cb
    feature_names = [f"f{i}" for i in range(ANCHOR_FEATS)]
    pool = _cb.Pool(X, label=y, feature_names=feature_names)
    model.fit(pool)
    elapsed = time.perf_counter() - t0
    log(f"CPU fit done in {elapsed:.2f}s")

    cpu_model_path = DATA_DIR / "cpu_model.json"
    model.save_model(str(cpu_model_path), format="json")
    log(f"CPU model saved: {cpu_model_path}")

    with open(cpu_model_path) as f:
        cpu_json = json.load(f)

    # Extract iter=2 splits (TARGET_TREE_IDX=1 in 0-indexed oblivious_trees)
    splits_out = extract_cpu_splits(cpu_json, TARGET_TREE_IDX)

    out_path = DATA_DIR / "cpu_iter2_splits.json"
    with open(out_path, "w") as f:
        json.dump(splits_out, f, indent=2)
    log(f"CPU iter=2 splits saved: {out_path}")

    for d in range(DEPTH):
        key = f"depth_{d}"
        s = splits_out[key]
        log(f"  CPU depth {d}: feat={s['feat']}, bin_idx={s['bin_idx']}, border={s['border']:.8g}")

    return splits_out


def extract_cpu_splits(cpu_json: dict, tree_idx: int) -> dict:
    """Extract per-depth splits from CatBoost CPU JSON for the given tree index.

    CatBoost CPU JSON layout (SymmetricTree):
      oblivious_trees[i].splits[d] = {float_feature_index, border, split_type}
      features_info.float_features[k].borders = sorted list of border values
        (k is flat_feature_index)

    bin_idx = position of border in the per-feature borders array (0-indexed).
    """
    trees = cpu_json["oblivious_trees"]
    if tree_idx >= len(trees):
        raise RuntimeError(
            f"CPU model has {len(trees)} trees; requested tree_idx={tree_idx}"
        )
    tree = trees[tree_idx]
    splits = tree["splits"]

    # Build per-feature border lookup: flat_feature_index -> sorted borders list
    ff = cpu_json["features_info"]["float_features"]
    feature_borders: dict[int, list[float]] = {
        entry["flat_feature_index"]: entry["borders"] for entry in ff
    }

    result: dict = {}
    for d, sp in enumerate(splits):
        feat_idx = sp["float_feature_index"]
        border = sp["border"]
        borders = feature_borders.get(feat_idx, [])

        # Exact lookup first (border is exact in JSON); bisect as fallback
        if border in borders:
            bin_idx = borders.index(border)
        else:
            # bisect_left gives insertion point = position of first border > this value,
            # which equals the number of borders <= border → bin_idx
            bin_idx = bisect.bisect_left(borders, border)
            # Verify it's within range
            if bin_idx >= len(borders) or abs(borders[bin_idx] - border) > 1e-7:
                # Use nearest
                bin_idx = bisect.bisect_right(borders, border) - 1

        result[f"depth_{d}"] = {
            "feat": feat_idx,
            "bin_idx": bin_idx,
            "border": border,
        }

    return result


# ---------------------------------------------------------------------------
# Step 2: MLX csv_train — run subprocess, save JSON, extract iter=2 splits
# ---------------------------------------------------------------------------

def run_mlx(anchor_csv: Path) -> dict:
    """Run MLX csv_train as subprocess, save model JSON, return extracted iter=2 split dict."""
    log_section("MLX csv_train run")

    if not BINARY.exists():
        raise RuntimeError(
            f"Binary not found: {BINARY}\n"
            "Rebuild with: bash docs/sprint38/probe-g/scripts/build_probe_g.sh"
        )

    mlx_model_path = DATA_DIR / "mlx_model.json"

    cmd = [
        str(BINARY), str(anchor_csv),
        "--iterations", str(ITERS),
        "--depth", str(DEPTH),
        "--lr", str(LR),
        "--bins", str(BINS),
        "--l2", str(L2),
        "--loss", LOSS,
        "--score-function", SCORE_FN,
        # SymmetricTree is the default grow policy; no explicit flag needed
        "--seed", str(ANCHOR_SEED),
        # S38-PROBE-Q-PHASE-2: MLX defaults to RandomStrength=1.0, but the
        # CPU CatBoost call below uses random_strength=0.0. Without this
        # explicit override the comparison is asymmetric and produces a
        # phantom ~14% drift at small N from MLX's noise-induced
        # suboptimality. ALWAYS match RS across runtimes for parity tests.
        "--random-strength", "0",
        "--output", str(mlx_model_path),
        "--verbose",
    ]

    log(f"MLX binary: {BINARY.name}")
    log(f"Command: {' '.join(cmd)}")

    env = os.environ.copy()
    # Sandbox any instrumentation outputs that the probe binary might emit
    # (COSINE_RESIDUAL_OUTDIR is checked by the binary; route to f2/data to
    # keep F2 self-contained rather than polluting probe-g data).
    env["COSINE_RESIDUAL_OUTDIR"] = str(DATA_DIR)
    env["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/mlx/lib"

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - t0

    log(f"MLX run done in {elapsed:.2f}s (exit={result.returncode})")
    log("--- MLX stdout ---")
    for line in result.stdout.strip().splitlines():
        log(line)
    if result.stderr.strip():
        log("--- MLX stderr ---")
        for line in result.stderr.strip().splitlines():
            log(line)

    if result.returncode != 0:
        raise RuntimeError(
            f"MLX csv_train exited {result.returncode}\n"
            f"stderr: {result.stderr[:1000]}"
        )

    if not mlx_model_path.exists():
        raise RuntimeError(
            f"MLX model JSON not written to {mlx_model_path}\n"
            "Check --output flag or binary version."
        )
    log(f"MLX model saved: {mlx_model_path}")

    with open(mlx_model_path) as f:
        mlx_json = json.load(f)

    splits_out = extract_mlx_splits(mlx_json, TARGET_TREE_IDX)

    out_path = DATA_DIR / "mlx_iter2_splits.json"
    with open(out_path, "w") as f:
        json.dump(splits_out, f, indent=2)
    log(f"MLX iter=2 splits saved: {out_path}")

    for d in range(DEPTH):
        key = f"depth_{d}"
        s = splits_out[key]
        log(f"  MLX depth {d}: feat={s['feat']}, bin_idx={s['bin_idx']}, border={s['border']:.8g}")

    return splits_out


def extract_mlx_splits(mlx_json: dict, tree_idx: int) -> dict:
    """Extract per-depth splits from MLX JSON for the given tree index.

    MLX JSON layout (SymmetricTree):
      trees[i].splits[d] = {feature_idx, bin_threshold, is_one_hot}
      features[k].borders = sorted list of border values (k is feature index)

    bin_threshold is 0-indexed into the per-feature borders array.
    border = features[feat_idx].borders[bin_threshold]
    """
    trees = mlx_json.get("trees", [])
    if tree_idx >= len(trees):
        raise RuntimeError(
            f"MLX model has {len(trees)} trees; requested tree_idx={tree_idx}"
        )
    tree = trees[tree_idx]
    splits = tree["splits"]

    # Build per-feature border lookup: feature index -> borders list
    feature_borders: dict[int, list[float]] = {
        feat["index"]: feat["borders"] for feat in mlx_json.get("features", [])
    }

    result: dict = {}
    for d, sp in enumerate(splits):
        feat_idx = sp["feature_idx"]
        bin_idx = sp["bin_threshold"]
        borders = feature_borders.get(feat_idx, [])

        if bin_idx < len(borders):
            border = borders[bin_idx]
        else:
            border = float("nan")
            log(
                f"  WARNING: MLX depth {d} bin_threshold={bin_idx} out of range "
                f"(feature {feat_idx} has {len(borders)} borders)"
            )

        result[f"depth_{d}"] = {
            "feat": feat_idx,
            "bin_idx": bin_idx,
            "border": border,
        }

    return result


# ---------------------------------------------------------------------------
# Step 3: Side-by-side comparison
# ---------------------------------------------------------------------------

def build_comparison(cpu_splits: dict, mlx_splits: dict) -> list[dict]:
    """Build side-by-side comparison rows for each depth 0..DEPTH-1.

    NOTE on bin_match: CPU CatBoost's JSON stores only the borders used by the
    model (not the full quantization grid), so cpu_bin_idx and mlx_bin_idx are
    on different scales. bin_match compares within-model bin indices and is
    INFORMATIONAL ONLY when quantization grids differ.

    feat_match is always valid (feature indices are global and comparable).
    border_match (|cpu_border - mlx_border| < 1e-4) tests numerical equivalence
    of the split threshold in the original feature space — this IS comparable
    across quantization grids as it is in original float32 units.

    The primary discriminator is: feat_match AND border_match.
    """
    rows = []
    for d in range(DEPTH):
        cpu_key = f"depth_{d}"
        mlx_key = f"depth_{d}"
        cs = cpu_splits.get(cpu_key, {})
        ms = mlx_splits.get(mlx_key, {})

        cpu_feat = cs.get("feat", -1)
        cpu_bin = cs.get("bin_idx", -1)
        cpu_border = cs.get("border", float("nan"))
        mlx_feat = ms.get("feat", -1)
        mlx_bin = ms.get("bin_idx", -1)
        mlx_border = ms.get("border", float("nan"))

        feat_match = cpu_feat == mlx_feat
        bin_match = cpu_bin == mlx_bin  # informational only; scales differ across models

        import math
        border_close = (
            not math.isnan(cpu_border) and not math.isnan(mlx_border)
            and abs(cpu_border - mlx_border) < 1e-4
        )
        # Primary discriminator: same feature AND close border value
        match = feat_match and border_close

        rows.append({
            "depth": d,
            "cpu_feat": cpu_feat,
            "cpu_bin": cpu_bin,
            "cpu_border": cpu_border,
            "mlx_feat": mlx_feat,
            "mlx_bin": mlx_bin,
            "mlx_border": mlx_border,
            "feat_match": feat_match,
            "bin_match": bin_match,
            "border_close": border_close,
            "match": match,
        })
    return rows


def write_comparison_csv(rows: list[dict], out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv_mod.DictWriter(
            f,
            fieldnames=["depth", "cpu_feat", "cpu_bin", "cpu_border",
                        "mlx_feat", "mlx_bin", "mlx_border",
                        "feat_match", "bin_match", "border_close", "match"],
        )
        w.writeheader()
        w.writerows(rows)


def print_comparison_table(rows: list[dict]) -> None:
    log_section("Comparison table: iter=2 splits (trees[1])")
    log("NOTE: bin_match is informational only — CPU and MLX have different quantization")
    log("      grid sizes (CPU stores only used-feature borders; MLX stores full grid).")
    log("      Primary discriminator = feat_match AND border_close (|delta|<1e-4 in feature space).")
    log("")
    hdr = (f"{'depth':<6} {'cpu_feat':>8} {'cpu_bin':>7} {'cpu_border':>14} "
           f"{'mlx_feat':>8} {'mlx_bin':>7} {'mlx_border':>14} "
           f"{'feat_m':>7} {'bdr_cl':>7} {'MATCH':>6}")
    log(hdr)
    log("-" * len(hdr))
    for r in rows:
        match_str = "YES" if r["match"] else "NO "
        feat_m = "Y" if r["feat_match"] else "N"
        bdr_cl = "Y" if r["border_close"] else "N"
        log(
            f"{r['depth']:<6} {r['cpu_feat']:>8} {r['cpu_bin']:>7} {r['cpu_border']:>14.8g} "
            f"{r['mlx_feat']:>8} {r['mlx_bin']:>7} {r['mlx_border']:>14.8g} "
            f"{feat_m:>7} {bdr_cl:>7} {match_str:>6}"
        )
    n_match = sum(1 for r in rows if r["match"])
    n_feat_match = sum(1 for r in rows if r["feat_match"])
    log(f"\nFeature-matching depths: {n_feat_match}/{len(rows)}")
    log(f"Full-match depths (feat+border): {n_match}/{len(rows)}")


# ---------------------------------------------------------------------------
# Step 4: Quantization border-policy check
# ---------------------------------------------------------------------------

def check_quantization_policy(anchor_csv: Path) -> None:
    """Log quantization border counts and first few borders for features 0 and 1
    from both CPU and MLX models, to flag any border-policy mismatch."""
    log_section("Quantization border-policy check")

    cpu_model_path = DATA_DIR / "cpu_model.json"
    mlx_model_path = DATA_DIR / "mlx_model.json"

    with open(cpu_model_path) as f:
        cpu_json = json.load(f)
    with open(mlx_model_path) as f:
        mlx_json = json.load(f)

    ff_cpu = cpu_json["features_info"]["float_features"]
    ff_mlx = mlx_json.get("features", [])

    log("CPU (CatBoost GreedyLogSum, default feature_border_type):")
    for ff in ff_cpu[:4]:
        fi = ff["flat_feature_index"]
        borders = ff["borders"]
        log(f"  feat {fi}: {len(borders)} borders, first 4: {borders[:4]}")

    log("MLX (GreedyLogSum port — csv_train.cpp GreedyLogSumBestSplit):")
    for ff in ff_mlx[:4]:
        fi = ff["index"]
        borders = ff["borders"]
        log(f"  feat {fi}: {len(borders)} borders, first 4: {borders[:4]}")

    # Check border count parity for f0 and f1 (the signal features)
    for feat_idx in [0, 1]:
        cpu_borders = ff_cpu[feat_idx]["borders"]
        mlx_borders = next(
            (ff["borders"] for ff in ff_mlx if ff["index"] == feat_idx), []
        )
        n_cpu = len(cpu_borders)
        n_mlx = len(mlx_borders)
        if n_cpu == n_mlx:
            # Check if borders are numerically close
            diffs = [abs(a - b) for a, b in zip(cpu_borders, mlx_borders)]
            max_diff = max(diffs) if diffs else 0.0
            log(f"  feat {feat_idx}: border count matches ({n_cpu}), max border diff = {max_diff:.6g}")
        else:
            log(
                f"  WARNING feat {feat_idx}: CPU has {n_cpu} borders, MLX has {n_mlx} — "
                "QUANTIZATION MISMATCH. Bin indices are not directly comparable."
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    import hashlib

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log_path = DATA_DIR / "run.log"
    global _log_file
    _log_file = open(log_path, "w")

    try:
        # -- Check catboost version early --
        import catboost
        log(f"catboost version: {catboost.__version__}")
        log(f"F2 harness start — {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"REPO_ROOT: {REPO_ROOT}")
        log(f"BINARY:    {BINARY}")
        log(f"Anchor: N={ANCHOR_N}, seed={ANCHOR_SEED}, feats={ANCHOR_FEATS}")
        log(f"Config: depth={DEPTH}, bins={BINS}, lr={LR}, l2={L2}, loss={LOSS}, score_fn={SCORE_FN}")
        log(f"Target tree index (0-based): {TARGET_TREE_IDX}  (= iter=2, the 2nd tree built)")

        # -- Wall time check --
        wall_start = time.perf_counter()

        # -- Step 0: anchor data --
        log_section("Step 0: Generate anchor data")
        import numpy as np
        X, y = make_anchor()
        anchor_csv = DATA_DIR / "anchor_n1000_seed42.csv"
        write_anchor_csv(anchor_csv, X, y)
        log(f"Anchor CSV written: {anchor_csv}")

        # md5 comparison with PROBE-G's anchor
        def md5_file(p: Path) -> str:
            h = hashlib.md5()
            with open(p, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            return h.hexdigest()

        f2_md5 = md5_file(anchor_csv)
        log(f"F2 anchor md5:      {f2_md5}")

        probe_g_anchor = PROBE_G_DATA / "anchor_n1000_seed42.csv"
        if probe_g_anchor.exists():
            pg_md5 = md5_file(probe_g_anchor)
            log(f"PROBE-G anchor md5: {pg_md5}")
            if f2_md5 == pg_md5:
                log("BYTE-IDENTICAL: YES — anchor files match.")
            else:
                log("BYTE-IDENTICAL: NO — anchor files differ! Halting.")
                log("  This means the data generation code diverged from PROBE-G.")
                return 1
        else:
            log(f"PROBE-G anchor not found at {probe_g_anchor} — skipping md5 cross-check.")

        # -- Step 1: CPU CatBoost --
        cpu_splits = run_cpu(X, y)

        # -- Step 2: MLX csv_train --
        mlx_splits = run_mlx(anchor_csv)

        # -- Step 3: Quantization check --
        check_quantization_policy(anchor_csv)

        # -- Step 4: Comparison --
        rows = build_comparison(cpu_splits, mlx_splits)
        print_comparison_table(rows)

        comparison_csv = DATA_DIR / "comparison.csv"
        write_comparison_csv(rows, comparison_csv)
        log(f"\nComparison CSV written: {comparison_csv}")

        # -- Wall time check --
        wall_elapsed = time.perf_counter() - wall_start
        log(f"\nTotal wall time: {wall_elapsed:.1f}s")
        if wall_elapsed > 300:
            log("WARNING: Wall time exceeded 300s budget.")

        # -- File inventory --
        log_section("File inventory")
        for p in sorted(DATA_DIR.iterdir()):
            if p.is_file():
                size_kb = p.stat().st_size / 1024
                log(f"  {p.name:<40} {size_kb:>8.1f} KB")

        log(f"\nF2 harness complete — {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return 0

    except Exception as exc:
        log(f"\nFATAL: {type(exc).__name__}: {exc}")
        import traceback
        log(traceback.format_exc())
        return 1

    finally:
        if _log_file is not None:
            _log_file.close()


if __name__ == "__main__":
    sys.exit(main())
