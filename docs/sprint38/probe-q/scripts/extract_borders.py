#!/usr/bin/env python3
"""
PROBE-Q Step 1 & 2: Extract full pre-training border sets for CPU CatBoost and MLX.

CPU side:
  1. Generate anchor data (N=1000, seed=42, 20 features — identical to F2/PROBE-G).
  2. Build catboost.Pool and quantize with border_count=128.
  3. Use pool.save_quantization_borders() to dump the full pre-training border grid.
  4. Parse into per-feature dict; save as data/cpu_borders_full.json.

MLX side:
  1. Read docs/sprint38/f2/data/mlx_model.json.
  2. Extract the 'borders' array for each of the 20 features.
  3. Save as data/mlx_borders_full.json.

Border-count notes:
  - CPU: border_count=128 → CatBoost produces 128 borders (defines 129 bins).
  - MLX: --bins 128 → csv_train.cpp caps maxBordersCount = min(128, 127) = 127
    (DEC-039 fix). So MLX produces 127 borders (defines 128 bins = T2_BIN_CAP).
  The off-by-one is intentional on MLX's side to avoid aliasing bin_value=128
  with the VALID_BIT in the histogram kernel's packed-byte scheme.

Quantization algorithm:
  - CPU: CatBoost default = GreedyLogSum (TGreedyBinarizer<MaxSumLog>).
  - MLX: csv_train.cpp GreedyLogSumBestSplit — port of the same algorithm
    (DEC-037 border fix, DEC-038 allVals-with-duplicates fix).
  Both use the unweighted document-count path.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
PROBE_Q_DIR = REPO_ROOT / "docs" / "sprint38" / "probe-q"
DATA_DIR = PROBE_Q_DIR / "data"
F2_DATA_DIR = REPO_ROOT / "docs" / "sprint38" / "f2" / "data"

# Anchor parameters — must match F2 / PROBE-G exactly
ANCHOR_SEED = 42
ANCHOR_N = 1_000
ANCHOR_FEATS = 20
BORDER_COUNT = 128  # CatBoost border_count param


def make_anchor() -> tuple[np.ndarray, np.ndarray]:
    """Canonical anchor: 20 features, y = 0.5 X[0] + 0.3 X[1] + 0.1 noise (fp32)."""
    rng = np.random.default_rng(ANCHOR_SEED)
    X = rng.standard_normal((ANCHOR_N, ANCHOR_FEATS)).astype(np.float32)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(ANCHOR_N) * 0.1).astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Step 1: CPU border extraction
# ---------------------------------------------------------------------------

def extract_cpu_borders(X: np.ndarray, y: np.ndarray) -> dict[str, list[float]]:
    """Quantize the pool with CatBoost and extract the full border grid.

    Uses pool.quantize(border_count=128) + pool.save_quantization_borders().
    The save file format is tab-separated lines: "<feat_index>\\t<border_value>".
    """
    try:
        import catboost
    except ImportError:
        print("ERROR: catboost not installed — run: pip install catboost", file=sys.stderr)
        sys.exit(1)

    pool = catboost.Pool(X, y)
    pool.quantize(border_count=BORDER_COUNT)

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as tf:
        borders_path = tf.name

    try:
        pool.save_quantization_borders(borders_path)
        borders: dict[str, list[float]] = defaultdict(list)
        with open(borders_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                feat_idx = int(parts[0])
                border_val = float(parts[1])
                borders[f"feat_{feat_idx}"].append(border_val)
    finally:
        os.unlink(borders_path)

    # Sort ascending within each feature (save_quantization_borders may not guarantee order)
    result: dict[str, list[float]] = {}
    for fi in range(ANCHOR_FEATS):
        key = f"feat_{fi}"
        vals = sorted(borders.get(key, []))
        result[key] = vals
        print(f"  CPU feat_{fi}: {len(vals)} borders  "
              f"[{vals[0]:.6f} .. {vals[-1]:.6f}]" if vals else f"  CPU feat_{fi}: 0 borders")

    return result


# ---------------------------------------------------------------------------
# Step 2: MLX border extraction
# ---------------------------------------------------------------------------

def extract_mlx_borders() -> dict[str, list[float]]:
    """Read full 127-border grid from mlx_model.json.

    Each feature entry has a 'borders' array. This is the grid that csv_train
    computed at quantization time and stored in the model JSON.
    """
    mlx_model_path = F2_DATA_DIR / "mlx_model.json"
    if not mlx_model_path.exists():
        print(f"ERROR: {mlx_model_path} not found.", file=sys.stderr)
        sys.exit(1)

    with open(mlx_model_path) as f:
        model = json.load(f)

    features = model.get("features", [])
    if len(features) != ANCHOR_FEATS:
        print(f"WARNING: expected {ANCHOR_FEATS} features, got {len(features)}", file=sys.stderr)

    result: dict[str, list[float]] = {}
    for feat in features:
        fi = feat["index"]
        borders = sorted(feat.get("borders", []))
        key = f"feat_{fi}"
        result[key] = borders
        print(f"  MLX feat_{fi}: {len(borders)} borders  "
              f"[{borders[0]:.6f} .. {borders[-1]:.6f}]" if borders else f"  MLX feat_{fi}: 0 borders")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[extract_borders] Anchor: N={ANCHOR_N}, seed={ANCHOR_SEED}, feats={ANCHOR_FEATS}")

    X, y = make_anchor()
    print(f"[extract_borders] Generated anchor data: X={X.shape}, y={y.shape}, dtype={X.dtype}")

    # --- CPU ---
    print(f"\n[extract_borders] Step 1: CPU CatBoost borders (border_count={BORDER_COUNT})...")
    cpu_borders = extract_cpu_borders(X, y)
    cpu_out = DATA_DIR / "cpu_borders_full.json"
    with open(cpu_out, "w") as f:
        json.dump(cpu_borders, f, indent=2)
    print(f"\n[extract_borders] Saved CPU borders -> {cpu_out}")

    # Cardinality summary
    n_cpu = {k: len(v) for k, v in cpu_borders.items()}
    unique_counts = set(n_cpu.values())
    print(f"[extract_borders] CPU cardinality: {unique_counts} (expected: {{128}})")

    # --- MLX ---
    print(f"\n[extract_borders] Step 2: MLX borders from mlx_model.json...")
    mlx_borders = extract_mlx_borders()
    mlx_out = DATA_DIR / "mlx_borders_full.json"
    with open(mlx_out, "w") as f:
        json.dump(mlx_borders, f, indent=2)
    print(f"\n[extract_borders] Saved MLX borders -> {mlx_out}")

    n_mlx = {k: len(v) for k, v in mlx_borders.items()}
    unique_counts_mlx = set(n_mlx.values())
    print(f"[extract_borders] MLX cardinality: {unique_counts_mlx} (expected: {{127}})")

    # Quick sanity: verify the known ULP-match at feat_0, border 0.10254748
    feat0_mlx = mlx_borders.get("feat_0", [])
    known_border = 0.10254748
    close = [b for b in feat0_mlx if abs(b - known_border) < 1e-6]
    print(f"\n[extract_borders] Sanity: MLX feat_0 border near {known_border} = {close}")

    feat0_cpu = cpu_borders.get("feat_0", [])
    close_cpu = [b for b in feat0_cpu if abs(b - known_border) < 1e-6]
    print(f"[extract_borders] Sanity: CPU feat_0 border near {known_border} = {close_cpu}")

    print("\n[extract_borders] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
