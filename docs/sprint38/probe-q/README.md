# PROBE-Q: Pre-Training Border Set Comparison (CPU vs MLX)

**Sprint 38 — 2026-04-25**
**Branch**: `mlx/sprint-38-lg-small-n`

## Purpose

Determine whether the 13.93% N=1k drift between MLX and CPU CatBoost is caused
by different quantization grids — i.e., whether CPU and MLX evaluate fundamentally
different split thresholds during training.

The question: do CPU's pre-training borders (128 per feature, GreedyLogSum) align
with MLX's pre-training borders (127 per feature, GreedyLogSum port)?

## Anchor

Identical to F2 / PROBE-G / PROBE-H:
- `np.random.default_rng(42)`, N=1000, 20 features
- `y = 0.5*X[0] + 0.3*X[1] + 0.1*noise` (standard normal, float32)
- ST/Cosine/RMSE, depth=6, bins=128 (border_count), l2=3, lr=0.03

## How to Reproduce

```bash
# From repo root on branch mlx/sprint-38-lg-small-n
python3 docs/sprint38/probe-q/scripts/extract_borders.py
python3 docs/sprint38/probe-q/scripts/compare_borders.py
```

Requirements: `catboost`, `numpy` (standard conda env).

Runtime: < 5 seconds total.

## What the Outputs Mean

| File | Description |
|------|-------------|
| `data/cpu_borders_full.json` | 128 borders per feature from CatBoost `pool.quantize(border_count=128)` |
| `data/mlx_borders_full.json` | 127 borders per feature from `docs/sprint38/f2/data/mlx_model.json` |
| `data/border_comparison.csv` | Per-feature stats: n_cpu, n_mlx, max_delta, mean_delta, n_close, n_far |
| `data/feat0_used_borders_alignment.csv` | For each F2 iter-2 CPU-stored border on feat_0, the nearest MLX border and delta |

### Alignment definition

- `n_close`: MLX borders with nearest-CPU distance < 1e-4
- `n_far`: MLX borders with nearest-CPU distance > 1e-2
- `ALIGNED`: max_delta < 1e-4 (all MLX borders have a CPU match within 1e-4)
- `DIVERGED`: at least one border pair with delta > 1e-2

### Cardinality note

CPU uses `border_count=128` → produces 128 borders (defines 129 bins).
MLX uses `--bins 128` → csv_train.cpp caps `maxBordersCount = min(128, 127) = 127`
(DEC-039 fix for histogram kernel T2_BIN_CAP contract) → produces 127 borders
(defines 128 bins). The off-by-one is intentional and structural. The 128th CPU
border has no MLX counterpart; for the comparison we nearest-match each of the
127 MLX borders to the 128-element CPU grid.

## Result Summary

- **CPU cardinality**: 128 borders per feature (all 20 features)
- **MLX cardinality**: 127 borders per feature (all 20 features)
- **Aligned features**: 20/20 — every MLX border has a CPU match within 5e-8
- **Diverged features**: 0/20 — no feature has any border pair with delta > 1e-2
- **Overall max_delta**: 5.0e-08 (float32 ULP-level rounding, not algorithmic divergence)
- **mean_delta** (typical): ~9-12e-09
- **Verdict**: **Q-ALIGNED**

The deltas are at the float32 representation boundary (~half a ULP for border
values in the range [-2.7, 2.6]). They reflect the difference between CatBoost
storing borders as float64 internally (then printing with more digits) and MLX
storing borders as float32 in the JSON. The quantization algorithm itself is
identical.

## Implication

Granularity (different quantization grids) is **NOT** the mechanism behind the
13.93% N=1k drift. Both sides evaluate essentially the same split thresholds.

The one structural difference — CPU has 128 borders, MLX has 127 — means CPU
evaluates one additional split per feature (the 128th border). This is a
systematic but very minor constraint; at most it costs one split candidate per
feature per depth, and that candidate would be in the tail of the distribution.
It cannot explain a 13.93% RMSE drift.

The d=4-5 divergence documented in PROBE-H must come from a different mechanism.
Candidates to investigate in Sprint 38+:
1. **basePred initialization**: MLX vs CPU may compute the base prediction
   differently for small N (documented in mlx_model.json: `base_prediction=0.01382460445`
   vs CPU's value — to be confirmed).
2. **Leaf value computation**: Newton step on small partitions may behave
   differently (floating-point sums over few documents).
3. **Score-function numerical differences**: Cosine gain accumulation at small N
   is more sensitive to per-document weight rounding.
