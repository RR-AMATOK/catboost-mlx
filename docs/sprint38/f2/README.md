# S38-F2: Cross-Runtime Tree-Split Comparison

**Purpose.** F2 is the cheap discriminator that answers whether CPU CatBoost and
MLX csv_train choose the SAME split at every depth of the iter=2 tree, at the
PROBE-G anchor (N=1000, seed=42, ST/Cosine/RMSE).

PROBE-G showed that d=2 skip rates match PROBE-E (Scenario C at d<=2) but
structurally cannot prove MLX-vs-CPU equivalence because its instrumentation
compares the *post-fix MLX formula* to a *pre-fix counterfactual* — not to the
real CPU CatBoost split decision. F2 closes that gap directly.

## Discriminator routing

| Outcome | Meaning | Next step |
|---------|---------|-----------|
| All 6 depths match (feat AND bin) | DEC-042 closed for d=2 cross-runtime | PROBE-I (precision/leaf-values) |
| Any depth differs | DEC-042 per-side formula != CPU CalcScoreOnSide | PROBE-H (CPU per-side instrumentation) |

## Reproducibility

```
python docs/sprint38/f2/scripts/run_f2.py
```

Requirements: `catboost` Python package, `csv_train_probe_g` binary at repo root
(falls back to `csv_train`). If the binary is missing:

```
bash docs/sprint38/probe-g/scripts/build_probe_g.sh
```

## Output inventory

All files written under `docs/sprint38/f2/data/`:

| File | Description |
|------|-------------|
| `anchor_n1000_seed42.csv` | Anchor data (byte-identical to PROBE-G's) |
| `cpu_model.json` | CatBoost CPU model in CatBoost JSON format |
| `cpu_iter2_splits.json` | iter=2 (trees[1]) per-depth splits from CPU model |
| `mlx_model.json` | MLX csv_train model in catboost-mlx-json format |
| `mlx_iter2_splits.json` | iter=2 (trees[1]) per-depth splits from MLX model |
| `comparison.csv` | Side-by-side depth/feat/bin/border/match table |
| `run.log` | Combined stdout+stderr log of the full run |

## Anchor parameters (must match PROBE-G exactly)

- `np.random.default_rng(42)`, N=1000, 20 features
- `y = 0.5*X[:,0] + 0.3*X[:,1] + 0.1 * rng.standard_normal(N)` (float32)
- ST/Cosine/RMSE, depth=6, bins=128, l2=3, lr=0.03, iters=2

## Quantization note

CPU uses CatBoost's GreedyLogSum border selection (default `feature_border_type`).
MLX uses an independent GreedyLogSum port in `csv_train.cpp:GreedyLogSumBestSplit`
(shipped in Sprint 31, S31-T2-PORT-GREEDYLOGSUM). F2 checks that border counts and
values for features 0 and 1 are consistent; any mismatch would make bin-index
comparison invalid and is reported as a quantization warning.

The `comparison.csv` compares `cpu_bin` vs `mlx_bin` using per-feature border arrays
from each respective model JSON as the authoritative bin-to-border mapping — bin
indices are not compared blindly across models if their border arrays differ.
