# F2 Finding — CPU-Tree Split Comparison at N=1k seed=42

**Date**: 2026-04-25
**Branch**: `mlx/sprint-38-lg-small-n`
**Anchor**: `np.random.default_rng(42)`, **N=1000**, 20 features,
y = 0.5·X[0] + 0.3·X[1] + 0.1·noise, ST/Cosine/RMSE, depth=6, bins=128, l2=3, lr=0.03
**Anchor CSV md5**: `0976a75cb621a74e55eb9480d35935b3` (`docs/sprint38/probe-g/data/anchor_n1000_seed42.csv`)
**Status**: COMPLETE — verdict C-PSF confirmed; C-QG falsified; C-LV falsified. Routes to **PROBE-H**.

---

## Why this probe

PROBE-G (2026-04-25) classified the small-N residual as Scenario C at d≤2, with d≥3 in an
unidentified regime. Its amended verdict noted that PROBE-G was MLX-internal and never observed
CPU CatBoost's actual runtime split choices. F2 is the cheap discriminator opened in
`docs/sprint38/probe-g/FINDING.md §Recommended next step`:

- If CPU and MLX both pick `(feat=0, bin≈21)` at d=2 → DEC-042 is equivalent to CPU at d=2 →
  open **PROBE-I** (precision/leaf-value at d≥3). Do NOT open PROBE-H.
- If CPU differs at d=2 → DEC-042's per-side formula diverges from CPU's `CalcScoreOnSide` →
  open **PROBE-H**.

The pre-F2 routing table also included a third candidate cause (C-QG: quantization grid
divergence) that required disambiguation using the model JSON files from both runtimes.

---

## Method

Standard `catboost` Python package (CatBoost 1.2.10, CPU, `thread_count=1`) ran on the
N=1k seed=42 anchor with identical hyperparameters (`depth=6, border_count=128, l2_leaf_reg=3,
learning_rate=0.03, grow_policy=SymmetricTree, score_function=Cosine, loss_function=RMSE,
iterations=2, random_seed=42, random_strength=0, bootstrap_type=No`). Both runtimes
were trained to `iterations=2`; the iter=2 tree is `oblivious_trees[1]` (0-indexed) in
`cpu_model.json` and `trees[1]` in `mlx_model.json`. The iter=1 tree is `oblivious_trees[0]`
and `trees[0]` respectively.

Comparison: per-depth split match on `(feat_index, border_value)`. Border values are compared
using `|cpu_border - mlx_border| < 1e-4` as the "close" criterion. Bin indices are NOT
directly comparable (CPU reports the index within its own sparse border list; MLX reports
the index within its 127-border full grid).

---

## Result — iter=2 splits

From `data/comparison.csv`:

| depth | cpu_feat | cpu_bin | cpu_border | mlx_feat | mlx_bin | mlx_border | feat_match | border_close | MATCH |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 2 | 0.10254748 | 0 | 73 | 0.18157052 | Y | N | NO |
| 1 | 1 | 2 | 0.00869883 | 1 | 91 | 0.65761542 | Y | N | NO |
| 2 | 0 | 0 | -1.046514 | 0 | 45 | -0.35235262 | Y | N | NO |
| 3 | 0 | 3 | 0.93405426 | 3 | 87 | 0.54884368 | N | N | NO |
| 4 | 1 | 4 | 0.96013570 | 4 | 102 | 0.78292483 | N | N | NO |
| 5 | 1 | 0 | -0.89314175 | 14 | 71 | 0.14395939 | N | N | NO |

**Feat match: 3/6 (d=0–2). Full match (feat + border close): 0/6.**

Note on the comparison.csv `cpu_bin` column: CPU serializes only the borders that appear in
splits (`float_features[i].borders`). feat=0 has 6 borders (split_index 0–5); feat=1 has 5
borders (split_index 6–10 in the flat array). The `cpu_bin` column in the CSV counts within
the CPU's own sparse list, which is NOT the same coordinate as `mlx_bin` (0-indexed within
the 127-border MLX grid). Border values (not bin indices) are the primary discriminator.

---

## Result — iter=1 splits (bisection: constant basePred, no leaf-value cascade possible)

Extracted from `cpu_model.json: oblivious_trees[0]` and `mlx_model.json: trees[0]`.
MLX border values resolved by looking up `mlx_borders[bin_threshold]` for each split.

| depth | cpu_feat | cpu_border | mlx_feat | mlx_bin | mlx_border | feat_match | border_close | MATCH |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.10254748 | 0 | 69 | 0.10254748 | Y | Y | YES |
| 1 | 1 | 0.43849730 | 1 | 64 | 0.09566100 | Y | N | NO |
| 2 | 0 | -0.81075025 | 0 | 29 | -0.70789814 | Y | N | NO |
| 3 | 0 | 1.03529096 | 15 | 28 | -0.86897051 | N | N | NO |
| 4 | 1 | -0.80015314 | 1 | 23 | -0.89314175 | Y | N | NO |
| 5 | 0 | 1.74658537 | 0 | 98 | 0.74944925 | Y | N | NO |

**iter=1 full match: 1/6 (d=0 only). Feat match: 5/6 (d=3 differs).**

**Critical implication**: iter=1 is computed from the constant basePred (= mean of y =
0.01382460). There are no iter=0 leaf values to cascade — any split divergence at iter=1
is purely a split-selection or quantization issue. The 5/6 iter=1 mismatch proves that
**C-LV (leaf-value cascade) is not necessary to explain the split divergence**. The
divergence mechanism operates from the very first tree and is upstream of any learned leaf
values. C-LV is falsified.

The single d=0 match at iter=1 is notable: CPU picks (feat=0, border=0.10254748) and MLX
picks (feat=0, bin=69, border=0.10254748). The borders are ULP-identical — this is the same
split by any measure. This shows the two runtimes CAN agree; the d=1–5 disagreements are
systematic, not random.

---

## The quantization confound

CPU CatBoost 1.2.10 serializes only the borders for features that appear in splits.
From `cpu_model.json: features_info.float_features`:

- **feat 0**: 6 borders: `[-1.04651403, -0.81075025, 0.10254748, 0.93405426, 1.03529096, 1.74658537]`
- **feat 1**: 5 borders: `[-0.89314175, -0.80015314, 0.00869883, 0.43849730, 0.96013570]`
- **feats 2–19**: empty (0 borders each)

MLX serializes the full 127-border grid for all 20 features. From `mlx_model.json: features[0].borders`
and `features[1].borders`:

- **feat 0**: 127 borders spanning `[-2.3936062, ..., 2.5203915]`
- **feat 1**: 127 borders spanning `[-2.521699, ..., 2.353653]`

Bin indices are therefore NOT directly comparable across runtimes. The `cpu_bin` and `mlx_bin`
columns in the comparison table are each local to their own runtime's border array.

---

## Disambiguation — Test 1 (border grid alignment) and Test 2 (which border MLX picks)

### Test 1: Are CPU's borders present in the MLX grid?

For each CPU border value, find the nearest value in the MLX grid and measure the distance.
**All 11 CPU borders (6 for feat 0, 5 for feat 1) are present in the MLX grid to within 3.5e-8.**

| CPU border | Feature | Nearest MLX index | Nearest MLX value | Distance |
|---|---|---|---|---|
| -1.04651403 | 0 | 18 | -1.04651400 | 3.43e-08 |
| -0.81075025 | 0 | 25 | -0.81075025 | 3.95e-09 |
| 0.10254748 | 0 | 69 | 0.10254748 | 1.66e-09 |
| 0.93405426 | 0 | 104 | 0.93405426 | 4.51e-09 |
| 1.03529096 | 0 | 106 | 1.03529100 | 4.35e-08 |
| 1.74658537 | 0 | 122 | 1.74658540 | 3.09e-08 |
| -0.89314175 | 1 | 23 | -0.89314175 | 3.48e-09 |
| -0.80015314 | 1 | 27 | -0.80015314 | 3.75e-09 |
| 0.00869883 | 1 | 60 | 0.00869883 | 4.43e-11 |
| 0.43849730 | 1 | 82 | 0.43849730 | 4.92e-09 |
| 0.96013570 | 1 | 102 | 0.96013570 | 1.68e-09 |

**Test 1 result: C-QG FALSIFIED.** The quantization algorithms (CPU's `GreedyLogSum` +
`border_count=128` and MLX's `csv_train.cpp` static 127-border grid) produce the same border
values at ULP-level agreement. The borders the CPU selects ARE available to MLX — the grids
are aligned. The confusion about 127-vs-128 (DEC-039) affects grid SIZE, not the values of the
borders that do exist.

### Test 2: Does MLX pick the nearest-to-CPU border, or a systematically different one?

For each of the three feature-matched iter=2 depths, compare the MLX-picked border to the
nearest MLX border to CPU's pick:

**d=0** (feat=0):
- CPU picks border `0.10254748` (MLX grid index 69).
- MLX picks bin=73, border `0.18157052`.
- |MLX-picked − CPU-border| = **0.0790**.
- MLX bin 69 (border `0.10254748`) exists and matches CPU exactly — MLX does NOT pick it.

**d=1** (feat=1):
- CPU picks border `0.00869883` (MLX grid index 60).
- MLX picks bin=91, border `0.65761542`.
- |MLX-picked − CPU-border| = **0.6489** (31 bins away from CPU's equivalent).
- MLX bin 60 (border `0.00869883`) exists and matches CPU exactly — MLX does NOT pick it.

**d=2** (feat=0):
- CPU picks border `-1.04651403` (MLX grid index 18).
- MLX picks bin=45, border `-0.35235262`.
- |MLX-picked − CPU-border| = **0.6942** (27 bins away from CPU's equivalent).
- MLX bin 18 (border `-1.04651403`) exists and matches CPU exactly — MLX does NOT pick it.

**Test 2 result: C-PSF CONFIRMED.** In all three feature-matched depths, the border that
CPU selects exists verbatim in the MLX grid, and MLX selects a different one — in one case
31 bins away, in another 27 bins away. This is not a grid-availability problem. MLX has CPU's
preferred border in its search space and ranks it lower. The scoring formula, not the border
pool, is the source of disagreement.

---

## Disambiguation summary

| Cause | Mechanism | Test | Result |
|---|---|---|---|
| **C-QG** (quantization grids differ) | MLX lacks CPU's preferred borders | Test 1: CPU borders in MLX grid? | **FALSIFIED** — all 11 borders present to 3.5e-8 |
| **C-LV** (leaf-value cascade) | iter=0 leaf values shift gradients → different iter=1 splits | iter=1 comparison from constant basePred | **FALSIFIED** — 5/6 mismatches at iter=1 from constant basePred |
| **C-PSF** (per-side formula divergence) | MLX scores bins differently than CPU's `CalcScoreOnSide` | Test 2: does MLX pick nearest-to-CPU border? | **CONFIRMED** — MLX ignores CPU's border (0–31 bins away) and picks a different one |

**C-PSF is the primary cause. The quantization grid and leaf-value cascade hypotheses are ruled out.**

---

## Verdict

**Route: PROBE-H.**

The pre-F2 routing rule was:
> "CPU differs at d=2 → DEC-042's per-side formula ≠ CPU's `CalcScoreOnSide` → PROBE-H."

At d=2, CPU picks `(feat=0, border=-1.04651403)` and MLX picks `(feat=0, bin=45, border=-0.35235262)`.
The CPU-equivalent border exists in MLX's grid at index 18 and MLX scores it lower. The
routing rule is triggered. PROBE-Q (quantization border alignment) is NOT needed — Test 1
showed the grids are aligned.

The iter=1 result further strengthens this: with only 1/6 matches from constant basePred,
the divergence is not a later-iteration effect. The scoring formula is wrong from the first
tree.

---

## Recommended next step — PROBE-H

**PROBE-H goal**: at the same N=1k seed=42 anchor, capture CPU CatBoost's runtime per-side
per-bin gain for iter=1 AND iter=2 (since both iterations show divergence) and compare against
MLX's post-DEC-042 picks. This is a cross-runtime comparison, not an MLX-internal counterfactual.

**New instrumentation needed**: a CPU-side hook in CatBoost's score calculator — specifically
in `catboost/private/libs/algo/score_calcers.cpp` inside `TCosineScoreCalcer::CalcMetric`
or the per-partition reducer `UpdateScoreBinKernelPlain` in
`catboost/libs/helpers/short_vector_ops.h:155+`. The hook should emit the same
`(feat, bin, partition, cosNum, cosDen, gain)` tuples that MLX emits via
`PROBE_E_INSTRUMENT`, armed at the same depth range (d=0..5) and anchor configuration.

**Prioritize iter=1 first**: since iter=1 diverges from constant basePred, the comparison
does not depend on any leaf-value state. iter=1 is the cleanest signal for a formula-level
divergence. If iter=1 is fully explained by the formula difference, iter=2 follows.

**Surprising structural insight**: at d=0 of iter=1, both runtimes pick `(feat=0, border=0.10254748)`
exactly. This means the runtimes agree when the scoring landscape is flat enough that the
top candidate is unambiguous. At d=1 they diverge (CPU: feat=1, border=0.43849730; MLX:
feat=1, bin=64, border=0.09566100). The formula divergence is gain-ranking sensitive — it
shifts the argmax but does not always do so.

**Estimate**: ~1 sprint (instrumentation ~2 days, anchor capture ~0.5 day, classification
~0.5 day).

**Directory**: `docs/sprint38/probe-h/` or `docs/sprint39/probe-h/` — Ramos decides sprint
boundary.

---

## Limitations

- **Single seed (42)**. The 0/6 iter=2 mismatch and 5/6 iter=1 mismatch are from one random
  draw. Multi-seed confirmation is warranted before treating the C-PSF verdict as definitive.
  However, the Test 1 (grid alignment) and Test 2 (nearest-border analysis) conclusions are
  deterministic given the two model files and are not seed-dependent.
- **Single anchor configuration** (N=1k, depth=6, ST+Cosine, RMSE). The verdict may not
  generalize across grow policies (though the same formula path runs for LG under post-DEC-042
  FBSPP fix). The 13.93% aggregate drift measured at N=1k is over 5 seeds and 50 iterations;
  the F2 anchor is iter=2 only.
- **iter=2 only in comparison.csv**. iter=1 was extracted manually from the model JSON files
  rather than from a purpose-built harness. The iter=1 analysis should be treated as
  confirmatory evidence for C-LV falsification, not as a primary result.
- **PROBE-G postfix vs comparison.csv**: PROBE-G's postfix column captured MLX picks in
  isolation (`PROBE_E_INSTRUMENT` armed at iter=2 only). The comparison.csv was built by the
  F2 harness independently. Both show MLX at iter=2 picking `(feat=0, bin=73, border=0.18157052)`
  at d=0 — consistent, confirming no harness artifact.

---

## Artifacts

All files under `docs/sprint38/f2/data/`:

- `comparison.csv` — 6-row iter=2 comparison table (primary F2 result)
- `cpu_model.json` — full CatBoost model JSON (both trees, borders, params)
- `mlx_model.json` — full MLX model JSON (both trees, 127-border grids, split gains)
- `cpu_iter2_splits.json` — extracted iter=2 CPU splits (auxiliary)
- `mlx_iter2_splits.json` — extracted iter=2 MLX splits (auxiliary)
- `anchor_n1000_seed42.csv` — anchor data (copy from probe-g/data/)
- `run.log` — F2 harness run log
- `cos_leaf_seed42_depth{0..5}.csv` — per-(feat, bin, partition) gain captures
- `cos_accum_seed42_depth{0..5}.csv` — fp32 vs fp64 accumulator shadows
- `leaf_sum_seed42.csv`, `approx_update_seed42.csv` — auxiliary per-iter dumps
