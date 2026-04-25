# S37 #113 — S30 T3 Gate Matrix Re-Run on Post-Fix Master

**Date**: 2026-04-25
**Branch**: `mlx/sprint-37-maintenance`
**Master tip**: `600e5b7285` (post-S33/S34/S35/S36 merge; DEC-042 closed)
**Binary**: `csv_train_t3` built with `-DCOSINE_T3_MEASURE`
**Kernel md5**: `9edaef45b99b9db3e2717da93800e76f` (unchanged)

## Verdict: PARTIAL — G3a PASS, G3b/G3c FAIL

The DEC-042 fix is **regime-dependent**. It collapsed N=50k drift as expected (53.30% → 1.27%). Small-N LG+Cosine (N=1k, N=2k) is essentially unchanged from pre-fix.

## Side-by-side

| Gate | Cell | Pre-fix (S30 verdict, 2026-04-24) | Post-fix (S37 re-run) | Threshold | Status |
|---|---|---|---|---|---|
| **G3a** | N=50k, depth=6, bins=128, ST/Cosine, 5 seeds | aggregate **53.30%** | aggregate **1.27%** | <2% | **PASS** (1941× → 42× improvement) |
| **G3b** | N=1k, depth=6, max_leaves=31, LG/Cosine, 5 seeds | ratios **1.274-1.314** (drift 27-31%) | ratios **1.274-1.311** (drift 27-31%) | [0.98, 1.02] | **FAIL** (no measurable improvement) |
| **G3c** | N=2k, depth=7, max_leaves=64, LG/Cosine, 3 seeds | ratios **1.440-1.452** (drift 44-45%) | ratios **1.432-1.446** (drift 43-45%) | [0.98, 1.02] | **FAIL** (no measurable improvement) |

## Per-seed detail

### G3a (post-fix)

| seed | MLX_RMSE | CPU_RMSE | ratio | drift |
|---|---|---|---|---|
| 42 | 0.19623200 | 0.19362645 | 1.013457 | 1.346% |
| 43 | 0.19601100 | 0.19357118 | 1.012604 | 1.260% |
| 44 | 0.19547900 | 0.19320460 | 1.011772 | 1.177% |
| 45 | 0.19513800 | 0.19250458 | 1.013680 | 1.368% |
| 46 | 0.19534000 | 0.19305704 | 1.011825 | 1.183% |

**Aggregate: 1.27% (was 53.30%)** — DEC-042 fix is working at this regime.

Note: G4b in S33 reported 0.027% on the same N=50k anchor. The difference is the seed (G4b seed=42, G3a 5-seed mean) and the harness internals — both are within the < 2% threshold; both confirm the fix.

### G3b (post-fix, N=1k)

| seed | MLX_RMSE | CPU_RMSE | ratio | drift |
|---|---|---|---|---|
| 42 | 0.24386000 | 0.18599921 | 1.311081 | 31.11% |
| 43 | 0.23797700 | 0.18680430 | 1.273938 | 27.39% |
| 44 | 0.24201000 | 0.18867654 | 1.282671 | 28.27% |
| 45 | 0.24460300 | 0.18713518 | 1.307093 | 30.71% |
| 46 | 0.24587000 | 0.18774096 | 1.309624 | 30.96% |

### G3c (post-fix, N=2k, depth=7)

| seed | MLX_RMSE | CPU_RMSE | ratio | drift |
|---|---|---|---|---|
| 0 | 0.13151200 | 0.09163246 | 1.435212 | 43.52% |
| 1 | 0.12796000 | 0.08936491 | 1.431882 | 43.19% |
| 2 | 0.13190300 | 0.09121844 | 1.446012 | 44.60% |

K2 conditional from DEC-035 fires (any G3c seed outside [0.98, 1.02]).

## Implication for S33 LG-guard removal

S33 Commit 3b (`d599e5b033`) removed the S28-LG-GUARD based on the iter=50 LG+Cosine drift measurement of **0.382%** — but that was on the **N=50k anchor**. Small-N LG+Cosine was not re-validated post-fix.

The S33 sprint-close documented the LG removal as conditional on the N=50k drift measurement. T3's re-run shows that gate was insufficient: the fix is real at large N, latent at small N.

## Possible mechanisms (hypotheses, not validated)

1. **Small-N partition-state regime**: at N=1k with depth=6 / max_leaves=31, average leaf size is ~32 docs. DEC-042's per-side mask helps when degenerate (p,k) partitions have meaningful signal in the non-empty side. At small leaf counts, the "non-empty side" may not have enough docs to produce a meaningful contribution either, and the joint cosine numerator/denominator picks up small-N noise.
2. **LG priority-queue × small-N interaction**: Lossguide's best-first leaf expansion at small N may concentrate drift in early splits where the cosine-gain ranking is most sensitive to per-bin floating-point noise.
3. **A second, independent mechanism** below the DEC-042 fix that surfaces at small N. PROBE-E only sampled the N=50k anchor; small-N counterfactual capture was never done.

## Recommended next step

Open **#130 S38-LG-SMALL-N-RESIDUAL** — investigate small-N LG+Cosine residual drift with the same math-first → code-first → empirical discipline used for S34/S35/S36:

- T0a (math): why does the per-side mask not collapse drift at small N? Is the parent-term-cancellation logic regime-dependent?
- T0b (code): trace the small-N path; is there a code branch that bypasses the DEC-042 fix?
- T1 (empirical): scaling probe at N ∈ {500, 1k, 2k, 5k, 10k, 50k} × LG/Cosine to find the boundary
- Decision: re-add LG+Cosine guard with N-threshold, or fix the small-N mechanism

Until S38 lands, **users running LG+Cosine at N < 10k will see significant drift vs CPU CatBoost**. Document this in `catboost/mlx/README.md` Known Limitations.

## Files

- `docs/sprint30/t3-measure/data/g3{a,b,c}_*.csv` — post-fix re-run data (overwrites S30 numbers; the sprint-close.md preserves the pre-fix history for audit)
- `docs/sprint37/t3-rerun/verdict.md` — this document
