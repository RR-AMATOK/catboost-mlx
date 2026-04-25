# Fix-Verify Finding — DEC-044 Joint-Skip Fix Verification

**Date**: 2026-04-25
**Branch**: `mlx/sprint-38-lg-small-n`
**Task**: Apply joint-skip → per-side independent accumulation fix in `FindBestSplit` ordinal
Cosine branch, verify via F2 and N=1k drift sweep.

> **Status**: COMPLETE — fix audit performed. **Verdict: NO CODE CHANGE REQUIRED.**
> The per-side mask fix (DEC-042, Sprint 33 commit `10c72b4e96`) was already present in
> the current source code and already compiled into the production binary. The rebuild
> from current source produced no change to tree structure or drift. The 13.93% drift
> at N=1k ST+Cosine is real and is driven by a mechanism OTHER than the joint-skip formula.

---

## Summary

PROBE-H's FINDING.md reported that `csv_train.cpp` `FindBestSplit` ordinal Cosine still used
the old joint-skip formula (`if (!wL_pos || !wR_pos) break`), and proposed a ~10-line fix.

Audit of the current source at HEAD (`4e56ddf6a3`) shows:

1. `FindBestSplit` ordinal Cosine (line 2078): already uses per-side mask — `if (!wL_pos && !wR_pos) break` with independent `if (wL_pos) {...}` and `if (wR_pos) {...}` blocks. Fixed in Sprint 33, commit `10c72b4e96` ("S33-L4-FIX Commit 1").

2. `FindBestSplitPerPartition` ordinal Cosine (line 2529): also uses per-side mask. Fixed in Sprint 38, commit `a481972529` (DEC-042 port).

3. OneHot Cosine paths (lines 1778 and 2413): intentionally retain joint-skip per S34-PROBE-F-LITE verdict. No change required.

The production binary (`csv_train`, Apr 23 2026) was compiled after Sprint 33's merge and already contained the S33-L4-FIX. Rebuilding from HEAD produced an identical binary (modulo timestamp).

---

## PROBE-H Analysis Discrepancy

PROBE-H's `divergence_iter1.csv` shows:
- Column `mlx_winner` = argmax under OLD joint-skip formula, applied in the Python analysis script.
- Column `picked` = what the PROBE-H binary actually picked at runtime.
- Column `cpu_winner` = argmax under CPU per-side formula, applied in the Python analysis script.

At d=2..5, `picked` ≠ `mlx_winner_old_formula`, which the FINDING.md interpreted as "the binary uses
the old formula." In fact it means the binary uses a DIFFERENT formula than what the analysis script
modeled as "mlx" — the binary uses the per-side mask (already fixed), not the old joint-skip.

At d=3, `picked` = feat=15 while `cpu_winner` = feat=0. This is a genuine divergence between
MLX and CPU, but it is NOT caused by the joint-skip formula. The residual gain under CPU formula
for the binary's pick (`picked_gain_under_cpu = 16.10`) is close to the CPU winner (`16.94`), but
not identical. The mechanism driving this ~0.84 gain difference is UNRELATED to joint-skip.

---

## Code Audit (no changes made)

Confirmed correct per-side mask in source at HEAD:

**FindBestSplit ordinal Cosine** (`catboost/mlx/tests/csv_train.cpp`, line 2078):
```cpp
if (!wL_pos && !wR_pos) break;      // skip only when BOTH sides empty
const double dL2   = static_cast<double>(l2RegLambda);
double termNum = 0.0;
double termDen = 0.0;
if (wL_pos) {
    const double dSL   = static_cast<double>(sumLeft);
    const double dWL   = static_cast<double>(weightLeft);
    const double dInvL = 1.0 / (dWL + dL2);
    termNum += dSL * dSL * dInvL;
    termDen += dSL * dSL * dWL * dInvL * dInvL;
}
if (wR_pos) {
    const double dSR   = static_cast<double>(sumRight);
    const double dWR   = static_cast<double>(weightRight);
    const double dInvR = 1.0 / (dWR + dL2);
    termNum += dSR * dSR * dInvR;
    termDen += dSR * dSR * dWR * dInvR * dInvR;
}
cosNum_d += termNum;
cosDen_d += termDen;
```

This exactly matches the CPU `UpdateScoreBinKernelPlain` formula and the DEC-042 target spec.

**FindBestSplitPerPartition ordinal Cosine** (`catboost/mlx/tests/csv_train.cpp`, line 2529):
identical pattern — per-side mask, independent contribution. Confirmed by DEC-042 commit `a481972529`.

---

## Build

Rebuilt production binary from HEAD (no source changes):

```
MLX_PREFIX=$(brew --prefix mlx)
clang++ -std=c++17 -O2 -I. -I${MLX_PREFIX}/include \
    -L${MLX_PREFIX}/lib -lmlx -framework Metal -framework Foundation \
    -Wno-c++20-extensions \
    catboost/mlx/tests/csv_train.cpp -o csv_train
```

- Build: clean, no warnings or errors.
- Binary timestamp: Apr 25 2026 16:37 (fresh rebuild).
- Kernel md5: `9edaef45b99b9db3e2717da93800e76f` — UNCHANGED.

---

## F2 Results — Pre-fix vs Post-fix

**Both runs used the same source (per-side mask already in place).** Results are byte-identical.

| depth | cpu_feat | cpu_bin | cpu_border   | mlx_pre_feat | mlx_pre_bin | mlx_pre_border | match_pre | mlx_post_feat | mlx_post_bin | mlx_post_border | match_post |
|-------|----------|---------|--------------|--------------|-------------|----------------|-----------|---------------|--------------|-----------------|------------|
| 0     | 0        | 2       | 0.10254748   | 0            | 73          | 0.18157052     | NO        | 0             | 73           | 0.18157052      | NO         |
| 1     | 1        | 2       | 0.00869883   | 1            | 91          | 0.65761542     | NO        | 1             | 91           | 0.65761542      | NO         |
| 2     | 0        | 0       | -1.04651403  | 0            | 45          | -0.35235262    | NO        | 0             | 45           | -0.35235262     | NO         |
| 3     | 0        | 3       | 0.93405426   | 3            | 87          | 0.54884368     | NO        | 3             | 87           | 0.54884368      | NO         |
| 4     | 1        | 4       | 0.96013570   | 4            | 102         | 0.78292483     | NO        | 4             | 102          | 0.78292483      | NO         |
| 5     | 1        | 0       | -0.89314175  | 14           | 71          | 0.14395939     | NO        | 14            | 71           | 0.14395939      | NO         |

F2 compares iter=2 (trees[1]) splits. 0/6 match pre-fix; 0/6 match post-fix. No change.

Note: F2 compares quantization-mismatched borders (CPU uses GreedyLogSum 6-border grid;
MLX uses uniform 127-border grid). The iter=1 tree (trees[0]) already shows feat-match
at 5/6 depths: feat 0,1,0,X,1,0 for MLX vs feat 0,1,0,0,1,0 for CPU — only d=3 diverges by
feature. This per-side fix contribution was already working.

---

## N=1k Drift Sweep

Seeds 42–46, N=1000, ST/Cosine/RMSE, iters=50, depth=6, bins=128, lr=0.03, l2=3.

| seed | mlx_rmse | cpu_rmse | drift_pct |
|------|----------|----------|-----------|
| 42   | 0.232996 | 0.204238 | 14.08%    |
| 43   | 0.229305 | 0.200425 | 14.41%    |
| 44   | 0.231307 | 0.204007 | 13.38%    |
| 45   | 0.233553 | 0.205136 | 13.85%    |
| 46   | 0.231817 | 0.203488 | 13.92%    |

**Aggregate drift (mean): 13.93%** — identical to PROBE-G baseline (13.96% with probe binary).
Gate threshold: 2.00%. **Gate: FAIL** — drift unchanged.

For reference, the PROBE-G scaling_sweep.csv at N=1k shows seed=42..46 drift as
14.08%, 14.41%, 13.38%, 13.85%, 13.92% — bit-for-bit identical to this sweep.

---

## Verdict

**The DEC-044 proposed fix is a NO-OP: the per-side mask was already in the source and binary.**

- The joint-skip formula was fixed in Sprint 33 (commit `10c72b4e96`) for `FindBestSplit` ordinal
  Cosine, and in Sprint 38 (commit `a481972529`) for `FindBestSplitPerPartition` ordinal Cosine.
- Both fixes are present in HEAD and in all compiled binaries since the Sprint 33 merge.
- The 13.93% drift at N=1k ST+Cosine is driven by a DIFFERENT mechanism.

---

## Implication for Sprint 39

The residual divergence at d=3 (feat=15 vs CPU feat=0 in iter=1) and the 13.93% drift
suggest a mechanism NOT captured by the formula fix. Candidates:

1. **Quantization border divergence**: CPU uses GreedyLogSum (dynamic, 6 borders for feat=0);
   MLX uses static uniform 127-border grid. Different borders → different split candidates →
   different argmax even with identical formulas. The F2 `WARNING: CPU has 6 borders, MLX has 127`
   confirms this mismatch persists.

2. **Random noise perturbation interaction**: `noiseScale * gradRms` adds noise to gain scores.
   At small N the gradient RMS is noisy, making perturbation large relative to gain differences.
   Different border grids mean different candidate orderings, so noise can flip the argmax.

3. **Leaf-value cascade**: even when iter=1 picks slightly different splits, leaf values accumulate
   differently, so iter=2's gradient residuals differ, leading to different iter=2 splits.

DEC-044 should be closed as INVALID (fix was already present). A new investigation targeting
the quantization border mismatch (C-QG hypothesis, partially addressed by F2's Scenario C) or
the random noise interaction at small N should be opened as a new ticket.

---

## Kernel md5 Invariant

`catboost/mlx/kernels/kernel_sources.h` md5: `9edaef45b99b9db3e2717da93800e76f`
— unchanged before and after rebuild. No kernel sources modified.

---

## Files

| File | Description |
|------|-------------|
| `data/comparison_postfix.csv` | F2 comparison.csv post-rebuild (identical to pre-fix) |
| `data/drift_n1k.csv` | N=1k seed=42..46 drift sweep with fresh production binary |
| `scripts/run_drift_n1k.py` | Drift sweep script |
| `FINDING.md` | This document |
