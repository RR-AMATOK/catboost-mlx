# S38 T0b — Code Reading: `FindBestSplitPerPartition` degenerate-child handling

**Date**: 2026-04-25
**Branch**: `mlx/sprint-38-lg-small-n`
**File under investigation**: `catboost/mlx/tests/csv_train.cpp`
**Read-only. No source modifications.**

---

## 1. Loop nesting: per-bin loop placement, accumulator scopes

### One-hot branch

```
FindBestSplitPerPartition (L2251)
  for feat in features              (L2279)
    if feat.OneHotFeature           (L2283)
      for bin in [0, feat.Folds)    (L2284)
        for p in [0, numPartitions) (L2285)
          double gain    = 0.0      (L2290)   <-- scoped per-(bin, p)
          double cosNum_d = 0.0     (L2293)   <-- scoped per-(bin, p)
          double cosDen_d = 1e-20   (L2294)   <-- scoped per-(bin, p)
          for k in [0, K)           (L2295)
            compute left/right stats
            if (wL < 1e-15 || wR < 1e-15) continue  (L2304) <-- JOINT SKIP
            accumulate gain / cosNum_d / cosDen_d
          finalize Cosine gain      (L2331-2333)
          update bestGains[p]       (L2341)
```

`gain`, `cosNum_d`, and `cosDen_d` are **re-initialized to zero for every (bin, p) pair**. The `continue` at L2304 skips only the current `k` iteration within that (bin, p) pair — it does not skip the entire (bin, p) evaluation. The argmax is per-partition (one `bestGains[p]` per partition).

### Ordinal branch

```
FindBestSplitPerPartition (L2251)
  for feat in features                  (L2279)
    else (ordinal)                      (L2350)
      precompute suffix sums (L2353-2368)
      for bin in [0, feat.Folds-1)      (L2371)
        for p in [0, numPartitions)     (L2373)
          double gain    = 0.0          (L2375)   <-- scoped per-(bin, p)
          double cosNum_d = 0.0         (L2378)   <-- scoped per-(bin, p)
          double cosDen_d = 1e-20       (L2379)   <-- scoped per-(bin, p)
          for k in [0, K)               (L2380)
            compute left/right from suffix sums
            if (wL < 1e-15 || wR < 1e-15) continue  (L2388) <-- JOINT SKIP
            accumulate gain / cosNum_d / cosDen_d
          finalize Cosine gain          (L2413-2415)
          update bestGains[p]           (L2423)
```

Same structure as one-hot. The `continue` at L2388 skips the current `k` within the current (bin, p) — not the entire (bin, p). Accumulators are per-(bin, p).

**Contrast with `FindBestSplit` (SymmetricTree, ordinal):** There, `totalGain`, `cosNum_d`, and `cosDen_d` are scoped per-**bin** (L1823-L1830), and the inner loop is `for p ... for k` (L1867, L1889). The aggregation crosses all partitions for a single bin candidate before one argmax winner is chosen for the whole tree.

The structural difference is: `FindBestSplit` computes one global winner per bin (sum over all p*K); FBSPP computes one winner **per partition** (sum over K only, one p at a time). Both have the same innermost `k`-loop with the same joint-skip `continue`.

---

## 2. The `continue` line: location, scope, and skip behavior

### One-hot branch

```
csv_train.cpp:2304
    if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;
```

**Scope**: `continue` terminates the current `k` iteration within the `for k in [0, K)` loop (L2295-L2329). For K=1 (single-output regression), skipping `k=0` means `cosNum_d` and `cosDen_d` remain at their initialized values (0.0 and 1e-20 respectively) for this (bin, p) pair. The `gain` finalizer at L2331-2333 still executes, yielding `gain = 0.0 / sqrt(1e-20) ≈ 0.0` for Cosine, or `gain = 0.0` for L2. `bestGains[p]` is initialized to `-inf`, so a zero gain may or may not win depending on other candidates.

**Scope clarification**: this is NOT skipping the entire (bin, p) evaluation — it skips only the k-th dimension contribution. For K>1 with some k degenerate and others not, the surviving k dimensions do contribute. But for K=1 (the RMSE regression case under investigation), a single skip at k=0 nullifies the entire (bin, p)'s cosNum/cosDen.

### Ordinal branch

```
csv_train.cpp:2388
    if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;
```

**Scope**: identical to one-hot. Terminates the current `k` within `for k in [0, K)` (L2380-L2412). For K=1, this nullifies the entire (bin, p)'s contribution — cosNum_d stays 0.0, cosDen_d stays 1e-20, `gain` finalizes to ~0.0.

**Both branches retain the old joint-skip `continue` that DEC-042 fixed in `FindBestSplit`.**

---

## 3. Per-side accumulation post-S33: FBSPP vs `FindBestSplit`

`FindBestSplit` ordinal (post-S33-L4-FIX):

```cpp
// csv_train.cpp:1961-2016
const bool wL_pos = (weightLeft  > 1e-15f);
const bool wR_pos = (weightRight > 1e-15f);
// ...
case EScoreFunction::Cosine: {
    if (!wL_pos && !wR_pos) break;   // skip only when BOTH empty
    // ...
    if (wL_pos) { /* add left-side terms to cosNum_d, cosDen_d */ }
    if (wR_pos) { /* add right-side terms to cosNum_d, cosDen_d */ }
```

`FindBestSplitPerPartition` ordinal (current, L2380-L2412):

```cpp
// csv_train.cpp:2388
if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;  // skip if EITHER empty
// followed by both-sides-always accumulation:
cosNum_d += dSL*dSL*dInvL + dSR*dSR*dInvR;
cosDen_d += dSL*dSL*dWL*dInvL*dInvL + dSR*dSR*dWR*dInvR*dInvR;
```

**S33-L4-FIX did not touch FBSPP.** The per-side mask (`if (wL_pos)` / `if (wR_pos)`) introduced by DEC-042 Commits 1 (Cosine) and 1.5 (L2) is absent from FBSPP. FBSPP still uses the pre-DEC-042 joint-skip pattern in both the one-hot branch (L2304) and the ordinal branch (L2388).

---

## 4. Cosine path specifically: does FBSPP have the per-side mask?

**One-hot branch Cosine (L2304, L2311-L2323):**

```cpp
// L2304 — joint skip:
if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;
// L2311-L2323 — both-sides accumulation (no per-side guard):
case EScoreFunction::Cosine: {
    const double dSL   = static_cast<double>(sumLeft);
    const double dSR   = static_cast<double>(sumRight);
    const double dWL   = static_cast<double>(weightLeft);
    const double dWR   = static_cast<double>(weightRight);
    const double dL2   = static_cast<double>(l2RegLambda);
    const double dInvL = 1.0 / (dWL + dL2);
    const double dInvR = 1.0 / (dWR + dL2);
    cosNum_d += dSL * dSL * dInvL + dSR * dSR * dInvR;
    cosDen_d += dSL * dSL * dWL * dInvL * dInvL
              + dSR * dSR * dWR * dInvR * dInvR;
    break;
}
```

**Ordinal branch Cosine (L2388, L2395-L2407):**

```cpp
// L2388 — joint skip:
if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;
// L2395-L2407 — both-sides accumulation (no per-side guard):
case EScoreFunction::Cosine: {
    const double dSL   = static_cast<double>(sumLeft);
    const double dSR   = static_cast<double>(sumRight);
    const double dWL   = static_cast<double>(weightLeft);
    const double dWR   = static_cast<double>(weightRight);
    const double dL2   = static_cast<double>(l2RegLambda);
    const double dInvL = 1.0 / (dWL + dL2);
    const double dInvR = 1.0 / (dWR + dL2);
    cosNum_d += dSL * dSL * dInvL + dSR * dSR * dInvR;
    cosDen_d += dSL * dSL * dWL * dInvL * dInvL
              + dSR * dSR * dWR * dInvR * dInvR;
    break;
}
```

Neither branch has:
- `if (!wL_pos && !wR_pos) break;`
- `if (wL_pos) { ... }` / `if (wR_pos) { ... }` conditional accumulation

Both have the OLD pattern: joint-skip `continue` followed by unconditional both-sides accumulation. This is **identical to the pre-S33 shape of `FindBestSplit`'s inner body**.

---

## 5. Parent-term subtraction: scope and presence

### L2 path

**`FindBestSplit` ordinal L2 (post-S33, L1984):**
```cpp
totalGain -= (totalSum * totalSum) / (totalWeight + l2RegLambda);
```
This is inside the per-(bin, p, k) loop, subtracted once per non-degenerate (p, k) pair after the per-side conditional adds.

**FBSPP ordinal L2 (L2391-L2393):**
```cpp
gain += (sumLeft * sumLeft) / (weightLeft + l2RegLambda)
      + (sumRight * sumRight) / (weightRight + l2RegLambda)
      - (totalSum * totalSum) / (totalWeight + l2RegLambda);
```
The parent term `totalSum²/(totalWeight+λ)` is present and subtracted **within the k-loop**, scoped to each (bin, p, k) triple. It is only reached when the joint-skip `continue` does NOT fire (i.e., both wL and wR are > 1e-15).

**FBSPP one-hot L2 (L2306-L2309):** Same structure — parent term present and subtracted within the k-loop, gated by the joint-skip.

**For Cosine:** neither FBSPP branch subtracts a parent term — consistent with `FindBestSplit`'s Cosine path which also has no parent-term subtraction. The Cosine score is `cosNum_d / sqrt(cosDen_d)`, a ratio rather than a difference-from-parent. This is the S34-PROBE-F-LITE finding: Cosine is "parentless."

---

## 6. CPU reference comparison: does FBSPP match CPU's degenerate behavior?

### CPU reference: `CalcAverage` (online_predictor.h:112-119)

```cpp
inline double CalcAverage(double sumDelta, double count, double scaledL2Regularizer) {
    double inv = count > 0 ? 1. / (count + scaledL2Regularizer) : 0;
    return sumDelta * inv;
}
```

When `count == 0`, `CalcAverage` returns `0.0`. In `UpdateScoreBinKernelPlain` (short_vector_ops.h:61-81, generic; short_vector_ops.h:155-175, SSE2):

```cpp
// Generic non-SSE path (short_vector_ops.h:67-80):
const double trueAvrg  = CalcAverage(trueStatsPtr[0],  trueStatsPtr[1],  scaledL2Regularizer);
const double falseAvrg = CalcAverage(falseStatsPtr[0], falseStatsPtr[1], scaledL2Regularizer);
scoreBinPtr[0] += trueAvrg  * trueStatsPtr[0];   // sum * avg_if_nonempty (= sum²/(w+λ))
scoreBinPtr[1] += trueAvrg  * trueAvrg  * trueStatsPtr[1];
scoreBinPtr[0] += falseAvrg * falseStatsPtr[0];
scoreBinPtr[1] += falseAvrg * falseAvrg * falseStatsPtr[1];
```

CPU calls both `trueAvrg` and `falseAvrg` unconditionally and adds both sides unconditionally. When one side is empty (weight=0), `CalcAverage` returns 0.0, so its contribution is exactly zero — but the non-empty side still contributes `sum²/(w+λ)`. **CPU never skips.** The mask is implicit through `CalcAverage`'s `count > 0` branch, which vectorizes cleanly in SSE2 via `_mm_cmpgt_pd` at `short_vector_ops.h:167`.

### FBSPP's Cosine path vs CPU

FBSPP's joint-skip `continue` (at L2304 / L2388) **diverges from CPU in exactly the same way** that the pre-S33 `FindBestSplit` diverged:

| Condition | CPU behavior | FBSPP behavior (current) | `FindBestSplit` post-S33 |
|-----------|-------------|--------------------------|--------------------------|
| wL > 0, wR > 0 | add both sides | add both sides | add both sides |
| wL = 0, wR > 0 | add right side only (CalcAverage(left)=0) | **skip entire k** | add right side only |
| wL > 0, wR = 0 | add left side only | **skip entire k** | add left side only |
| wL = 0, wR = 0 | add nothing (both CalcAverage=0) | skip entire k | skip (break) |

FBSPP does not add the non-empty side when one child is degenerate. This matches the pre-DEC-042 bug in `FindBestSplit` exactly.

---

## What did S33-L4-FIX leave at FBSPP?

S33-L4-FIX (DEC-042, Commits 1 and 1.5) replaced the joint-skip `continue` with a per-side mask in `FindBestSplit` only. The change was applied at `csv_train.cpp:1961-2016` (ordinal) and `csv_train.cpp:1698-1734` (one-hot). `FindBestSplitPerPartition` starts at L2251 and contains structurally identical joint-skip `continue` lines at **L2304** (one-hot) and **L2388** (ordinal). S33-L4-FIX **left FBSPP entirely untouched**. Both FBSPP branches still carry the pre-DEC-042 pattern: joint skip when either child has weight < 1e-15, followed by unconditional both-sides accumulation. This is the same bug that caused the 52.6% → 0.027% drift collapse in `FindBestSplit`'s ordinal SymmetricTree path. Since LG calls `FindBestSplitPerPartition` exclusively (structure_searcher.cpp:625) and DW shares it, neither Lossguide nor Depthwise received the DEC-042 correction. The CR-S33-S1 note that FBSPP is "structurally immune to the cross-partition running-sum bug class" was correct in the narrow sense (FBSPP cannot corrupt a different partition's running sum), but FBSPP does suffer from the same single-partition under-attribution: when a (bin, p, k) has one degenerate child, FBSPP contributes zero to that partition's cosNum/cosDen instead of the non-empty side's true value. At small N, degenerate children per (partition, bin) are denser because depth-aware partitions have fewer documents — the fraction of (p, k) cells where exactly one child is empty grows monotonically with depth (mirroring the 0% → 14.6% skip-rate progression documented in PROBE-E for `FindBestSplit`), which explains the N-dependent magnitude of the LG+Cosine drift.

---

## File:line citations summary

| Claim | Location |
|-------|----------|
| FBSPP function signature | `csv_train.cpp:2251` |
| One-hot branch entry | `csv_train.cpp:2283` |
| One-hot per-(bin,p) accumulator init | `csv_train.cpp:2290-2294` |
| One-hot k-loop | `csv_train.cpp:2295` |
| **One-hot joint-skip `continue`** | **`csv_train.cpp:2304`** |
| One-hot Cosine both-sides accumulation | `csv_train.cpp:2311-2323` |
| One-hot Cosine gain finalization | `csv_train.cpp:2331-2333` |
| Ordinal branch suffix-sum precompute | `csv_train.cpp:2353-2368` |
| Ordinal per-(bin,p) accumulator init | `csv_train.cpp:2375-2379` |
| Ordinal k-loop | `csv_train.cpp:2380` |
| **Ordinal joint-skip `continue`** | **`csv_train.cpp:2388`** |
| Ordinal Cosine both-sides accumulation | `csv_train.cpp:2395-2407` |
| Ordinal Cosine gain finalization | `csv_train.cpp:2413-2415` |
| `FindBestSplit` ordinal per-side mask (Cosine) | `csv_train.cpp:1997-2016` |
| `FindBestSplit` ordinal per-side mask (L2) | `csv_train.cpp:1977-1984` |
| `FindBestSplit` one-hot per-side mask (L2) | `csv_train.cpp:1709-1712` |
| `FindBestSplit` one-hot joint-skip retained (Cosine) | `csv_train.cpp:1717` |
| `FindBestSplit` ordinal accumulator scope (per-bin) | `csv_train.cpp:1823-1830` |
| CPU `CalcAverage` mask | `catboost/private/libs/algo_helpers/online_predictor.h:117` |
| CPU `UpdateScoreBinKernelPlain` both-sides unconditional | `catboost/libs/helpers/short_vector_ops.h:67-80` |
| CPU SSE2 per-side mask via `_mm_cmpgt_pd` | `catboost/libs/helpers/short_vector_ops.h:167-168` |
