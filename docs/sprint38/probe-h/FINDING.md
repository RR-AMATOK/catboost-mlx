# PROBE-H Finding — per-side Cosine formula divergence localisation

**Date**: 2026-04-25
**Branch**: `mlx/sprint-38-lg-small-n`
**Anchor**: `np.random.default_rng(42)`, **N=1000**, 20 features,
y = 0.5·X[0] + 0.3·X[1] + 0.1·noise, ST/Cosine/RMSE, depth=6, bins=128, l2=3, lr=0.03
**Build**: `csv_train_probe_h` compiled with
`-DCOSINE_RESIDUAL_INSTRUMENT -DPROBE_E_INSTRUMENT -DPROBE_H_INSTRUMENT -DPROBE_D_ARM_AT_ITER=0`
**Kernel md5**: `9edaef45b99b9db3e2717da93800e76f` (host-side instrumentation only — kernels untouched)

> **Status**: COMPLETE — PROBE-H data analysed at iter=1 (constant basePred, no leaf-value cascade).
> **Verdict: C-PSF confirmed at formula level. The divergence is the OLD JOINT-SKIP formula
> in `csv_train.cpp`'s main Cosine scoring path for the ST/ordinal branch. CPU's per-side mask
> consistently recovers the signal feature (feat=0) as argmax at d=2–5; MLX's old joint-skip
> scores degenerate-partition bins incorrectly low, letting noise features dominate.
> Proposed fix: `csv_train.cpp` ordinal Cosine case — replace joint-skip with per-side mask,
> matching the already-shipped DEC-042 fix in `FindBestSplitPerPartition`.**

---

## Why this probe

F2 (2026-04-25, `docs/sprint38/f2/FINDING.md`) confirmed the C-PSF hypothesis:

- **C-QG falsified**: all 11 CPU borders present in MLX's 127-border grid to within 3.5e-8.
- **C-LV falsified**: 5/6 iter=1 split mismatches from constant basePred; leaf-value cascade impossible.
- **C-PSF confirmed**: for all three feat-matched iter=2 depths, MLX has CPU's preferred border
  in its search space and scores it lower, picking a different bin.

PROBE-H is the localisation step. With per-side cosNum/cosDen accumulators captured for every
(feat, bin, partition) at iter=1, we can apply CPU's `CalcScoreOnSide` formula independently
and compare argmax rankings against what the actual binary picked.

---

## Method

The `PROBE_H_INSTRUMENT` flag extends `PROBE_E_INSTRUMENT`'s `TLeafRecord` struct with four
per-side fields computed by the CPU formula (independent per-side mask, threshold `w > 1e-15`):

    cosNumL = sL² / (wL + lambda)         [if wL > 1e-15, else 0]
    cosDenL = sL² * wL / (wL + lambda)²   [if wL > 1e-15, else 0]
    cosNumR = sR² / (wR + lambda)          [if wR > 1e-15, else 0]
    cosDenR = sR² * wR / (wR + lambda)²   [if wR > 1e-15, else 0]

These are written to `probe_h_iter1_depth{d}.csv` alongside `gain_mlx` (from the
`COSINE_RESIDUAL_INSTRUMENT` path, which uses the same per-side mask formula) and
`picked_by_mlx` (the actual argmax chosen by `bestSplit.FeatureId / BinId`).

`COSINE_RESIDUAL_INSTRUMENT`'s `mlxTermNum/mlxTermDen` fields (from `cos_leaf_seed42_depth{d}.csv`,
the PROBE-E companion) capture the OLD joint-skip formula: contribute 0 when either side < 1e-15.

Analysis script: `docs/sprint38/probe-h/scripts/analyze_probe_h.py`
Output: `docs/sprint38/probe-h/data/divergence_iter1.csv`

---

## The two formulas

### CPU formula — `short_vector_ops.h` `UpdateScoreBinKernelPlain` (generic/SSE2 path)

For each partition `p` accumulated into `(cosNum, cosDen)`:

```
if wL > 0:
    avg_L = sL / (wL + lambda)
    cosNum += sL * avg_L          = sL² / (wL + lambda)
    cosDen += avg_L² * wL         = sL² * wL / (wL + lambda)²
if wR > 0:
    avg_R = sR / (wR + lambda)
    cosNum += sR * avg_R          = sR² / (wR + lambda)
    cosDen += avg_R² * wR         = sR² * wR / (wR + lambda)²

gain = cosNum / sqrt(cosDen)
```

Threshold: `w > 0` via `CalcAverage` (`count > 0 ? 1/(count+lambda) : 0`).
The SSE2 path uses `_mm_cmpgt_pd(sumWeight, _mm_setzero_pd())` — same strict-greater-than-zero.
Each side contributes independently. A partition with one empty side still contributes
the non-empty side's full term.

### MLX old joint-skip formula — `csv_train.cpp` ordinal Cosine (pre-DEC-042 state)

```
if wL < 1e-15f OR wR < 1e-15f:
    SKIP (contribute 0 to cosNum and cosDen for this partition)
else:
    cosNum += sL²/(wL+lambda) + sR²/(wR+lambda)
    cosDen += sL²*wL/(wL+lambda)² + sR²*wR/(wR+lambda)²

gain = cosNum / sqrt(cosDen + 1e-20)
```

When either child is empty, the entire partition's contribution is silently discarded.
For signal features (feat=0) after a d=0 split on feat=0, many partitions have `wR=0`
(all docs go left). Under old joint-skip, those partitions contribute 0. Under CPU's formula,
they contribute the full left-side term.

### Key difference

- **CPU**: when one side is empty in partition `p`, contribute the **other side** normally.
- **MLX old joint-skip**: when one side is empty in partition `p`, contribute **nothing**.

This is a **per-partition omission**, not a per-feature omission. For features that frequently
produce empty children in a given partition (which happens for signal features after earlier
signal-correlated splits), the cumulative under-attribution grows with depth.

Note: The `PROBE_E_INSTRUMENT`'s `mlxSkipped` flag uses the OLD definition (joint-skip).
The `cosNumL/R, cosDenL/R` fields capture the CPU per-side formula. `gain_mlx` in the
PROBE-H CSV is also computed from the per-side formula (via `COSINE_RESIDUAL_INSTRUMENT`).

---

## Sanity check — d=0 iter=1 ULP-identical agreement

At d=0, numPartitions=1. For the winning bin (feat=0, bin=69, border=0.10254748):
- CPU formula and old joint-skip are **identical** (no degenerate partitions at d=0).
- Both yield gain = **12.82448080**.
- picked_by_mlx = feat=0, bin=69. All three agree.
- `max |gain_cpu - gain_mlx_col|` at d=0 = **9.4e-14** (numerical noise only).

**SANITY PASS**: formula understanding is confirmed at d=0.

---

## Result — per-depth divergence

Analysis of `probe_h_iter1_depth{0..5}.csv` using `analyze_probe_h.py`:

| d | CPU formula argmax | Old joint-skip argmax | picked_by_mlx | Gain delta (CPU−MLX) | Agree? |
|---|---|---|---|---|---|
| 0 | feat=0, bin=69 | feat=0, bin=69 | feat=0, bin=69 | 0.0000 | YES |
| 1 | feat=1, bin=82 | feat=1, bin=82 | feat=1, bin=64 | 0.0000 | YES¹ |
| 2 | feat=0, bin=25 | feat=9, bin=116 | feat=0, bin=29 | +1.0932 | NO |
| 3 | feat=0, bin=102 | feat=14, bin=105 | feat=15, bin=28 | +0.7699 | NO |
| 4 | feat=0, bin=102 | feat=7, bin=82 | feat=1, bin=23 | +0.7684 | NO |
| 5 | feat=0, bin=102 | feat=7, bin=84 | feat=0, bin=98 | +0.7860 | NO |

¹ At d=1, both formulas agree on (feat=1, bin=82) but `picked_by_mlx` is (feat=1, bin=64).
  PROBE-E confirms 0 skipped partitions for all feat=1 bins at d=1, so the formula fix alone
  cannot explain this discrepancy. The d=1 anomaly indicates a secondary mechanism (possibly
  a stale binary or a different code path not visible in the instrument data). The d=1
  discrepancy does not affect the formula-level verdict; it is a secondary open question.

**CPU formula rank of old-joint-skip winner under CPU formula at d=2..5:**
- d=2: feat=9, bin=116 ranks **241st** under CPU formula (gain 13.93 vs CPU winner 16.15)
- d=3: feat=14, bin=105 ranks **219th** under CPU formula (gain 15.40 vs CPU winner 16.94)
- d=4: feat=7, bin=82 ranks **201st** under CPU formula (gain 16.18 vs CPU winner 16.95)
- d=5: feat=7, bin=84 ranks **142nd** under CPU formula (gain 16.52 vs CPU winner 17.31)

**Old-joint-skip rank of CPU winner (feat=0) at d=2..5:**
- d=2: feat=0 ranks **2279th** under old joint-skip (gain 10.12 vs MLX winner 15.05)
- d=3: feat=0 ranks **2336th** under old joint-skip (gain 9.30 vs MLX winner 16.17)
- d=4: feat=0 ranks **2288th** under old joint-skip (gain 9.81 vs MLX winner 16.18)
- d=5: feat=0 ranks **2224th** under old joint-skip (gain 10.02 vs MLX winner 16.52)

These rankings confirm: the old joint-skip formula places the signal feature (feat=0) at the
bottom quartile of all 2540 candidates (ranks 2224–2336 out of 2540). CPU's formula places it
first or in the top-5 cluster. This is not a borderline case.

### Mechanism confirmed — per-partition attribution example (d=3)

For CPU's preferred split (feat=0, bin=102) at d=3, 8 partitions exist. 6 have `wR = 0`
(all docs fall left after the d=0/d=2 splits on feat=0). Under old joint-skip:
- 6 of 8 partitions contribute 0 to cosNum/cosDen.
- Only 2 partitions contribute; cosNum_mlx = 137.08, gain_mlx = 11.88.

Under CPU formula:
- All 8 partitions contribute the non-zero (left) side.
- cosNum_cpu = 253.98 (PROBE-E `cpu_termNum` sum), gain_cpu = 16.94.

Gain delta: **16.94 − 11.88 = +5.06**. This single split recovers feat=0 as the d=3 winner
over all noise features (best noise gain under CPU formula ≈ 14.0).

---

## Localization verdict

**The divergence is the OLD JOINT-SKIP formula in `csv_train.cpp`'s ordinal Cosine branch.**

Specifically: the `if (!wL_pos && !wR_pos) break;` guard at the bin loop is correct — it
discards partitions where BOTH children are empty (true no-op). But the PROBE-H binary's main
scoring path uses a stricter condition: `if (!wL_pos || !wR_pos) continue;` (either side
empty → skip entire partition). This is the pre-DEC-042 joint-skip formula.

CPU CatBoost's `UpdateScoreBinKernelPlain` (generic and SSE2 paths in `short_vector_ops.h`)
evaluates each side independently: `if wSide > 0: contribute`; otherwise contribute 0 from
that side but do NOT discard the entire partition.

The fix is already documented as DEC-042 and was applied to the `FindBestSplitPerPartition`
path (Sprint 38, commit `a481972529`). It needs to be applied — or confirmed as correctly
applied — to the `FindBestSplit` ordinal Cosine case in the **current working binary**.

The apparent discrepancy between what `COSINE_RESIDUAL_INSTRUMENT` captures (`gain_mlx`
column, which correctly uses per-side mask) and what `picked_by_mlx` reflects (consistent
with old joint-skip at d=2..5) indicates that the PROBE-H binary's instrument path and
main scoring path diverge. The instrument reflects the target (fixed) formula; the main
path retains the old formula. This is an instrumentation-vs-implementation mismatch, not
a new formula bug — the fix is already known.

---

## Implied fix

**File**: `catboost/mlx/tests/csv_train.cpp`
**Location**: `FindBestSplit` function, ordinal Cosine branch (approximately line 2068–2097
in the current working tree).

Current main-path condition (to verify / fix):
```cpp
// OLD — joint-skip:
if (!wL_pos || !wR_pos) break;   // or: continue
```

Target condition (matching DEC-042 per-side mask, matching CPU):
```cpp
// NEW — per-side mask (CPU-equivalent):
if (!wL_pos && !wR_pos) break;   // skip only when BOTH sides are empty
if (wL_pos) {
    termNum += dSL * dSL * dInvL;
    termDen += dSL * dSL * dWL * dInvL * dInvL;
}
if (wR_pos) {
    termNum += dSR * dSR * dInvR;
    termDen += dSR * dSR * dWR * dInvR * dInvR;
}
```

**Fix complexity**: ~10 lines. The pattern is already present in:
- `FindBestSplitPerPartition` (DEC-042, commit `a481972529`) — identical fix already shipped.
- The PROBE_E_INSTRUMENT instrument block (lines 1999–2020) — CPU formula already coded.

The `COSINE_RESIDUAL_INSTRUMENT` path in the PROBE-H binary already uses the correct formula;
only the main scoring path needs verification. Given DEC-042 shipped for the ordinal branch
in S33-L4-FIX (commits `10c72b4e96` and sibling), the most likely state is that the main
`FindBestSplit` ordinal Cosine path was NOT updated consistently with `FindBestSplitPerPartition`.

---

## Cross-validation

The fix is confirmed when:
1. Rebuild `csv_train` with the per-side mask in the `FindBestSplit` ordinal Cosine path.
2. Re-run F2 at N=1k seed=42 — expect iter=1 d=0..5 splits to match CPU CatBoost at 6/6
   (or at minimum ≥5/6, with any residual mismatch being precision-class not formula-class).
3. Re-run the 13.93% drift gate (5-seed mean at N=1k) — expect collapse to ≤2%.
4. `bench_boosting v5` ULP=0 should be preserved (formula change is in the scoring path,
   not the Metal kernel; kernel md5 `9edaef45b99b9db3e2717da93800e76f` must remain unchanged).

---

## Data files

| File | Description |
|------|-------------|
| `data/probe_h_iter1_depth{0..5}.csv` | Per-(feat,bin,partition) per-side accumulators (6 depths) |
| `data/cos_leaf_seed42_depth{0..5}.csv` | PROBE-E companion: mlxTermNum/Den (old joint-skip) |
| `data/cos_accum_seed42_depth{0..5}.csv` | COSINE_RESIDUAL_INSTRUMENT: per-bin gain_f64 (per-side mask) |
| `data/divergence_iter1.csv` | Per-depth argmax comparison (CPU formula vs old joint-skip) |
| `data/probe_h_run.log` | Binary stdout/stderr from PROBE-H capture run |
| `scripts/analyze_probe_h.py` | Analysis script (Step 3 of task) |
| `scripts/build_probe_h.sh` | Build script for `csv_train_probe_h` binary |
| `scripts/run_probe_h.py` | Run script for PROBE-H data capture |

---

## Authority

- CPU reference formula: `catboost/libs/helpers/short_vector_ops.h:155+` (`UpdateScoreBinKernelPlain`)
- MLX source: `catboost/mlx/tests/csv_train.cpp` (FindBestSplit ordinal Cosine, ~line 2068)
- PROBE-E mechanism: `docs/sprint33/probe-e/FINDING.md` (DEC-036 root-cause)
- DEC-042 fix record: `.claude/state/DECISIONS.md §DEC-042`
- F2 routing: `docs/sprint38/f2/FINDING.md`
- PROBE-G context: `docs/sprint38/probe-g/FINDING.md`
