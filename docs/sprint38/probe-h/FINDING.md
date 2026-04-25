# PROBE-H Finding — per-side Cosine formula divergence localisation

**Date**: 2026-04-25
**Branch**: `mlx/sprint-38-lg-small-n`
**Anchor**: `np.random.default_rng(42)`, **N=1000**, 20 features,
y = 0.5·X[0] + 0.3·X[1] + 0.1·noise, ST/Cosine/RMSE, depth=6, bins=128, l2=3, lr=0.03
**Build**: `csv_train_probe_h` compiled with
`-DCOSINE_RESIDUAL_INSTRUMENT -DPROBE_E_INSTRUMENT -DPROBE_H_INSTRUMENT -DPROBE_D_ARM_AT_ITER=0`
**Kernel md5**: `9edaef45b99b9db3e2717da93800e76f` (host-side instrumentation only — kernels untouched)

> **Status**: REVISED 2026-04-25 — original verdict WITHDRAWN. See §Original analysis error below.
>
> **Corrected verdict**: The binary's per-side mask formula (`csv_train.cpp:2068-2097`,
> S33-L4-FIX commit `10c72b4e96`) and CPU's `UpdateScoreBinKernelPlain` are **mathematically
> identical** on identical input (max delta 1.37e-13, all 6 depths). DEC-044 is WITHDRAWN.
>
> The 13.93% N=1k drift is **not caused by a formula difference**. Correction 2 (granularity
> test) finds that restricting MLX to only CPU-equivalent bins matches CPU in 4/6 depths.
> At d=4 and d=5 the restricted argmax also misses CPU's pick. The divergence mechanism is
> **partially explained by MLX's finer grid** (CPU has 6 borders for feat 0, 5 for feat 1,
> none for feats 2–19; MLX has 127 borders for all 20 features) but a residual of 2/6 depths
> remains unexplained. The 13.93% drift mechanism is **not fully identified**.

---

## Original analysis error

The first version of this finding (and `analyze_probe_h.py`) made a **counterfactual / observational
confusion**:

- `analyze_probe_h.py` computed a column called `gain_mlx_formula` by applying the OLD
  joint-skip formula to `PROBE_E_INSTRUMENT`'s `mlxTermNum/mlxTermDen` fields. Those fields
  capture the OLD formula by construction (the `PROBE_E_INSTRUMENT` code block applies
  `if (wL < 1e-15f || wR < 1e-15f) continue;` before emitting).
- The column `picked_by_mlx` reflects the binary's **actual** output, which uses the
  **correct per-side mask** (shipped in S33-L4-FIX, `csv_train.cpp:2068-2097`, commit `10c72b4e96`).
- The script compared the counterfactual `gain_mlx_formula` (old joint-skip) against
  `picked_by_mlx` (correct per-side mask) and reported their divergence as evidence the
  binary used the old formula. The logic was inverted: the divergence was evidence that
  the two formulas are DIFFERENT, not that the binary uses the old one.

Direct code reading at `csv_train.cpp:2068-2097` confirms the per-side mask structure:
```cpp
if (!wL_pos && !wR_pos) break;   // skip only when BOTH are empty
if (wL_pos) {
    const double dSL = ...; const double dWL = ...;
    const double dInvL = 1.0 / (dWL + dL2);
    termNum += dSL * dSL * dInvL;
    termDen += dSL * dSL * dWL * dInvL * dInvL;
}
if (wR_pos) {
    const double dSR = ...; const double dWR = ...;
    const double dInvR = 1.0 / (dWR + dL2);
    termNum += dSR * dSR * dInvR;
    termDen += dSR * dSR * dWR * dInvR * dInvR;
}
cosNum_d += termNum; cosDen_d += termDen;
```

This has been the code since Sprint 33 (`10c72b4e96`). The "fix" described in the original
finding was already present.

---

## Why this probe

F2 (2026-04-25, `docs/sprint38/f2/FINDING.md`) confirmed the C-PSF hypothesis:

- **C-QG falsified**: all 11 CPU borders present in MLX's 127-border grid to within 3.5e-8.
- **C-LV falsified**: 5/6 iter=1 split mismatches from constant basePred; leaf-value cascade impossible.
- **C-PSF confirmed**: for all three feat-matched iter=2 depths, MLX has CPU's preferred border
  in its search space and scores it lower, picking a different bin.

PROBE-H was opened as the localisation step: capture per-side accumulators and apply both
formulas to the data to identify which formula diverges. The instrumentation is valid and
the captured data is good. The error was in how the analysis script used the data.

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

---

## The two formulas side-by-side

### CPU formula — `short_vector_ops.h` `UpdateScoreBinKernelPlain` (generic/SSE2 path)

```
if wL > 0:
    cosNum += sL² / (wL + lambda)
    cosDen += sL² * wL / (wL + lambda)²
if wR > 0:
    cosNum += sR² / (wR + lambda)
    cosDen += sR² * wR / (wR + lambda)²
gain = cosNum / sqrt(cosDen)
```

### MLX formula — `csv_train.cpp:2068-2097` per-side mask (S33-L4-FIX)

```
if !wL_pos && !wR_pos: skip                // both empty → true no-op
if wL_pos: termNum += sL²/(wL+λ);  termDen += sL²·wL/(wL+λ)²
if wR_pos: termNum += sR²/(wR+λ);  termDen += sR²·wR/(wR+λ)²
cosNum_d += termNum; cosDen_d += termDen
gain = cosNum_d / sqrt(cosDen_d + 1e-20)
```

These are **mathematically identical** on the same input. The only structural difference is
the `1e-20` guard in the denominator, which is numerically negligible.

---

## Result — Correction 1: Formula equivalence

`analyze_probe_h_v2.py` recomputes both formulas from the stored `cosNumL/R, cosDenL/R` fields
and verifies:

| d | max \|gain_per_side_mask − gain_calc_score_on_side\| | max \|gain_computed − gain_mlx_captured\| | max \|cosNumTotal − (cosNumL+cosNumR)\| |
|---|---|---|---|
| 0 | 6.40e-14 | 9.24e-14 | 7.11e-13 |
| 1 | 9.59e-14 | 9.24e-14 | 1.19e-12 |
| 2 | 1.37e-13 | 1.07e-13 | 1.62e-12 |
| 3 | 7.11e-14 | 9.59e-14 | 1.28e-12 |
| 4 | 3.20e-14 | 6.40e-14 | 3.41e-13 |
| 5 | 3.20e-14 | 6.40e-14 | 3.98e-13 |

**Max across all depths: 1.37e-13 — well within double-precision rounding noise.**

**CONCLUSION: Formulas are IDENTICAL.** The formula claimed in the original PROBE-H as the
divergence source does not exist in the current binary. DEC-044 is WITHDRAWN.

---

## Result — Correction 2: Granularity hypothesis

**Hypothesis**: MLX finds a higher-scoring border in its finer 127-bin grid that CPU never
evaluates (CPU has only 6 borders for feat 0, 5 for feat 1, and 0 for feats 2–19;
MLX has 127 borders for all 20 features).

**Test**: for each depth, compute the MLX "restricted argmax" over only bins where the MLX
border is within 1e-4 of a CPU border (11 bins total across feats 0–1 at each depth).
If the restricted argmax matches CPU's pick → granularity explains the divergence.

CPU borders present in MLX grid: feat 0 at 6/127 bins, feat 1 at 5/127 bins; feats 2–19 have 0.

| d | MLX actual pick | MLX restricted pick | CPU actual pick | restricted = CPU |
|---|---|---|---|---|
| 0 | feat=0, bin=69 (gain=12.8245) | feat=0, bin=69 (gain=12.8245) | feat=0, border=0.102547 (bin=69) | YES |
| 1 | feat=1, bin=82 (gain=15.0317) | feat=1, bin=82 (gain=15.0317) | feat=1, border=0.438497 (bin=82) | YES |
| 2 | feat=0, bin=25 (gain=16.1478) | feat=0, bin=25 (gain=16.1478) | feat=0, border=−0.810750 (bin=25) | YES |
| 3 | feat=0, bin=102 (gain=16.9426) | feat=0, bin=106 (gain=16.9402) | feat=0, border=1.035291 (bin=106) | YES |
| 4 | feat=0, bin=102 (gain=16.9506) | feat=0, bin=106 (gain=16.9463) | feat=1, border=−0.800153 (bin=27) | NO |
| 5 | feat=0, bin=102 (gain=17.3101) | feat=0, bin=104 (gain=17.3031) | feat=0, border=1.746585 (bin=122) | NO |

**Restricted matches CPU: 4/6 depths.**

- **d=0–3**: restricting MLX to CPU-equivalent bins recovers CPU's pick (feat match AND
  border match within 1e-4). At d=0–2 the restricted pick equals the unrestricted MLX pick
  (no finer-grid advantage at those depths). At d=3 the unrestricted pick is bin=102 vs
  restricted bin=106 (CPU's equivalent): the finer grid lets MLX prefer bin=102 by a tiny
  0.0024 gain margin (16.9426 vs 16.9402), but restricting to CPU's grid recovers the
  correct answer.

- **d=4**: CPU picks feat=1 (border=−0.800153, bin=27, gain=16.42). MLX restricted argmax
  still picks feat=0, bin=106 (gain=16.95). Even in the restricted space, feat=0 bin=106
  scores higher than feat=1 bin=27 (rank of feat=1 bin=27 = 75 out of 2540). Granularity
  alone does not explain the d=4 divergence.

- **d=5**: CPU picks feat=0 (border=1.746585, bin=122, gain=17.009). MLX restricted pick is
  feat=0, bin=104 (gain=17.303). The restricted pick misses CPU's bin=122 despite both
  being on feat=0 and having a CPU-equivalent. The finer grid lets MLX find bins with higher
  gain on the same feature — confirming grid granularity as the cause at d=5 but the
  restricted pick lands on a different CPU-absent bin (feat=0 bin=104), not the CPU border.

**CONCLUSION: Granularity is a partial (4/6) explanation.** The divergence at d=4–5 is not
fully explained by the finer grid alone. At d=4 the restricted argmax chooses the wrong
feature even without the finer-grid advantage. A secondary mechanism is active at d≥4.

---

## Secondary anomaly at d=1

Both the recomputed gain and `gain_mlx_captured` rank feat=1, bin=82 first at d=1 (gain=15.032),
but `picked_by_mlx` is feat=1, bin=64 (gain=14.976, rank=19). The gap is 0.056 gain units —
not noise. This means the binary chose a different bin than what the instrument records as the
gain winner. This anomaly was noted in the original PROBE-H finding and remains unexplained.
It does not affect the formula equivalence conclusion of Correction 1, but it does show the
binary's argmax path and the COSINE_RESIDUAL_INSTRUMENT path disagree at d=1.

---

## Localization verdict (corrected)

**The 13.93% N=1k drift is NOT caused by a formula divergence between MLX and CPU.**
Both formulas are identical to within 1.37e-13 at all 6 depths.

**Partial explanation via grid granularity**: restricting MLX to CPU-equivalent bins
recovers CPU's pick in 4/6 depths. At d=3 the finer grid causes a minor drift (0.0024
gain margin favoring a non-CPU bin over the CPU-equivalent bin). At d=0–2 the grids produce
the same argmax.

**Residual mechanism (2/6 depths, d=4–5)**: not identified. At d=4 the restricted argmax
chooses feat=0 over CPU's feat=1, even within the CPU-equivalent bin pool. At d=5 the
restricted argmax picks the right feature but a different border. A third mechanism (not
C-QG, not C-LV, not C-PSF formula) is active.

---

## Implied next probe — PROBE-Q (revised scope)

The F2 falsification of C-QG was: "all 11 CPU borders are present in MLX's grid to within
3.5e-8." That is correct and stands. But F2 tested the wrong direction of C-QG. The real
C-QG question is: **why does MLX have 127 borders for all features while CPU has 6/5/0?**
CPU's `GreedyLogSum` algorithm generates borders from the data distribution; MLX uses a
static uniform grid. When MLX's uniform grid places more evaluable bins on noise features
(feats 2–19 each have 127 MLX bins but 0 CPU bins), it opens up 127×18=2286 candidates
that CPU never considers. The argmax over these may find higher scores on noise features.

**PROBE-Q goal**: replace MLX's static uniform-grid quantization with CPU's `GreedyLogSum`
border generation, or cap MLX's border grid per feature to match CPU's actual border count.
Expected outcome: if grid alignment is the primary driver, drift should collapse close to 0%
once MLX evaluates the same candidate set as CPU.

The d=4 depth=4 residual (wrong feature even in restricted space) will require a separate
investigation if PROBE-Q does not close it.

---

## Data files

| File | Description |
|------|-------------|
| `data/probe_h_iter1_depth{0..5}.csv` | Per-(feat,bin,partition) per-side accumulators (6 depths) |
| `data/cos_leaf_seed42_depth{0..5}.csv` | PROBE-E companion: mlxTermNum/Den (old joint-skip counterfactual) |
| `data/cos_accum_seed42_depth{0..5}.csv` | COSINE_RESIDUAL_INSTRUMENT: per-bin gain_f64 (per-side mask) |
| `data/divergence_iter1.csv` | Original PROBE-H output (based on inverted counterfactual analysis — do not use) |
| `data/granularity_test.csv` | Correction 2 output: per-depth restricted vs actual MLX argmax vs CPU pick |
| `data/probe_h_run.log` | Binary stdout/stderr from PROBE-H capture run |
| `scripts/analyze_probe_h.py` | Original analysis script — INVALID (counterfactual / observational confusion) |
| `scripts/analyze_probe_h_v2.py` | Corrected analysis script (Corrections 1 + 2) |
| `scripts/build_probe_h.sh` | Build script for `csv_train_probe_h` binary |
| `scripts/run_probe_h.py` | Run script for PROBE-H data capture |

---

## Authority

- Binary code-reading confirming per-side mask: `catboost/mlx/tests/csv_train.cpp:2068-2097`
- S33-L4-FIX commit: `10c72b4e96`
- CPU reference formula: `catboost/libs/helpers/short_vector_ops.h:155+` (`UpdateScoreBinKernelPlain`)
- F2 routing: `docs/sprint38/f2/FINDING.md`
- DEC-042 fix record: `.claude/state/DECISIONS.md §DEC-042`
- DEC-044 withdrawal: `.claude/state/DECISIONS.md §DEC-044`
- PROBE-G context: `docs/sprint38/probe-g/FINDING.md`
