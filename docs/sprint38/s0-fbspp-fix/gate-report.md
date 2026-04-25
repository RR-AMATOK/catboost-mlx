# S38-S0 Gate Report — DEC-042 FBSPP Port

**Date**: 2026-04-25
**Branch**: `mlx/sprint-38-lg-small-n`
**Commit**: S38-S0 (pending; this document is pre-commit)
**Kernel md5**: `9edaef45b99b9db3e2717da93800e76f` (host-side fix only — no Metal shader changes)

---

## Summary

Ported DEC-042's per-side mask from `FindBestSplit` (SymmetricTree) to
`FindBestSplitPerPartition` (FBSPP), which is used by Lossguide and Depthwise grow policies.
T0b code-reading (`docs/sprint38/lg-small-n/code-reading.md`, 2026-04-25) confirmed FBSPP
carries the same unfixed joint-skip `continue` at `csv_train.cpp:2304` (one-hot) and
`csv_train.cpp:2388` (ordinal).

The fix is asymmetric, mirroring the S33/S34/S35 asymmetry exactly:

| Branch + Score | Fix applied |
|---|---|
| Ordinal Cosine | Per-side mask — `if (!wL_pos && !wR_pos) break;` + conditional `if(wL_pos)`/`if(wR_pos)` accumulation |
| Ordinal L2 | Per-side mask — same as above + unconditional parent subtraction |
| One-hot L2 | Per-side mask — mirrors FindBestSplit one-hot S35-#129 |
| One-hot Cosine | **Joint-skip preserved** — S34-PROBE-F-LITE parentless argument applies unchanged |

---

## Asymmetry Rationale (One-hot Cosine)

The S34-PROBE-F-LITE verdict that prevents fixing one-hot Cosine in `FindBestSplit` applies
identically to FBSPP's one-hot Cosine:

- `FindBestSplit` one-hot Cosine retains joint-skip because Cosine has no parent-term
  subtraction. When wR=0 (one child degenerate), per-side mask would add
  `totalSum²/(totalWeight+λ)` to cosNum_d with nothing cancelling it. This injects a
  bin-dependent bias — bins that produce more one-sided splits (rare categories) get a
  higher cosNum_d spuriously.

- FBSPP one-hot Cosine has the **same structure**: no parent term, same cosNum/cosDen
  accumulation, same one-hot split semantics. The CR's "structurally immune" claim applied
  to the cross-partition running-sum class; the parentless-Cosine argument is independent
  of accumulator scope (per-partition vs per-bin) and applies in both FBSPP and FindBestSplit.

Ordinal Cosine IS fixed in both FindBestSplit (Commit 1) and FBSPP (this commit) because
ordinal features share the same `totalSum`/`totalWeight` across all bins of a given (feat, p, k)
triple. The injected term when wR=0 would be `totalSum²/(totalWeight+λ)`, which is constant
across all bins of the same (feat, p, k) — it does not bias the per-feature argmax.

---

## Gate Specifications

### G3a — N=50k ST+Cosine (regression guard)

**Spec**: ST uses `FindBestSplit`, not FBSPP. Fix should have zero effect on ST.
**Expected**: ~1.27% (S37 baseline; unchanged)
**Criterion**: drift delta vs S37 < 0.5% (within gate noise)

### G3b — N=1k LG+Cosine

**Spec**: LG calls FBSPP exclusively. This fix removes H3 (FBSPP joint-skip). H1 residual
remains (separately investigated in S38 PROBE-G).
**Pre-fix (S37)**: 27-31% drift (ratios 1.274-1.311)
**Expected post-fix**: drop toward ST baseline (~14%) — FBSPP degenerate-partition skips
no longer suppress signal feature gains at small N
**Criterion**: observed drop from S37 baseline; absolute target is ST baseline ± noise

### G3c — N=2k LG+Cosine

**Pre-fix (S37)**: 43-45% drift (ratios 1.432-1.446)
**Expected post-fix**: drop from S37 baseline toward LG-at-N=2k H1 floor
**Criterion**: observed drop; same structural reasoning as G3b

### LG+Cosine N=50k smoke

**Pre-fix**: 0.382% (S33 Commit 3b `d599e5b033`)
**Expected**: ~0.382% or slight improvement; FBSPP fix only helps at small N where
degenerate-partition skip rate is higher (per-PROBE-E: skip rate grows monotonically
with depth; larger N = fewer degenerate leaves at any given depth)
**Criterion**: drift stays within [0%, 2%]

### L2 path smoke

**Spec**: Per-side mask for L2 is math no-op-but-correct (parent term cancels non-empty
contribution exactly for degenerate (p,k) cells). Loss curve should be byte-identical
pre/post fix.
**Expected**: byte-identical RMSE loss curve on any regression anchor
**Criterion**: byte-identical (0 ULP change)

### Kernel md5

`9edaef45b99b9db3e2717da93800e72f` (actual md5 to be confirmed post-build; must match
`9edaef45b99b9db3e2717da93800e76f` — ALL Metal shader files unchanged)

---

## Files Changed

- `catboost/mlx/tests/csv_train.cpp`: FBSPP one-hot k-loop (~L2295-2350) and ordinal
  k-loop (~L2380-2474) — joint-skip `continue` replaced with per-side mask declarations
  + asymmetric switch cases for L2 and Cosine. No other files changed.

---

## Note on H1

This commit is **sibling to S38 PROBE-G** (H1 investigation). H1 is a separate dominant
mechanism responsible for ~14% residual LG+Cosine drift at N=1k that persists even through
the FBSPP path. PROBE-G is investigating H1 independently and its findings do not affect
the correctness of this H3 fix. After both commits land, the LG+Cosine drift at N=1k is
expected to equal the H1 residual alone (~14%), and at N=50k to remain ~0.382%.
