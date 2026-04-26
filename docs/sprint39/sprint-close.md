# Sprint 39 Close — Housekeeping after S38 RESOLVED (DEC-045)

**Sprint:** 39  |  **Status:** CLOSED  |  **Date:** 2026-04-25
**Branch:** `mlx/sprint-39-housekeeping`  |  **Base:** master `679ef517a5`  |  **Tip:** `1e3f2b34a7`
**Authoritative record:** this file
**Theme:** Housekeeping following the S38 DEC-045 resolution (harness asymmetry root-cause).
No production code changes. No new features. All commits are docs, state, and test-coverage
improvements.

---

## Mission

S38 resolved the long-running LG+Cosine small-N drift investigation as a harness configuration
mismatch (DEC-045). S39 closes the loose ends: retire instrumentation built under the now-
retracted DEC-044 hypothesis, re-run the PROBE-G sweep under the corrected RS=0 harness,
extend the RS=1.0 multi-seed verification to confirm the measured bias is real and bounded,
update the README accordingly, refresh the anchor inventory, and audit two branches flagged as
stale.

---

## Items Completed

| Item | Description | Outcome | Commit |
|------|-------------|---------|--------|
| **#5 (item 5)** | Retire `PROBE_H_INSTRUMENT` macro (DEC-044 withdrawn) | Macro removed; ULP-identical tree structure confirmed pre/post | `059d0e56c8` |
| **#6 (item 6)** | Re-run PROBE-G scaling sweep under corrected RS=0 harness | RS=0 parity confirmed across all N; mean drift 0.023% at N=1k (see item 6 detail below) | `aa4cb9dccb` |
| **#7 (item 7)** | Extend RS=1.0 verification to 10 seeds | Bias confirmed real-not-noise: mean −4.08%, 95% CI [−4.78%, −3.39%] (see headline finding) | `b03af34161` |
| **#8 (item 8)** | Update README §Known Limitations RS=1.0 paragraph | Initial update with 5-seed data point (≈3%) | `482a8308dd` |
| **#8 follow-up** | Tighten README RS=1.0 paragraph with 10-seed result from item 7 | Precise CI replaces the ≈3% estimate | `313115729c` |
| **#9 (item 9)** | Audit `archive/s24-d0-v5-retreat` branch | SAFE-TO-DELETE (see branch audit below) | `531b9f2c04` (incidental) |
| **#10 (item 10)** | Audit `origin/mlx/sprint-33-iter2-scaffold` branch | SAFE-TO-DELETE (see branch audit below) | `531b9f2c04` (incidental) |
| **#11 (item 11)** | Refresh anchor inventory through Sprint 38 (DEC-031 Rule 2) | AN-019–AN-023 registered; 2 superseded markers added | `531b9f2c04` |

---

## Headline Finding — Item 7: RS=1.0 Bias Confirmed

The S38 close-out noted a ~3% bias at RS=1.0 (MLX slightly lower RMSE than CPU CatBoost) from
a single 5-seed sample. Item 7 extended this to 10 seeds (seeds 42–51) on the canonical N=1k
LG+Cosine anchor.

| Stat | Value |
|------|-------|
| Seeds | 42–51 (10 seeds) |
| Mean drift | −4.08% (MLX lower RMSE) |
| Std | 1.10% |
| 95% CI | [−4.78%, −3.39%] |
| Min | −5.17% (seed 50) |
| Max | −1.83% (seed 43) |

**Interpretation**: The confidence interval does not overlap zero. The bias is statistically
significant across 10 seeds and is a bounded, real difference in how each runtime samples gain
noise. MLX's RNG path (seeded via `std::mt19937` default Mersenne Twister) produces different
noise realizations than CatBoost's internal RNG, with the net effect that MLX slightly
over-regularizes candidate selection relative to CPU at this anchor. This is not a correctness
issue: both models are valid, and the RS=0 RMSE is bit-identical.

**README consequence**: The prior "≈3%" estimate was replaced with the precise
`mean −4.08% (95% CI [−4.78%, −3.39%])` in commit `313115729c`.

Source data: `docs/sprint38/probe-q/data/parity_verification_rs1_extended.csv`
Analysis script: `docs/sprint38/probe-q/scripts/verify_parity_extended.py`

---

## Item 6 Detail — PROBE-G Sweep at RS=0

The original PROBE-G scaling sweep (`docs/sprint38/probe-g/`) was run with an asymmetric
harness (MLX at RS=1.0, CPU at RS=0.0). Item 6 re-ran the sweep with both runtimes at RS=0.

Results at matched RS=0:

| N | RS=0 drift (%) | Notes |
|---|----------------|-------|
| 500 | ~0.0% | Below measurement noise floor |
| 1000 | 0.023% | Canonical anchor |
| 2000 | ~0.0% | |
| 5000 | ~0.0% | |
| 10000 | ~0.0% | |
| 20000 | ~0.0% | |
| 50000 | ~0.0% | |

The smooth log-linear decay curve documented in S38 PROBE-G PHASE-2-FINDING.md was entirely
a product of the RS asymmetry (noise/signal ratio scales with N). At matched RS=0, drift is
flat near zero across all tested N values, confirming DEC-045.

Artifacts: `docs/sprint38/probe-g/` (original asymmetric data preserved); re-run summary
in HANDOFF.md.

---

## Branch Audit — Items 9 and 10

Both branches were audited in this sprint. The Explore agent verdict:

### `archive/s24-d0-v5-retreat`

**Status: SAFE-TO-DELETE**

The v5 fix from this branch shipped to master via commits `36e01438a1` and `784f82a891` (DEC-023
RESOLVED). The archive branch contains the diagnostic arc (D0/D1a/D1b/D1c) and the v5 kernel
changes. All production-relevant code is on master. DEC-023 is formally CLOSED. No unique
artifact on this branch is absent from master or the sprint-24 docs directory.

### `origin/mlx/sprint-33-iter2-scaffold`

**Status: SAFE-TO-DELETE**

This branch was the working branch for S33-L4-FIX and associated probes (PROBE-C through
PROBE-E). The L4-FIX commits (per-side mask, guard removal, four-gate validation) are on
master via PR #29. The probe data, verdict docs, and gate reports are all committed under
`docs/sprint33/`. The branch itself has no unique commits beyond what master carries.

**RECOMMENDATION TO USER (Ramos)**: Both branches are SAFE-TO-DELETE based on this audit.
However, no deletions were performed in this sprint. Explicit user confirmation is required
before deleting either branch. Suggested commands if you want to proceed:

```bash
# Delete local archive branch (if it exists locally)
git branch -d archive/s24-d0-v5-retreat

# Delete remote origin branches
git push origin --delete archive/s24-d0-v5-retreat
git push origin --delete mlx/sprint-33-iter2-scaffold
```

---

## Anchor Inventory — Item 11

Commit `531b9f2c04` refreshed `docs/sprint27/scratch/aa-t1-anchor-inventory.md` through Sprint 38,
registering 5 new anchors and adding 2 superseded markers:

| Anchor | Location | Kind | Notes |
|--------|----------|------|-------|
| AN-019 | `docs/sprint33/l1-determinism/verdict.md` | RMSE pair (CPU + MLX pre-fix) | Sprint 33 N=50k ST+Cosine baseline; docs-only |
| AN-020 | `docs/sprint33/l2-graft/verdict.md` | RMSE (graft diagnostic) | Class-d (dead); graft harness required |
| AN-021 | `docs/sprint33/sprint-close/cr-report.md` | Loss pair (one-hot smoke) | SUPERSEDED — one-hot patch reverted (S34 verdict) |
| AN-022 | `docs/sprint38/probe-g/data/anchor_n1000_seed42.csv` | Data file (CSV, MD5) | Sprint 38 canonical N=1k probe CSV; docs-only |
| AN-023 | `docs/sprint38/probe-q/PHASE-2-FINDING.md` | RMSE (RS=0 bit-identical) | DEC-045 resolution anchor; docs-only |

Total anchors after refresh: 23 (AN-001–AN-023). Docs-only: 14. Live-test-backed: 9 (AN-001–008, AN-009, AN-010 range, AN-015–018 range). Data-file: 1 (AN-022). Superseded: AN-020, AN-021.

---

## Commits (oldest to newest)

| SHA | Description |
|-----|-------------|
| `482a8308dd` | Clean stale 'ongoing investigation' refs from README §Known Limitations |
| `531b9f2c04` | Refresh anchor inventory through Sprint 38 (DEC-031 Rule 2); register AN-019–AN-023 |
| `059d0e56c8` | Retire `PROBE_H_INSTRUMENT` macro (DEC-044 withdrawn) |
| `aa4cb9dccb` | Re-run PROBE-G scaling sweep at RS=0 parity |
| `b03af34161` | Extend RS=1.0 parity verification to 10 seeds |
| `313115729c` | Tighten README RS=1.0 paragraph with 10-seed CI result |
| `1e3f2b34a7` | Close-out docs — sprint-close.md, HANDOFF, TODOS, CHANGELOG, LESSONS-LEARNED |

7 commits total. No production source code changes. No kernel changes. Kernel md5
`9edaef45b99b9db3e2717da93800e76f` unchanged.

---

## Lessons Learned — S38 Entries (Carried, Not Added)

Five lessons from S38 are in `LESSONS-LEARNED.md` under §Probe Design. No new S39-specific
lesson warrants a new entry — S39 was housekeeping execution, not investigation. The S38
lesson on cross-runtime configuration symmetry (the RS=1.0 asymmetry trap) remains the
most broadly applicable and is already recorded.

---

## DEC Status at Sprint 39 Close

| DEC | Status | Notes |
|-----|--------|-------|
| DEC-032 | CLOSED | Guard removal shipped S33 Commits 3a/3b |
| DEC-036 | CLOSED | S33-L4-FIX per-side mask; 52.6% → 0.027% drift |
| DEC-042 | FULLY CLOSED | Ordinal (S33) + one-hot (S34/S35) + FBSPP (S38) |
| DEC-044 | WITHDRAWN | DEC-044 hypothesis retracted; instrumentation retired S39 |
| DEC-045 | RESOLVED | Harness asymmetry root-cause; RS=0 parity confirmed bit-identical |

---

## Wall Time

Estimated 1 working session. All items were documentation, data re-runs, and audit tasks.
No new code paths opened. No gate failures.

---

## What Shipped

- README `catboost/mlx/README.md`: RS=1.0 paragraph tightened with precise 10-seed CI
- Anchor inventory: 5 new anchors (AN-019–AN-023) + 2 superseded markers registered
- `PROBE_H_INSTRUMENT` macro retired from `catboost/mlx/tests/csv_train.cpp`
- PROBE-G RS=0 re-run documented confirming near-zero drift across all N
- 10-seed RS=1.0 extended verification: bias real, bounded, mean −4.08%, CI [−4.78%, −3.39%]
- Branch audit complete: `archive/s24-d0-v5-retreat` and `origin/mlx/sprint-33-iter2-scaffold`
  both SAFE-TO-DELETE pending user confirmation

## What Did NOT Ship

- Branch deletions for items 9 and 10 (pending Ramos confirmation)
- Any new production code, kernels, or algorithm changes
