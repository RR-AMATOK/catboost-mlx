# S49-T4 — Quick Bundle 2 Measurement (Higgs-1M Logloss)

**Date:** 2026-05-18
**Sprint:** S49
**Branch:** `mlx/sprint-49-c6-engineering`
**Verdict:** **Outcome C — RETIRED-EMPIRICALLY**

---

## Method

User selected (b) Quick T4 first per S49-T4 sprint-plan §3 decision rule — single-seed Higgs-1M iter=200 Logloss measurement on baseline (Option B patch active, C6 disabled) vs C6 (patch reverted), to validate the speedup story before committing 2-3 hours to full T4.

## Result

| Configuration | Train time |
|---|---|
| Baseline (C6 disabled) | 27.08s |
| C6 active | 27.14s |
| **C6/baseline ratio** | **1.002× (C6 ~0.2% SLOWER)** |

Per Q3 lock (DEC-052 T0c AMENDMENTS):
- **Outcome A trigger:** ≥1.7× iter speedup on Higgs-1M → NOT MET (1.002× << 1.7×)
- **Outcome B trap-zone:** 1.5–1.7× → auto-retire under sunk-cost rail → NOT REACHED
- **Outcome C:** <1.5× → retire empirically; pivot to ordered boosting per T0c Q3 → **FIRED**

## Sanity check: C6 path was exercised

Confirmed via T3 max-1-ULP delta on Logloss (S49-T3 envelope sweep). If C6 path had been silently inactive, parity would be 0 ULP everywhere. The 1-ULP delta proves the subtract kernel ran for Logloss configurations.

Verbose training output also confirmed: training completed normally, no errors, no crashes; `hist=13.1ms` per-iter profile consistent with v0.7.0 baseline production path.

## Why no measurable speedup (hypotheses)

Three plausible explanations for the disappointing result:

1. **Dispatch overhead absorbs the savings.** C6 adds ~5+ MLX primitive ops per depth (smaller-child mask construction, smaller-child dispatch, subtract, assembly via where/tile/reshape). On Higgs-1M's already-fast histogram (~135ms per iter at depth 6), the per-depth additional dispatch latency may equal the per-depth savings from halving histogram work.

2. **Lazy-graph evaluation materializes the same work.** Even though we "skip" larger-child computation, the lazy graph may evaluate the same data path through other consumers downstream (score-calc, partition-update).

3. **Downstream consumer requires both children.** If score-calc or another stage reads BOTH children's histograms (rather than just the assembled output), the "saved" larger-child histogram has to be computed regardless.

**These hypotheses are documented for potential future revival, not investigated here.** Per Q3 lock, the sunk-cost rail fires automatically at <1.5×; further diagnosis would risk the exact sunk-cost ratchet the rail was designed to prevent.

## What this measurement is NOT

- NOT a full 3-seed median (single seed)
- NOT iter=1000 (would be 5× longer, scaling linear in iter count; per-iter ratio constant)
- NOT Epsilon iter=2000 (Bundle 2 hard gate also requires Epsilon ≤5×)
- NOT a per-kernel profile breakdown (couldn't decompose where the savings went)

But the per-iter ratio is what matters for Outcome A. The 1.002× ratio at iter=200 will not improve at iter=1000 — the ratio is dispatch-cost-vs-savings per depth, not a function of tree count.

Full T4 would have produced more precise numbers (3-seed median CV<8%, both datasets) but the verdict would be the same: nowhere near 1.7×.

## Result classification

**Type of failure:** 8th throughput-hypothesis falsification on this codebase. First one that:
- Passed pre-flight gates (S48 T0/T1/T2/T3) without falsification
- Passed engineering implementation (S49 T1/T2)
- Passed all parity gates (T2.12 Branch-B GREEN, T2.13 RMSE bit-identical, T3 envelope ALL_PASS with max 2 ULP)
- Was code-inspection verified (10/10 pre-conditions VERIFIED in static analysis)
- ...but failed the final empirical measurement gate

**New failure category for LESSONS-LEARNED:** "Workload-reduction lever with correct mechanism + perfect parity that delivers no measured speedup due to dispatch/synchronization overhead absorbing the theoretical savings."

## DEC-052 OUTCOME revision

DEC-052 OUTCOME A (locked 2026-05-13) revised to **RETIRED-EMPIRICALLY** at S49-T4 (2026-05-18).

## Next step (per Q3 lock)

S50 pivots to **ordered boosting** kickoff (DEC-052 T0c Q3 pre-decided pivot target). Throughput epic for v0.8.0 is retired. PyPI publish per DEC-051 remains gated; no path via histogram-internal levers.
