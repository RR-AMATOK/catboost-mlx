# S45-T3 Decision Synthesis — DEC-048 KILL on H-Dispatch

**Date:** 2026-05-04
**Sprint:** 45
**Branch:** `mlx/sprint-45-perf-spike-and-decide`
**Owner:** @strategist (this document)
**Companion DEC:** DEC-048 in `.claude/state/DECISIONS.md`
**Verdict source:** `docs/sprint45/T2/probe-verdict.md`, `docs/sprint45/T2/analysis.json`

---

## Summary

S45 set out to test a single load-bearing throughput hypothesis (H-Dispatch) on the
v0.7.0 release path. T2's Probe-D execution falsified it by code inspection alone:
`DispatchHistogramBatched` already batches all feature groups into a single Metal
dispatch per depth level, the dispatch count is 6/iter on both Epsilon and Higgs-1M
(not ~3,000 as hypothesized), and dispatch overhead is 0.008% of iter wall-clock —
2,500× below the 20% Outcome C threshold. The "single multi-group dispatch" engineering
proposed for S46 is already the production implementation. **Decision: KILL the
dispatch-fusion lever permanently (DEC-048).** v0.7.0 path narrows to either holding
indefinitely or committing to a multi-sprint warp-shuffle redesign of the kernel — the
choice is the user's, post-S45 close-out.

---

## Numerical evidence

| Hypothesis claim | Predicted | Measured | Threshold | Outcome |
|---|---|---|---|---|
| Dispatches/iter on Epsilon (2000 feat) | ~3,000 | **6** | — | falsified (500× off) |
| Dispatches/iter on Higgs-1M (28 feat) | ~42 | **6** | — | falsified |
| Dispatch overhead vs iter wall-clock | "load-bearing" | **0.008%** | <20% (Outcome C) | C triggered (2,500× margin) |
| Step 3 single-dispatch upper bound | 1-day fix → ≥3× speedup | **already production code** | ≥3× (Outcome A) | A NOT triggered; C triggered |

Wall-clock baselines (seeds 42/43/44, mean): Epsilon iter=2000 = 2241.2 ms/iter MLX vs
140.9 ms/iter CPU (15.9×); Higgs-1M iter=200 = 132.9 ms/iter MLX vs 24.6 ms/iter CPU
(5.4×). Dispatch overhead = 6 × ~30 µs = 0.18 ms/iter — independent of feature count.

Code evidence: `catboost/mlx/methods/histogram.cpp:31–109` (`numGroups` in dispatch
grid X dimension); `catboost/mlx/methods/structure_searcher.cpp:60–108` (depth × approxDim
loop). Production dispatcher already fuses; no engineering remains.

---

## Process lesson

Six independent advisory agents (silicon-architect, mathematician, performance-engineer,
devils-advocate, strategist, visionary) converged on H-Dispatch from incorrect
arithmetic. The "2000 features ÷ 4 per group × 6 depth = 3,000 dispatches/iter"
calculation describes per-group dispatches; the production implementation has fused
this into one multi-group dispatch since well before S45. **One `grep` against
`histogram.cpp` would have refuted the entire hypothesis before the sprint scaffold
was cut.**

The arithmetic was correct. The architectural model was wrong. Consensus did not catch
it because consensus was operating on the same flawed mental model. The cost: one
sprint cycle and the 1-day Probe-D budget. The cost-avoidance: the Probe-D protocol
itself, which mandates code-path verification before engineering — exactly the check
that fired here at Step 1 and prevented the 1-day "single-dispatch upper bound"
implementation from being written and committed to no effect.

This is preserved in `Frameworks/LESSONS-LEARNED.md` at T6 close-out under the framing
"6-agent advisory consensus is not a substitute for code-path verification on
mechanism claims."

---

## Why this is NOT "throughput is permanently dead"

The sprint plan's risk-register meta-criterion uses the phrase "throughput epic is
retired permanently" conditional on T2 falsifying all hypotheses. This DEC must be
read precisely:

- **Killed by DEC-048:** The dispatch-fusion lever specifically. There is no version
  of "fuse the dispatches" that buys back >0.008% of iter wall-clock on Epsilon. This
  hypothesis is closed and forbidden from re-opening absent evidence of a regression
  to per-group dispatching.
- **NOT killed by DEC-048:** The `simd_shuffle_xor` serial chain in histogram
  accumulation, which S19-01c (DEC-020) attributed as 86% of accumulation cost and
  which scales linearly with batch-TG-ops (~200M for Epsilon vs ~438k for Higgs — a
  456× ratio that closely tracks the observed cross-class wall-clock gap). This lever
  was not in the S45 hypothesis set and was not measured by Probe-D.

The throughput epic is **narrowed** to one remaining lever, not "permanently retired."
Future readers must not infer from DEC-048 that all GPU-throughput work on
catboost-mlx is closed. The dispatch route is closed. The kernel-architecture route
remains open as a separate scope decision.

---

## What's next

The v0.7.0 release path narrows to two options, both pre-flagged by the user as
acceptable. T3 records the verdict and frames the choice; T3 does NOT make the choice
between A and C — that is the user's call after T6 close-out.

### Path A — Hold

v0.7.0 deferred indefinitely. The project remains at v0.6.1 until either a
research-grade lever appears or the audience proposition shifts. Branch-B regression
gate (T1) protects v0.6.1's bit-equivalence indefinitely. T4 (cross-class CUDA
writeup) and T5 (`catboost-tripoint` parity oracle) ship as v0.6.x extensions of the
reproducibility-grade frame. PyPI publish does not happen on the dispatch route.

**When this is the right call:** if the user values shipping the reproducibility-grade
frame as the durable v0.x story over committing multi-sprint engineering to a
hardware-class-bound gap; if the cross-class CUDA delta (88× on Epsilon) is judged
structurally bound by hardware physics rather than addressable by software.

### Path C — simd_shuffle warp-shuffle redesign

Multi-sprint redesign of the histogram accumulation kernel targeting the S19-01c
bottleneck (86% of accumulation cost). New scope decision; NOT gated by DEC-048.
Requires its own probe + decision sequence: identify the warp-shuffle reduction
strategy, prototype, validate against Branch-B regression test, measure delta on
Epsilon iter=2000 at production shape.

**When this is the right call:** if the user judges that the 456× batch-TG-ops ratio
between Epsilon and Higgs is the actual addressable mechanism (per S19-01c), is
willing to commit a multi-sprint engineering window to test it, and can accept the
risk that the redesign may itself fail to land ≥3× on Epsilon (in which case Path A
becomes the fallback).

### Decision blockers

Neither path is recommended in this T3 document. Both are framed honestly with their
trigger conditions. The strategist's role here is to record the verdict cleanly, not
to pre-empt the user's call between A and C. T6 close-out happens regardless of which
path is chosen — DEC-048 stands either way.
