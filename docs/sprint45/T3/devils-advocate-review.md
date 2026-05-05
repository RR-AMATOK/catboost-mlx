# S45-T3 Devil's Advocate Review — DEC-048 Stress Test

**Date:** 2026-04-30 | **Reviewer:** @devils-advocate
**Subject:** @strategist's proposed DEC-048 = KILL with "narrowed not killed" scope

## 1. Verdict color: YELLOW

GREEN on `KILL` for the dispatch lever — T2 is decisive (`histogram.cpp:31` numGroups grid is production code, not hypothesis). YELLOW on "narrowed not killed": defensible only with §4's hard gate. Without it, "narrowed" is the seventh escape hatch in a chain.

## 2. Biggest weakness in "narrowed not killed"

**Definitional drift evades the meta-criterion.** Plan said "if T2 falsifies all hypotheses, the epic is retired permanently." It was narrowed *pre-execution* from {H-Sibling, H-Bandwidth, H-Sync} to {H-Dispatch alone} — citing priors that already adjudicated the others (Bandwidth: 0.007–0.13% roof; Sibling: 10–15% per S22; Sync: silicon-architect April). T2 falsified the survivor. "Broader question open" reopens what was closed *precisely* to justify the narrow probe.

This is the DEC-013/014/015/017/019 pattern: each lever dies with evidence, the epic survives as "next lever might work." Five prior falsifications; T2 is the sixth in three sprints. Plain reading: Outcome C ⇒ retire epic.

## 3. Pre-mortem — 6 months if simd_shuffle also falsifies

S19-01c's own conclusion: shuffle chain "requires algorithmic restructuring." Sprint 20 was supposed to address it; didn't ship. DEC-017 (T3b atomic-CAS alternative) regressed +42.3%. No one-sprint shuffle plan exists 6 months on. **Likely state:** S46 verdict shape-identical to T2. DEC-049 = KILL with "MSLR pivot remains." S47–S48 burn on next narrowed lever. v0.6.1 indefinitely. Sunk cost is no longer hours — it is epistemic credibility.

## 4. Gate required before any future perf-pivot plan

**MANDATORY-CODE-INSPECTION:** Before agent-panel consensus on a perf hypothesis enters a sprint plan, the named function/kernel/dispatch site MUST be read end-to-end by one agent and cited (file:line) in the hypothesis. Arithmetic-derived mechanism claims without a source-line refutation step are inadmissible.

Six agents converged on H-Dispatch from arithmetic; one grep refuted it. Missing-gate cost: a sprint. Gate cost: ~15 min per hypothesis. T6 must commit this to LESSONS-LEARNED.md as a hard rule.

## 5. v0.7.0 honest call

User's "no PyPI without perf" **does not survive T2 = Outcome C unmodified.** Three options, ordered by honesty:

1. **Retire v0.7.0 throughput framing.** Ship v0.7.0 = reproducibility extension + parity oracle + cross-class CUDA receipts (T4 + T5). Drop perf gate. Proposition shifts from "fast GBDT" to "auditable cross-platform GBDT." Consistent with v0.6.1. **Recommended.**
2. **Defer v0.7.0 indefinitely.** Honest but cedes momentum.
3. **Lower the perf bar** (e.g. ≥1.2×). Loses user trust. **Reject.**

User's position is perf-as-*gate*, not perf-as-*goal*. Option 1 honors the gate by removing it from v0.7.0's path while preserving it for v0.8.0 *if* a real lever emerges. Surface explicitly to user — do not paper over.

## Bottom line

GREEN on KILL dispatch. YELLOW on "narrowed not killed" unless §4 gate commits. Recommend v0.7.0 reframe to reproducibility-extension. Throughput epic is one falsification from permanently dead by the meta-criterion's plain reading; future perf work should bear the burden of proof.
