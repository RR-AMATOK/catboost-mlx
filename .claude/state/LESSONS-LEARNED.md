# LESSONS-LEARNED.md — Cross-Project Knowledge Base

> A durable, additive repository of principle-level lessons. Not a diary; not a changelog.
> Every entry is a lesson that should transfer to the next project.

---

## Meta — Contribution Protocol

### What belongs here
- Principles, patterns, traps, and methodologies that transfer across projects.
- Framework / hardware gotchas that are not obvious from documentation.
- Debugging methodologies that proved their worth on a real bug.
- Sprint / process lessons that improved delivery cadence.
- Error loop resolutions — what was tried, what failed, what eventually worked.
- Negative results — approaches that were disproven with evidence.

### What does NOT belong here
- Project-specific file paths, function names, or commit hashes (use only in `(source: …)` footer).
- Results tables scoped to one benchmark harness.
- Architecture summaries (those belong in ARCHITECTURE.md).
- Open TODOs or active-sprint state (those belong in `.claude/state/`).

### How to append
- **New lesson in existing category** → add an `h3` under the matching `h2`.
- **New category** → add an `h2`, and log it in the Contribution Log below.
- **Superseding a prior lesson** → do not delete; append `Superseded YYYY-MM-DD by: …` and add the new lesson below.
- Each `h3` entry must have: one-line principle as the title, 2-3 sentences of context, optional code/formula, and a `(source: …)` footer.

---

## Probe Design

### Probes that compare two MLX paths cannot prove MLX-vs-CPU equivalence

**Date**: 2026-04-25
**Sprint**: [sprint-38]
**Tags**: [probe-design] [mlx-vs-cpu]

PROBE-G compared post-DEC-042 MLX against a pre-DEC-042 counterfactual constructed entirely
within MLX's execution path. Both the "postfix" and "prefix" columns in the diagnostics table
assumed MLX's code; neither column reflected CPU CatBoost's actual runtime behavior.
As a result, PROBE-G correctly confirmed the DEC-042 structural mechanism at small N (Scenario C,
d≥2 gap ≠ 0) but could not tell whether the 13.93% aggregate residual drift is caused by a
formula difference between MLX's per-side mask and CPU's `UpdateScoreBinKernelPlain`, or by
something else entirely (quantization borders, basePred initialization, leaf-value precision).

**Why this matters**: Cross-runtime divergence hunts require CPU-side instrumentation emitting
the same `(feat, bin, partition, gain)` tuples as the MLX-side capture. A probe that compares
two MLX paths (one actual, one counterfactual) can confirm whether a structural change within
MLX is active and correct, but it cannot confirm equivalence with CPU because it never observes
CPU's actual values. The confirmation and the equivalence proof are two different claims
requiring two different instruments.

**How to apply**: Future probes targeting MLX-vs-CPU drift should plan for the CPU hook
upfront. The hook design (file/function, tuple format, arming condition) should be specified
at probe-scoping time, not discovered after the MLX side returns inconclusive. Reusing MLX's
existing `PROBE_E_INSTRUMENT` column format as the reference schema for the CPU hook is the
recommended pattern — it makes the cross-runtime comparison a direct column join rather than
a semantic reconciliation.

*(source: `docs/sprint38/probe-g/FINDING.md` §Critical caveat; PROBE-H scope in `docs/sprint38/probe-g/FINDING.md` §Recommended next step)*

### Probe rubrics imported from a prior regime can mis-classify in a new regime

**Date**: 2026-04-25
**Sprint**: [sprint-38]
**Tags**: [probe-design] [classification] [rubric-validity]

PROBE-G imported PROBE-E's classification rubric (≫5% skip rate means active, similar skip
rate means Scenario C, etc.) without re-deriving the rubric's assumptions. PROBE-E operated
at N=50k where the degenerate-partition skip mechanism is the dominant structural driver. At
N=1k the rubric matched at d≤2 — the d=2 skip rate (5.36%) agreed with PROBE-E's 5.00%,
confirming Scenario C there. But at d≥3 the rubric produced a false-positive Scenario C
verdict: skip rates were 4× higher than PROBE-E's reference, but the per-bin contribution
magnitude had collapsed 25× and the drift curve showed no threshold knee. The rubric said
"high skip rate = active mechanism = same class"; the data said "high skip rate, noise-scale
per-cell effect, smooth curve = different class (precision/noise), higher skip count is a
symptom of leaf exhaustion, not topology."

**Why this matters**: Rubric portability requires checking that the rubric's implicit
assumptions hold in the new regime — not just that the observable inputs (skip rate) are
in the expected range. PROBE-E's Scenario C rubric assumed that skip-rate magnitude was
proportional to accumulated effect on the argmax. At small N that assumption breaks because
per-cell contribution magnitude drops independently of skip count.

**How to apply**: When importing a classification rubric from a prior probe or regime, list
the rubric's load-bearing assumptions explicitly before applying it. For each assumption,
check whether it holds in the new regime. If an assumption fails (e.g., "skip rate ∝
accumulated effect"), re-derive the rubric from first principles rather than importing the
verdict. The data tables (skip rates, gaps, magnitudes, drift curve shape) are facts; the
rubric and the verdict derived from them are not.

*(source: `docs/sprint38/probe-g/FINDING.md` §Classification — AMENDED; @devils-advocate stress-test, 2026-04-25)*

### Cross-runtime tree comparisons must control for quantization grid alignment before attributing split divergence to algorithm divergence

**Date**: 2026-04-25
**Sprint**: [sprint-38]
**Tags**: [probe-design] [cross-runtime] [quantization] [split-divergence]

When two runtimes produce different split choices on the same dataset, it is tempting to
attribute the divergence immediately to scoring formula differences. But a confound exists:
if the two runtimes use different quantization grids, their bin indices are not directly
comparable and their border values may not overlap. A runtime that "picks bin 73" may be
picking a border value that the other runtime never considered, not scoring the same border
differently.

F2 resolved this confound before attributing the split divergence to the scoring formula
(C-PSF). Test 1 checked whether CPU's borders are present in MLX's grid — all 11 CPU
borders (6 for feat=0, 5 for feat=1) were found in MLX's 127-entry grid to within 3.5e-8.
Only after confirming grid alignment did Test 2 ask whether MLX picks the nearest-to-CPU
border or a different one. In all three feature-matched depths, MLX has the CPU-preferred
border available and scores it lower — confirming the formula divergence. Without Test 1,
Test 2 would be meaningless: "MLX picked a different border" could mean "MLX doesn't have
that border" or "MLX has it and ranks it lower" — two very different causes requiring
different fixes.

**How to apply**: For any cross-runtime split comparison, first verify that the target
border appears in both border arrays (to ULP-level tolerance). If it does not, the
quantization grids diverge and a grid-alignment probe (like PROBE-Q) is required before
any formula-level comparison. If it does appear in both, proceed to Test 2: does the
other runtime pick that border or a different one? The two tests are sequential and each
rules out a distinct failure class (C-QG vs C-PSF). Building both into any cross-runtime
harness is cheap — it requires only a nearest-neighbor lookup in the border array — and
it prevents false attribution of formula errors to grid errors or vice versa.

*(source: `docs/sprint38/f2/FINDING.md` §Disambiguation; F2 analysis 2026-04-25)*

---

## Contribution Log

| Date | Change | Contributor |
|------|--------|-------------|
| 2026-04-25 | Added § Probe Design — first real entry | sprint-38 |
| 2026-04-25 | Added § Probe Design — rubric portability lesson (PROBE-G amendment) | sprint-38 |
| 2026-04-25 | Added § Probe Design — cross-runtime quantization grid alignment lesson (F2) | sprint-38 |
