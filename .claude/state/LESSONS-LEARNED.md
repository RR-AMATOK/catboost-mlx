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

### Formula equivalence proofs must treat the boundary case wSide ≈ 0 as a first-class test vector

**Date**: 2026-04-25
**Sprint**: [sprint-38]
**Tags**: [probe-design] [formula-equivalence] [boundary-conditions] [small-n]

Two Cosine scoring formulas that agree on the main path (no degenerate partitions) can produce
radically different argmax rankings once degenerate partitions appear. PROBE-H showed that the
old joint-skip formula and CPU's per-side mask formula produce identical gains at d=0 (no
degenerates). At d=2–5 where degenerate partitions appear, the old joint-skip places the signal
feature (feat=0) in rank 2224–2336 out of 2540 candidates (bottom decile), while the per-side
mask places it first. The gain delta reaches +5.06 units at d=3. A math equivalence argument
that ignores `wSide = 0` cases would declare these formulas identical; an empirical probe with
real data at N=1k (where wSide = 0 is frequent after earlier signal-correlated splits) immediately
refutes it.

**Why this matters**: For any pair of scoring formulas differing only in how they handle degenerate
inputs (empty leaves, zero weights, zero gradients), the boundary case IS the case that matters
at small N and at large depth. Two formulas may agree asymptotically yet diverge catastrophically
at finite samples. A proof of equivalence that doesn't enumerate the boundary cases is
incomplete — especially when the boundary case is not pathological but structurally guaranteed
(e.g., after a split on feat=0, every subsequent evaluation of feat=0 bins will produce empty
right-children in the partitions where the d=0 split already sent everyone to the right).

**How to apply**: When proposing that formula A is equivalent to formula B:
1. List all inputs for which formula A and B diverge (boundary cases: zeros, infinities, ties).
2. Check whether those inputs arise in the target training regime (N size, depth, feature correlation).
3. Write an explicit test case at the boundary. For Cosine scoring: run at depth ≥ 2 with a
   feature that was used at depth 0, verify that the argmax rankings match at ALL depths.
4. Any fp comparison `w < eps` (for eps > 0) creates a small-N divergence class that any
   correctness proof must account for.

*(source: `docs/sprint38/probe-h/FINDING.md`; PROBE-H analysis 2026-04-25; DEC-044)*

### Counterfactual analysis must be clearly distinguished from observational analysis in probe scripts

**Date**: 2026-04-25
**Sprint**: [sprint-38]
**Tags**: [probe-design] [analysis-error] [counterfactual]

PROBE-H's `analyze_probe_h.py` computed "what MLX would produce under the OLD joint-skip formula"
by applying the joint-skip formula to PROBE-E's `mlxTermNum/mlxTermDen` fields (which capture the
old formula by construction) and named the result `gain_mlx_formula`. It then compared this
counterfactual column against `picked_by_mlx` (the binary's actual output under the correct
per-side mask, shipped since S33-L4-FIX commit `10c72b4e96`). The difference was misread as
evidence the binary used the old formula. In reality, the difference was expected: the two
columns measured different things. `analyze_probe_h_v2.py` Correction 1 confirmed the binary's
formula produces gain values agreeing with CPU's formula to within 1.37e-13 — numerical noise.

**Why this matters**: Counterfactual columns simulate "what would happen under a different code
path." Observational columns record "what the binary actually did." When a script computes both
and names them without making the distinction explicit, a reader (or future agent) can easily
confuse the counterfactual for the observational. If the two disagree, the natural interpretation
is that the binary uses the hypothetical path — when in fact it just means the two formulas
produce different numbers on the same input, which is always true when they differ in semantics.

**How to apply**: Counterfactual columns in probe scripts must have names that make the
hypothetical explicit, e.g. `gain_under_old_joint_skip_counterfactual` not `gain_mlx_formula`.
Any code block that compares a counterfactual column against an observational column must
include a comment: `# NOTE: comparing counterfactual (simulated) vs observational (binary actual)`.
Before concluding "the binary uses formula X because counterfactual_X matches observational_Y",
verify by direct code-reading which formula is actually in the binary.

*(source: `docs/sprint38/probe-h/FINDING.md` §Original analysis error; `analyze_probe_h.py` vs
`analyze_probe_h_v2.py`; DEC-044 withdrawal 2026-04-25)*

### Cross-runtime parity tests must verify SYMMETRIC configuration before interpreting drift

**Date**: 2026-04-25
**Sprint**: [sprint-38]
**Tags**: [probe-design] [harness] [cross-runtime] [configuration-asymmetry]

Sprint 37 #113 T3 G3b/G3c flagged a 13-44% drift at small N for LG+Cosine. Sprint 38 spent
four probes (PROBE-G, F2, PROBE-H, PROBE-Q phase 1) hunting algorithm-class mechanisms:
formula divergence, quantization granularity, per-bin precision, instrumentation contamination.
Each probe produced an internally-consistent verdict that was later retracted. The actual
mechanism: the comparison harness invoked CPU CatBoost with `random_strength=0.0` (explicit
deterministic) but invoked MLX `csv_train` at its default `RandomStrength=1.0` (noisy).
MLX's argmax was perturbed by random noise; CPU's was not. The 13.93% drift was the gap
between MLX-with-noise and CPU-deterministic — entirely a configuration artifact.

**Why this matters**: A 13% drift number across 5 seeds feels like a real bug because it's
reproducible and seed-stable. The reproducibility came from the asymmetry being deterministic,
not from an algorithm defect. Probes built on the contaminated harness can produce verdicts
that look causal but are observing the same configuration drift through different lenses.
Verdict diversity does not imply mechanism diversity if all probes share the same input
pipeline.

**How to apply**: The first move in any cross-runtime drift investigation is a configuration
sanity-print:

```
For each parameter that controls training behavior:
  print(name, value_passed_to_runtime_A, value_passed_to_runtime_B, "EQUAL" if equal else "ASYMMETRIC")
```

If any line says ASYMMETRIC, fix it before opening any probe. Special attention to: random
seeds, random strength / regularization noise, bootstrap type, sub-sampling rates, default-
parameter handling (one runtime's default may differ from the other's). For configurations
where defaults differ between runtimes, ALWAYS pass the value explicitly on both sides
even when it equals the default — defaults are the most common source of silent asymmetry.

**Configuration bisection rule**: when drift is mysterious, vary ONE parameter at a time
between MATCHED and DEFAULT. Halve the difference. The parameter at which drift collapses
is the cause. This is cheaper than any algorithm-class probe and should be the FIRST check,
not the last.

*(source: `docs/sprint38/probe-q/PHASE-2-FINDING.md`; PROBE-Q phase 2 analysis 2026-04-25;
S38 close-out: 12/12 tree splits match at parity, 0.000% RMSE drift)*

---

## Noise-Driven Algorithms

### RNG-implementation differences between runtimes can produce bounded but persistent bias — verify with multi-seed sweep before declaring parity

**Date**: 2026-04-25
**Sprint**: [sprint-39]
**Tags**: [cross-runtime] [rng] [bias] [multi-seed]

After Sprint 38 confirmed that the small-N LG+Cosine drift was a harness configuration
mismatch (asymmetric RandomStrength), a residual question remained: with both runtimes at
default RS=1.0, is the single-seed RMSE delta (≈ −3.66%) just sampling noise, or a real
structural bias? A 5-seed sample produced a mean of −3.43% (sprint 38 close-out). Sprint 39
extended this to 10 seeds and found: mean −4.08%, 95% CI [−4.78%, −3.39%]. The CI excludes
zero — the bias is real, bounded, and reproducible.

**Why this matters**: Two runtimes sharing the same algorithm can still produce systematically
different outputs when their underlying RNGs draw from different distributions or have different
seeding conventions. A single-seed comparison cannot distinguish "this seed's noise happened to
favor one side" from "this algorithm systematically favors one side". Even a 5-seed sample may
not be enough if the effect size is small relative to per-seed variance. A 10-seed sweep with
a 95% CI explicitly testing whether zero is excluded is the minimum credible parity claim for
noise-driven algorithms. For CatBoost-MLX specifically: MLX's `std::mt19937` (Mersenne Twister)
and CatBoost's internal RNG produce different noise realizations at the same integer seed, with
the net effect that MLX slightly over-regularizes gain selection relative to CPU at small N.

**How to apply**: For any two runtimes that share a noise/regularization parameter (RandomStrength,
dropout rate, Bernoulli sampling, etc.): (1) run RS=0 parity first to confirm algorithm
correctness; (2) run RS>0 multi-seed sweep (minimum 10 seeds) to characterize the RNG-induced
bias; (3) compute the 95% CI; (4) if the CI excludes zero, document the bias as a known
bounded difference, not a correctness issue; (5) update documentation with the precise CI
rather than a single-seed estimate. Do not report "≈X%" based on fewer than 5 seeds for
noise-driven algorithms — the per-seed variance is typically comparable to or larger than
the bias magnitude.

*(source: `docs/sprint38/probe-q/data/parity_verification_rs1_extended.csv`;
`docs/sprint38/probe-q/scripts/verify_parity_extended.py`; sprint-39 RS=1.0 10-seed
verification 2026-04-25)*

---

## Cross-Runtime Triage

### Decompose a cross-runtime ML port "floor" into seed-noise + architectural + encoding layers before opening a mechanism investigation

**Date**: 2026-04-26
**Sprint**: [sprint-40]
**Tags**: [cross-runtime] [decomposition] [triage] [methodology] [release-readiness]

When a port (CatBoost-MLX vs CatBoost-CPU, or any analogous cross-runtime ML
implementation) shows a residual prediction-disagreement floor that cannot be
attributed by code reading alone, the temptation is to open a mechanism
investigation — a multi-week PROBE-style escalation that examines candidates
sequentially (M1 multiclass dispatch, M2 encoding RNG, M3 fp precision, M4
quantization borders, …). Sprint 38 history shows this approach has a base rate
of ~50% verdict retraction and routinely overruns its time-box (5-day cap →
4-week reality). Three cheap experiments, run in ~1 day combined, can replace
the multi-week investigation entirely *for the purpose of a release decision*
and often partially *for the purpose of mechanism identification*.

**The triage**:

1. **Arithmetic reconciliation** (5 min): if the residual gap is reported via a
   single headline metric, verify the metric definition and confirm the gap
   magnitude is consistent with the disagreement count. Mismatches frequently
   trace to (a) plain-vs-balanced accuracy confusion on imbalanced data,
   (b) public-vs-private leaderboard subset scoring, (c) tie-breaking
   differences. *All three are silent; all three look like algorithm bugs.*

2. **Encoding-mechanism discriminator** (~30 min wall time): re-train both
   runtimes with the suspected encoding mechanism removed (for CatBoost,
   `cat_features=[]`; for any port with categorical/embedding/CTR paths, drop
   to numeric-only). The delta in disagreement count and rare-class shift
   isolates the encoding-attributable contribution from the architectural
   floor. If the rare-class asymmetry collapses dramatically, the encoding
   path is the dominant driver — and the architectural floor is bounded by
   the residual.

3. **Same-runtime noise floor** (~10–15 min wall time): train the *reference*
   runtime alone at 5 different seeds (with the same noise-injection settings
   as the comparison run); compute pairwise prediction agreement and rare-class
   shift across all 10 pairs. This establishes the irreducible RNG-driven floor
   for that algorithm class on that dataset. Cross-runtime gaps of less than
   ~2× this floor in disagreement count rarely require mechanism investigation;
   gaps of 5–10× in rare-class shift point at a specific encoding asymmetry
   rather than uniform architectural drift.

**Why this works**: the three experiments produce a *3-row decomposition table*
that names every component and bounds its contribution. A release narrative
built on "X% of the gap is seed-noise, Y% is architectural floor, Z% is one
specific identified mechanism" is materially stronger than "we couldn't close
a 0.28pp gap, but synthetic anchors agree." The decomposition either
sufficiently characterizes the variant for a release (Lane B exit) or sharpens
the mechanism investigation to a single isolated component (narrow Lane D
scope) — in either case, the 1-day investment dominates the multi-week
alternative.

**How to apply**: at the end of a sprint that closes synthetic-anchor parity
but leaves a real-world floor open, run all three experiments before triggering
a follow-on mechanism sprint. Report the decomposition. If the three layers add
to ≥80% of the observed gap and no single layer is unbounded, the variant is
characterized — proceed to release. If a single layer dominates and is
isolatable (e.g. CTR encoding path), open a *narrow* targeted investigation
limited to that mechanism only, with a 3-day kill-switch. Skip the multi-month
PROBE-style cascade unless the decomposition shows a layer larger than 50% AND
not isolatable AND likely to grow on different datasets.

**Counter-example check** (when this triage is insufficient): if disagreements
are concentrated on the *common* classes rather than rare classes, OR if
removing categoricals does *not* meaningfully reduce the rare-class shift, OR
if same-seed CPU-vs-CPU comparison is suspiciously deterministic
(suggesting RNG was somehow disabled), then the decomposition is not
representative and a mechanism investigation IS warranted. The triage is a
release-readiness filter, not a substitute for fundamental correctness review.

*(source: `docs/sprint40/pre_lane_check/FINDING.md`; `docs/sprint40/pre_lane_check/scripts/exp{2,3}_*.py`; `docs/sprint40/pre_lane_check/results/exp{2,3}_*.json`; DEC-046)*

---

---

## SIMD Histogram Kernel Routing Invariant

### In SIMD histogram kernels, processor-lane and owner-lane are independent random variables

**Date**: 2026-05-05
**Sprint**: [sprint-46]
**Tags**: [simd] [histogram] [kernel-correctness] [routing]

In histogram accumulation kernels with per-lane register accumulators, the lane that
processes a document (selected by doc index, e.g. `d & 31`) and the lane that owns
that document's target bin (selected by bin value, e.g. `bin & 31`) are statistically
independent under any realistic bin distribution. P(processor == owner) = 1/SIMD_SIZE.

S46 Probe B (`kernel_sources.h:1374-1407`) removed the `simd_shuffle(packed, src)`
broadcast believing "each lane's partial sum was already complete for its owned bins."
That comment was wrong. Without routing — either shuffle (the cost Probe B tried to
remove) or pre-sort by bin — approximately (1 − 1/SIMD_SIZE) of contributions are
silently discarded. Probe B measured a 9.79× iter "speedup" that was 96.9% missing
work, parity broken from iteration 0.

**How to apply:**
1. When reviewing any kernel touching `simd_shuffle` / `simd_shuffle_xor` in a
   histogram or scatter context, require an explicit account of how each (bin, value)
   pair reaches the lane that owns that bin.
2. Treat any code comment asserting "per-lane partial IS the full sum for that lane"
   as suspect until verified against the conditional that gates accumulation.
3. If a probe shows speedup before parity sign-off is complete, treat the speedup
   as zero. Parity-before-speedup is mandatory.
4. Any "remove the broadcast" proposal must disclose its routing replacement.
   No replacement = structurally incorrect — not a candidate.

**Process gap surfaced:** code inspection of Probe B happened post-measurement, not
pre-sweep. Future probe specs MUST require code-inspection sign-off on the probe
kernel's accumulation invariant BEFORE the sweep runs. The sign-off question is:
"How does each (bin, stat) pair reach the lane that owns that bin?" Standing rule
added S46-T6.

*(source: catboost-mlx S46-T4, DEC-049 OUTCOME, `kernel_sources.h:1374-1407`, 2026-05-05)*

---

## Contribution Log

| Date | Change | Contributor |
|------|--------|-------------|
| 2026-04-25 | Added § Probe Design — first real entry | sprint-38 |
| 2026-04-25 | Added § Probe Design — rubric portability lesson (PROBE-G amendment) | sprint-38 |
| 2026-04-25 | Added § Probe Design — cross-runtime quantization grid alignment lesson (F2) | sprint-38 |
| 2026-04-25 | Added § Probe Design — formula equivalence boundary-case lesson (PROBE-H) | sprint-38 |
| 2026-04-25 | Added § Probe Design — counterfactual vs observational confusion lesson (PROBE-H v2) | sprint-38 |
| 2026-04-25 | Added § Probe Design — cross-runtime configuration symmetry (PROBE-Q phase 2) | sprint-38 |
| 2026-04-25 | Added § Noise-Driven Algorithms — RNG-implementation bias multi-seed verification | sprint-39 |
| 2026-04-26 | Added § Cross-Runtime Triage — 3-experiment decomposition methodology (release-readiness filter) | sprint-40 |
| 2026-05-05 | Added § SIMD Histogram Kernel Routing Invariant — processor-lane vs owner-lane independence trap (S46 Probe B) | sprint-46 |
