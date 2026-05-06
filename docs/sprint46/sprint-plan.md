# Sprint 46 Plan — simd_shuffle Redesign Research Arc

**Sprint:** 46  |  **Status:** DRAFT — pending user approval before T0 starts  |  **Branch:** `mlx/sprint-46-simd-shuffle-research`
**Cut from:** master at S45 PR #46 merge (post DEC-048)
**Theme:** Bounded research/scoping sprint. Determine whether the histogram-accumulation `simd_shuffle` broadcast chain can be restructured to deliver ≥3× MLX/CPU iter speedup at production dispatch shape — OR rule it out empirically via a probe-D upper-bound measurement before committing engineering scope.
**Mode:** RESEARCH. NOT engineering. NO production-code commits in S46. Engineering is conditional on T5 = COMMIT, executed in S47+ as a separate sprint.
**Duration target:** 1 week.

## Strategic context

S45 falsified H-Dispatch (DEC-048): dispatch fusion is already production code in `DispatchHistogramBatched` (`catboost/mlx/methods/histogram.cpp:31`), 6 dispatches/iter on both Epsilon and Higgs, dispatch overhead = 0.008% of iter wall-clock. The throughput-pivot framing (W1/W2/W3) is retired.

Per the S45-T2 verdict §"Root Cause of the Cross-Class Gap" (`docs/sprint45/T2/probe-verdict.md` lines 106–115) and DEC-048 §"What survives", the actual cost driver — empirically established in S19-01c (DEC-020) — is the `simd_shuffle` serial chain inside the histogram accumulation kernel. S19-01c attributed it at **86% of accumulation time** and **~80% of `histogram_ms`** at the 50k/RMSE/d6/128b gate config.

DEC-048 line 2546 explicitly states: *"Scope of this KILL is the dispatch-fusion lever specifically. The simd_shuffle_xor serial chain identified as the actual cost driver was NOT in the S45 hypothesis set and is NOT killed by this DEC. Addressing it requires a multi-sprint warp-shuffle redesign of the histogram accumulation kernel; that is a separate v0.7.x scope decision to be made post-S45 close-out, not a DEC-048 outcome."*

S46 is that scope decision. It is bounded: produce a probe-D upper-bound measurement and decide go/no-go for engineering in S47+.

### Terminology correction (load-bearing)

S45-T2 verdict and DEC-048 both refer to "`simd_shuffle_xor` serial chain." This terminology is a legacy carry-forward from S17-era D1c (which DID use `simd_shuffle_xor`). The current production kernel at `kernel_sources.h:209–211` uses `simd_shuffle` (broadcast), NOT `simd_shuffle_xor` (butterfly). The xor butterfly was REMOVED in S18 BUG-S18-001 fix (DEC-012):

> *"Reduction phase (Sprint 18, BUG-S18-001 fix): single 8-term cross-SIMD linear fold (DEC-009, fixed g=0..7 order, 7 addition levels → γ_7 ≈ 4.2e-7 FP32). The D1c intra-SIMD simd_shuffle_xor butterfly was REMOVED when the layout changed."* — `kernel_sources.h:87–98`.

The actual hot loop is the 32-iter `src` broadcast at `kernel_sources.h:209–224`:

```metal
for (uint src = 0u; src < SIMD_SIZE; ++src) {
    const uint  p_s = simd_shuffle(packed, src);
    const float s_s = simd_shuffle(stat,   src);
    if ((p_s & VALID_BIT) == 0u) continue;
    const uint p_clean = p_s & 0x7FFFFFFFu;
    for (uint f = 0u; f < FEATURES_PER_PACK; ++f) {
        const uint bin = (p_clean >> (24u - 8u * f)) & 0xFFu;
        if (bin < foldCountsFlat[foldBase + f] + 1u &&
            (bin & (SIMD_SIZE - 1u)) == lane) {
            simdHist[simd_id][f * BINS_PER_BYTE + bin] += s_s;
        }
    }
}
```

This is **two** `simd_shuffle` broadcasts per `src` iteration (DEC-016 dropped from 3 → 2 via the MSB-fused VALID_BIT). The chain runs 32 times per outer batch, FEATURES_PER_PACK=4 inner iterations, gated by ownership predicate `(bin & 31) == lane`. **This loop is the lever.** S46 names it the "src-broadcast chain" to avoid the legacy "xor" misnomer in plan text and DEC-049.

DEC-048's text ("simd_shuffle_xor") will be cross-referenced in DEC-049's opening section so future readers find both terms.

## What changed vs the original throughput-pivot framing

The W1/W2/W3 framing committed multi-sprint scope based on arithmetic mechanism claims. S45 demonstrated that arithmetic-derived hypotheses without code inspection are inadmissible (one `grep` for `numGroups` would have refuted H-Dispatch in 5 minutes; instead, six advisory agents converged on the wrong mechanism over two days).

**S46 inherits MANDATORY-CODE-INSPECTION from `Frameworks/LESSONS-LEARNED.md`:** every hypothesis statement in this sprint cites file:line. Every sub-agent task output is rejected if it makes a mechanism claim without code citation. This is a hard rule.

**S46 commits no engineering until probe-D measures the upper bound at production shape.** This mirrors S45-T2 spike-then-commit (probe before commit). DEC-017 (T3b retired) is the load-bearing precedent for why: T3b projected −84.4% from toy isolation, measured +42.3% regression at production. Toy-to-production transfer is unreliable in this codebase (standing rule, DEC-017 §"Standing warning for Sprint 21+").

## Items table

| # | Description | Status | Owner | Reviewers |
|---|---|---|---|---|
| **T0** | Sprint scaffold — branch cut, plan committed, DEC-049 entry opened (status: OPEN), TODOS.md S46 row added | TODO | @ml-product-owner | — |
| **T1** | Code-inspection-grounded current-state characterization. Cite file:line for every claim. Document src-broadcast chain mechanics, register pressure, threadgroup-memory utilization, ownership predicate, parity invariants. NO arithmetic-only claims. | TODO | @performance-engineer | @silicon-architect |
| **T2** | Theoretical-alternatives feasibility analysis. For each of 4 candidates: code-inspection-grounded sketch (cite affected files), upper-bound estimate from first principles, parity stance (Branch-B preservation strategy), risk register (DEC-017 / DEC-023 cross-references). | TODO | @silicon-architect | @performance-engineer, @mathematician |
| **T3** | Probe-D experiment design. For each surviving candidate from T2, specify how to bound the upper improvement WITHOUT engineering — ablation strategy, dispatch-shape requirement (multi-TG production), measurement cadence (3 seeds × 3 runs warm), kill threshold. | TODO | @performance-engineer | @silicon-architect, @mathematician |
| **T4** | Probe-D execution. NO production engineering commits. Probe artifacts under `docs/sprint46/T4/` only. Measure upper-bound improvement per surviving candidate at Epsilon iter=200 (subset for speed) AND iter=2000 (full anchor). Branch-B regression test may break temporarily on probe branch. | TODO | @performance-engineer | @silicon-architect |
| **T5** | Decision gate (DEC-049 outcome). @strategist synthesizes T4 results; @devils-advocate stress-tests against LESSONS-LEARNED toy-to-production-transfer rule and DEC-017/DEC-023 parity-blocker precedents. Outcome A (COMMIT to S47), B (user-call), or C (HALT, DEC-049 = KILL). | TODO | @strategist | @devils-advocate |
| **T6** | Sprint close-out. DEC-049 entry finalized with verdict and empirical justification. HANDOFF/TODOS/CHANGELOG-DEV updated. LESSONS-LEARNED.md entry on whichever path lands. PR `mlx/sprint-46-simd-shuffle-research` → master. | TODO | @ml-product-owner + @technical-writer | @strategist |

## T1 — Current-state characterization

**Owner:** @performance-engineer (carries S19-01c context). **Reviewer:** @silicon-architect.

**Mandatory inputs (read end-to-end before drafting):**
- `catboost/mlx/kernels/kernel_sources.h:107–283` — full `kHistOneByteSource`. Map every barrier, every shuffle, every threadgroup memory reference.
- `catboost/mlx/methods/histogram.cpp:31–217` — `DispatchHistogramBatched` and `ComputeHistogramsImpl`. Confirm DEC-048 finding; document `numGroups`/`maxBlocksPerPart`/grid topology.
- `catboost/mlx/methods/structure_searcher.cpp:60–198` — `SearchTreeStructure` outer loops. Confirm 6 dispatches/iter (1 per depth × 1 approxDim for binary).
- `docs/sprint45/T2/probe-verdict.md` (full).
- DEC-016 (S19, MSB-fused VALID_BIT — current shuffle count).
- DEC-020 + DEC-023 (T2 sort-by-bin shipped, then v5 reverted).
- DEC-017 (T3b atomic-CAS retired — load-bearing precedent for toy-vs-production).
- DEC-009, DEC-011, DEC-012 (reduction layout; 32 KB threadgroup ceiling; xor butterfly removed).

**Deliverable:** `docs/sprint46/T1/current-state.md` covering at minimum:

1. **Src-broadcast chain mechanics.** Cite `kernel_sources.h:191–225`. Describe data dependencies, what's serial vs parallel, why all 32 lanes execute the inner predicate every src iteration.
2. **Ownership predicate cost.** `(bin & 31) == lane` at line 220 — quantify the fraction of work each lane discards. For FEATURES_PER_PACK=4 inner iterations, each lane writes ~4/32 = 12.5% of inner-loop iterations (lower bound; valid-doc dependent). Cite the line; describe the register pressure of the conditional.
3. **Threadgroup memory utilization.** 32 KB hard ceiling per DEC-011. `simdHist[8][1024]` consumes the entire ceiling. Any candidate that adds threadgroup memory must either (a) re-tile, (b) shrink an existing buffer, or (c) re-negotiate DEC-011 (which forces 1 TG/SM, currently exact).
4. **Reduction phase (`kernel_sources.h:240–255`).** 4-tile cross-SIMD linear fold; 1 barrier/tile = 4 barriers + accumulation barriers = 6 total. γ_7 = 4.2e-7 FP32 per DEC-009. Any candidate that changes reduction depth crosses DEC-008 boundary.
5. **Writeback (`kernel_sources.h:263–281`).** atomic_fetch_add on device atomic_float; gated by `maxBlocksPerPart == 1` (NIT-4, KNOWN_BUGS S-1). Any candidate that uses cross-TG atomic accumulation re-opens the latent race.
6. **S19-01c re-attribution numbers.** State the 86% / 80% measurements; cite supporting evidence. Note that these are at the 50k/RMSE/d6/128b gate config — Epsilon at 400k/2000 features may amplify or attenuate (T3 must measure, not assume).
7. **DEC-008 envelope constraints.** Any candidate must preserve DEC-008 (RMSE/Logloss ulp ≤ 4, MultiClass ulp ≤ 8) OR explicitly renegotiate the envelope with full parity sweep. No exceptions.

**Exit criteria for T1:** A code-inspection-grounded spec doc that any sub-agent in T2 can use to evaluate alternatives WITHOUT re-reading the kernel. Every claim in T1 cites file:line.

## T2 — Theoretical-alternatives feasibility analysis

**Owner:** @silicon-architect. **Reviewer:** @performance-engineer (mechanism), @mathematician (parity invariants).

**Bounded candidate set (S46 evaluates only these four; novel architecture proposals are out-of-scope per user constraint):**

### Candidate A — Atomic-add accumulation (revisits DEC-017 in light of S22 evidence)

**Sketch:** Replace per-SIMD `simdHist[g][bin]` writes with `threadgroup atomic_uint` accumulation into a single 1024-slot threadgroup buffer. Each lane processes its own doc (no src broadcast). Eliminates the 32-iter src loop entirely.

**Affected files:** `kernel_sources.h:158, 191–225` (accumulation phase); `kernel_sources.h:240–255` (cross-SIMD fold becomes single-pass since there's only one buffer).

**Upper-bound estimate from first principles:**
- Eliminates 32-iter src broadcast (currently 86% of accumulation per S19-01c).
- Adds atomic-CAS contention. DEC-017 measured this lever at +42.3% regression at production shape due to per-TG fixed overhead at 3 docs/thread (depth 6 partition-fragmented). Per-TG zero-init + writeback overhead is ~8 mem ops/thread; at 3 docs/thread × 4 features = 12 inner ops/thread, overhead ratio is 67%, dominates.
- **Sprint 22 D0 evidence (DEC-020):** T2 sort-by-bin made the same lever measurable at 0.317× hist_ms ratio at 50k gate config (in-situ multi-TG production shape) BUT was retroactively reverted in DEC-023 v5 due to atomic-float race at config #8 (N=10k bimodal).
- **Net upper-bound:** 1.0×–3.5× depending on whether (a) Epsilon's 400k×2000 dispatch shape behaves like S22 D0 (favorable per-TG amortization) or like DEC-017 production shape (per-TG overhead dominates). Per DEC-017 standing rule, the answer requires probe-D measurement at Epsilon shape, not extrapolation.

**Parity stance:** Atomic-add reorders accumulation non-deterministically across TGs. Higham γ_N where N = docs/bin at Epsilon scale (~3000 docs/bin/partition at root, scaling down by depth). Almost certainly exceeds DEC-008 ulp ≤ 4. **Branch-B regression WILL break.** DEC-023 v5 fix path (sort-by-bin + T1-style accumulation for features 1-3) is the precedent; it shipped at 1.01× hist_ms (no speedup). Recovering the 0.317× number requires re-engaging DEC-026 (cascade-robust GAIN, FALSIFIED in S25 G1) or DEC-027 (deferred radix-sum).

**Risk classification:** HIGH — DEC-017 / DEC-023 precedent says this design space has been explored twice, both falsified. Re-entry without new mechanism evidence violates DEC-025 re-entry policy. T2 should evaluate only whether new evidence (e.g., Epsilon shape changes the per-TG ratio) is plausible; if not, candidate A is RETIRED at T2.

### Candidate B — Hierarchical reduction (segmented per-bin parallel reduction)

**Sketch:** Within each SIMD group, lanes process disjoint doc subsets and accumulate into per-lane register histograms (small — e.g., 32 bins/lane × 4 features = 128 floats × 4 B = 512 B/lane register pressure). Then a per-bin reduction across the SIMD group via `simd_shuffle_xor` butterfly (re-introducing it with the correct layout per DEC-012 §"Future trigger"). Cross-SIMD fold remains as today.

**Affected files:** `kernel_sources.h:158` (replace `simdHist[8][1024]` with per-lane register state), `kernel_sources.h:191–225` (replace src broadcast with per-lane accumulation), `kernel_sources.h:240–255` (cross-SIMD fold).

**Upper-bound estimate from first principles:**
- Eliminates the 32-iter src broadcast (current 86%).
- Adds per-lane accumulation work + intra-SIMD butterfly reduction (5 levels of `simd_shuffle_xor`).
- Register pressure: 128 floats/lane = 32 VGPRs/lane. AGX has ~256 VGPRs/thread budget; current kernel uses ~9 VGPRs for doc state. Adding 32 fits but compounds with the existing live state. **Risk of register spill** (DEC-014 precedent: A1 at +6 VGPR/lane caused +9.4% regression due to spill).
- **Net upper-bound:** 1.5×–2.5× IF no spill, 0.7×–1.0× IF spill. Probe-D must measure the spill threshold via Metal compiler register-allocation report (DEC-014 §"S19-03 must verify no spill via Metal compiler register-allocation report" precedent).

**Parity stance:** Intra-SIMD butterfly (`simd_shuffle_xor`) was REMOVED in DEC-012 because it was incorrect under the per-SIMD shared layout. Per DEC-012 §"Future trigger": *"Any future kernel that accumulates into per-lane register state (not shared threadgroup memory) should re-introduce the intra-SIMD butterfly for that phase."* This candidate is in-scope of that trigger. γ bound: γ_5 (intra-SIMD butterfly, 5 levels) + γ_7 (cross-SIMD linear) = γ_12 ≈ 7.2e-7 FP32 — same as S17 D1c. Tolerance-wise compliant with DEC-008.
- Branch-B regression: order-deterministic per `simd_shuffle_xor` semantics within a SIMD group; final value differs from current kernel in the LAST decimal but bit-equivalence is NOT preserved. Branch-B test will need re-baselining if this lands.

**Risk classification:** MEDIUM — register-spill risk is real but measurable; reduction-order is principled; DEC-012 §"Future trigger" explicitly opens this path.

### Candidate C — Sort-by-bin extension (reuse DEC-020 T2 sort + apply to all features deterministically)

**Sketch:** DEC-023 v5 reverted T2-accum back to T1-shuffle topology because features 1-3 atomic-scatter raced. The per-feature ratio of 0.317× hist_ms (DEC-020 D1-R2) was real at 50k gate config but only for feature 0's bin-range scan. C is the question: can the bin-range-scan path be extended to all 4 features in a per-pack with an alternative deterministic-reduction mechanism that DEC-023 / DEC-026 missed?

**Affected files:** `kernel_sources.h` (full kernel restructure as in DEC-020/023 v3-v5 history; would require new kernel source separate from `kHistOneByteSource`); `histogram.cpp` (new dispatch path); `structure_searcher.cpp` (consumer is unchanged).

**Upper-bound estimate from first principles:**
- DEC-020 measured 0.317× hist_ms at gate (50k/RMSE/d6/128b) in-situ multi-TG production shape. This IS at production shape, unlike candidate A — strong precedent.
- 1.778× e2e iter speedup at gate (DEC-020 §"Final numbers"). At Epsilon 2000-feature shape, the kernel-work plurality is even higher (per S45-T2 attribution chain), so e2e ratio could be >1.778×.
- **Net upper-bound at Epsilon iter=2000:** 2.5×–4.0× IF feature-0-only accumulation handles all 4 features deterministically. Probe-D measures whether the deterministic-reduction mechanism survives.

**Parity stance:** DEC-023 G3 v5 measurements showed 18/18 ULP=0 + 100/100 deterministic at gate config when features 1-3 use T1-style accumulation. The KILL was at config #8 (N=10k, RMSE, 128b — singleton race footprint, see DEC-023 §"Footprint"). Either (i) Epsilon shape avoids the race envelope by virtue of N=400k (DEC-023 §H1: "larger bin counts resolve additions in consistent order" — 18/18 deterministic at N=50k), or (ii) the race re-emerges at Epsilon scale. T3 must measure determinism under Epsilon shape with ≥5 runs at config-equivalent settings, NOT extrapolate.

**Risk classification:** MEDIUM-HIGH — design space has been visited (DEC-020 ship → DEC-023 revert → DEC-026 falsify). C re-enters with the explicit question "does Epsilon shape avoid the race envelope?" — this is new evidence per DEC-025 re-entry policy. T2 must justify the re-entry against DEC-025 §"Re-entry policy" criteria.

### Candidate D — Split-K accumulation (per-block-of-docs partial histograms reduced via on-chip merge)

**Sketch:** Partition docs into K blocks (e.g., K=4 or K=8); each block's TG produces a partial per-bin histogram in threadgroup memory; a final on-chip reduction pass merges K partial histograms into the per-partition output. Reduces per-TG src-loop work by factor K but adds a second-stage merge dispatch.

**Affected files:** `kernel_sources.h` (new kernel for partial-hist + new kernel for merge-K). `histogram.cpp` (new dispatcher with two-stage). `structure_searcher.cpp` unchanged.

**Upper-bound estimate from first principles:**
- Reduces per-TG accumulation work by factor K.
- Adds K-1 merge passes (4 or 8 atomic-add passes; each pass is bounded by `numPartitions × numBinFeatures` outputs).
- Increases total dispatches from 6 to 6×2 = 12/iter (still well below the DEC-048 dispatch-overhead threshold; 0.36 ms vs 2241 ms).
- **Net upper-bound:** 1.5×–2.5× IF K-merge cost is dominated by the K-fold reduction in src-broadcast cost. Probe-D measures the actual ratio.

**Parity stance:** K-block partition introduces a new reduction order (block-major vs current SIMD-group-major). γ bound: γ_K + γ_7 = γ_(K+7). For K=4: γ_11 ≈ 6.6e-7 — within DEC-008. For K=8: γ_15 ≈ 9e-7 — also within. Branch-B regression bit-equivalence breaks; re-baseline required.

**Risk classification:** MEDIUM — adds dispatch count (S45 verified this is low-cost), introduces new reduction tier (manageable parity risk), but is the most architecturally novel of the four — least precedent in DEC log.

### T2 deliverable

`docs/sprint46/T2/feasibility.md` covering:
- Per-candidate code-inspection-grounded sketch (cite affected files for each).
- First-principles upper-bound estimate with explicit assumptions cited.
- Parity stance: bit-equivalence path / re-baseline path / DEC-008 envelope position.
- Risk classification: HIGH/MEDIUM/LOW vs DEC-017/DEC-023 precedents.
- T2 verdict: which candidates survive to T3 probe-D? Recommend RETIRE at T2 if a candidate is structurally blocked (e.g., A retired if no new evidence vs DEC-017/DEC-023 design space).

**Exit criteria for T2:** at least 2 candidates survive to T3 (to give T4 a real measurement). If 0 or 1 survives, the sprint MAY HALT early (DEC-049 = KILL with empirical justification from T1+T2 alone).

## T3 — Probe-D experiment design

**Owner:** @performance-engineer. **Reviewers:** @silicon-architect, @mathematician.

**Mandatory inputs:** T1 spec doc, T2 feasibility doc, DEC-017 §"Standing warning" (toy-to-production transfer rule), DEC-014 §"Register pressure delta" (Metal compiler register-allocation verification).

**For each surviving T2 candidate, T3 specifies:**

1. **Ablation strategy.** What is the cheapest measurement that bounds the upper improvement WITHOUT shipping engineering? (S45-T2 precedent: replace dispatch with `mx::zeros()` to isolate dispatch overhead.) Examples per candidate:
   - **A (atomic-add):** prototype atomic-add kernel as a probe-only `kernel_sources.h` variant; build with `-DPROBE_F_ATOMIC_ACCUM` guard; dispatch only in probe build; never compiled into release.
   - **B (hierarchical):** prototype per-lane register accumulation; verify register-allocation via Metal compiler dump (`-print-ir-after-llvm` or equivalent); measure hist_ms.
   - **C (sort-by-bin extension):** revive DEC-020 T2 kernel from `kernel_sources.h` git history (commit 73baadf445 era); benchmark Epsilon shape WITHOUT shipping. Determinism sweep at Epsilon required (mirror S22 D3 protocol).
   - **D (split-K):** prototype K=4 partial-histogram + merge kernel; measure hist_ms; verify γ bound numerically against double-precision reference.

2. **Dispatch shape.** All measurements MUST be at production dispatch shape per DEC-017 §"Standing warning":
   - Higgs-1M iter=200 (cheap repro).
   - Epsilon iter=200 (cheap Epsilon-shape).
   - Epsilon iter=2000 (full anchor; Outcome A is decided here).
   - For Outcome A, ≥3 seeds × ≥3 warm runs is required at iter=2000; warmup matches `bench_boosting`.

3. **Parity sweep.** Branch-B regression test at v0.6.1 baselines. If candidate breaks bit-equivalence:
   - DEC-008 envelope sweep on the 18-config matrix (as DEC-020 did at S22 D1).
   - 100/100 determinism at gate config (50k/RMSE/d6/128b).
   - Special-case audit at config #8 (N=10k/RMSE/128b) per DEC-023 atomic-float race precedent.

4. **Kill threshold per candidate:** If ablation upper-bound on Epsilon iter=2000 is <1.5× MLX iter speedup, candidate is RETIRED at T4. <3× and ≥1.5× routes to Outcome B (user-call). ≥3× routes to Outcome A (COMMIT to S47).

**T3 deliverable:** `docs/sprint46/T3/probe-d-spec.md` with the experiment plan, dispatch shape table, kill thresholds per candidate, and reproduction commands.

## T4 — Probe-D execution

**Owner:** @performance-engineer. **Reviewer:** @silicon-architect.

**No production-code commits.** Probe artifacts under `docs/sprint46/T4/` only:
- `docs/sprint46/T4/<candidate>/probe-results.md` per candidate
- `docs/sprint46/T4/<candidate>/data/*.json` raw measurement
- `docs/sprint46/T4/<candidate>/scripts/*.sh` reproduction commands
- `docs/sprint46/T4/<candidate>/kernel-variant.metal` (in scratch/, NOT linked from production)

Probe builds use `#ifdef SIMD_SHUFFLE_PROBE_<X>` guards (analogous to S33 PROBE-E pattern). Production builds compile bit-identical to v0.6.1 baseline; Branch-B regression test green throughout T4.

**T4 deliverable per candidate:** A quantitative verdict — measured upper-bound MLX iter speedup at Epsilon iter=2000 production dispatch shape, parity stance, and recommendation (proceed to T5 with this candidate / retire).

## T5 — Decision gate (DEC-049 outcome)

**Owner:** @strategist. **Stress-test:** @devils-advocate.

@strategist synthesizes T4 verdicts. @devils-advocate stress-tests against:
- LESSONS-LEARNED.md MANDATORY-CODE-INSPECTION rule (every claim cited).
- DEC-017 / DEC-023 toy-to-production-transfer precedents (was probe at production shape?).
- DEC-025 re-entry policy (if a candidate re-enters a previously-killed design space, did T2 justify with new evidence?).

**Outcome A (COMMIT to S47):** Best candidate's T4 measurement shows ≥3× MLX iter speedup upper-bound on Epsilon iter=2000 at production dispatch shape, AND parity preservation path is plausible (either bit-equivalence preserved or re-baseline scope is 1-sprint). DEC-049 = COMMIT. S47 plan opens with the named candidate as the engineering scope.

**Outcome B (user-call):** Best candidate shows 1.5×–3× upper-bound. Marginal — does not close v0.7.0 perf gate alone but may compose with a future lever. User decides.

**Outcome C (HALT, DEC-049 = KILL):** No candidate shows ≥1.5× upper-bound at production shape. simd_shuffle redesign is empirically not the load-bearing lever for the v0.7.0 ≥3× gate. Throughput epic is permanently retired (the meta-criterion from S45 plan §"Risk register" fully fires — gap is hardware-class-bound). v0.7.0 path becomes "reproducibility extension only" or remains held.

**T5 deliverable:** DEC-049 entry with verdict, empirical justification, and authority pointers.

## T6 — Sprint close-out

- DEC-049 entry in `.claude/state/DECISIONS.md` with empirical justification.
- `.claude/state/HANDOFF.md` updated with T5 outcome.
- `.claude/state/TODOS.md` updated; S47 row added if Outcome A.
- `.claude/state/CHANGELOG-DEV.md`: 2026-05-XX session entry covering T0-T6.
- `Frameworks/LESSONS-LEARNED.md`: at least one entry. Likely candidates:
  - "Probe-D upper-bound measurement at production shape is non-negotiable for kernel-internal levers" (echoes DEC-017 standing rule, reinforces with simd_shuffle evidence).
  - If Outcome C: "Histogram-kernel restructure design space is empirically exhausted under DEC-008 envelope."
  - If Outcome A: "Candidate <X>'s mechanism passes the production-shape bar — distinguishing pattern for future kernel-restructure proposals."
- Single PR `mlx/sprint-46-simd-shuffle-research` → master once T0–T6 land.

A sprint that closes with T5 = HALT and no engineering work IS a successful research sprint by the plan's Definition of Done. Negative results are legitimate deliverables (project rule per CLAUDE.md "negative results are legitimate deliverables").

## Files in scope vs explicitly NOT in scope

**In scope (S46):**
- `docs/sprint46/sprint-plan.md` (this file)
- `docs/sprint46/T1/current-state.md`
- `docs/sprint46/T2/feasibility.md`
- `docs/sprint46/T3/probe-d-spec.md`
- `docs/sprint46/T4/<candidate>/...` per surviving candidate
- `docs/sprint46/T4/scratch/*.metal` probe-only kernel variants (NOT linked from production)
- `docs/sprint46/sprint-close.md`
- `.claude/state/{HANDOFF,TODOS,CHANGELOG-DEV,DECISIONS}.md`
- `Frameworks/LESSONS-LEARNED.md`
- `python/tests/regression/test_branch_b_regression.py` (must stay green)

**Explicitly NOT in scope (deferred to S47 conditional on T5 = COMMIT):**
- `catboost/mlx/kernels/kernel_sources.h` production source changes
- `catboost/mlx/methods/histogram.cpp` production logic changes
- `catboost/mlx/methods/structure_searcher.cpp` changes
- `python/catboost_mlx/**` API changes
- Any change that flips a bit in `predict()` output (would break Branch-B regression test)

**Explicitly NOT in S46 under any outcome:**
- PyPI v0.7.0 publish — gated on S47 perf gate landing per DEC-048 release condition.
- Novel algorithm proposals beyond the bounded T2 candidate set (out-of-scope per user constraint).
- Multi-sprint engineering commitments (T5 = COMMIT scope must be 1-sprint per S45 lessons).

## Risk register

| Risk | Mitigation | Owner |
|---|---|---|
| **Toy-bench upper bound does not transfer to production shape** (DEC-017 precedent) | T3 mandates production dispatch shape (multi-TG, depth 6, partition-fragmented) for all measurements. T4 probes Epsilon iter=2000 directly. NO single-TG isolation results enter T5 decision. | @performance-engineer |
| **Branch-B regression breaks during T4 ablation** | Allowed temporarily on probe branch with `#ifdef` guards. T5 = COMMIT requires bit-equivalence preservation OR a 1-sprint re-baseline plan in S47 scope. | @qa-engineer |
| **DEC-023 atomic-float race re-emerges at Epsilon shape** for candidates A/C | T3 includes determinism sweep at Epsilon-equivalent settings (≥5 runs). Candidate FAILS T4 if any race fires; result documented in DEC-049 regardless. | @performance-engineer |
| **Register spill regression for candidate B** (DEC-014 precedent) | T3 requires Metal compiler register-allocation verification before measurement. Candidate B FAILS T4 if spill is detected. | @silicon-architect |
| **All four candidates fall <1.5×** at production shape | Outcome C is a pre-defined valid outcome. DEC-049 = KILL. v0.7.0 path narrows further. | @strategist |
| **Sprint creeps to 2+ weeks** | Hard 1-week budget. T5 fires at end of Day 5 even if only 2 candidates measured. Remaining candidates documented as "not measured" in DEC-049, not "indeterminate". | @ml-product-owner |
| **T2 retires all candidates** without probe-D measurement | Sprint may HALT early at T2 with DEC-049 = KILL, citing DEC-025 re-entry policy violations and code-inspection-grounded structural blockers. | @silicon-architect + @ml-product-owner |
| **MANDATORY-CODE-INSPECTION rule violation by sub-agent** | Plan explicitly forbids arithmetic-only mechanism claims. Reviewer (one of @silicon-architect / @performance-engineer) rejects any T1/T2/T3 deliverable that lacks file:line citations. | All reviewers |

## Definition of Done

**S46 closes successfully if:**
- T1 current-state spec exists and cites file:line for every mechanism claim.
- T2 feasibility analysis exists with at least 2 candidates evaluated against DEC-017/023/025 precedents (or with explicit early-halt verdict).
- T3 probe-D spec exists with production-shape dispatch requirements.
- T4 probe-D verdicts exist per surviving candidate (or T2 early-halt closure).
- T5 DEC-049 entry exists with one of {COMMIT, user-call, KILL} and empirical justification.
- T6 close-out artifacts updated; PR opened.
- Branch-B regression test GREEN on master throughout (probe branches may break temporarily).

**S47 (engineering) opens only if (separately, post-S46):**
- T5 = COMMIT with named candidate.
- 1-sprint engineering scope plausible per T4 evidence.
- Branch-B preservation OR 1-sprint re-baseline plan in S47 scope.

If S47 fails to land the perf gate, v0.7.0 does not ship; project remains at v0.6.1 indefinitely until either a research-grade lever appears or the audience proposition shifts (per DEC-048 §"Implication for v0.7.0").

## Agent panel

**Load-bearing trio:** @performance-engineer (T1 owner; T2/T3/T4 advisor; carries S19-01c context), @silicon-architect (T2 owner; T3/T4 advisor; Apple Silicon SIMD/threadgroup levers), @mathematician (T2/T3 reviewer; parity-invariant analysis under each candidate; γ-bound derivations).

**Decision authorities:** @strategist (T5 synthesis), @devils-advocate (T5 stress-test against LESSONS-LEARNED + DEC precedents).

**Implementation:** @qa-engineer (Branch-B regression preservation), @code-reviewer (every commit, even probe-only).

**Writing:** @technical-writer (T6 close-out + LESSONS-LEARNED entries).

**NOT consulted in S46:** @visionary (out-of-scope per user constraint; novel architecture proposals are S47+ scope), @research-scientist (same), @hardware-researcher (S45 hardware writeup is sufficient), @security-auditor (no security-surface change in research mode), @mlops-engineer (no infra change), @data-scientist (no benchmark-suite change), @ml-engineer (no implementation in S46).

## Branch + PR plan

- Single branch `mlx/sprint-46-simd-shuffle-research` cut from master at S45 PR #46 merge.
- T0–T6 commits on this branch as atomic per-task commits per DEC-012 atomicity rule.
- T4 probe artifacts commit as `docs/sprint46/T4/...` (NOT production code).
- Single PR after T6 lands, regardless of T5 direction.
- Branch-B regression test runs on every PR commit; failure on master-reachable code is a hard block.

## What this plan does NOT repeat from prior failure patterns

| S45 pattern | S46 mitigation |
|---|---|
| Arithmetic-derived hypothesis without code inspection | T1 mandatory; every T2/T3/T4 claim cites file:line |
| Multi-sprint commit before measurement | S46 produces no production code; S47 conditional on T5 |
| Toy-bench projection of production speedup | T3/T4 mandate production dispatch shape; DEC-017 precedent cited |
| Re-entering previously-killed design space without new evidence | T2 explicitly cross-references DEC-017/023/025/026; candidate A retired if no new evidence |
| Sprint over-runs past 1 week | Hard T5 deadline at Day 5; unmeasured candidates documented as such, not deferred |

This plan is **not** "throughput epic continuation." It is a bounded research investigation of a specific kernel-internal lever, with binary go/no-go gating at T5.
