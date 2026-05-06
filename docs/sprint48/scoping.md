# Sprint 48 — v0.8.0 Throughput Research Arc — KICKOFF SCOPING

**Sprint:** 48 (kickoff of multi-sprint v0.8.0 throughput arc)
**Status:** v2 PLAN APPROVED 2026-05-06 by user. Awaiting PR #48 merge to cut S48 branch.
**Branch (proposed):** `mlx/sprint-48-v0.8.0-arc-kickoff`
**Cut-from:** master after PR #48 merges (`mlx/sprint-47-release-0.7.0` → master).
**Authority chain:** DEC-049 OUTCOME (RETIRED) + DEC-050 (Option α, reproducibility-grade v0.7.0) + DEC-051 (PyPI publish gated on CUDA-class throughput).
**Mode:** RESEARCH. **3-day spike + 10-day total v0.8.0 throughput budget** (revised from initial 7-day mirror — see §v2 below).
**Theme:** Identify and stress-test structurally-NEW lever classes for histogram throughput. If a survivor passes premise + silicon gates in 3 days, greenlight S49 probe build. Otherwise retire arc, pivot v0.8.0 to non-throughput milestone.

---

## §v2 — Panel Synthesis (2026-05-06, post-multi-agent review)

After advisory board review (visionary + devils-advocate + silicon-architect + strategist), the plan below is the **v2 canonical structure** for S48. The v1 framing in §0–§11 is preserved as historical record; **v2 supersedes** where they conflict.

### What changed from v1

**Structure: 7-day full mirror → 3-day spike.** Cheaper-to-falsify. Probes only fire in S49 on greenlit candidates, not in S48.

**Candidate roster restructured:**

| Candidate | Origin | v2 Status | Reason |
|---|---|---|---|
| **L1** Bin-distributed dispatch | v1 | **PRE-RETIRED** | Silicon-architect: 2,000× over DEC-018 amortization regime; flips DEC-017's failed axis |
| **L2** Sort-then-scan | v1 | **PRE-RETIRED** | Silicon-architect: 6.4 GB sort I/O at Epsilon ≥ entire current iter cost; BW floor exceeds savings |
| **L3** Prefix-sum / scan | v1 | **PRE-RETIRED** | `mx::segmented_sum` does NOT exist in MLX 0.31.x (verified `mlx/mlx/ops.h`; only `segmented_mm`); custom path collapses to current src-broadcast (DEC-013/049 territory) |
| **L4** Hybrid two-stage | v1 | **MARGINAL — KEEP** | Only v1 candidate where TG-mem + VGPR + dispatch + BW all clear silicon math |
| **L5** Sparse representation | v1 | **PRE-RETIRED** | 76% density at Epsilon (already in v1 §1) |
| **C1** Inverted-index histogram | NEW (visionary) | **CANDIDATE** | Per-feature, per-bin precomputed sample-index lists; histogram = `gather + segmented_reduce`. Sidesteps scatter entirely (Postgres GIN / Lucene posting lists analogy). Buildable in MLX today. |
| **C4** Persistent-kernel pipelining | NEW (visionary) | **CANDIDATE** | Single persistent Metal kernel owns all iters of tree-build cycle. Changes CPU↔GPU sync topology — never measured in 7-falsification chain (Triton/FlashAttention-2 analogy). |
| **C5** Leaf-wise tree growth | NEW (visionary) | **CANDIDATE — BUT product-changing** | LightGBM-style best-first growth. NOT a faster histogram — *fewer histograms* (O(F·B)/split vs O(2^d·F·B)). Highest ceiling but **breaks oblivious-tree contract**; needs explicit user-call at T0. |
| **L6** Hybrid CPU+GPU concurrent | NEW (silicon-architect) | **CANDIDATE** | Split feature-set: GPU does feat 0..K, CPU+AMX does K..F concurrently, merge via UMA zero-copy. Apple Silicon's defining feature (UMA) unused by all of L1–L5. |

**Net candidate pool for S48 T0:** **L4 (carryover) + C1 + C4 + C5 + L6** = **5 candidates**, with @visionary potentially adding more during T0a brainstorm.

### Sprint structure (3-day spike + 10-day total budget)

```
S48 (3 days):
  Day 1 (T0): @visionary brainstorm + @devils-advocate stress-test
              + USER candidate-list approval + threshold lock + pre-commit pivot rail
  Day 2 (T1+T2): @performance-engineer fresh f_hist (hours)
                 + @silicon-architect roofline + occupancy on survivors (paper only)
  Day 3 (T3): @research-scientist probe-spec drafts for 1-2 finalists (specs only, NO build)
              + @strategist synthesis + USER decision (A/B/C/D)

S49 (≤7 days, conditional on Outcome A): full probe build + measurement on greenlit candidate
S48 + S49 = 10 days HARD TIMEBOX. If no ≥1.5× measured speedup on Higgs-1M iter=1000 after 10 days → retire arc.
```

### 4-Outcome decision structure (refined from v1's 3-outcome)

- **A — Greenlight S49**: ≥1 candidate clears all gates AND projects ≥2× theoretical speedup → build probes
- **B — User-call**: 1 candidate clears with 1.2–2× theoretical (THE TRAP ZONE — default to retire unless explicit invest)
- **C — Retire + Pivot to non-throughput milestone** (LIKELY OUTCOME, P≈0.55) — v0.8.0 reframes to ordered boosting / DEC-046 / MSLR-WEB10K
- **D — Retire hard** (no compelling alternative; team takes research break)

### Sunk-cost pre-commit rail (anti-LESSONS-LEARNED-violation)

> *"If T0a+T0b produce 0–1 surviving candidates, S48 closes Day 3 with DEC-052 = RETIRED-AT-PREMISE, **no user re-deliberation**, and v0.8.0 re-scopes automatically to the pre-decided pivot target."*

This pre-commit is mandatory per devils-advocate's sunk-cost diagnosis. Rationalizing continuation after a candidate-empty T0 is the failure mode of the prior 4 sprints.

### Tiered threshold (LOCK at T0)

- **Hard gate (PyPI publish per DEC-051):** MLX/CUDA wall-clock ratio ≤ 5× on Higgs-1M iter=1000
- **Stretch:** ≤ 3× (structural win, not incremental)
- **Stop-loss floor:** ≤ 8× after S49 measurement → retire arc, accept indefinite PyPI deferral

Plus G2–G5 from v1 §4 (Branch-B parity, ±0.0003 logloss, Epsilon ≤15× cross-shape, DEC-008 envelope).

### Probability-weighted outcomes (strategist)

| Outcome | P | What happens |
|---|---|---|
| **C — Retire + Pivot** | **~0.55** | All candidates retire at T0/T2; v0.8.0 ships ordered-boosting or DEC-046; PyPI defers to v0.9.0+ |
| **B — Marginal user-call** | ~0.25 | 1 finalist with marginal projection; default retire to avoid sunk-cost trap |
| **A — Measured ≥2× in S49** | ~0.20 | Closes gap to ~10×; still short of ≤5× gate but enough for S50 continuation |

**The team should plan emotionally and technically for Outcome C.**

### Approved configuration (user 2026-05-06)

- ✅ Option (b) 3-day spike + 10-day total timebox
- ✅ Sunk-cost pre-commit rail (auto-pivot if 0-1 candidates survive)
- ✅ Tiered threshold (≤5× hard / ≤3× stretch / ≤8× stop-loss)
- ⏳ Pivot target choice (ordered-boosting / DEC-046 / MSLR-WEB10K) — **deferred to T0c user-call**; orchestrator default recommendation: **ordered boosting** (most leverage on perf-relevant axis without changing throughput question; LightGBM evidence shows the 5-10× exists)
- ⏳ C5 leaf-wise admission — **deferred to T0c user-call** (oblivious-tree contract is product-defining; user must explicitly approve breaking before C5 enters T2 feasibility)

### Factual corrections to v1

- v1 §1 L3 references `mx::segmented_sum` — does NOT exist in MLX 0.31.x. Only `mx::segmented_mm` exists (`mlx/mlx/ops.h:1579`). L3's library-primitive composition path was always more limited than v1 framed.
- v1 §1 L1 dispatch math says "5,000× more TGs" — correct order of magnitude but silicon-architect re-derived as 2,000× over DEC-018 amortization regime. Either way: pre-retired.
- v1 §1 L4 claim that "Stage A 8× lower VGPR than DEC-014" — devils-advocate flagged the arithmetic as suspect; DEC-014 spilled at 4 features × 64 docs, Stage A is 4 features × 8 bins = same order of magnitude, not 8× lower. L4 still passes silicon math but its key claim needs T2 verification.
- v1 §1 missed UMA / CPU+GPU concurrency (now L6) and persistent-kernel pipelining (now C4). Apple Silicon-specific features that the 7-falsification chain never explored.
- v1 §6 Risk pre-mortem assumed Outcome C as a fallback; v2 reframes Outcome C as the **modal expected outcome** (P≈0.55).

### What remains unchanged from v1

- §0 Premise gate (rejection rules for retired DEC overlap, routing-honest requirement, MANDATORY-CODE-INSPECTION carry-forward).
- §3 f_hist methodology (cheap; runs Day 2 in hours; same shapes as S46-T4).
- Parity envelope (DEC-008: RMSE/Logloss ulp ≤ 4, MultiClass ulp ≤ 8).
- Hard rules: NO production commits in S48; all probe code (if any in S49) `#ifdef`-gated; Branch-B regression GREEN on master throughout.
- Multi-sprint roadmap (§7 below): S49 engineering → S50 parity → S51 cutover → S52 publish gate, conditional on Outcome A.

---

## 0. Premise gate — what makes a candidate "structurally new"

DEC-049 OUTCOME (`docs/sprint46/T5/decision.md` §"Why RETIRE, Not Defer") established that the premise *"eliminate src-broadcast serialization without restoring routing"* is logically impossible on the current `kHistOneByteSource` topology. Any v0.8.0 candidate must therefore satisfy ALL of the following:

1. **Routing-honest.** The candidate must explain how every (data-element, accumulator-slot) pair reaches its owner. The S46-T6 SIMD routing invariant rule (every probe spec must include explicit account of the routing path before measurement) carries forward.
2. **Not in the 7-falsification chain.**
   - DEC-013 (writeback-plurality / batched-atomic) — REJECTED by S19-01c attribution; writeback is 5%, not plurality.
   - DEC-014 (BATCH_DOCS=64 wider batch / accumulation-redesign-via-register-state) — REJECTED at production: +9.4% regression from VGPR spill.
   - DEC-015 (col-major compressedIndex transposed view) — REJECTED at production: ~0.98× (no improvement); AGX prefetcher hides the gather already.
   - DEC-017 (T3b threadgroup-atomic-CAS) — REJECTED at production: +42.3% regression from per-TG fixed-overhead amortization mismatch.
   - DEC-019 (L2 stats pre-permute) — REJECTED at production: zero-gather upper bound was +2.6% slower; AGX prefetcher hides the latency.
   - DEC-048 (H-Dispatch fusion) — REJECTED by code inspection: dispatch fusion already production code via `numGroups` parameter at `histogram.cpp:31`.
   - DEC-049 (simd_shuffle B/C/D family) — REJECTED structurally: routing requires shuffle or pre-sort; no third route.
3. **Not a rebrand.** "Another atomic-add probe with a different reduction tree" is REJECTED a priori under DEC-013/023. "Another shuffle butterfly with different stride" is REJECTED under DEC-049 OUTCOME. Repackaging falsified mechanisms with new names does not pass the premise gate.
4. **Production-shape feasible on Apple Silicon.**
   - Threadgroup memory ≤ 32 KB per TG (DEC-011 hard ceiling).
   - VGPR ≤ 64/lane budget (DEC-014 spill rule).
   - No kernel/structure that needs > 1 TG/SM occupancy under the 32 KB layout.
   - Metal API surface — no operations that don't exist in MLX 0.31.x or Metal 3.x.
5. **Branch-B parity preservable.** The candidate must have a plausible parity story under DEC-008 (RMSE/Logloss ulp ≤ 4, MultiClass ulp ≤ 8). Bit-identity to v0.6.1 baseline is NOT required (the histogram kernel changes), but Branch-B regression (`python/tests/regression/test_branch_b_regression.py`) must pass on the Higgs-1M + Epsilon checkpoint with documented loss-equivalence (≤0.0003 logloss delta vs v0.7.0).

**Examples of what would be REJECTED at the premise gate (not exhaustive):**

- "Another atomic-add accumulator with merge-kernel" → REJECTED. Merge-kernel was D2; production-shape flat (DEC-049). Atomic-add per-bin was DEC-013/023; race-bound at production.
- "Wider batch / register-residency variant of the existing src-broadcast loop" → REJECTED under DEC-014 (BATCH_DOCS=64 already falsified for VGPR spill).
- "Pre-permute stats by feature column" → REJECTED under DEC-019 (zero-gather upper bound shows AGX prefetcher hides gather already).
- "Multi-dispatch tiling that subdivides the existing topology" → REJECTED under DEC-017 (toy-to-production transfer rule).
- "Simd_shuffle butterfly variant with different fan-out" → REJECTED under DEC-049 OUTCOME (routing impossibility).

---

## 1. Candidate lever classes (5 candidates)

Each candidate ships through T0–T5 as an option, not a commitment. T2 feasibility analysis culls obviously-infeasible ones; T3 specifies probes; T4 measures; T5 decides.

### Candidate L1 — Bin-distributed dispatch (one TG per (bin-range, partition))

**Mechanism:** The current production topology dispatches one TG per `(feature-group, block, partition, stat)` tuple (`histogram.cpp:79–88`); each TG processes ~3 docs/thread at depth 6 and uses the src-broadcast routing chain to deliver each doc's stat to the owner-lane for each of its 4 packed feature-bins. **L1 inverts the topology:** dispatch one TG per `(bin-range, partition, stat)`. Each TG is responsible for accumulating into a small contiguous bin range (e.g. 32 bins per TG = 4 TGs per feature). Within a TG, every lane reads docs in stride and tests `bin ∈ my_range`; only matching docs contribute. Per-bin contention is eliminated by construction because each TG owns exactly one bin range.

**Why structurally new:**
- Not DEC-013/023 (no atomic accumulation; per-TG bin range has single writer = the TG itself).
- Not DEC-014 (no per-doc register tiling; the savings are dispatch-shape, not accumulator-state).
- Not DEC-019 (no stats permute; gather pattern unchanged).
- Not DEC-017 (no atomic-CAS; per-TG amortization is by *bin range*, not by *doc count*).
- Not DEC-049 (no src-broadcast shuffle chain at all — eliminated, not redesigned).
- Topology change is at the dispatch level (`histogram.cpp:79–88` grid construction), not just the kernel hot loop.

**Apple Silicon feasibility:**
- TG memory: per-TG accumulator is 32 bins × 4B × 8 SIMD groups = 1 KB. Well under the 32 KB ceiling.
- Dispatch overhead: at Epsilon (2000 features, 128 bins, 64 partitions, 2 stats), naive bin-distributed = 2000 × 4 × 64 × 2 = 1,024,000 TGs. **High dispatch count is the primary risk.** Mitigation: per-feature-group TGs each owning a bin-range, total = 500 × 4 × 64 × 2 = 256,000 TGs. Still 5,000× current 64 × 2 = 128 TGs/dispatch. Per-TG fixed cost (DEC-018 measured: 2.5% of `histogram_ms`) would scale linearly — risk this kills the lever before measurement.
- Read-bandwidth: each TG re-reads all docs in its partition slice. At Epsilon (400k docs / 64 partitions = 6250 docs/partition, 4 TGs/feature/partition), each doc is re-read 4× per feature group.

**Theoretical upper-bound speedup:** Highly uncertain pre-measurement. Dispatch-overhead scaling may negate the savings. f_hist at Epsilon = 0.9772 sets the iter ceiling at ~3.9× even for an infinitely fast histogram.

**Risks:**
- (R1) Dispatch overhead explosion — 5,000× more TGs than current. Per-TG amortization regime *flips* relative to DEC-017.
- (R2) Read-amplification 4× (each doc seen by 4 TGs/feature). Bandwidth-bound at high feature counts.
- (R3) Parity story unclear: with 4 TGs writing different bin ranges, accumulation order changes. Branch-B regression needs DEC-008 envelope verification.

**Pre-flight gates (T2/T3, before any code):**
- (P1) Dispatch-count arithmetic at Gate, Higgs-1M, Epsilon must not exceed 5× S22 T2 dispatch count (~1,664 TGs at Gate). If yes → L1 retires at T2 without measurement.
- (P2) Code inspection of `mx::fast::metal_kernel` grid argument — confirm Metal can dispatch >100k TGs in a single submit on M3 Max without driver-side throttling.
- (P3) Routing argument written and code-inspection sign-off (S46-T6 rule).

### Candidate L2 — Sort-then-scan with race-free atomics (radix-sort + scan accumulator)

**Mechanism:** Pre-sort docs within each (partition, feature) by bin value using a radix-sort variant on the GPU (a separate dispatch). Then accumulate via a scan-style pass: each lane sees a bin-sorted stream and writes contiguously, eliminating the need for atomic-fetch-add. The race that DEC-023 hit (features 1–3 racing on the shared accumulator after a feature-0 sort) is resolved here by sorting **within each (partition, feature) independently**, accepting the sort cost as the price of a race-free accumulator.

**Why structurally new:**
- DEC-023's race fired on the *shared* accumulator; L2 sorts per-feature independently, so each feature's accumulation is single-writer by construction.
- DEC-020 T2 was per-feature-0-only sort + atomic for features 1–3; L2 is per-feature-N sort. Different cost structure.
- Not in DEC-013/014/015/017/019/048/049 (none used radix-sort).

**Apple Silicon feasibility:**
- Radix-sort kernel: MLX has `mx::sort` and `mx::argsort`; bin values are 7-bit (DEC-016), so a 1-pass 128-bucket radix is feasible.
- Sort cost: O(N log B) per partition per feature; at Epsilon, 2000 features × 64 partitions × O(6250 × log(128)) = ~5.6 billion ops total. Memory-bandwidth bound: ~140 GB at 4 B/element bin index. At 400 GB/s peak, ~0.35s minimum. **This is the kill scenario — L2 is likely BW-bound.**
- TG memory: standard sort kernel; no exotic layout.

**Theoretical upper-bound:** Unlikely > 2× iter at Epsilon if sort is BW-bound. Better at Higgs-1M (only 28 features → 28-feature sort cost is small).

**Risks:**
- (R1) Bandwidth-bound at high-feature-count (Epsilon). Sort cost may exceed savings.
- (R2) MLX `mx::sort` on Metal-backed `mx::array` — confirm performance via micro-benchmark before T3.
- (R3) Parity: sort changes accumulation order; Higham bound under sorted-stream is different from current src-broadcast.

**Pre-flight gates:**
- (P1) MLX sort throughput micro-benchmark on M3 Max — must achieve ≥ 5 G-elements/s on bin-sized arrays.
- (P2) Per-feature sort vs. per-(feature,partition) sort cost analysis at Epsilon dispatch shape.
- (P3) Routing argument: "After sort, bin-b docs are contiguous in `sortedDocs[partOffset+s_b..partOffset+s_{b+1}]`. Lane l accumulates `sum(stat[d] for d in sortedDocs[s_b..s_{b+1}])` into `hist[b]`."

### Candidate L3 — Prefix-sum / scan-aggregation histogram

**Mechanism:** Replace the per-doc scatter pattern with a prefix-sum-based aggregation. For each (partition, feature) and each bin b, compute `hist[b] = sum over docs where bin(d) == b of stat[d]` via a parallel scan. This decouples accumulation entirely from the routing/contention question: there is no scatter, only a structured reduction. MLX library has primitive support for segmented reductions (`mx::cumsum`, `mx::reduce`, `mx::segmented_sum`) which can be composed.

**Why structurally new:**
- Eliminates the scatter pattern entirely. There is no "owner lane" question.
- Not a kernel-internal optimization; it replaces the kernel with library primitives + a custom segmented-reduction kernel.
- Not in DEC-013/014/015/017/019/048/049.

**Apple Silicon feasibility:**
- For each bin b, this is a per-partition reduction over N docs. At Epsilon with 128 bins × 2000 features = 256,000 reductions per iteration. Each reduction is over ~6,250 docs/partition at depth 6. Total ops: 256k × 6.25k = 1.6 G-ops/iter — but this is across all docs for *each* bin, naively 128× the work of the current kernel.
- Smarter form: combine bin selection with reduction in a single kernel using segmented sum keyed on bin value. MLX may not have a direct primitive; would require a custom Metal kernel.

**Theoretical upper-bound:** Depends on how much MLX's reduction kernels are optimized. Best case: matches `mx::sum` throughput (~50 GB/s effective on M3 Max) → equivalent to current. Worst case: 128× slower because of the per-bin loop.

**Risks:**
- (R1) Library-primitive composition is highly likely to be slower than the current bespoke kernel.
- (R2) Numerical stability of large parallel reductions — Higham γ_log(N) for N=400k = γ_18 ≈ 1.2e-6 → DEC-008 envelope tight on RMSE/Logloss but acceptable.
- (R3) MLX framework overhead per primitive call.

**Pre-flight gates:**
- (P1) MLX primitive throughput micro-benchmark (`mx::sum`, `mx::segmented_sum` if it exists).
- (P2) Code inspection of MLX sources (`../mlx/`) to identify whether segmented reduction primitives exist or must be written.
- (P3) Routing argument: "For bin b, partition p, feature f: `hist[f][p][b] = reduce(d in part_p, bin(d, f) == b ? stat[d] : 0)`. No scatter, no shuffle, single output per kernel cell."

### Candidate L4 — Hybrid two-stage: coarse-bin + refine

**Mechanism:** Two-pass histogram. Stage A: coarse histogram with 8 bins (3 bits) per feature instead of 128, using the current src-broadcast pattern. Output is small enough to fit in registers — no threadgroup memory needed for the coarse pass. Stage B: refine within each coarse bin using a different, lower-overhead kernel (since the per-bin doc count is known and small from stage A). Mirrors radix-sort idea from L2 but avoids the BW cost of sorting all docs.

**Why structurally new:**
- Two-stage decomposition not present in any of the 7 retired levers.
- Coarse pass uses register-residency (DEC-014 territory) BUT at much-reduced bin count (8 vs 128) — VGPR pressure is 8× lower than DEC-014's failed BATCH_DOCS=64.
- Fine pass operates on coarse-bin slices, so per-TG amortization is *increased* (the regime DEC-017 fell into is *avoided*).

**Apple Silicon feasibility:**
- Stage A registers: 8 bins × 4 features × 4 B = 128 B per lane. Within VGPR budget.
- Stage A→B handoff: coarse histogram + per-coarse-bin doc indices. Memory: 8 coarse bins × 4 features × 64 partitions × 4 B per count = 8 KB total (negligible).
- Stage B: 16 docs/coarse-bin/partition (avg) at Epsilon. Per-TG amortization at this granularity matches gate-config at depth-3 (≈195 docs/TG, where DEC-017 toy-isolation worked). **This is the load-bearing argument for L4.**

**Theoretical upper-bound:** 1.5–2.5× iter at Epsilon if both stages cleanly compose. Stage A is ~16× faster than full pass. Stage B at coarse-bin granularity should match toy-kernel projections from DEC-017's isolated-shape measurement (~84% accumulation reduction).

**Risks:**
- (R1) Hand-off overhead between Stage A and Stage B may absorb savings. Two dispatches, two synchronizations, intermediate buffer reads.
- (R2) Stage B is essentially a re-dispatch at a different granularity — re-introduces the dispatch-overhead question.
- (R3) Parity: two-stage accumulation has a different reduction order than single-pass.

**Pre-flight gates:**
- (P1) Stage-A register pressure analysis via Metal compiler dump. Must show ≤ 64 VGPR/lane.
- (P2) Stage-B per-TG amortization arithmetic at Epsilon, Higgs-1M, Gate. Must show docs/thread ≥ 30 (well into DEC-017's "winning regime").
- (P3) Routing argument for both stages independently.

### Candidate L5 — Sparse representation (only-non-zero-bins)

**Mechanism:** Observe that at Epsilon (400k docs × 2000 features × 128 bins = 102 M histogram cells), most cells are zero or near-zero — a typical histogram is sparse. Instead of materializing the full dense histogram, accumulate only the non-zero contributions into a sparse format (CSR or hash-map style) and decompress only when needed. The src-broadcast loop becomes a streaming append into a per-TG sparse buffer; merge happens in a separate pass.

**Why structurally new:**
- Not a representation change in any of the 7 retired levers (all worked with dense histograms).
- Eliminates the bin-owner-routing problem if the sparse representation is per-doc rather than per-bin.

**Apple Silicon feasibility:**
- Sparse-buffer size estimation: each doc contributes 4 (feature, bin, stat) tuples per feature group. At Epsilon: 400k × 2000 × 12 B = 9.6 GB raw stream. **This is the kill scenario** — without compression, a sparse representation is more expensive than dense.
- Density check: at depth 6 with 400k docs / 64 partitions = 6250 docs/partition × 2000 features = 12.5 M (feature, partition) pairs. With 128 bins, density ratio = 12.5M / (2000 × 64 × 128) = ~76% — actually quite dense at the (partition, feature, bin) granularity in Epsilon.
- L5 may be viable on Adult / Higgs-1M (smaller, denser per-bin) but unlikely on Epsilon.

**Theoretical upper-bound:** Best case 3× if density < 25% (not the Epsilon case). Likely BELOW 1× at Epsilon due to representation overhead.

**Risks:**
- (R1) Density analysis kills L5 at Epsilon before T3.
- (R2) Sparse-format merge kernel may have its own race-condition class (DEC-023 echo).
- (R3) Score-calc downstream kernel must be modified to accept sparse input — large blast radius.

**Pre-flight gates:**
- (P1) Density measurement at Epsilon, Higgs-1M, Gate config. Density < 25% required for L5 to be considered.
- (P2) If density is high, L5 retires at T2 without further analysis.
- (P3) Routing argument.

---

## 2. Recommended advisory board sequence

The 7-falsification pattern shows that single-perspective optimism is the failure mode. Mirror S46's advisory board sequence:

| Stage | Agent | Output |
|---|---|---|
| **T0a — Visionary brainstorm** | `@visionary` | Expand candidate-class enumeration. Produce 2–3 *additional* candidates beyond L1–L5. Cross-domain ideas welcome (graph algorithms, signal-processing primitives, bioinformatics histogram methods). |
| **T0b — Devils-advocate stress-test on candidate list** | `@devils-advocate` | For each candidate (L1–L5 + visionary additions): apply the 7-falsification cross-reference. Reject any that maps to a retired DEC. Apply MANDATORY-CODE-INSPECTION rule: if a candidate makes a mechanism claim, demand file:line citation BEFORE T2. Surviving candidates feed T1. |
| **T1 — Current-state at v0.7.0 baseline** | `@performance-engineer` | Fresh f_hist measurement at v0.7.0 master baseline. Document drift from S46-T4 numbers if any. |
| **T2 — Silicon-architect realism** | `@silicon-architect` | For each surviving candidate, evaluate Apple Silicon feasibility: TG memory, VGPR pressure, dispatch overhead, Metal API surface. Output: shortlist of 1–3 candidates that pass the silicon gate. |
| **T3 — Probe spec writing** | `@research-scientist` | For each shortlisted candidate, write a probe spec with explicit kill-thresholds. Mirror `docs/sprint46/T3/probe-d-spec.md` structure. **Routing-completeness argument is mandatory** (S46-T6 rule). Code-inspection sign-off mandatory before any code is written. |
| **T4 — Strategist synthesis (if multi-candidate at T3)** | `@strategist` | If T2 emits >1 candidate, sequence them by ROI / risk. Identify shared infrastructure. Define exit criteria per candidate. |
| **T5 — Decision** | User + `@ml-product-owner` | Greenlight ZERO, ONE, or MORE candidates for S49+ engineering. ZERO is a legitimate outcome. |

**Why insert T1 (f_hist) BEFORE T2 (silicon-architect):** silicon-architect's analysis depends on the f_hist number to compute realistic upper-bound iter speedup at v0.7.0 baseline.

---

## 3. f_hist methodology at v0.7.0 baseline

**Reference:** `docs/sprint46/T4/f_hist/analysis.md`. The S46 measurement tool (`bench_boosting_s46` with `-DCATBOOST_MLX_STAGE_PROFILE` and `--per-kernel-profile`) is reusable.

**Required measurement (T1):**

| Shape | rows × features | classes | depth | bins | Why |
|---|---|---|---|---|---|
| Gate-config | 50,000 × 100 | 2 | 6 | 128 | Carryover anchor from S19 onward; calibration reference. |
| Higgs-1M-proxy | 1,000,000 × 28 | 2 | 6 | 128 | The DEC-051 working-hypothesis anchor (≤5× MLX/CUDA ratio). |
| Epsilon-proxy | 400,000 × 2,000 | 2 | 6 | 128 | The high-feature-dim cost-driver anchor; primary v0.8.0 perf-gate candidate shape. |

**Methodology:** 3 seeds × 12 iters × per-kernel profile. 10%-trimmed warm mean. Same as S46-T4.

**Acceptance criterion for T1:**
- f_hist measured at v0.7.0 master baseline. Three-shape table.
- Drift vs. S46-T4 reported (likely zero — `kernel_sources.h` byte-identical from S30 through v0.7.0).
- If f_hist drifted significantly (>0.05 at any shape), document and flag for advisory-board review BEFORE T2.

**What f_hist tells us:**
- f_hist ≥ 0.95: any candidate with histogram-only ≥3× speedup maps to ~2.93× iter speedup. Headroom for Outcome A.
- 0.80 ≤ f_hist < 0.95: candidates need >3× histogram speedup OR must compose with a non-histogram lever.
- f_hist < 0.80: histogram is no longer the plurality cost; v0.8.0 should consider non-histogram levers.

---

## 4. Threshold-lock proposal

Per DEC-051 explicit clause: *"The threshold must be set BEFORE the v0.8.0 lever-research arc opens, to avoid retroactive goalpost-moving."* This must lock at **T0 of the kickoff sprint, before T1 begins.**

### Working hypothesis (from DEC-051)

**MLX/CUDA wall-clock ratio ≤ 5× on Higgs-1M iter=1000, parity intact.**

### Endorsement / refinement analysis

**Current Higgs-1M iter=1000 numbers** (from `docs/benchmarks/cross-class-cuda-comparison.md` §3.2):
- catboost_mlx (M3 Max): 128.79 s
- catboost_cuda (RTX 5070 Ti): 5.55 s
- Current MLX/CUDA ratio: **23.2×**

**Hardware physics floor:**
- RTX 5070 Ti FP32 ~30 TFLOPS vs M3 Max GPU die ~5–7 TFLOPS = ~4–6× compute differential
- Memory bandwidth: 896 GB/s vs ~400 GB/s = ~2.24× differential
- Physics floor for hardware-class gap: ~4–6× on compute-bound, ~2× on bandwidth-bound

### Recommendation: ENDORSE the working hypothesis (≤5× MLX/CUDA on Higgs-1M iter=1000)

**Rationale:**
- 5× sits at the upper end of the hardware-physics floor (4–6×). Achieving it means MLX has closed the *implementation* gap and only the *hardware-class* gap remains.
- Higgs-1M iter=1000 is a clean numeric workload (no categoricals); it isolates the kernel-throughput question from the categorical-encoding asymmetry (DEC-046).
- Higgs-1M is large enough (28 features × 1M rows) to be representative but small enough to iterate fast in research.
- Current state 23.2× → target ≤ 5× = **~4.6× improvement required.** Per f_hist headroom analysis, this is achievable IF a structurally-new histogram lever delivers ~5× histogram speedup (with f_hist Higgs-1M ≈ 0.90, that maps to ~4.5× iter speedup).

### Success rubric (LOCK at T0)

For PyPI publish to unblock per DEC-051, ALL of the following must hold at v0.8.0 release:

| Gate | Specification | Required |
|---|---|---|
| **G1 — Throughput** | MLX/CUDA wall-clock ratio ≤ 5× on Higgs-1M iter=1000, 3-seed mean | YES |
| **G2 — Branch-B parity** | `python/tests/regression/test_branch_b_regression.py` PASS at v0.8.0 baseline | YES |
| **G3 — Logloss equivalence** | Higgs-1M iter=1000 logloss within ±0.0003 of v0.7.0 | YES |
| **G4 — Cross-shape robustness** | Epsilon iter=2000 MLX/CUDA wall-clock ratio ≤ 15× | YES |
| **G5 — DEC-008 envelope** | RMSE ulp ≤ 4, Logloss ulp ≤ 4, MultiClass ulp ≤ 8 across DEC-008 envelope | YES |

**Why include G4:** The 88× MLX/CUDA gap on Epsilon is far worse than Higgs-1M's 23.2×. A v0.8.0 win that closes Higgs-1M to 5× but leaves Epsilon at 88× is a partial victory.

**Anti-goalpost-moving rule:** Once locked at T0, these gates do not move *down* (relax) without a documented amendment with explicit rationale. They may move *up* (tighten) if early measurements show the working hypothesis is too conservative.

---

## 5. Sprint structure proposal (T0–T6)

Mirrors S46 structure where it fits. **Hard 1-week budget.**

| Day | Task | Owner | Output |
|---|---|---|---|
| **D1** | **T0 — Scaffold + threshold lock + advisory board kickoff** | `@ml-product-owner` | Branch cut. `docs/sprint48/sprint-plan.md`. DEC-052 OPEN filed. Threshold rubric §4 LOCKED. `@visionary` brainstorm session results filed to `docs/sprint48/T0/visionary-brainstorm.md`. `@devils-advocate` candidate-list stress-test filed to `docs/sprint48/T0/dac-stress-test.md`. **Hard gate: this list of candidates is the only set considered for T1+.** |
| **D2** | **T1 — Current-state characterization + f_hist at v0.7.0 baseline** | `@performance-engineer` | `docs/sprint48/T1/current-state.md` + `docs/sprint48/T1/f_hist/analysis.md`. |
| **D3** | **T2 — Candidate feasibility** | `@silicon-architect` (lead) + `@research-scientist` + `@mathematician` | `docs/sprint48/T2/feasibility.md`. Output: shortlist of 1–3 candidates. |
| **D4** | **T3 — Probe spec(s)** | `@research-scientist` | `docs/sprint48/T3/probe-spec-<candidate>.md` per shortlisted candidate. Routing-completeness argument is mandatory. |
| **D5** | **T4 — Probe-D-style measurement (NO production code commits)** | `@performance-engineer` + `@silicon-architect` | `docs/sprint48/T4/<candidate>/probe-verdict.md`. Probe binaries gated under `#ifdef V0_8_0_PROBE_<X>`. 27-run sweep per candidate. **Parity must be checked** (S46 Probe B was 9.79× BOGUS without parity gate). |
| **D6** | **T5 — Decision gate (DEC-052 verdict)** | `@strategist` (synthesis) + `@devils-advocate` (stress-test) + user | `docs/sprint48/T5/decision.md`. Outcome A (greenlight S49+ engineering on candidate X), B (user-call: marginal speedup), or C (HALT, all candidates falsified). |
| **D7** | **T6 — Sprint close-out** | `@ml-product-owner` + `@technical-writer` | DEC-052 finalized. State files + LESSONS-LEARNED updated. PR `mlx/sprint-48-v0.8.0-arc-kickoff` → master. |

**Sequential gates:** T0 → T1 → T2 → T3 → T4 → T5 → T6.

**Hard rules carried from S46:**
- NO production-code commits in S48. All probe code is `#ifdef`-gated.
- MANDATORY-CODE-INSPECTION at every mechanism claim (file:line citation required).
- Pre-sweep code-inspection sign-off on accumulation/routing invariant per S46-T6.
- Branch-B regression GREEN on master throughout.

---

## 6. Risk pre-mortem — the 8th-falsification scenario

DEC-013, DEC-014, DEC-015, DEC-017, DEC-019, DEC-048, DEC-049 = 7 throughput hypothesis falsifications. **The 8th-falsification pattern is the failure-mode this scoping doc is most likely to fall into.**

### Scenario 8a — All 5 candidates fail the premise gate at T0

Most likely failure: visionary expands the candidate list, but devils-advocate's 7-falsification cross-reference rejects every candidate, including L1–L5, as "echo of DEC-XXX." **What to do:** T6 closes the arc with DEC-052 = RETIRED-AT-PREMISE. Product implication is significant: PyPI publish per DEC-051 may stay deferred indefinitely unless either (i) hardware advances (M5/M6 Apple Silicon with substantively different microarchitecture), or (ii) algorithmic advances elsewhere (non-histogram-class lever, e.g. tree-search restructure, score-calc fusion). User re-decides v0.8.0 scope at that point.

### Scenario 8b — All candidates pass T0 but fail T2 silicon-realism

Candidates survive premise gate but silicon-architect rejects all of them on TG memory, VGPR, or dispatch-overhead grounds. **What to do:** Same as 8a — DEC-052 = RETIRED-AT-SILICON-FEASIBILITY.

### Scenario 8c — 1+ candidate passes T2, all fail T4 measurement

Probe-D-style measurement shows no candidate clears the kill threshold. **What to do:** DEC-052 = RETIRED-EMPIRICALLY. Sprint is still a successful research sprint (negative results with quantitative evidence are legitimate deliverables).

### Scenario 8d — Candidate passes T4 but fails parity

Per S46 Probe B — speedup is real, but accumulation order changes beyond DEC-008 envelope. **What to do:** DEC-052 = CONTINGENT — engineering can proceed IF parity envelope can be tightened (e.g. Kahan summation, double-precision intermediate). User-call at T5.

### Scenario 8e — All gates pass at T5 but T6 user decides not to proceed

User reviews the candidate, weighs engineering cost vs. throughput delta, and decides v0.8.0 isn't worth the implementation effort. **What to do:** DEC-052 = DEFERRED. Candidate documented for future re-entry.

### EARLY HALT triggers (during T0–T5, before reaching T6)

Halt if any of:
- T0a visionary brainstorm + L1–L5 produces 0 candidates that pass devils-advocate's gate. → Skip to T6, file DEC-052 = RETIRED-AT-PREMISE.
- T1 measurement shows f_hist drift > 0.10 at any shape vs. S46-T4. → Pause, advisory-board review, possibly re-scope.
- T2 silicon-architect rejects all shortlisted candidates. → Skip to T6, file DEC-052 = RETIRED-AT-SILICON.
- T4 probe shows MANDATORY-CODE-INSPECTION violation. → Stop that probe, root-cause, file LESSONS-LEARNED entry.
- 3 days into T4 with no candidate clearing 1.2× iter speedup at any shape. → Strategic pause, advisory-board review.

---

## 7. Multi-sprint roadmap (if kickoff sprint succeeds)

Conditional on T5 = Outcome A or B:

| Sprint | Theme | Outputs |
|---|---|---|
| **S48 (kickoff)** | This scoping sprint. T0–T6 produce a candidate-or-retire verdict. | DEC-052 + scoping doc + probe verdicts. |
| **S49 (engineering)** | Engineering implementation of the surviving candidate. Branch-B regression GREEN throughout. Atomic commits per DEC-012. | New histogram kernel ships behind feature flag (`use_v0_8_0_histogram=True`). |
| **S50 (parity gate)** | Full DEC-008 envelope parity sweep on the new kernel. 18-config sweep. 100/100 determinism on gate config. | Parity report. Gate to S51 cutover. |
| **S51 (cutover)** | Switch default histogram kernel from current production to v0.8.0. Update `kernel_sources.h` md5 anchor. Full benchmark sweep. | New v0.8.0 baseline; cross-class CUDA writeup refresh. |
| **S52 (publish gate)** | Verify G1–G5 from §4 hold at v0.8.0 baseline. If yes → PyPI publish unblock per DEC-051. | v0.8.0 release ceremony: tag, GitHub Release, PyPI publish (per DEC-051 amendment). |

**Total estimate: 5 sprints (S48 + 4 more) ≈ 5 weeks** if every sprint runs to budget. **Risk-adjusted: 7–10 sprints** given the 7-falsification track record.

If T5 = Outcome C (RETIRED), only S48 fires. Total: 1 week.

---

## 8. Time / budget estimate

| Phase | Sprints | Wall-clock | Confidence |
|---|---|---|---|
| Kickoff (S48) | 1 | 1 week | High |
| Engineering (S49–S51) if Outcome A | 3 | 3 weeks | Medium |
| Cutover + publish (S52) | 1 | 1 week | High |
| **Total v0.8.0 to PyPI publish** | **5 sprints** | **~5 weeks** | Medium |

Compare to: S46 was 1 day actual against 1-week budget; S43–S47 averaged ~3 days/sprint. Plausible compression to **3 weeks** if all gates pass on first attempt.

If kickoff outcome is C (RETIRED), v0.8.0 = 1 sprint, and the v0.8.0 milestone re-scopes to non-throughput work at user discretion.

---

## 9. Deferred items — perf-gate robustness + `stage_profile` API drift

Two items are already in the S48 backlog:

1. **Perf-gate robustness** — `mlx-perf-regression.yaml` ratio-invariance assumption breaks at 50k×50 due to CPU-time variance. Fix: scale to N≥200k, median-of-3, or noise-tolerant threshold.

2. **`stage_profile` API drift** — `bench_boosting.cpp` API has drifted from the in-process Python path. Needs a unified profiling surface.

**Recommendation:** **KEEP THESE SEPARATE FROM S48**, do not roll into the v0.8.0 kickoff scope. Reasons:
- S48 is a research/scoping sprint with hard 1-week budget. Adding 2 engineering tasks dilutes focus.
- Perf-gate robustness depends on the v0.8.0 cutover (S51) — fixing it before knowing the kernel topology is wasted work.
- `stage_profile` API drift is independent of the lever class chosen and can be done in parallel by an unblocked agent.

**Suggested home:** Track both as "S48 follow-up" items; pick up in S49 (if Outcome A) or in a dedicated polish sprint (if Outcome C).

---

## 10. Authority chain summary

| Reference | Role in this scoping |
|---|---|
| `docs/sprint46/T5/decision.md` (DEC-049 OUTCOME) | Defines what is OUT of scope (simd_shuffle family) and explicitly lists candidate classes that remain. |
| DEC-050 | v0.7.0 reproducibility-grade ships; throughput deferred to v0.8.0+. |
| DEC-051 | PyPI publish gated on CUDA-class throughput. ≤5× MLX/CUDA on Higgs-1M iter=1000 working hypothesis. Threshold must lock BEFORE arc opens. |
| DEC-008 | Parity envelope (RMSE ulp ≤ 4, MultiClass ulp ≤ 8). |
| DEC-011 | 32 KB threadgroup memory hard ceiling. |
| DEC-014 | VGPR/lane budget rule. |
| DEC-017 | Toy-to-production transfer rule. |
| `Frameworks/LESSONS-LEARNED.md` MANDATORY-CODE-INSPECTION | Mechanism claims require file:line citation. |
| S46-T6 SIMD routing invariant | Probe specs must include explicit routing argument BEFORE measurement. |

---

## 11. Open questions for advisory board / user input

1. **Threshold lock — endorse or refine ≤5× MLX/CUDA on Higgs-1M iter=1000?**
   - Recommend: ENDORSE. See §4 rationale.
2. **Should `@visionary` get a 1-day session BEFORE T0 commits the candidate list?**
   - Recommend: YES. Pattern matches S46. Insert as T0a.
3. **Should the perf-gate / stage_profile items be folded into S48 or kept separate?**
   - Recommend: SEPARATE. See §9.
4. **If Outcome C (all candidates retire) at T5, does v0.8.0 re-scope to non-throughput milestones?**
   - User decision. Pre-decide framework: "if Outcome C, milestone v0.8.0 = ?" candidates: ordered boosting (E2), categorical-encoding rework (DEC-046 closure), MSLR-WEB10K ranking support (S44 carryover).
5. **Is the 5-sprint roadmap acceptable, or should v0.8.0 be timeboxed (e.g., "if not done in 4 sprints, retire")?**
   - User decision.

---

## Definition of Done (for this kickoff sprint)

- [ ] T0–T6 documented under `docs/sprint48/`.
- [ ] DEC-052 filed in `.claude/state/DECISIONS.md` with verdict.
- [ ] Threshold rubric §4 LOCKED at T0 and unchanged at T6.
- [ ] HANDOFF/TODOS/CHANGELOG-DEV/MEMORY/LESSONS-LEARNED updated.
- [ ] Branch-B regression GREEN on master throughout.
- [ ] PR `mlx/sprint-48-v0.8.0-arc-kickoff` → master prepared for user review.
- [ ] If Outcome A: S49 sprint plan drafted (engineering).
- [ ] If Outcome C: arc retired with DEC-052 = RETIRED, scoping doc archived, v0.8.0 milestone re-scoped at user discretion.

**Negative results are legitimate deliverables.** A sprint that rules out the entire structurally-new-lever design space, with quantitative evidence, is a successful sprint. The 7-falsification track record makes Outcome C a likely outcome — the project rule treats this as a positive contribution.
