# S46-T5 Decision Gate — DEC-049 OUTCOME

**Date:** 2026-05-05
**Sprint:** 46
**Branch:** `mlx/sprint-46-simd-shuffle-research`
**Owner:** @strategist (synthesis); @devils-advocate (stress-test)
**Companion DEC:** DEC-049 in `.claude/state/DECISIONS.md`
**Evidence source:** `docs/sprint46/T4/probe-d/results.json`, `docs/sprint46/T4/B/probe-verdict.md`, `docs/sprint46/T4/D/probe-verdict.md`

---

## Summary

All four bounded candidates from the S46 research arc are falsified. After a build-environment fix (MLX 0.31.2 / Darwin 25.3 SDK incompatibility resolved via CMake integration — `docs/sprint46/T4/build-env/status.md`) and a dispatcher rewrite (mx::add sequential chain → pre-allocated K-slot + mx::concatenate + parallel eval), a 27-run sweep at production dispatch shape measured:

- **Probe B** (per-lane register accumulation): **9.79× iter speedup — BOGUS.** Parity FAIL from iter-0. Speedup is artifact of dropping 96.9% of histogram contributions, not genuine acceleration.
- **Probe D2** (split-K with merge kernel): **1.006× iter speedup.** Parity FAIL (Δloss 0.03+ at Epsilon). Empirically flat — no measurable speedup after the dispatcher rewrite.
- **Probe C** (sort-by-bin): RETIRED structurally before measurement per DEC-023 (race condition on shared accumulator).
- **Probe D1** (intra-kernel K-split with private threadgroup partials): RETIRED in T3 erratum — `partialHist[K][8][1024]` at K=4 requires 128 KB, 4× over the 32 KB ceiling (DEC-011).

**Verdict: Outcome C — RETIRED.** DEC-049 = KILL. The premise "eliminate src-broadcast serialization without restoring routing" is structurally impossible. The simd_shuffle-family throughput research arc on the existing histogram kernel topology is closed.

---

## T4 Measurements Summary

### Sweep parameters

- **Shape matrix:** Gate-config (50k×100), Higgs-1M-proxy (1M×28), Epsilon-proxy (400k×2000)
- **Seeds:** 42, 43, 44
- **Iters per run:** 12 (1 cold + 11 warm, 10%-trimmed warm mean)
- **Probes measured:** Baseline, Probe B, Probe D2
- **Total runs:** 3 shapes × 3 seeds × 3 probes = 27 runs
- **Data:** `docs/sprint46/T4/probe-d/results.json`

### Performance table

| Probe | Shape | hist_ms mean | iter_ms mean | hist speedup | iter speedup |
|---|---|---|---|---|---|
| Baseline | Gate (50k×100) | 37.75 | 51.43 | — | — |
| Baseline | Higgs-1M (1M×28) | 405.95 | 455.37 | — | — |
| Baseline | Epsilon (400k×2000) | 2057.61 | 2097.63 | — | — |
| **Probe B** | Gate | 4.27 | 19.43 | 8.85× | 2.65× |
| **Probe B** | Higgs-1M | 28.03 | 70.20 | 14.48× | 6.49× |
| **Probe B** | **Epsilon** | **174.01** | **214.17** | **11.82×** | **9.79×** |
| **Probe D2** | Gate | 38.48 | 52.17 | 0.98× | 0.99× |
| **Probe D2** | Higgs-1M | 393.18 | 435.30 | 1.03× | 1.05× |
| **Probe D2** | **Epsilon** | **2044.49** | **2085.27** | **1.01×** | **1.006×** |

### Parity results

| Probe | Loss delta vs baseline | First observed | Verdict |
|---|---|---|---|
| Probe B | Δloss 0.005–0.008 across all shapes | iter-0 | PARITY FAIL |
| Probe D2 | Δloss 0.03+ at Epsilon | iter-5 onward | PARITY FAIL |

---

## Probe B: The Load-Bearing Finding

### What Probe B did

Probe B replaced the 32-iter `simd_shuffle` src-broadcast loop (`kernel_sources.h:209–224`) with per-lane direct accumulation. Each lane processes only its own document (`d = batch_start + lane`) and writes to `laneAccum[j]` when it owns that bin (`(bin & 31) == lane`). After the batch loop, `laneAccum[j]` is written directly to `simdHist[simd_id][j*32+lane]` — no broadcast needed because "each lane's partial IS the full sum for its owned bins."

### Why the 9.79× speedup is bogus

Read `kernel_sources.h:1374-1407` (the Probe B accumulation block under `#ifdef SIMD_SHUFFLE_PROBE_B`):

```metal
const uint d = batch_start + lane;           // lane processes doc by INDEX
const uint packed = /* ... packed bins for doc d ... */;
const uint stat   = /* ... stat for doc d ... */;
for (uint j = 0u; j < FEATURES_PER_PACK; ++j) {
    const uint bin = (packed >> (24u - 8u * j)) & 0x7Fu;
    if ((bin & (SIMD_SIZE - 1u)) == lane) {  // lane writes only if it OWNS bin
        laneAccum[j] += stat;
    }
}
```

The processor lane is determined by `d & 31` — the doc index modulo 32. The owner lane is determined by `bin & 31` — the bin value modulo 32. These are **statistically independent** under any realistic bin distribution (bins are assigned by quantization, not by doc index).

P(processor == owner) = 1/32 ≈ 3.1%.

**Probe B silently discards approximately 96.9% of histogram contributions.** The 9.79× speedup is explained entirely by doing 1/32 the work. The histograms are structurally wrong from iteration 0 — not from numerical drift, not from floating-point re-ordering, but from missing entries. The final loss diverges from baseline starting at the first iteration; no amount of numerical tolerance negotiation could save it.

### The misleading comment

The code comment at `kernel_sources.h:1423-1434` states: "per-lane partial IS the full sum for that lane." This claim confused the silicon-architect during the T2 pre-flight by conflating two distinct concepts:

- **"lane processes doc"** — determined by doc index; lane `l` sees `d = batch_start + l`
- **"lane owns bin"** — determined by bin value; lane `l` owns bins where `bin & 31 == l`

In production (`kernel_sources.h:209–224`), the `simd_shuffle(packed, src)` broadcast exists *precisely* to route every document to the lane that owns its bin, decoupling the two concepts. Removing the broadcast without restoring the routing is not an optimization — it breaks the kernel's invariant that every (bin, stat) pair reaches its owner.

The comment's claim is only true in the specific case where every document happens to be owned by the lane that processes it. Under uniform bin assignment this occurs with probability 1/32 — exactly the fraction of work Probe B actually performs.

### Why this matters for the verdict

Under the sprint plan's kill criteria (`docs/sprint46/T3/probe-d-spec.md` §4): "Kill threshold: if Epsilon iter speedup < 1.5×, RETIRE." Probe B measured 9.79× — which by the numeric criterion alone would appear to be a candidate for Outcome A. The parity gate prevents a wrong answer from being treated as a correct fast answer.

The devils-advocate meta-criterion clarification (from `project_sprint46_t5_verdict.md`): "speedup-at-wrong-answer is zero useful speedup." A kernel that runs 9.79× faster by doing 3.1% of the work and producing incorrect histograms has zero useful speedup. Parity failure at iter-0 is the strongest possible falsification signal — the kernel is not computing the correct function at all.

**Probe B is FALSIFIED.** The speedup is artifact; the lever is structurally broken.

---

## Probe D2: Empirical Flat

### What Probe D2 did

After the dispatcher rewrite (Phase 2 of the build-environment fix), Probe D2 allocated K=4 independent per-slice partial histogram buffers, dispatched K accumulation kernels simultaneously via `mx::eval(kSlices)`, then merged with a separate merge kernel via `mx::concatenate`. This eliminates the original `mx::add` sequential accumulation chain that would have serialized the K dispatches in the MLX compute graph.

### Result

Epsilon-proxy iter speedup: **1.006×.** Epsilon hist speedup: **1.006×.** Both within measurement noise of 1.0×.

The theoretical mechanism — K=4 parallel dispatch reduces per-TG doc count by 4×, so the src-broadcast loop runs 4× fewer iterations — did not materialize. Most likely causes:

1. The K accumulation dispatches amortize the same fixed per-TG overhead that DEC-017 identified as the bottleneck in production-shape atomic-CAS designs. At Epsilon's doc/TG ratio, dispatch overhead absorption cancels the K-fold reduction in loop iterations.
2. `mx::concatenate` forces a memory copy of the K partial slices before the merge kernel reads them. At 262 MB aggregate partial buffer size (K=4 × 64 partitions × ~1M floats), the copy may absorb the wall-clock savings from the K-fold src-loop reduction.
3. MLX's lazy evaluation graph may still serialize the K dispatches at the Metal command-buffer level even after the `mx::eval(kSlices)` call.

Parity also failed (Δloss 0.03+ at Epsilon). This is likely related to merge-kernel atomic ordering — the merge kernel writes to final histogram slots via `atomic_fetch_add`, introducing non-deterministic accumulation order across merge TGs. The Higham bound at γ_10 (7 fold + 3 merge additions) ≈ 6.0e-7 is theoretically within DEC-008 for MultiClass, but the empirical loss delta of 0.03 exceeds the ULP=4 RMSE ceiling by a large margin.

**Probe D2 is FALSIFIED** on both grounds: no measurable speedup, parity breach.

---

## Devils-Advocate Meta-Criterion Check

The sprint plan (`docs/sprint46/sprint-plan.md` §T5) defined three outcome thresholds:

- **Outcome A (COMMIT):** ≥3× iter speedup at Epsilon, parity path plausible
- **Outcome B (user-call):** 1.5–3× iter speedup
- **Outcome C (HALT):** <1.5× iter speedup

**Probe B numeric check:** 9.79× — would trigger Outcome A by the number alone. This is the exact failure mode the meta-criterion guards against. Per devils-advocate review: "speedup-without-parity counts as FALSIFICATION, not 'invalid measurement.' Lever's contract includes correctness; speedup at wrong answer is zero useful speedup." Probe B is classified as Outcome C (falsified), not Outcome A.

**Probe D2 numeric check:** 1.006× — well below the 1.5× Outcome C threshold. Clean falsification regardless of parity status.

**Probe C:** Structurally RETIRED per DEC-023. The sort-by-bin approach requires bins to be pre-sorted so each lane accumulates into a contiguous range. This introduces a shared accumulator race when the sort is applied to all 4 features simultaneously. DEC-023 v5 resolution shipped feature-0-only sort + T1-shuffle for features 1-3, yielding 1.01× speedup — not the 0.317× hist_ms ratio originally measured. No surviving configuration of Probe C avoids the race without collapsing to the production topology.

**Probe D1:** RETIRED in T3 erratum. `partialHist[K=4][8 SIMD groups][1024 bins] × 4 bytes = 128 KB` — 4× the 32 KB threadgroup memory ceiling (DEC-011). Not measurable; structurally infeasible.

All four candidates satisfy Outcome C criteria. The meta-criterion "MLX/CUDA gap > 50×" clause (which would override Outcome C to force a user decision) is NOT load-bearing here — it exists to prevent premature retirement when the gap might close via software. The gap is 88× on Epsilon; but all candidate designs have been empirically falsified at production shape, meaning there is no software path to close it within this research arc. "All candidates falsified at production shape" is itself the Outcome C signal, independent of the hardware-class gap.

---

## Why RETIRE, Not Defer

The sprint plan's Outcome C definition: "No candidate shows ≥1.5× upper-bound at production shape. simd_shuffle redesign is empirically not the load-bearing lever."

The deeper argument for retirement over deferral is **structural impossibility.** The premise of the entire research arc was: "eliminate src-broadcast serialization without restoring routing." After T4, this premise is falsified not just empirically but logically:

- Bin-owner mapping is intrinsic to bin values (`bin & 31`). It has no relationship to doc-index lane assignment.
- Routing every (doc, stat) pair to the lane that owns the target bin REQUIRES either:
  - (a) `simd_shuffle(packed, src)` broadcasting each doc's packed value to all 32 lanes so each lane checks ownership — the production approach, the cost we tried to eliminate; or
  - (b) Pre-sorting docs by bin before accumulation so the lane that processes each doc already owns its bins — Probe C, which has an unresolvable race at config #8 per DEC-023.
- There is no third route. Probe B (no routing, ownership predicate only) silently discards 96.9% of contributions. Probe D2 (K-split) reduces per-TG doc count but each K-slice still uses the src-broadcast internally — it does not eliminate the cost, only subdivides it, and the subdivision overhead cancels the gain.

No B/C/D-class variant changes this analysis. Any future proposal to "remove the simd_shuffle broadcast" must be rejected unless it discloses a routing replacement that does not itself reintroduce the broadcast cost. No such replacement exists in the existing kernel topology.

Deferral would consume another research arc to reach the same conclusion with more evidence. The structural argument is sufficient.

---

## Cross-References

| DEC | Relevance to T5 verdict |
|---|---|
| DEC-008 | Parity envelope (RMSE ulp≤4, MultiClass ulp≤8). Both B and D2 breach it. |
| DEC-011 | 32 KB threadgroup ceiling. D1 infeasibility derives from this. |
| DEC-017 | Toy-to-production transfer rule. D2's merge-overhead optimization did not transfer at production shape. |
| DEC-023 | Sort-by-bin race. Probe C structurally retired under this. |
| DEC-048 | Dispatch-fusion KILL (S45). Confirms this is the 7th throughput hypothesis falsification in this codebase. |
| DEC-049 | This DEC. OUTCOME = RETIRED. |

**Falsification chain:** DEC-013 (writeback plurality), DEC-014 (gather-latency bound), DEC-015 (col-major cache-line reduction), DEC-017 (T3b atomic-CAS, +42.3% production regression), DEC-019 (D2 sibling-subtraction), DEC-048 (H-Dispatch), DEC-049 (simd_shuffle arc) — **7 throughput falsifications** in this codebase. DEC-049 is the first caught by empirical-measurement-before-merge; all six prior falsifications were caught during or after engineering implementation. The LESSONS-LEARNED MANDATORY-CODE-INSPECTION rule and the Probe-D protocol fired correctly.

---

## What This Closes

**Closed by DEC-049 RETIRED:**

The simd_shuffle-family throughput research arc on the existing `kHistOneByteSource` / `kT2AccumSource` kernel topology. Specifically: per-lane register accumulation (B), sort-by-bin extension (C), intra-kernel K-split (D1), and split-K with merge kernel (D2). No further probes in this class are warranted without a structural change to the kernel that introduces a new routing mechanism.

**NOT closed by DEC-049:**

The broader v0.7.0 throughput question. Alternative levers — bin-distributed dispatch, sort-then-scan with race-free atomics, or a fundamentally different kernel topology replacing `kHistOneByteSource` entirely — remain unevaluated. They would require fresh f_hist analysis, a new DEC entry, and a new research arc. None of these is recommended in this document; the choice is the user's.

---

## Final Verdict

**DEC-049 = RETIRED. Outcome C — HALT.**

**S46 is a successful research sprint.** The Definition of Done from `docs/sprint46/sprint-plan.md` §"Definition of Done" is satisfied: T1–T5 deliverables exist, all candidates evaluated, empirical justification documented, verdict filed. Negative results with quantitative evidence that foreclose future re-exploration are legitimate deliverables (project rule, `CLAUDE.md`; `Frameworks/LESSONS-LEARNED.md` §"Negative results are legitimate deliverables").

**v0.7.0 path:** BLOCKING DECISION required from user. See HANDOFF.md §"BLOCKING DECISION FOR S47."
