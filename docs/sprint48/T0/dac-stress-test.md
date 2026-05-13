# S48-T0b — Devils-Advocate Formal Stress-Test of v0.8.0 Candidate Roster

**Date:** 2026-05-08
**Sprint:** 48 (Day 1, T0b)
**Branch:** `mlx/sprint-48-t0-brainstorm`
**Author:** @devils-advocate
**Authority:** DEC-049 OUTCOME (RETIRED), DEC-052 OPEN, scoping `docs/sprint48/scoping.md` §0 + §v2
**Companion:** `docs/sprint48/T0/visionary-brainstorm.md` (parallel; this stress-test ran on the v2 roster)
**Prior position:** YELLOW-leaning-RED on opening this arc at all (panel review 2026-05-06)

---

## §0 — Methodological note

The v2 panel synthesis already pre-retired L1 (dispatch explosion 2,000× over DEC-018), L2 (BW-bound sort 6.4 GB I/O at Epsilon), L3 (`mx::segmented_sum` doesn't exist in MLX 0.31.x), and L5 (76% density at Epsilon) at the silicon-math layer. **I do NOT re-stress-test those four — they are RETIRED-AT-PREMISE.** The roster I am stress-testing is the v2 surviving 5: **L4, C1, C4, C5, L6.**

Verdict scale: **KEEP** / **RETIRE-AT-PREMISE** / **FLAGGED-FOR-T2** (clears rejection rules but has specific question that MUST be answered YES at T2 by code-inspection or silicon-architect roofline).

The S46 catastrophe was Probe B's 9.79× "speedup" that survived T2 silicon analysis and only died at T4 measurement (parity check post-sweep). Cost of accepting a hidden rebrand here = entire 7-day S49 engineering budget. I retire at T0b when candidates fail the seven-DEC rejection lookup OR have suspect arithmetic from v2 itself.

---

## §1 — Seven-falsification cross-reference applied to each candidate

### Rejection lookup table (carryover from scoping §0)

| Retired DEC | Pattern | Reject if candidate's mechanism… |
|---|---|---|
| DEC-013 | Writeback-plurality / batched-atomic | …relies on per-bin atomic accumulation OR batched writeback as primary lever |
| DEC-014 | Wider batch / per-doc register tiling | …holds per-doc state in per-lane VGPR across the inner shuffle loop |
| DEC-015 | Col-major transpose | …primary win is gather-pattern locality (AGX prefetcher hides it) |
| DEC-017 | T3b atomic-CAS / per-TG amortization regime flip | …per-TG fixed overhead amortization regime *flips* without production-shape validation |
| DEC-019 | Stats pre-permute | …primary win is hiding gather latency |
| DEC-048 | H-Dispatch fusion | …proposes to fuse what `histogram.cpp:31` already fuses |
| DEC-049 | simd_shuffle B/C/D family | …removes broadcast routing without naming a routing replacement |

### Candidate L4 — Hybrid two-stage (coarse-bin + refine)

**Verdict: RETIRE-AT-PREMISE.**

L4 is a composite of three retired DEC patterns dressed up as one new lever.

*Stage A* uses "the current src-broadcast pattern" with registers holding 8 bins × 4 features = 32 accumulator slots × 4B = **128 B per lane**. This is *exactly* DEC-014's failure regime. DEC-014 was BATCH_DOCS=64 with per-doc register state of ~32 B per lane and the production port measured **+9.4% regression from VGPR spill** (DECISIONS.md:131-136). L4 Stage A increases per-lane register pressure to **128 B (4× DEC-014's failure point)** on top of existing production live-ness. The "8× lower than DEC-014" claim in scoping §1 L4 R1 is arithmetically wrong: DEC-014 spilled at 32 B accumulator state; L4 Stage A holds 128 B. Stage A is in DEC-014's failure regime, **not 8× clear of it**. v2 already flagged this as suspect arithmetic in its "Factual corrections to v1" section — but then proceeded to call L4 "marginal — KEEP" anyway. Once you accept the corrected arithmetic, L4 fails the premise gate, not just the T2 gate.

*Stage A also still requires `simd_shuffle` for routing.* The src-broadcast loop at `kernel_sources.h:209–224` is the routing mechanism — without it, the kernel hits the S46 Probe B silent-drop bug (LESSONS-LEARNED SIMD routing invariant). Stage A inherits the full 32-iter shuffle chain at the cost of holding a 4× larger accumulator. The "16× faster than full pass" claim has no path: bin count drops 128→8 but doc count is unchanged; writeback was already DEC-013-falsified at 5%.

*Stage B* claims to "match DEC-017's winning regime" of 195 docs/thread. The arithmetic doesn't check out. Scoping §1 L4 says "16 docs/coarse-bin/partition (avg) at Epsilon" — that's 16, not 195. DEC-017 was -42.3% in production at 3 docs/thread (DECISIONS.md:212-214). At 16 docs/thread, Stage B is in DEC-017's *failure* regime, not winning regime. Toy-to-production transfer rule explicitly forbids extrapolating from toy isolation at this regime gap.

**DEC echoes:** DEC-014 (Stage A VGPR), DEC-017 (Stage B amortization regime), DEC-049 (Stage A still needs simd_shuffle routing).

### Candidate C1 — Inverted-index histogram

**Verdict: RETIRE-AT-PREMISE.**

Three independent fatal flaws, any one sufficient.

*(F1) Required primitive does not exist.* Scoping §v2 "Factual corrections" notes: `mx::segmented_sum` does NOT exist in MLX 0.31.x. C1's mechanism statement says `gather + segmented_reduce`. Same missing-primitive class. Without `segmented_reduce`, the only path is a custom Metal kernel — at which point C1 collapses to "the current src-broadcast pattern but you also paid for an inverted-index build first," i.e., L2 with extra steps.

*(F2) Index-build cost echoes L2 retirement.* GBDT histogram-build is per-iteration AND per-depth — partition assignment changes when each tree level adds a split, so the inverted index must be rebuilt 6× per iteration. At Epsilon, sorting 400k docs by bin into 128 buckets per (feature, partition) = the same 6.4 GB I/O cost that pre-retired L2 at silicon math. Postgres GIN analogy fails because Postgres amortizes index-build across millions of queries; GBDT amortizes across exactly one histogram-build per depth-level.

*(F3) DEC-019 echo on the gather phase.* C1's per-bin gather is the exact gather pattern DEC-019 falsified. AGX prefetcher already hides gather latency, so eliminating gather provides ~no win.

**DEC echoes:** DEC-019, L2 territory, L3 territory.

### Candidate C4 — Persistent-kernel pipelining

**Verdict: FLAGGED-FOR-T2.**

C4 is NOT a kernel-internal optimization. The 7-falsification chain is *all* kernel-internal: re-design the histogram kernel's accumulation, routing, layout, dispatch granularity, or shuffle topology. C4 keeps the kernel byte-identical and changes the *invocation topology*. This is genuinely outside the rejection lookup table.

**Three blockers must clear at T2.**

*(Q1 — MANDATORY-CODE-INSPECTION) Does Metal support kernel persistence the way CUDA does?* Triton/FlashAttention-2 work because CUDA has cooperative-group primitives with cross-SM synchronization. Metal command queues are different: each `MTLCommandBuffer` is submitted, runs, and completes. There is no production-supported "persistent kernel" primitive in Metal 3.x or `mx::fast::metal_kernel` API surface. Silicon-architect or research-scientist must cite specific Metal API + MLX API path that supports kernel persistence across CPU work, with file:line. If the answer is "use larger command buffers," that's not persistent-kernel pipelining.

*(Q2) Upper-bound savings.* DEC-018 measured per-TG fixed overhead at 2.5% ± 1.3% of `histogram_ms`. At depth 6, six dispatches per iteration. Even amortizing ALL per-dispatch fixed overhead, upper bound is ~6 × 2.5% × histogram_ms = ~15% of histogram time → ~10% of iter time (f_hist≈0.97). **Upper bound is well below 2×.** This places C4 pre-emptively in Outcome B trap zone.

*(Q3) Tree-build has CPU branch points that can't be subsumed.* Best-split selection produces a CPU-side decision feeding back into next depth's partition assignment. A persistent kernel can't subsume the branch without porting CatBoost CPU greedy-search into GPU code. Best case: C4 amortizes within-depth dispatches only.

### Candidate C5 — Leaf-wise tree growth

**Verdict: FLAGGED-FOR-T2 (conditional on user admit at T0c).**

C5 is not a kernel-level lever at all. It changes the algorithmic structure that wraps the kernel. Where every entry in the 7-falsification chain attacks histogram cost-per-call, C5 attacks the call-count. None of the seven retired DECs touched the call-count axis. Only candidate addressing the *asymptotic* shape of the cost function rather than the constant factor.

**Three blockers (cannot be resolved by stress-test alone).**

*(Q1) Oblivious-tree contract is product-defining.* CatBoost's defining feature vs LightGBM/XGBoost is oblivious trees. This isn't an implementation detail — it's the property that makes DEC-030's depthwise leaf-index encoding work, that makes the model-format compatible with upstream catboost-cuda/catboost-cpu, and that gates DEC-008. C5 breaks all of these.

*(Q2) The benchmarking question is unfairly framed.* DEC-051's threshold ("≤5× MLX/CUDA on Higgs-1M iter=1000") was specified for oblivious trees on both sides. If C5 ships, apples-to-apples comparison reframes to leaf-wise-MLX vs leaf-wise-LightGBM. Product question becomes "ship as a LightGBM clone with CatBoost branding?"

*(Q3) Parity envelope (DEC-008) does not apply.* DEC-008 is RMSE/Logloss ulp ≤ 4 against an oblivious-tree reference. Leaf-wise produces structurally different trees; loss curves diverge at iter-1 from algorithmic difference, not numerical drift. G3 "Logloss equivalence within ±0.0003" gate cannot hold.

### Candidate L6 — Hybrid CPU+GPU concurrent histogram

**Verdict: FLAGGED-FOR-T2.**

L6 exploits Apple Silicon's UMA, the platform's defining hardware feature, which none of L1–L5 used. The 7-falsification chain assumes a discrete-GPU model. L6 is the first candidate splitting workload across silicon classes.

**Four blockers must clear at T2.**

*(Q1) Amdahl ceiling at ~2× iter.* Best case (perfect load balance, t_cpu ≈ t_gpu): wall-clock = 0.5 × t_gpu_alone → **2× iter speedup MAX**. Achieving the ≤5× MLX/CUDA gap-close (4.6× iter improvement from 23.2×) needs CPU faster than GPU on its share — implausible given M3 Max GPU ~5 TFLOPS and M3 CPU+AMX ~2-3 TFLOPS effective. Realistic load balance more like 1.5×. Outcome B trap zone like C4 but with higher upper bound.

*(Q2 — MANDATORY-CODE-INSPECTION) MLX UMA buffer-sharing API surface.* Apple Silicon has UMA at hardware level, but the API path for shared CPU/GPU access in MLX 0.31.x is non-obvious. Silicon-architect must cite specific MLX API path supporting concurrent CPU read/write and GPU read/write to the same `mx::array` without explicit copy. If the API requires `mx::eval()` between CPU and GPU accesses, "concurrent" claim collapses to "do CPU and GPU sequentially, hide CPU cost behind GPU eval" — that's CPU prefetch, not concurrency.

*(Q3 — MANDATORY-CODE-INSPECTION) Production CPU histogram entry-point not currently wired.* CatBoost's CPU histogram code exists at `catboost/private/libs/algo/score_calcers.cpp` and supporting paths, but the catboost-mlx build path doesn't dispatch into it from the histogram pipeline. Silicon-architect must cite the dispatch-fork point in `histogram.cpp` that supports a CPU/GPU split without invasive surgery.

*(Q4) Parity merge order.* CPU and GPU histogram have different reduction orders. Merge step adds two cross-class accumulations with different Higham bounds. DEC-008 envelope (RMSE ulp ≤ 4) is tight at γ_8; cross-class merge might push to γ_9 or γ_10. Probable PASS for RMSE/Logloss (≈4.77e-7 ceiling, γ_9 ≈ 5.4e-7 — barely), more slack for MultiClass.

---

## §2 — Hidden-rebrand check

For each FLAGGED candidate, the strongest argument for retirement that I do NOT fully accept but acknowledge.

**C4** — *"C4 is a DEC-018 echo."* DEC-018 amortized within a single dispatch (TG count reduction); C4 amortizes across dispatches (sync topology). Mechanism class is arithmetically distinct in DEC-018's measurement methodology. **Why I acknowledge:** if T2 (Q2) shows the upper bound is ~15%, that's small enough that the rebrand argument becomes effectively true at the outcome level — same gain ceiling as DEC-018, just measured differently. If silicon-architect re-derives upper bound at T2 and it's <10% iter, retire.

**C5** — *"C5 is the LightGBM milestone in disguise."* User's pivot-target shortlist lists ordered boosting, DEC-046, MSLR-WEB10K. Adding C5 = "implement LightGBM in MLX" — *another* pivot-target candidate, not a v0.8.0 throughput lever. **Why I acknowledge:** if user picks C5, v0.8.0 has effectively re-scoped from "throughput" to "tree-shape diversification" — a product decision masquerading as a throughput decision.

**L6** — *"L6 is just CPU-fallback with extra steps."* DEC-007 (small-N CPU fallback below 5k rows) already routes some workloads to CPU. L6 generalizes that to "whenever feature count exceeds K, route excess features to CPU." **Why I acknowledge:** if MLX UMA API (Q2) doesn't support real concurrency and degrades to sequential, L6 collapses to DEC-007 with a wider threshold — and DEC-007's ROI was characterized as "small-N only." UMA concurrency is the entire mechanism; if it falls, L6 falls.

---

## §3 — LESSONS-LEARNED rule check

| Rule | C4 | C5 | L6 |
|---|---|---|---|
| **Routing-honest** (S46-T6) | PASS — inherits production routing inside persistent kernel | N/A — C5 changes call count, not kernel internals | PASS — GPU side inherits production routing |
| **Toy-to-production transfer** (DEC-017) | **HIGH RISK** — Triton/FlashAttention-2 = toy reference; Apple Silicon Metal = production target | LightGBM is the production reference — speedup target moves from CUDA-CatBoost to LightGBM | UMA + AMX **never measured at GBDT histogram workload** |
| **MANDATORY-CODE-INSPECTION** (S45) | YES — Metal API path for kernel persistence; file:line citation required | YES — DEC-030 depthwise leaf-index encoding assumes oblivious | YES — (a) MLX UMA API; (b) histogram.cpp dispatch-fork point; (c) CPU histogram entry-point |

**Cross-cutting lesson observation.** All three FLAGGED candidates require MANDATORY-CODE-INSPECTION at T2. None can pass T2 by analytical reasoning alone. Silicon-architect's Day 2 work must be a *code read*, not just a roofline derivation. If silicon-architect produces a T2 feasibility report without file:line citations for load-bearing API claims, send it back.

---

## §4 — Final shortlist

```
SHORTLIST FOR T2 (in priority order):

1. L6 — Hybrid CPU+GPU concurrent histogram (UMA exploit)
   - Genuinely outside 7-falsification chain (Apple-Silicon-specific axis)
   - Upper bound ~2× iter (Outcome B trap zone, but mechanism is real)
   - MANDATORY-CODE-INSPECTION at T2: MLX UMA API + histogram.cpp dispatch-fork

2. C4 — Persistent-kernel pipelining (sync topology change)
   - Genuinely outside 7-falsification chain (inter-dispatch axis)
   - Upper bound ~10–15% iter (Outcome B trap zone, smaller upper bound than L6)
   - MANDATORY-CODE-INSPECTION at T2: Metal kernel-persistence API path

3. C5 — Leaf-wise tree growth (CONDITIONAL on user admit at T0c)
   - Genuinely outside 7-falsification chain (call-count axis)
   - Highest ceiling but breaks oblivious-tree contract + DEC-008 + DEC-030
   - Threshold rubric must be amended (LightGBM target, not CUDA-CatBoost)
   - If user says "no" at T0c → REMOVE FROM SHORTLIST (becomes 2 candidates)

RETIRED AT PREMISE GATE:

- L4 — DEC-014 echo (Stage A VGPR pressure 4× DEC-014 failure regime, not 8× clear)
       + DEC-017 echo (Stage B at 16 docs/thread is in failure regime, not 195)
       + DEC-049 echo (Stage A still uses src-broadcast; can't escape routing cost)

- C1 — Missing primitive (segmented_reduce doesn't exist in MLX 0.31.x — verbatim L3 problem)
       + L2 territory (per-iter per-depth index rebuild = 6.4 GB I/O at Epsilon)
       + DEC-019 echo (gather as primary win, but AGX prefetcher hides it already)
```

**Shortlist size:** 2 unconditional (L6, C4) + 1 conditional on user admit (C5).

**Auto-pivot triggered? NO** — unconditional shortlist has 2 candidates, which is ≥2.

**However:** both unconditional candidates are pre-emptively in Outcome B's trap zone (upper bound ≤2× iter). Pre-commit rail says Outcome B defaults to retire. Expected outcome at T5 is still C (retire + pivot), even though we proceed to T2. This is not a contradiction — proceeding through T2/T3/T4 generates measurement evidence that informs the pivot-target choice and provides quantitative grounding for DEC-052 = RETIRED-EMPIRICALLY rather than RETIRED-AT-PREMISE.

---

## §5 — Honest meta-verdict on opening this arc

**Prior verdict:** YELLOW leaning RED.
**Current verdict (post formal T0b cross-reference):** YELLOW.

**What changed:**
1. v2 candidate roster is genuinely cleaner than v1. Four pre-retired (L1, L2, L3, L5) all RIGHTLY died at silicon math without engineering cost.
2. C4 and L6 are genuinely outside the 7-falsification chain. After formal cross-reference, L4 and C1 ARE rebrands (retiring at T0b). C4 (sync topology) and L6 (UMA concurrency) attack axes the seven retired DECs never touched.

**What did NOT change:** Strategic concern is unchanged. P(C) ≈ 0.55 expected outcome. Both unconditional shortlist candidates have upper bounds in trap zone.

**Net:** arc opens, but the team should plan emotionally and technically for Outcome C at T5, as v2 already says. Of 7 candidates total (v1 5 + v2 added 2 net), 5 retire before any code is written or any measurement runs. **That's the rail working.**

---

## §6 — Open questions for @strategist + user-call (T0c)

### Pre-known (from scoping §v2)

1. **Endorse ≤5× MLX/CUDA on Higgs-1M iter=1000 threshold?** ENDORSE if shortlist remains oblivious-only (L6 + C4). If user admits C5, threshold MUST be amended (Q5).

2. **Approve shortlist (this §4)?** 2 unconditional + 1 conditional. Auto-pivot does NOT fire.

3. **Confirm pivot target = ordered boosting (or pick alternative)?** Recommend ordered boosting. LightGBM evidence shows 5–10× exists on this axis. DEC-046 is also legitimate alternative if user prefers product-impact over throughput.

4. **C5 leaf-wise admit?** Recommend REJECT. Oblivious-tree is product-defining, breaks DEC-008 / DEC-030, throughput comparison reframes (LightGBM ≠ CUDA-CatBoost).

### New (surfaced by formal T0b)

5. **(C5-conditional) If C5 admitted, does DEC-051 threshold rubric amend to "≤2× LightGBM-CPU on Higgs-1M iter=1000" instead of "≤5× CUDA-CatBoost"?** Coupled decision with Q4. Cannot admit C5 without re-locking threshold.

6. **(L6-conditional) Is the Outcome B trap zone (≤2× iter upper bound) acceptable for L6 if MANDATORY-CODE-INSPECTION passes at T2?** Per pre-commit rail, Outcome B defaults to retire unless explicit user invest. Pre-decide: if L6 measures 1.5–2× at T4 with parity intact, does v0.8.0 ship that win OR retire and pivot?

7. **(C4-conditional) If Metal API does NOT support kernel persistence (Q1 in C4), does C4 retire at T2 or is there a fallback "larger command buffer" mechanism worth measuring?** Pre-decide: API negative result = retire C4 at T2, do NOT measure fallback.

---

**STATUS: READY FOR T0c USER-CALL — shortlist of 2-3 candidates proceeds to T1/T2.**
