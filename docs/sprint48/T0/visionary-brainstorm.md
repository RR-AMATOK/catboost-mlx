# S48-T0a — Visionary Brainstorm (formal deliverable)

**Sprint:** 48 (v0.8.0 throughput research arc kickoff)
**Date:** 2026-05-08
**Branch:** `mlx/sprint-48-t0-brainstorm` (cut from master `0131b598b6` post PR #49)
**Owner:** @visionary
**Authority:** DEC-049 OUTCOME (RETIRED) / DEC-050 / DEC-051 / DEC-052 OPEN
**Companion docs:** `docs/sprint48/scoping.md` (v2 panel synthesis), `docs/sprint46/T5/decision.md`

This document extends `docs/sprint48/scoping.md` §1. It does NOT recapitulate the scoping doc — only stresses what the panel review committed to roster, then commits or rules out the second-wave backlog.

---

## §0 — Frame check before generating ideas

Re-reading the structural-impossibility argument (`docs/sprint46/T5/decision.md` §"Why RETIRE, Not Defer"):

> "Routing every (doc, stat) pair to the lane that owns the target bin REQUIRES either (a) `simd_shuffle(packed, src)` broadcasting … or (b) Pre-sorting docs by bin … There is no third route."

The visionary's job is to challenge the framing. Two observations:

1. The "no third route" claim is true **only conditional on the kernel topology being `kHistOneByteSource` (one TG per (feature-group, block, partition, stat)) and the routing target being a per-bin SIMD-lane within that TG**. Change the topology and the impossibility evaporates — the routing target itself moves. C1, C4, C5, L4, L6 below all do exactly that, in different ways.
2. The 7-falsification chain (DEC-013/014/015/017/019/048/049) is entirely **inside the histogram kernel**. None of them touched the **outer dispatch graph**, the **CPU/GPU work split**, the **tree-growth schedule**, or the **representation of the problem itself** (sparse vs dense, dense vs delta-from-parent). The design space outside the kernel is genuinely unexplored.

So: the burden on every candidate below is to identify which axis it changes and why that axis is not yet falsified. "Inside the kernel, but redesigned" is rejected a priori. "Outside the kernel" is the productive frontier.

---

## §1 — Stress test of the existing 5 candidates

### L4 — Hybrid two-stage (coarse-bin then refine)

**Mechanism (refined):** Stage A computes a coarse-bin histogram (8 bins per feature, 3 bits) using the production src-broadcast path BUT at register residency only. Stage B refines per coarse-bin slice via second kernel.

**Refinement to v1 claim:** v1 "8× lower VGPR than DEC-014" — DEC-014 spilled at BATCH_DOCS=64 × 4 features = 256 register-resident state per lane (per-doc state). L4 Stage A is 8 bins × 4 features = 32 register state per lane (per-bin state). 8× lower in absolute count but the scaling axis differs. T2 needs Metal compiler dump confirming ≤64 VGPR/lane.

**Single most likely killing risk:** Stage A→Stage B handoff. Two dispatches + intermediate buffer + sync = same regime that killed Probe D2.

**Cross-domain prior art:** GPU CUB DeviceHistogram (per-block private histograms reduced to global); radix sort coarse-then-fine; hierarchical clustering / k-d trees.

### C1 — Inverted-index histogram (posting-list / GIN analogy)

**Mechanism (re-affirmed):** Pre-compute an inverted index: for each (feature, bin) pair, store the list of doc indices. Histogram becomes `for each (f, b): hist[f][b] = sum(stats[d] for d in postingList[f][b])` — `gather + segmented_reduce`. **No scatter, no atomic, no SIMD-lane ownership question.**

**Single most likely killing risk:** **Build cost amortization.** Posting list must be rebuilt every time partition assignment changes — every level of every tree. At depth 6 × 1000 trees = 6000 rebuilds per training run. If rebuild cost > histogram savings, C1 collapses.

**Cross-domain prior art:** Postgres GIN; Lucene posting lists; GraphBLAS sparse semiring multiplication; DuckDB/ClickHouse columnar GROUP BY for low-cardinality.

### C4 — Persistent-kernel pipelining (Triton / FlashAttention-2 analogy)

**Mechanism (refined):** Replace "dispatch a kernel per histogram pass, return to host, decide split, dispatch next" with **single long-running Metal kernel that owns the entire tree-build cycle**. Kernel reads control flow from a small device-side state buffer that CPU updates between depths via low-latency UMA writes. All stages fused into one persistent Metal command buffer.

**Why structurally different:** 7-falsification chain measured `histogram_ms`. C4 doesn't try to make any single dispatch faster — eliminates **dispatch overhead, command-buffer queuing, MLX lazy-graph rebuild, Metal-driver round-trip**.

**Single most likely killing risk:** **MLX framework opacity.** MLX 0.31.x doesn't expose persistent-kernel primitives. We'd have to either bypass `mx::fast::metal_kernel` entirely OR hack a long-loop polling kernel. T2 silicon-architect will likely reject both.

**Cross-domain prior art:** Triton persistent matmul; FlashAttention-2 kernel fusion; CUDA persistent kernels (Merrill & Garland 2012).

### C5 — Leaf-wise tree growth (LightGBM analogy)

**Mechanism (re-affirmed):** Switch from depth-wise (oblivious) to leaf-wise (best-first) tree construction. **Not a faster histogram kernel — fewer histograms.**

**Single most likely killing risk:** **Product contract.** CatBoost's defining product feature is the oblivious tree (gives prediction-time efficiency, model interpretability, DEC-046 categorical-encoding semantics). Breaking this is a v1.0 decision, not a v0.8.0 throughput optimization.

**Cross-domain prior art:** LightGBM (Ke et al. 2017); LightGBM histogram subtraction trick (independent of tree-growth choice — applicable to oblivious too — see C6 below).

### L6 — Hybrid CPU+GPU concurrent (UMA exploit)

**Mechanism (re-affirmed):** Split feature-set: GPU handles features `0..K`, CPU+AMX handles `K..F` concurrently, merged via UMA. AMX provides 1+ TFLOPS additional FP32 on CPU side currently wasted.

**Single most likely killing risk:** **CPU histogram kernel quality.** CatBoost's CPU histogram path is mature but not designed to be a "concurrent partial worker." Re-using it requires either calling existing CPU-path APIs (which expect to own pipeline) or new CPU histogram kernel that interoperates with MLX state buffer. The latter is multi-week engineering not in S48-S52 budget.

**Cross-domain prior art:** Heterogeneous compute literature (HSA, OpenCL 2.0); PyTorch MPS fallback path; MLX itself (CPU and GPU primitives exist but not used for concurrent split today).

---

## §2 — New candidates

### C6 — LightGBM histogram subtraction trick (parent-minus-sibling)

**Mechanism:** When parent partition `P` splits into children `L` and `R`, instead of building `hist[L]` and `hist[R]` independently, build only the smaller child (say `L`, with fewer docs), then compute `hist[R] = hist[P] - hist[L]` via cheap dense subtraction. Per-level histogram work drops to **the smaller-child fraction** of full work — average ~50%, sometimes much less for high-information splits.

**Cross-domain analogy:** Inclusion-exclusion principle. Same trick used in **integral images / summed-area tables** (Crow 1984, Viola-Jones 2001). Also: **LightGBM** documents this as one of its three primary throughput levers (Ke et al. 2017 §3.2).

**Why it's structurally NEW:**
- **Not in the 7-falsification chain.** None of DEC-013/014/015/017/019/048/049 changed the per-iteration histogram WORKLOAD; they only changed how that fixed workload was scheduled across the GPU.
- **Not L1–L5.** All five candidates accept the current workload (full per-partition histograms at every level) as fixed and try to make it run faster.
- **Workload reduction, not throughput optimization.** This is the cleanest case for "we were solving the wrong problem."

**Apple Silicon feasibility hot-take:** Survives. Subtraction kernel is trivially parallel `histR[i] = histP[i] - histL[i]` over `F·B·numStats` elements — bandwidth-bound, cache-friendly, no routing issues, no atomics. Only new question is **selecting which child to compute directly** — needs doc count per child after partition split.

**Single most likely killing risk:** **Savings depend on sibling chosen being substantially smaller.** Worst case: 50/50 splits everywhere → 50% reduction in histogram work. At f_hist=0.95 and 50% histogram reduction, iter speedup = `1 / (0.05 + 0.475) = 1.90×`. At highly unbalanced splits (common for one-hot or near-constant features), speedup approaches `1 / (0.05 + ε) ≈ 20×`. The **mean** across realistic datasets is the kill axis.

**Pre-flight gate:**
- T1 instrumentation: log per-split child-size ratio across Higgs-1M and Epsilon for 100 trees of v0.7.0 baseline. If geomean `min(|L|,|R|)/|P|` < 0.4, ship to T2 with high confidence. If ≈ 0.5 (random splits), mark as marginal.

**Feasibility:** 🟢 Buildable. Subtraction kernel is one of the simplest GPU primitives possible. Integration challenge is in dispatch graph (compute parent BEFORE children, choose smaller child, schedule subtraction before downstream score-calc), not in any kernel.

### C7 — Cross-iteration histogram reuse (delta-from-previous-tree)

**Mechanism:** Maintain "previous-tree histogram cache" and update incrementally between trees. Per-iteration cost: instead of recomputing F·B·partitions histograms from scratch, update the cache by `O(N · numFeatures)` operations.

**Cross-domain analogy:** Streaming systems / materialized view maintenance (Flink, Materialize); Kalman filter / online learning; Differential dataflow (McSherry).

**Why structurally NEW:** No prior CatBoost lever has touched cross-iteration state. Every iter recomputes histograms from scratch.

**Apple Silicon feasibility hot-take:** Memory-cost-dominated. At Epsilon: cache size ≈ 130 MB persistent state per cache. Update kernel BW potentially 2× current full recompute.

**Single most likely killing risk:** **The "delta" is potentially as expensive as the recompute.** Every doc's stat changes every iteration (gradient changes). Unless there's exploitable structure (e.g., leaf-constant changes), C7 collapses.

**Pre-flight gate:** T1 instrumentation: measure correlation between `hist[T]` and `hist[T+1]` at fixed (partition, feature, bin). If correlation > 0.9, incremental delta is small enough. If < 0.5, delta IS the histogram.

**Feasibility:** 🟡 Research needed. Lower priority than C6.

### C8 — AMX matmul reformulation (boolean-feature matmul)

**Mechanism:** Reframe histogram as boolean matrix multiplication. Define `A[d, (f,b)] = 1[bin(d, f) == b]` (sparse boolean matrix `[N, F·B]`) and stat vector `s[d]`. Then `hist[f, b] = sum_d A[d, (f,b)] · s[d]` — sparse-dense matrix-vector product. Apple's AMX on CPU side and GPU's `simdgroup_matrix` path are designed for exactly this.

**Cross-domain analogy:** GraphBLAS (histogram = SpMV in (+, ×) semiring); MonetDB/Vectorwise radix-bucket aggregation; sparse matrix-vector multiplication on GPUs (Bell & Garland 2009).

**Why structurally NEW:** Reformulates the problem out of the histogram-kernel domain entirely. Reuses most-optimized primitive on the platform.

**Apple Silicon feasibility hot-take:** Sparse matrix is enormous: `A` at Epsilon `[400k, 256k]` with 800M non-zeros (3.2 GB). Comparable to existing `compressedIndex` storage.

**Single most likely killing risk:** **MLX 0.31.x's sparse-matmul story is incomplete.** Same library-completeness gap that retired L3.

**Pre-flight gate:** Code inspection of `mlx/mlx/ops.h` for `sparse`, `coo`, `csr`, `scatter_reduce`. If only `scatter_add` exists, C8 retires.

**Feasibility:** 🟡 Research needed.

### Candidates I rule OUT after stressing them

- **C3 — Probabilistic histogram sketch (CountSketch / Count-Min):** Parity with DEC-008 envelope (RMSE ulp ≤ 4) is impossible — sketches give probabilistic accuracy, not bit-equivalent.
- **Mesh shaders / object shaders:** Apple's GPU mesh-shader support is graphics-pipeline only as of M3; no compute path.
- **Neural Engine (ANE) int8 histogram:** ANE is fixed-function for CoreML graphs. No general-purpose compute API.
- **GPU cooperative groups:** Metal has `simdgroup_*` primitives but not full grid-level cooperative API.
- **Ant-colony / swarm split selection:** Cute analogy, but split-selection cost is not the bottleneck.

---

## §3 — Ranking for @devils-advocate stress-test

| Rank | Candidate | Expected ceiling at Higgs-1M | P(survive 7-falsification) | Probe-spec shape | Notes |
|---|---|---|---|---|---|
| **1** | **C6 — Histogram subtraction** | **2× iter (mean) / up to 4×** | **HIGH (~0.7)** | **CLEAN — instrumentation + simple subtract kernel** | Single cleanest "we were solving the wrong problem" candidate. Cross-domain prior art is decisive (LightGBM ships this in production). Lowest engineering cost. |
| 2 | L4 — Hybrid two-stage | 1.5–2.5× iter | 0.4 | Medium — two dispatches + handoff buffer | Only v1 candidate clearing all silicon math |
| 3 | C1 — Inverted-index histogram | 1.5–3× iter | 0.4 | Medium — needs index build + gather kernel | Highest novelty |
| 4 | L6 — Hybrid CPU+GPU concurrent | 1.5–2× iter (theoretical) | 0.3 | Heavy — needs CPU histogram interop | Apple-Silicon-specific lever |
| 5 | C4 — Persistent-kernel pipelining | 1.5–4× iter (highly speculative) | 0.2 | Heavy — bypasses MLX surface | Big upside if it works |
| 6 | C8 — AMX matmul reformulation | uncertain (depends on MLX) | 0.15 | Medium-heavy — depends on library | Conditional on MLX library surface |
| 7 | C7 — Cross-iteration delta | 1.5–3× iter (speculative) | 0.15 | Heavy — persistent state + correlation analysis | Mechanism plausible but unclear |
| **8** | **C5 — Leaf-wise tree growth** | 2–10× iter | high mechanism, 0.0 product-fit | N/A — product-changing | **DEFER to T0c user-call** |

---

## §4 — Honest meta

**My honest P(at least 1 of {L4, C1, C4, C5, L6, C6, C7, C8} survives the 7-falsification cross-reference and the silicon-realism gate combined): ~0.40.**

This is **slightly below** the panel's 0.45 (i.e., 1 - 0.55) survival prior. Three reasons:

1. **C6 (histogram subtraction) is structurally different from anything in the 7-failure chain.** It is a workload reduction, not a kernel optimization. **It is also the lowest-engineering-cost lever and has independent cross-domain proof of concept (LightGBM).** This raises my P significantly — but only into the 0.4 range.
2. **C1 has clean cross-domain prior art** but build-cost amortization gate is real (L2 BW collapse precedent).
3. **My new candidates have first-order gates.** C7 needs cross-iter correlation. C8 needs MLX library surface. C5 needs product-contract approval.

**My most likely scenario:** C6 survives to T4, delivers ~1.7× iter speedup measured (mean child-imbalance is favorable but not extreme), and lands in the **Outcome B "user-call zone."** I do not expect to see Outcome A (≥2× projection) unless C6 + C1 compose (a strong S49 question, not S48).

---

## §7 — What I'd challenge about the current plan

**The threshold `≤2×` for greenlight Outcome A may be too high for the C6 class of candidate.** If C6 delivers a measured 1.7× at Higgs-1M with low engineering cost (1-2 sprints), that is a legitimate v0.8.0 ship even though it does not project to ≥2×. Current rubric forces it into Outcome B "user-call default-retire" — which under sunk-cost pre-commit becomes effective Outcome C.

**A 1.7× lever at low-cost-with-clean-parity-and-cross-domain-prior-art may be the team's BEST realistic outcome.** The pre-commit rail should not retire it by accident.

**Recommendation:** Add a clause at T0c: "If a candidate has (a) low engineering cost ≤2 sprints, (b) cross-domain industrial validation, AND (c) measured ≥1.5× iter speedup on Higgs-1M, it is eligible for Outcome A regardless of whether it projects to ≥2×."

This is not goalpost-moving — the threshold for **CUDA-class throughput claim per DEC-051** stays at ≤5× MLX/CUDA. This is just clarifying that incremental wins compose toward that target.

---

**END OF VISIONARY BRAINSTORM. READY FOR @DEVILS-ADVOCATE.**
