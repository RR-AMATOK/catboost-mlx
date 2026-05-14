# Sprint 48 T2 — MANDATORY-CODE-INSPECTION Feasibility Verdicts

**Date:** 2026-05-13
**Branch:** `mlx/sprint-48-t0-brainstorm`
**Author:** @silicon-architect
**Authority:** DEC-052 T0c LOCK; LESSONS-LEARNED MANDATORY-CODE-INSPECTION rule
**No production-code commits.**

---

## §1 — C6 (Histogram subtraction, parent-minus-sibling)

### Q1 — Parent-cache memory feasibility on M3 Max — **PASS**

MLX's `MetalAllocator` allocates with `MTL::ResourceStorageModeShared` at `mlx/mlx/backend/metal/allocator.cpp:14-15` — every `mx::array` is UMA-shared. Epsilon depth-6 cache: 2000 features × 128 bins × 64 partitions × 2 stats × 4B = 130 MB per snapshot; worst-case all-ancestors = 780 MB. On M3 Max 36 GB, `recommendedMaxWorkingSetSize` ≈ 27 GB; well within budget.

### Q2 — Smaller-child selection: PASS

`structure_searcher.cpp:11-44` (`ComputePartitionLayout`) keeps partition counts on GPU (`scatter_add_axis` + `cumsum`, no CPU sync per comment at L41-43). MLX has `less`/`where`/`take` at `ops.h:461, 510, 1063` — smaller-child mask buildable with **zero new CPU-GPU sync points**.

### Q3 — DEC-017 amortization at smaller-child shapes — **MARGINAL (T1-gated)**

`histogram.cpp:133-137` enforces `maxBlocksPerPart=1` static_assert (Sibling S-1 race guard). DEC-017 cliff: +42.3% production regression at 3 docs/thread.

| Config | docs/thread | DEC-017 outcome |
|---|---|---|
| Production depth-6 Higgs (DEC-017 actual) | 3 | LOSE (+42.3%) |
| C6 smaller-child Epsilon avg-skew (0.45) | ~11 | between cliff — MARGINAL |
| C6 smaller-child Epsilon high-skew (0.05) | ~1.2 | WORSE than DEC-017 cliff |

T1 ≤ 0.35 → C6 wins (large child elided). T1 ≥ 0.45 → C6 loses (smaller-child re-enters DEC-017 cliff).

### §1 verdict: **MARGINAL** — depends on T1 child-imbalance.

---

## §2 — L6 (Hybrid CPU+GPU concurrent histogram) — **FAIL on all 3**

### Q1 — MLX UMA concurrent-fence API: FAIL
`allocator.cpp:14-15` declares `HazardTrackingModeUntracked` — MLX opts OUT of automatic CPU/GPU hazard sync. `array::item()` calls `eval()` (hard sync). MLX's `event.h` is private; only internal scheduling events exposed, no CPU↔GPU `MTLSharedEvent`. L6 requires raw Metal `MTLSharedEvent` (multi-week, outside timebox) OR `mx::eval()` boundaries (collapses to sequential CPU-prefetch).

### Q2 — CPU histogram fork-point: FAIL
`scoring.cpp:773-790` (`CalcStatsAndScores`) + inner `CalcStatsKernel` at `:252-312` is a monolithic CPU-fold abstraction with `TQuantizedObjectsDataProvider`, `TBucketStatsCache`, `TStatsIndexer`. No fork-point; 2–4 sprint engineering to reconcile data layouts.

### Q3 — AMX accessibility: FAIL
MLX uses Accelerate for GEMM/BLAS workloads (`cblas.cpp`, `bnns.cpp`). AMX is BLAS-shaped only. Histogram is scatter-add — no Accelerate/AMX API maps.

### §2 verdict: **L6 RETIRES AT T2.**

---

## §3 — C4 (Persistent-kernel pipelining) — **RETIRE AT T2** per DEC-052 T0c Q6

### Q1 — Metal kernel-persistence API: NO confirmed

`fast.h:60-78` CustomKernelFunction — no persistence parameter, no shared_memory (CUDA variant has it; Metal does not), no cooperative-groups handle. `device.cpp:334-348` dispatch is standard MTL one-shot. `memoryBarrier(MTL::BarrierScopeBuffers)` is intra-encoder only.

`<cooperative_groups.h>` included in CUDA backend only (`mlx/mlx/backend/cuda/...`). **Zero matches in `mlx/mlx/backend/metal/`.**

Metal 3.x exposes no API for cross-threadgroup grid-wide synchronization. Persistent kernels are unimplementable in MLX-Metal without raw Metal + custom atomics — multi-week production-stability work.

### §3 verdict: **C4 RETIRED AT T2.** Fallback NOT authorized per DEC-052 T0c Q6.

---

## §4 — Updated shortlist for T3

```
SURVIVORS FOR T3:
- C6 (MARGINAL): depends entirely on T1 child-imbalance result
                 T1 geomean ≤ 0.35 → C6 advances to T3 probe-spec
                 T1 geomean ≥ 0.45 → C6 retires at T1 gate
                 0.35 < geomean < 0.45 → C6 advances with T4-load-bearing caveat

RETIRED AT T2:
- L6: FAIL Q1 (no concurrent-fence API), Q2 (no fork-point), Q3 (AMX is BLAS-only).
- C4: FAIL Q1 (Metal has no cross-TG grid-wide sync; MLX one-shot dispatch).
```

**Sunk-cost rail status:** Conditional on T1.
- T1 ≥ 0.45 → C6 retires → 0 survivors → **auto-pivot to ordered boosting per Q3**
- T1 ≤ 0.35 → C6 advances → 1 survivor with rubric clause applying (Outcome A pre-certified)
- 0.35 < T1 < 0.45 → C6 advances with T4-measurement load-bearing

---

## §5 — Cross-cutting findings

**F1.** MLX UMA is real and consistent (`MTL::ResourceStorageModeShared` universal). Eliminates copy overhead but does NOT solve CPU/GPU sync.

**F2.** MLX `metal_kernel` is one-shot dispatch only. **Invalidates entire classes of CUDA-port-style optimizations going forward.**

**F3.** `maxBlocksPerPart=1` lock (`histogram.cpp:133-137`) is **load-bearing for C6**: at small smaller-child sizes, dispatch shape is forced to 1 TG/partition with potentially <2 docs/thread.

**F4.** Smaller-child selection has all needed MLX ops. **No MLX upstream changes required for C6.**

**F5 (load-bearing positive).** `scoring.cpp:315-332` `FixUpStats` already implements parent-minus-sibling subtraction on CPU CatBoost, gated by `fold.SmallestSplitSideValue` from `calc_score_cache.cpp:1152-1176`. **Not just LightGBM — CatBoost's own CPU code uses this trick.** C6 risk is dispatch-shape, not algorithmic novelty.

**F6.** DEC-017 cliff is the dominant C6 risk surface. Smaller-child dispatch re-enters this regime at Epsilon high-skew (≈1.2 docs/thread). Mitigation requires T4 production measurement.

---

**OUTCOME PENDING ON T1.** If T1 geomean ≥ 0.45 → AUTO-PIVOT to ordered boosting. If T1 geomean < 0.45 → C6 advances to T3.
