# S48-T3 — Probe Spec for Candidate C6 (Histogram Subtraction / Parent-minus-Sibling)

**Branch:** `mlx/sprint-48-t0-brainstorm`
**Status:** OPEN — T3 deliverable (specification only; engineering is S49)
**Authority:** DEC-049 OUTCOME / DEC-050 / DEC-051 / DEC-052 T0c LOCK (2026-05-12)
**Surviving candidate:** C6 — sole survivor of T0 + T1 + T2
**Retired this sprint:** L1–L5, C1, C4, C5, C7, C8, L6
**Inputs:**
- T0 visionary brainstorm: `docs/sprint48/T0/visionary-brainstorm.md` §C6
- T0 devils-advocate stress-test: `docs/sprint48/T0/dac-stress-test-delta.md`
- T1 child-imbalance empirical: `docs/sprint48/T1/child-imbalance/analysis.md`
- T2 feasibility (silicon-architect): `docs/sprint48/T2/feasibility.md` §1, §5
**Template:** `docs/sprint46/T3/probe-d-spec.md`

**Mode reminder:** SPEC ONLY. Per DEC-052, no production-code commits in S48. S49 is the engineering sprint that implements what this document defines.

---

## §0  Frame and authority

C6 cleared three independent gates this sprint:

1. **T0 ranking #1** (visionary): expected ceiling ~2× iter mean / up to 4× at Higgs-1M; survives 7-falsification cross-reference because it is a **workload reduction**, not a kernel optimization. The 7-failure chain (DEC-013/014/015/017/019/048/049) is entirely inside `kHistOneByteSource`; none of them touched the per-iteration histogram WORKLOAD. C6 changes the workload.
2. **T1 empirical KEEP** (instrumented baseline run): geomean `min(|L|,|R|) / (|L|+|R|)` = **0.3064 on Higgs-1M** and **0.2830 on Epsilon** across 100 trees × 3 seeds × depth 6. Both are `≤ 0.35` → C6 projects ≥ 2× under DEC-052 T0c Q7. Per-depth breakdown (T1 §"Per-depth breakdown") confirms smaller-child fraction monotone-increases from 0.04 at depth 0 to ~0.47 at depth 5, but never crosses 0.5 — the smaller side is always strictly smaller.
3. **T2 PASS on Q1 + Q2; MARGINAL on Q3** (silicon-architect): parent-cache memory feasible (130 MB/level peak at Epsilon, 780 MB worst-case all-ancestors; well within M3 Max `recommendedMaxWorkingSetSize` ≈ 27 GB — `mlx/mlx/backend/metal/allocator.cpp:14-15,83`). Smaller-child selection is sync-free (`mlx/mlx/ops.h:461,510,1063`). The DEC-017 cliff concern (Q3) is resolved by T1's per-depth data: at depth 5 the smaller child still holds ~`31250 × 0.47 ≈ 14700` docs across `256 threads = ~57 docs/thread` — well above the DEC-017 cliff (which fired at 3 docs/thread, `DECISIONS.md:208`).

**T2 F5 (load-bearing positive finding):** CatBoost's own CPU path already implements parent-minus-sibling subtraction. `catboost/private/libs/algo/scoring.cpp:315-332` (`FixUpStats`) does `stats[i].Remove(stats[i + halfOfStats])` and conditionally `DoSwap` based on `fold.SmallestSplitSideValue`. The selector lives in `catboost/private/libs/algo/calc_score_cache.cpp:1155-1176` (`TCalcScoreFold::SelectSmallestSplitSide` at `:533`). C6 is **not** a port of LightGBM into CatBoost-MLX — it is a port of CatBoost-CPU's own optimization into the MLX backend. Algorithmic novelty risk: zero. The only novelty is the dispatch shape.

**Rubric clause (DEC-052 T0c LOCK, lines 2890–2898):** C6 qualifies for Outcome A if S49 measures **≥1.7× iter speedup on Higgs-1M iter=1000** AND (a) engineering cost ≤2 sprints — PRE-CERTIFIED at T0c — AND (b) cross-domain industrial validation — PRE-CERTIFIED at T0c (LightGBM ships in production; CatBoost-CPU F5). Below 1.7× → Outcome B (user-call, default RETIRE under sunk-cost rail). Stop-loss floor: > 8× MLX/CUDA on Higgs-1M after S49 → retire arc.

---

## §1  Mechanism

### §1.1  Current production dispatch graph

At each tree depth in the oblivious search loop (`catboost/mlx/methods/structure_searcher.cpp:60-108`), the code:

1. Computes a fresh partition layout from the current `dataset.GetPartitions()` (`structure_searcher.cpp:69-74`). `ComputePartitionLayout` (`:11-44`) returns `{DocIndices, PartOffsets, PartSizes}` — three GPU-resident `mx::array`s, no CPU sync.
2. For each approx-dimension `k`, invokes `ComputeHistograms(...)` (`structure_searcher.cpp:86-107` calling `histogram.cpp:236-329`), which dispatches `histogram_one_byte_features` (`histogram.cpp:63-77`) — one Metal kernel covering **all `numPartitions = 2^depth` leaves at the current level**, all feature groups, all stats.
3. The output histogram array shape is `[numPartitions × numStats × totalBinFeatures]` of float32 (`histogram.cpp:125`), produced atomically (`histogram.cpp:76` `atomic_outputs=true`).
4. `FindBestSplitGPU` (`score_calcer.cpp:10-11,187-188`) consumes the histogram array and emits the single best `(featureId, binId, gain)` for the level (oblivious: all leaves share one split).

The kernel dispatch grid is `(256 × maxBlocksPerPart × numGroups, numPartitions, numStats)` divided by threadgroup `(256, 1, 1)` (`histogram.cpp:83-88`), with `maxBlocksPerPart = 1` locked by `static_assert` (`histogram.cpp:133-137`, Sibling S-1 race guard). Each TG handles one `(feature-group, partition, stat)` tuple.

**Key quantitative fact (T2 F3, T1 §"Per-depth breakdown"):**  At depth 6, half of the `numPartitions = 64` leaves carry the **larger** of each parent's split — collectively the larger children hold ~70% of all docs (Higgs `1 - 0.3064 ≈ 0.69`; Epsilon `1 - 0.2830 ≈ 0.72`). Computing histograms for those larger children directly from doc data is the workload that C6 eliminates.

### §1.2  C6 dispatch-graph rewrite

At depth `d ≥ 1`, the rewritten loop:

1. **Parent cache.** Retain `parentHistograms[k]` from depth `d-1` — the histogram array used to derive the best split that just produced the children at depth `d`. (At depth 0 the parent is the empty/root partition; depth-0 histograms are built from doc data as today. The cache becomes live at depth 1.)
2. **Smaller-child selection.** Each leaf `p` at depth `d` is a child of leaf `p_parent = p >> 1` at depth `d-1` (oblivious-tree convention: depth-`d` bit `d-1` distinguishes the two children of `p_parent`). Given `layout.PartSizes` at depth `d` (`structure_searcher.h:22`, `structure_searcher.cpp:38`), pair siblings and select the index with the smaller `PartSizes` value. This is a GPU-resident operation using `mx::reshape(PartSizes, {numParents, 2})`, `mx::less(left, right)` (`ops.h:461`), `mx::where(...)` (`ops.h:510`) and `mx::take` (`ops.h:1063,1067`) — **zero new CPU-GPU sync points** (T2 Q2 PASS).
3. **Smaller-child histogram build.** Issue ONE batched `histogram_one_byte_features` dispatch sized over the **smaller children only**. The current kernel dispatches over all `numPartitions = 2^d` leaves; the C6 dispatch covers `numPartitions / 2 = 2^(d-1)` smaller children. Dispatch shape changes from `(256 × numGroups, numPartitions, numStats)` to `(256 × numGroups, numPartitions/2, numStats)`. The kernel itself — `kHistOneByteSource` at `catboost/mlx/kernels/kernel_sources.h:107` — is **unchanged**. C6 changes how often and over what doc slice the kernel is called, not the kernel body.
4. **Larger-child derivation via subtraction.** For each parent `p_parent` and its two children `(p_small, p_large)`, compute `hist[p_large] = hist[p_parent] - hist[p_small]` as a dense elementwise subtraction over `[numStats × totalBinFeatures]` floats. The subtract is bandwidth-bound: at Epsilon depth 6, payload is `32 leaves × 2 stats × 64000 bin-features × 4 B ≈ 16 MB` per subtract, ~0.04 ms at ~400 GB/s; at Higgs-1M depth 6, `32 × 2 × 28 × 128 × 4 B ≈ 0.9 MB`, ~0.002 ms. **Far below 1 ms.** The brainstorm estimate of "~1 ms at Epsilon depth 6" was per-iter accumulated across all depths, not per-depth.
5. **Assembly.** Concatenate / scatter the smaller-child and derived-larger-child slices into a single `[numPartitions × numStats × totalBinFeatures]` array matching the current output shape. This array is then handed to the unchanged downstream `FindBestSplitGPU` (`score_calcer.cpp:187-188`) and `ComputeLeafSumsGPU` (`structure_searcher.cpp:120-125`).
6. **Cache update.** After the best split at depth `d` is determined and partitions are updated (`structure_searcher.cpp:163-195`), the new full histogram array becomes the parent cache for depth `d+1`. The old depth-`d-1` cache can be released.

The mechanism is the LightGBM trick (Ke et al. 2017 §3.2) and equivalently the integral-image / summed-area-table subtraction pattern (Crow 1984). It is also CatBoost-CPU's own `FixUpStats` strategy (T2 F5) — algorithmically identical, just dispatched on GPU.

### §1.3  Where the components attach

| Component | File:line | Modification |
|---|---|---|
| Outer depth loop | `structure_searcher.cpp:60-108` | Insert parent-cache state + smaller-child selection + subtract |
| Per-dim histogram build | `structure_searcher.cpp:86-107` | Dispatch over smaller-children-only at `d ≥ 1`; assemble larger-children via subtract |
| Histogram dispatch entry | `histogram.cpp:236-285` (and `:288-329`) | New overload `ComputeHistogramsSmallerChild(...)` taking `smallerChildPartitionIds` array (or new fields on `TPartitionLayout`); existing entry stays for depth 0 |
| Dispatch grid math | `histogram.cpp:83-88` | Grid `Y` dim becomes `numSmallerChildren = numPartitions / 2` for `d ≥ 1` |
| Subtract op | new — `histogram.cpp` or new `histogram_subtract.cpp` | `mx::subtract` (or `mx::ops.h:890`) over rank-1 slices |
| Smaller-child selection | new — in `structure_searcher.cpp` body | `mx::reshape(PartSizes, {-1, 2})` + `mx::less` + `mx::where` |
| Parent cache | new — local `mx::array parentHistograms[k]` inside `SearchTreeStructure` scope | One `mx::array` per approx-dim; released at end of depth `d+1` |
| Output assembly | new — in `structure_searcher.cpp` body | `mx::concatenate` or `mx::scatter` to rebuild full `[numPartitions × ...]` shape |
| Downstream score-calc | `score_calcer.cpp:10-11, 187-188` | **Unchanged.** Consumes the assembled full-shape histogram array. |
| Partition update | `structure_searcher.cpp:163-195` | **Unchanged.** Same `bitwise_or` of go-right bits. |
| `maxBlocksPerPart=1` lock | `histogram.cpp:133-137` | **Unchanged.** C6 still dispatches at `maxBlocksPerPart=1`; the lock is orthogonal to which partitions are dispatched. |

### §1.4  Memory budget for the parent cache

T2 Q1 (PASS): on M3 Max 36 GB, `recommendedMaxWorkingSetSize ≈ 27 GB` (`mlx/mlx/backend/metal/allocator.cpp:83`).

Per-level cache size = `numPartitions × numStats × totalBinFeatures × 4 B`.

| Shape | depth | numPartitions | totalBinFeatures (max) | numStats | bytes |
|---|---|---|---|---|---|
| Higgs-1M | 6 | 64 | 28 × 128 ≈ 3584 | 2 | ≈ 1.8 MB |
| Higgs-1M iter=1000 | 6 | 64 | 3584 | 2 | ≈ 1.8 MB |
| Epsilon | 6 | 64 | 2000 × 128 ≈ 256000 | 2 | ≈ 130 MB |
| Epsilon iter=2000 | 6 | 64 | 256000 | 2 | ≈ 130 MB |
| Amazon iter=1000 | 6 | 64 | ~10 features × 128 ≈ 1280 (sparse) | 2 | < 1 MB |

The cache only needs the immediate parent (depth `d-1` → depth `d`), not all ancestors — once depth `d`'s full histogram is assembled, the depth `d-2` cache is released. Single-level retention is sufficient because the subtraction at depth `d` only requires `hist[parent]` from depth `d-1`. T2's 780 MB worst-case was an upper bound that assumed retaining the full ancestor chain, which is not required.

**Worst-case peak across S49 datasets: ~130 MB (Epsilon).** Two-level retention (transient: hold both `d-1` and partially-built `d`) doubles this to ~260 MB — still 0.96% of `recommendedMaxWorkingSetSize`.

---

## §2  Routing-completeness argument (S46-T6 SIMD invariant)

The S46-T6 SIMD routing invariant requires that every `(doc, stat)` pair contribute exactly once to the bin owned by the lane responsible for that bin. This invariant is enforced **inside** `kHistOneByteSource` (`catboost/mlx/kernels/kernel_sources.h:107`) and is the structural reason for the 7-falsification chain.

**C6 does not modify the histogram kernel.** It dispatches the unchanged kernel less often — once per smaller-child partition, instead of once per child. Within each dispatch, the kernel's routing logic is byte-identical to the current production kernel — `simd_shuffle_xor`, the cross-SIMD linear fold, the MSB sentinel — all unchanged. **The routing invariant is inherited from production verbatim.** There is no new SIMD-lane routing concern.

The **subtract kernel** introduces no routing question. It is a dense rank-1 elementwise: `out[i] = parent[i] - smaller[i]` over a contiguous range of `numStats × totalBinFeatures` floats. Each output cell has a single writer (single MLX thread or simdgroup lane); there is no scatter, no atomic, no cross-lane reduction. Race-freedom is trivial.

The **smaller-child selection** uses MLX standard ops (`less`, `where`, `take`, `reshape`) over `[numParents, 2]` int32. These are validated MLX primitives with documented routing (they compile to standard Metal compute kernels with no histogram-style atomic scatter).

**Conclusion:** C6 inherits the production routing invariant inside the kernel; introduces no new routing question in the subtract or selection paths. SIMD routing-completeness holds.

---

## §3  MANDATORY-CODE-INSPECTION sign-off

Every mechanism claim above is grounded in code citations from this repo + MLX reference repo. The table below is the bound checklist for S49 review (any S49 PR must point each item to current line numbers at the commit it targets).

| Claim | File:line | Notes |
|---|---|---|
| Outer depth loop entry | `catboost/mlx/methods/structure_searcher.cpp:60` | `for (ui32 depth = 0; depth < maxDepth; ++depth)` |
| Current partition-layout call | `catboost/mlx/methods/structure_searcher.cpp:69-74` | `ComputePartitionLayout(...)` |
| GPU-resident partition layout | `catboost/mlx/methods/structure_searcher.cpp:20-43` | argsort + scatter_add + cumsum; comment at `:41-43` "No EvalNow — consumed lazily" |
| `TPartitionLayout.PartSizes` field | `catboost/mlx/methods/structure_searcher.h:22` | `mx::array PartSizes; // [numPartitions] uint32` |
| Per-dim histogram dispatch | `catboost/mlx/methods/structure_searcher.cpp:86-107` | The C6 modification scope; current code dispatches over all `numPartitions` |
| `ComputeHistograms` entry | `catboost/mlx/methods/histogram.cpp:236-329` | Two overloads — both go through `ComputeHistogramsImpl` (`:112-217`) |
| Kernel dispatch | `catboost/mlx/methods/histogram.cpp:63-77` | `mx::fast::metal_kernel(...)` registering `kHistOneByteSource` |
| Kernel source location | `catboost/mlx/kernels/kernel_sources.h:107` | `static const std::string kHistOneByteSource = R"metal(...` |
| Grid shape | `catboost/mlx/methods/histogram.cpp:83-88` | `(256 × maxBlocksPerPart × numGroups, numPartitions, numStats)` |
| `maxBlocksPerPart=1` Sibling-S1 guard | `catboost/mlx/methods/histogram.cpp:133-137` | `static_assert` — must remain at 1; C6 does not touch this |
| `atomic_outputs=true` (writeback path) | `catboost/mlx/methods/histogram.cpp:76` | unchanged in C6 |
| `FindBestSplitGPU` consumer | `catboost/mlx/methods/score_calcer.cpp:10-11, 187-188` | Takes `TVector<THistogramResult>` — C6 must hand it the assembled full-shape array |
| `ComputeLeafSumsGPU` consumer | `catboost/mlx/methods/structure_searcher.cpp:120-125` | Operates on `dataset.GetGradients()/GetHessians()` directly; **not** consumer of the histogram array. C6 does not affect this path. |
| `MetalAllocator::malloc` | `mlx/mlx/backend/metal/allocator.cpp:98` | All `mx::array` allocations go here; UMA-shared (`:14-15`) |
| `MTL::ResourceStorageModeShared` (UMA) | `mlx/mlx/backend/metal/allocator.cpp:14-15` | Universal across the MLX surface |
| `recommendedMaxWorkingSetSize` | `mlx/mlx/backend/metal/allocator.cpp:83` | `0.95 * device_->recommendedMaxWorkingSetSize()` is the memory limit |
| `mx::less` signature | `mlx/mlx/ops.h:461` | `array less(const array& a, const array& b, StreamOrDevice s = {})` |
| `mx::where` signature | `mlx/mlx/ops.h:510-514` | `array where(condition, x, y, StreamOrDevice s = {})` |
| `mx::take` signature | `mlx/mlx/ops.h:1063, 1067` | `array take(a, indices, axis)` and `array take(a, indices)` |
| `mx::subtract` signature | `mlx/mlx/ops.h:890` | `array subtract(a, b, StreamOrDevice s = {})` |
| `mx::reshape` signature | `mlx/mlx/ops.h:133` | `array reshape(a, Shape shape, StreamOrDevice s = {})` |
| CatBoost-CPU subtraction precedent | `catboost/private/libs/algo/scoring.cpp:315-332` | `FixUpStats`: `stats[i].Remove(stats[i + halfOfStats])` |
| CatBoost-CPU smaller-side selector | `catboost/private/libs/algo/calc_score_cache.cpp:533, 1155-1168` | `TCalcScoreFold::SelectSmallestSplitSide`; conditional `SmallestSplitSideValue = true` when `trueCount * 2 ≤ docCount` |
| Branch-B regression test | `python/tests/regression/test_branch_b_regression.py:54` | `np.array_equal(actual.astype(np.float32), reference)` — byte-identity gate |
| Branch-B baseline pickle | `python/tests/regression/v0.6.1_predict_baselines.pkl` | Generated at commit `d3bc0e1d02` (v0.6.1 release merge) by `generate_v061_baselines.py` |
| Branch-B test rationale (load+predict, not retrain) | `python/tests/regression/test_branch_b_regression.py:11-30` | Documents MLX Metal training non-determinism; predict is deterministic given fixed model weights |
| DEC-008 envelope (RMSE/Logloss ulp≤4, MultiClass ulp≤8) | `.claude/state/DECISIONS.md:71-81` | Authoritative parity gate |
| DEC-017 cliff precedent | `.claude/state/DECISIONS.md:208` | "+42.3% regression at gate config in production dispatch" at 3 docs/thread |
| DEC-052 T0c LOCK (rubric clause) | `.claude/state/DECISIONS.md:2890-2898` | ≥1.7× iter speedup + (a) + (b) → Outcome A |

Items I could not verify by code inspection (flagged here, not speculated):
- The exact MLX op that performs the smaller-child selection most efficiently (`take_along_axis` vs `where` + `take`) — both are present in `ops.h` but the optimal lowering depends on Metal kernel fusion. **Defer to S49 T2.**
- Whether `mx::concatenate` along axis 0 over `numPartitions/2` smaller + `numPartitions/2` derived-larger slices fuses into the existing graph without an `mx::eval()` boundary. The current code comments at `structure_searcher.cpp:41-43` and `histogram.cpp:206-209` document that the histogram path is held lazy through `FindBestSplitGPU`. C6 must preserve this property — **S49 must verify with `mlx::print_graph` or instrumentation.**

---

## §4  Pre-flight gates for S49

Before any S49 production-code commit:

### §4.1  Gate A — Smaller-child docs/thread floor

**Threshold:** smaller-child docs/thread ≥ 10 at every dispatch.

**Why:** DEC-017 (`DECISIONS.md:208`) shows +42.3% regression at 3 docs/thread. T1 §"Per-depth breakdown" gives geomean smaller-child fractions per depth; we need the runtime fraction, not the geomean, on every dispatch.

**How:** Instrument the smaller-child dispatch in S49 to log `smallerSize / numThreadsPerTG` at each depth, run on Higgs-1M, Epsilon, Amazon at iter=1000/2000/1000 with seeds 42/43/44. Compute per-dispatch docs/thread distribution.

**Action:**
- If `> 95%` of dispatches have docs/thread ≥ 10 → PASS. Proceed.
- If `5% ≤ failures < 20%` → consider runtime fallback: when smallerSize/threads < threshold for a particular partition, fall back to direct histogram build for that leaf (no subtract). Defer fallback decision to S49 T3.
- If `≥ 20%` failures → RETIRE C6. The DEC-017 cliff is structural for this dispatch.

### §4.2  Gate B — Parity envelope (DEC-008)

**Threshold:** RMSE/Logloss ulp ≤ 4; MultiClass ulp ≤ 8 (`DECISIONS.md:71-81`).

**Why:** C6 introduces a new reduction order. The current production γ_7 reduction (7-level cross-SIMD linear fold, `DEC-014` `:167`) accumulates each child's histogram from doc data independently. C6 derives the larger child as `hist[parent] - hist[smaller]` — adding **one** new subtraction (rank-1 elementwise, single op, no compound reduction) on top of γ_7. The composite Higham bound is `γ_7 + 1 ≈ γ_8 ≈ 4.77e-7` per cell. This is **at** the DEC-008 RMSE ceiling; not over.

**Why not γ_9 (as my brief claimed):** The brief's "γ_9" framing assumed parent and smaller-child each accumulate independently with γ_7 and then compose. In fact, in C6 the parent is **cached from the prior level**, not recomputed. Its γ history is fixed at the moment it was used to find the parent's split — it does not accumulate further. The new error at level `d` is the single subtraction step only: γ_{parent} + 1 ULP. The 1-ULP propagation across `d ≤ 6` levels gives γ_{7+6} = γ_13 ≈ 7.7e-7 in the worst case — still well under MultiClass ulp ≤ 8 (γ_14 ≈ 8.3e-7), and just slightly over RMSE γ_8. This is the parity-risk region.

**How:** Run the 18-config DEC-008 envelope sweep (same suite used in S17–S22 gates; reference `DECISIONS.md:111, 129`). At each config, compare C6 output BENCH_FINAL_LOSS against the v0.7.0 baseline.

**Action:**
- All 18 configs ulp within envelope → PASS.
- Any RMSE/Logloss config > 4 ulp but ≤ 8 ulp → MultiClass-only deployment OR rebaseline (see §7 risk register).
- Any config > 8 ulp → investigate before any further work. Likely a bug, not γ propagation.

### §4.3  Gate C — Branch-B regression

**Threshold:** `test_higgs_1m_predict_byte_equivalent_to_v061` and `test_epsilon_subset_predict_byte_equivalent_to_v061` PASS (byte-equivalent on float32 predict output, `python/tests/regression/test_branch_b_regression.py:54`).

**Why:** Branch-B is the v0.6.1 reproducibility-grade frame. The test loads a checked-in model and re-runs `predict`. Since C6 only changes the **training** dispatch graph, and the Branch-B test only exercises `predict`, **C6 should pass Branch-B trivially**: it does not change `predict()` behavior at all.

**Conditional risk:** If S49 introduces any incidental change to predict — e.g., reorganizing `_predict_inprocess`, touching `_predict_utils.py` — the gate may fire on those changes, not on C6 proper. S49 engineering MUST keep `python/catboost_mlx/_predict_utils.py` and predict-path code untouched.

**Action:**
- Branch-B PASS → continue.
- Branch-B FAIL → STOP. The failure is in predict-path code, not in C6's training-path scope. Revert any predict-path edits before reproposing.
- If the user later asks to re-baseline v0.7.0 → v0.8.0 for an unrelated reason (e.g., training-output drift desired), that decision is handled outside C6. See §7 risk register: this is the same problem that DEC-008 envelope solves at the loss level; Branch-B is the predict-bit-identity level and is independent.

**Note on the brief's worry:** The brief speculated that "C6 will change tree-build order through different per-depth dispatch shape, potentially producing different float-precision-bit predict output." This is wrong: predict consumes the **trained model**, not the training dispatch graph. Different training output ≠ different predict output for a given model. Branch-B's load-and-predict design (`test_branch_b_regression.py:11-30`) is exactly the right gate; C6 cannot break it unless predict-path code is also touched.

### §4.4  Gate D (NEW) — Lazy-graph fusion preservation

**Threshold:** No new `mx::eval()` or `EvalAtBoundary` calls inside the per-depth loop relative to v0.7.0.

**Why:** Current production keeps the histogram path lazy from `ComputePartitionLayout` through `FindBestSplitGPU` (comments at `structure_searcher.cpp:41-43`, `histogram.cpp:206-209`). Inserting smaller-child selection + assembly between them risks a forced CPU-side materialization if `mx::concatenate` or `mx::scatter` lazy-graph-incompatibility is hit.

**How:** Build C6 with `CATBOOST_MLX_STAGE_PROFILE` enabled (see `stage_profiler.h`); count sync points per iter and compare to v0.7.0.

**Action:**
- Same sync count or fewer → PASS.
- One or more new sync points → STOP. Use `mx::where` / `take_along_axis` to avoid `concatenate`-style materialization. This is structurally fixable; do not ship with new sync points.

---

## §5  Measurement protocol for S49 T4-equivalent

Per Bundle 2 protocol (DEC-052 T0c LOCK, `DECISIONS.md:2870-2898`):

### §5.1  Hardware and seeding

- **Hardware anchor:** Apple M3 Max (current dev box). Cross-chip extrapolation deferred to v0.8.0 release validation.
- **3-seed median:** seeds `{42, 43, 44}`.
- **CV across seeds:** required < 8% (Bundle 2 hard rule). If CV ≥ 8%, increase to 5 seeds and re-evaluate.
- **Warm runs only:** at least 1 untimed warmup iter; discard cold-iter timings (compile + caches).
- **Ratio:** `median(MLX) / median(CUDA)`, NOT mean of per-seed ratios.

### §5.2  Multi-dataset hard gate (Bundle 2)

≤ 5× MLX/CUDA on EVERY dataset:

| Dataset | iters | Notes |
|---|---|---|
| Higgs-1M | 1000 | Primary anchor; rubric trigger uses this dataset |
| Epsilon | 2000 | Worst case for histogram payload (2000 features × 128 bins) |
| Amazon | 1000 | Sparse/categorical-heavy; cross-domain check; T2 risk identified low total bin features |

**Stretch:** ≤ 3× on Higgs-1M iter=1000.
**Stop-loss floor:** > 8× on Higgs-1M iter=1000 after S49 → retire entire v0.8.0 throughput arc.

### §5.3  C6-specific measurement additions

Beyond the Bundle 2 wall-clock ratio, S49 T4 must measure:

| Metric | How | Why |
|---|---|---|
| **Smaller-child histogram time** (`hist_small_ms`) | `bench_boosting --per-kernel-profile` with C6 build, tagged dispatches | Confirm the "savings story" — must be < ~50% of v0.7.0 `hist_ms` |
| **Subtraction kernel time** (`hist_sub_ms`) | Same profile flag, new tag for the subtract op | Sanity check: ≤ 0.1 ms at Higgs-1M depth 6; ≤ 0.05 ms at Epsilon depth 6 |
| **Assembly time** (`hist_asm_ms`) | Same flag, new tag for concatenate/scatter | If > 5% of `hist_ms`, redesign the assembly path |
| **f_hist post-C6** | `(hist_small_ms + hist_sub_ms + hist_asm_ms) / iter_total_ms` | Tracks where the new bottleneck has moved |
| **Smaller-child docs/thread distribution** | Histogram of `smallerSize / 256` per dispatch | Gate A live-check |
| **Parent-cache peak resident** | `MetalAllocator` stats via MLX | Memory-pressure validation; T2 Q1 follow-up |

### §5.4  f_hist re-projection

Current production f_hist (Epsilon-proxy) ≈ 0.95 (S46-T3 §0.4 decision table threshold). If C6 reduces histogram total work by 50% (geomean child-imbalance 0.30 → smaller side carries 0.30 → directly-built fraction is 0.30 + small assembly cost), then post-C6 expected `f_hist ≈ 0.30 × 0.95 / (0.05 + 0.30 × 0.95) ≈ 0.85` — histogram still dominates but is no longer the runaway leader.

**Iter speedup projection:** `1 / (0.05 + 0.30 × 0.95) ≈ 1 / 0.335 ≈ 2.99×` at the ideal child-imbalance geomean. This is the upper bound. Realistic outcome (overhead from assembly, lazy-graph friction, gate-config CV) likely lands in **1.7× – 2.2×** at Higgs-1M iter=1000. That window crosses the DEC-052 rubric trigger at 1.7×.

### §5.5  Decision table (S49 T5-equivalent user call)

| Higgs-1M iter speedup | Multi-dataset gate (≤5× CUDA) | Decision |
|---|---|---|
| ≥ 3× | PASS all 3 datasets | Outcome A: COMMIT, default merge |
| ≥ 1.7× and < 3× | PASS all 3 datasets | Outcome A via rubric (qualifiers (a), (b) pre-certified): COMMIT |
| ≥ 1.5× and < 1.7× | PASS all 3 datasets | Outcome B: user call. Default retire under sunk-cost rail. |
| < 1.5× | irrelevant | Outcome B → RETIRE per sunk-cost rail |
| any speedup | > 8× MLX/CUDA on Higgs-1M | Outcome D: retire v0.8.0 throughput arc (stop-loss floor) |

---

## §6  Engineering scope estimate (qualifier (a) verification)

T0c pre-certified qualifier (a) as "≤2 sprints." Validation by concrete task breakdown:

### §6.1  Sprint 1 (S49 — engineering)

| Task | File scope | Estimate (days) |
|---|---|---|
| T0 — Outer loop refactor: introduce `parentHistograms[k]` state | `structure_searcher.cpp:60-108` | 0.5 |
| T1 — Smaller-child selection (MLX ops sequence) | `structure_searcher.cpp` body (new helper) | 0.5 |
| T2 — Histogram dispatch overload for smaller-children | `histogram.cpp:236-329` + `histogram.h` | 1 |
| T3 — Subtraction integration + output assembly | `structure_searcher.cpp` body (new helper) | 0.5 |
| T4 — `bench_boosting` instrumentation for §5.3 metrics | `catboost/mlx/tests/bench_boosting.cpp` | 0.5 |
| T5 — Gate A measurement (docs/thread distribution) | run-and-analyze | 0.5 |
| T6 — Gate D measurement (lazy-graph fusion) | run-and-analyze | 0.25 |
| T7 — Branch-B regression validation (Gate C) | `python/tests/regression/` | 0.25 |
| T8 — Single-seed Higgs-1M speedup check (1000 iters) | bench + analysis | 0.25 |
| Buffer / debug | — | 1 |
| **S49 total** | — | **~5.25 days = 1 sprint** |

### §6.2  Sprint 2 (S50 — validation + cutover)

| Task | File scope | Estimate (days) |
|---|---|---|
| T1 — DEC-008 18-config parity sweep (Gate B) | parity tooling | 1 |
| T2 — Bundle 2 multi-dataset benchmark (3 datasets × 3 seeds × full iters) | `benchmarks/upstream/` | 1.5 |
| T3 — Downstream score-calc + leaf-estimator validation (no regression) | full training-loop check | 0.5 |
| T4 — `bench_boosting` baseline refresh post-C6 | benchmark suite | 0.5 |
| T5 — Documentation: DECISIONS.md update, CHANGELOG-DEV.md, technical-writer pass | docs | 1 |
| Buffer | — | 0.5 |
| **S50 total** | — | **~5 days = 1 sprint** |

**Scope total: 2 sprints.** Qualifier (a) holds. If S49 trips Gate B (DEC-008 envelope) requiring re-baselining v0.7.0 → v0.8.0, S50 absorbs the baseline refresh; this is included in S50 T2 + T4.

**Flag if scope creeps:** If Gate D (lazy-graph fusion) fails and requires bypassing `mx::concatenate` with a custom Metal kernel, add ~3 days to S49 T3. Still within 2 sprints. If Gate A fails and requires per-partition fallback dispatch logic, add ~2 days. Still within 2 sprints. If both fail jointly, scope tips to 2.5 sprints — flag for orchestrator decision.

---

## §7  Risk register

### §7.1  Branch-B parity bit-equivalence (Risk 1)

The brief flagged: "C6 will change tree-build order through different per-depth dispatch shape, potentially producing different float-precision-bit predict output."

**Assessment:** This conflates two independent concerns.
- **Training output (loss / model weights):** C6 will produce different weights in low-order bits (γ propagation), within DEC-008 envelope. This is expected and acceptable.
- **Predict output (Branch-B):** Branch-B loads a saved model and runs predict only (`test_branch_b_regression.py:11-30`); training is not exercised. C6 does not touch the predict path. Branch-B should PASS without amendment.

**Mitigation:** S49 engineering must not touch `python/catboost_mlx/_predict_utils.py` or any predict-side code. The current branch already has a pending modification to `_predict_utils.py` (in `git status`); confirm it is unrelated to C6 before S49 starts.

**Fallback decision tree:**
- Branch-B PASS in S49 → ship as is.
- Branch-B FAIL in S49 due to incidental predict-path change → revert the predict-path change.
- Branch-B FAIL in S49 due to load-format change (would require a v0.8.0 baseline) → this scenario is implausible for C6 scope; if it happens, treat as separate problem and escalate to orchestrator.

### §7.2  Memory pressure at deep trees (Risk 2)

T2 Q1 PASS at depth 6 worst-case (Epsilon, 130 MB single-level). At greater depths (depth 8: `numPartitions = 256`, 4× the cache size) the cache grows to ~520 MB at Epsilon, still well within `recommendedMaxWorkingSetSize`.

**Hot-path allocation concern:** if S49 implements the parent cache as a per-iter `mx::array` allocation (no buffer reuse), the `MetalAllocator::malloc` cost (`mlx/mlx/backend/metal/allocator.cpp:98`) may show up in profile. Mitigation: hoist the cache allocation out of the per-iter scope; reuse the buffer across iterations. T2 F1 (UMA shared memory) and the existing MLX allocator caching make this straightforward.

### §7.3  Cross-dataset variance — Amazon (Risk 3)

T1 measured child-imbalance on Higgs-1M and Epsilon only. **Amazon was NOT measured in T1.** The Bundle 2 hard gate requires Amazon ≤ 5× CUDA. If Amazon's child-imbalance distribution differs materially from Higgs/Epsilon (e.g., categorical features produce more balanced splits, geomean → 0.45+), C6 may **fail to reach the rubric on Amazon even if Higgs passes**.

**Action item for S49 T0 (pre-engineering):** Re-run T1's instrumented binary on Amazon (Sprint 44 already has Amazon datasets in `benchmarks/upstream/results/`). If Amazon geomean child-imbalance > 0.45, flag to orchestrator before S49 engineering begins. C6 is then conditional on dataset shape.

**Note:** Amazon's small `totalBinFeatures` (~1280 vs Epsilon's ~256000) means the per-leaf histogram cost is already low; C6's savings are smaller in absolute ms even if the ratio is good. C6 may show smaller iter speedup on Amazon even with favorable child-imbalance. This is OK as long as the multi-dataset hard gate (≤ 5× CUDA) holds; the rubric trigger uses Higgs-1M only.

### §7.4  Top 3 risks (summary)

1. **Amazon dataset cross-domain validation (Risk 3).** Highest priority for S49 T0 pre-flight — measure before engineering.
2. **Gate D lazy-graph fusion (Risk 2).** Insertion of smaller-child selection + assembly in the lazy graph is novel; a forced sync could erase the speedup. Address in S49 T2/T3.
3. **DEC-008 envelope at Gate B (Risk 4 below).** γ_13 ≈ 7.7e-7 is close to the MultiClass ceiling. If S49 introduces any extra accumulation (e.g., a Kahan-compensated subtract for safety), γ could cross the envelope. Keep the subtract single-op.

### §7.5  Lower-priority risks

- **Risk 4: DEC-008 envelope (γ propagation).** Subtraction at depth 6 propagates γ_13 ≈ 7.7e-7. RMSE ulp ≤ 4 (γ_8) is the tight ceiling. Mitigation: parity sweep at Gate B. Fallback: MultiClass-only deployment (γ_14 ≈ 8.3e-7 ceiling).
- **Risk 5: Lossguide / depthwise paths.** `SearchTreeStructure` (oblivious) is the C6 scope. `SearchDepthwiseTreeStructure` (`structure_searcher.cpp:201-453`) and `SearchLossguideTreeStructure` (`:455-747`) also call `ComputeHistograms` per-leaf. C6 in oblivious-only mode: depthwise/lossguide users get no speedup but no regression. S50 may extend C6 to depthwise (siblings still well-defined by BFS index pairing); lossguide is harder (parent/child relationship is via `NodeSplitMap`/BFS, not bit-position). **Scope C6 to oblivious only for v0.8.0; defer depthwise/lossguide to S52+.**
- **Risk 6: Categorical features.** F5 (CatBoost CPU `FixUpStats`) works on continuous and categorical features alike. The histogram kernel `kHistOneByteSource` is feature-type-agnostic (treats all features as bin-indexed). No expected interaction with DEC-046 categorical encoding.

---

## §8  Open questions for S49 (NOT to be resolved in this spec)

These are unresolved by code inspection at T3 and must be decided in S49:

1. **Per-dispatch graceful degradation.** Should the subtract path run on **every** smaller child, or only when smaller-child docs/thread > threshold (e.g., 20)? A hybrid (subtract for the favorable partitions, direct build for the unfavorable ones) could increase robustness against unbalanced trees. Gate A's measurement will inform this; S49 T3 decides.
2. **Interaction with T2-sort kernel (DEC-020 / `kT2AccumSource`).** Production currently uses `kHistOneByteSource`; the T2-sort path (`kernel_sources.h:1008, 1156`) is not in production. C6 modifies dispatch frequency, not kernel choice. If S52+ revives T2 (per DEC-020 0.317× hist_ms record), C6 should compose cleanly with it (subtract is orthogonal to which kernel produces the smaller-child histogram). **Defer composability question; not on critical path.**
3. **Ordered boosting composability.** If S49 / S50 also land ordered boosting (the auto-pivot target if C6 had failed; per DEC-052 T0c Q3), does C6 stack with ordered boosting? Ordered boosting changes the training data flow (per-doc rolling-fold gradients) but does NOT change the histogram dispatch structure. They should stack additively, but verification is S52+ work. **Out of scope here.**
4. **Larger-children cache for the next level.** At depth `d`, we cache the **full** histogram (smaller + derived larger) to serve depth `d+1`'s parents. Alternative: cache only the smaller-child slices and recompute derived-larger on demand. This saves ~50% memory at the cost of a second subtract per depth. **Likely not worth it given memory is plentiful; defer to S49 T3 if profiling shows allocator pressure.**
5. **Depth-0 special case.** At depth 0, there is no parent. The current path builds histogram for the root partition (size = numDocs); C6 retains this. Is there an opportunity to use the iter-level "parent of root" (e.g., the previous tree's root histogram) as a starting point? **No — every iter re-quantizes gradients/hessians per the boosting update; the histograms are not iter-stable.** Confirmed by code inspection of `structure_searcher.cpp:86-107` (per-dim slice of `dataset.GetGradients()` each iter). The C7 candidate explored this and was retired at T0c (`DECISIONS.md:2916`).

---

## §A  Appendix — Key file:line index

| Symbol / Concept | File:line | Role in C6 |
|---|---|---|
| Oblivious depth loop | `catboost/mlx/methods/structure_searcher.cpp:60-108` | Primary C6 modification target |
| `ComputePartitionLayout` | `catboost/mlx/methods/structure_searcher.cpp:11-44` | Source of GPU-resident `PartSizes` for smaller-child selection |
| `TPartitionLayout` definition | `catboost/mlx/methods/structure_searcher.h:19-23` | `PartSizes` field at `:22` |
| `ComputeHistograms` entries | `catboost/mlx/methods/histogram.cpp:236-329` | C6 adds a smaller-child overload |
| `ComputeHistogramsImpl` core | `catboost/mlx/methods/histogram.cpp:112-217` | Shared dispatch path |
| `DispatchHistogramBatched` | `catboost/mlx/methods/histogram.cpp:32-109` | Direct kernel-dispatch wrapper |
| Kernel grid math | `catboost/mlx/methods/histogram.cpp:83-88` | Y-dim is `numPartitions` today; C6 changes to `numPartitions/2` |
| `maxBlocksPerPart=1` lock | `catboost/mlx/methods/histogram.cpp:133-137` | Unchanged by C6 |
| `atomic_outputs` | `catboost/mlx/methods/histogram.cpp:76` | Unchanged |
| `kHistOneByteSource` | `catboost/mlx/kernels/kernel_sources.h:107` | Unchanged kernel body |
| `FindBestSplitGPU` | `catboost/mlx/methods/score_calcer.cpp:10-11, 187-188` | Unchanged consumer |
| `ComputeLeafSumsGPU` call site | `catboost/mlx/methods/structure_searcher.cpp:120-125` | Independent of histogram array |
| Partition update | `catboost/mlx/methods/structure_searcher.cpp:163-195` | Unchanged |
| MLX `less` | `mlx/mlx/ops.h:461` | Smaller-child selection primitive |
| MLX `where` | `mlx/mlx/ops.h:510-514` | Smaller-child selection primitive |
| MLX `take` | `mlx/mlx/ops.h:1063, 1067` | Smaller-child selection primitive |
| MLX `subtract` | `mlx/mlx/ops.h:890` | Larger-child derivation |
| MLX `reshape` | `mlx/mlx/ops.h:133` | `PartSizes` reshape to `[-1, 2]` |
| `MetalAllocator` shared-mode | `mlx/mlx/backend/metal/allocator.cpp:14-15` | UMA cache backing |
| `recommendedMaxWorkingSetSize` | `mlx/mlx/backend/metal/allocator.cpp:83` | Memory ceiling |
| CatBoost CPU `FixUpStats` | `catboost/private/libs/algo/scoring.cpp:315-332` | F5 algorithmic precedent |
| CatBoost CPU `SelectSmallestSplitSide` | `catboost/private/libs/algo/calc_score_cache.cpp:533, 1155-1168` | F5 selector precedent |
| `FixUpStats` call site | `catboost/private/libs/algo/scoring.cpp:566` | Gated by `fold.SmallestSplitSideValue` |
| Branch-B test definition | `python/tests/regression/test_branch_b_regression.py` | Gate C |
| Branch-B baseline pickle | `python/tests/regression/v0.6.1_predict_baselines.pkl` | Generated at `d3bc0e1d02` |
| DEC-008 parity envelope | `.claude/state/DECISIONS.md:71-81` | Gate B authority |
| DEC-017 cliff data | `.claude/state/DECISIONS.md:208` | Gate A precedent |
| DEC-049 Bundle 1 retire | `.claude/state/DECISIONS.md` | Sprint-46 outcome that opens v0.8.0 arc |
| DEC-052 T0c LOCK | `.claude/state/DECISIONS.md:2828-2916` | Rubric, gates, sunk-cost rail |

---

**END OF C6 PROBE SPEC. READY FOR T5 USER-CALL — DEC-052 OUTCOME A / B / C / D decision.**
