# S49-T1 — C6 Implementation Design (Histogram Subtraction / Parent-minus-Sibling)

**Sprint:** 49
**Branch:** `mlx/sprint-49-c6-engineering`
**Authority:** DEC-052 OUTCOME A + S49-T0c AMENDMENTS (2026-05-14) — Q1 (Amazon carve-out), Q3 (Outcome-B auto-retire), Q4 (loss-conditional dispatch, NO runtime flag), Q5 (Modified-β shape, T0 SKIPPED).
**Inputs:** `docs/sprint48/T3/probe-spec-c6.md`, `docs/sprint48/T2/feasibility.md`, `docs/sprint49/sprint-plan.md`.
**Mode:** DESIGN ONLY — no production-code commits at T1. T2 engineering translates this document into code with zero ambiguity.

---

## §0  Scope

C6 rewrites the per-depth dispatch graph of `SearchTreeStructure` (oblivious-tree path only) such that at depth `d ≥ 1` the histogram kernel `kHistOneByteSource` is dispatched over **smaller children only** (`numPartitions/2` partitions instead of `numPartitions`), and the larger-child histograms are derived via dense rank-1 subtraction `hist[R] = hist[parent] - hist[L]`. The kernel body is UNCHANGED; the only changes are dispatch shape, a depth-resident parent cache, and a smaller-child selection / output-assembly graph fragment.

**Out of scope at S49:** depthwise (`SearchDepthwiseTreeStructure`), lossguide (`SearchLossguideTreeStructure`). Per probe-spec §7.5 Risk 5, those paths fall through to v0.7.0 production. Loss types outside Logloss + MultiClass also fall through to v0.7.0 production per Q4 lock.

---

## §1  MLX primitive sequence (file:line citations)

The C6 graph fragment inserted at `catboost/mlx/methods/structure_searcher.cpp:86-107` (the per-dim histogram build block) for `depth ≥ 1` is the following sequence. All primitives are MLX core ops in `mlx/mlx/ops.h` with the cited signatures; all operate on `mx::array` lazily and never force CPU sync.

### §1.1  Smaller-child mask (per-parent pair)

```cpp
// layout.PartSizes  : [numPartitions] uint32 at current depth d
// numParents = numPartitions / 2 = 1u << (depth - 1)
auto sizesPairs = mx::reshape(
    layout.PartSizes,
    {static_cast<int>(numParents), 2}
);                                                          // mlx/ops.h:133
// sizesPairs[p, 0] = size of left child of parent p  (partition index 2p)
// sizesPairs[p, 1] = size of right child of parent p (partition index 2p+1)
auto leftSizes  = mx::slice(sizesPairs, {0, 0}, {static_cast<int>(numParents), 1});
auto rightSizes = mx::slice(sizesPairs, {0, 1}, {static_cast<int>(numParents), 2});
auto smallerIsLeft = mx::less(leftSizes, rightSizes);       // mlx/ops.h:461
// smallerIsLeft : [numParents, 1] bool — true ⇒ left child is the smaller side
```

**Oblivious-tree partition-index convention (CRITICAL).** At depth `d` after the update at `structure_searcher.cpp:189` (`bits = leftShift(goRight, depth)` followed by `partitions |= bits` at `:191`), each leaf partition at depth `d-1` with index `p_parent` produces two children at depth `d`:
- Left child (go-left bit = 0): partition index `p_parent`
- Right child (go-right bit = 1): partition index `p_parent | (1u << (d-1))` = `p_parent + numParents`

The two siblings are therefore **NOT adjacent** in `layout.PartSizes`; they are `numParents` apart. The `mx::reshape(PartSizes, {numParents, 2})` formulation above is therefore WRONG — sibling pairs are `(p_parent, p_parent + numParents)`, not `(2*p_parent, 2*p_parent+1)`.

**Correct mask construction:**

```cpp
// View as two halves: first half = left children of all parents, second half = right children.
auto leftSizes  = mx::slice(layout.PartSizes,
    {0}, {static_cast<int>(numParents)});                   // [numParents] uint32 — left siblings
auto rightSizes = mx::slice(layout.PartSizes,
    {static_cast<int>(numParents)},
    {static_cast<int>(2 * numParents)});                    // [numParents] uint32 — right siblings
auto smallerIsLeft = mx::less(leftSizes, rightSizes);       // mlx/ops.h:461
// smallerIsLeft : [numParents] bool
```

This matches the depth-update convention at `structure_searcher.cpp:189-191` exactly and is sync-free.

### §1.2  Smaller-child partition-index array

```cpp
// Build the partition indices of the smaller and larger sides per parent.
// leftIdxArr[p]  = p  (left child of parent p has partition index p at depth d)
// rightIdxArr[p] = p + numParents
auto leftIdxArr = mx::arange(0, static_cast<int>(numParents), mx::uint32);
auto rightIdxArr = mx::add(
    leftIdxArr,
    mx::array(static_cast<uint32_t>(numParents), mx::uint32)
);

auto smallerIndices = mx::where(smallerIsLeft, leftIdxArr, rightIdxArr);  // mlx/ops.h:510-514
auto largerIndices  = mx::where(smallerIsLeft, rightIdxArr, leftIdxArr);  // mlx/ops.h:510-514
// Both : [numParents] uint32 — partition indices into the depth-d full histogram
```

`mx::arange` and `mx::add` produce lazy arrays; `mx::where` is documented at `mlx/mlx/ops.h:510-514` and is the canonical lazy-fusable conditional-select. **No CPU sync.**

### §1.3  Smaller-child histogram dispatch

The current production entry point is `ComputeHistograms(...)` (`histogram.cpp:288-329`) which calls `ComputeHistogramsImpl(...)` (`histogram.cpp:112-217`) which calls `DispatchHistogramBatched(...)` (`histogram.cpp:32-109`).

C6 introduces ONE new entry-point overload:

```cpp
// catboost/mlx/methods/histogram.h — new public overload
THistogramResult ComputeHistogramsSmallerChild(
    const TMLXDataSet& dataset,
    const mx::array& gradients,
    const mx::array& hessians,
    const TPartitionLayout& fullLayout,        // depth-d layout (all numPartitions = 2*numParents partitions)
    const mx::array& smallerIndices,           // [numParents] uint32 — partition index of smaller child per parent
    ui32 numParents                            // = numPartitions / 2
);
```

Implementation (skeleton, citing existing code paths to reuse):

```cpp
// catboost/mlx/methods/histogram.cpp — new function, body parallels ComputeHistograms at :288-329
//
// 1. Build (smallerDocIndices, smallerPartOffsets, smallerPartSizes) — gathered slices.
//    smallerPartSizes[p]   = take(fullLayout.PartSizes,   smallerIndices[p])
//    smallerPartOffsets[p] = take(fullLayout.PartOffsets, smallerIndices[p])
//    For DocIndices we DO NOT re-gather; the depth-d partition layout already
//    sorts ALL docs by partition. We only need to dispatch over the contiguous
//    ranges of the smaller-side partitions. The kernel reads doc indices via
//    partOffsets + partSizes — so we simply hand it the COMPACTED partOffsets/PartSizes
//    indexed by smallerIndices.
auto smallerPartSizes   = mx::take(fullLayout.PartSizes,   smallerIndices, 0);
                                                                          // mlx/ops.h:1061
auto smallerPartOffsets = mx::take(fullLayout.PartOffsets, smallerIndices, 0);
                                                                          // mlx/ops.h:1061

// 2. Call ComputeHistogramsImpl with numPartitions = numParents (NOT 2*numParents).
//    Pass the FULL DocIndices array — docs are already sorted by partition at depth d.
//    The kernel reads docIndices[partOffsets[p] .. partOffsets[p]+partSizes[p]] for each p,
//    so as long as smallerPartOffsets/Sizes point into the SAME docIndices buffer
//    they did at depth d, the routing invariant is preserved.
return ComputeHistogramsImpl(
    compressedIndex, features, statsArr,
    fullLayout.DocIndices,        // unchanged — full depth-d sort
    smallerPartOffsets,           // [numParents] — gathered
    smallerPartSizes,             // [numParents] — gathered
    numDocs, lineSize, numStats=2, totalBinFeatures,
    /*numPartitions=*/numParents  // dispatch shape change — half the partitions
);
```

This reuses 100% of `ComputeHistogramsImpl` (`histogram.cpp:112-217`) and `DispatchHistogramBatched` (`histogram.cpp:32-109`); the **only** difference is the values of `numPartitions`, `partitionOffsets`, `partitionSizes` passed in. Grid math at `histogram.cpp:83-88` automatically scales to `numParents` along the Y dimension. The `maxBlocksPerPart = 1` lock at `histogram.cpp:133-137` is untouched.

### §1.4  Parent histogram cache access

The cache is keyed by `(depth, dimensionIndex k)`. At depth `d ≥ 1`, the cache holds the **depth-(d-1)** full histogram for dimension `k`, of shape `[(numPartitions/2) × numStats × totalBinFeatures]` flat float32.

```cpp
// In SearchTreeStructure scope (catboost/mlx/methods/structure_searcher.cpp:60-108)
// Allocated ONCE outside the depth loop; reused across depths.
std::vector<mx::array> parentHistograms;  // size = approxDimension; valid for d ≥ 1
parentHistograms.reserve(approxDimension);
```

At depth `d ≥ 1`, for dimension `k`:
```cpp
const mx::array& histParent = parentHistograms[k];
// shape: [numParents * numStats * totalBinFeatures] flat float32
```

### §1.5  Larger-child derivation via subtract

The parent histogram at depth `d-1` is laid out as `[(numParents) × numStats × totalBinFeatures]` flat (matching the production output of `ComputeHistogramsImpl` at `histogram.cpp:125`). For each parent `p_parent ∈ [0, numParents)`, the parent's flat slice is:

```
histParent[p_parent * numStats * totalBinFeatures .. (p_parent + 1) * numStats * totalBinFeatures]
```

The smaller-child histogram at depth `d` (output of §1.3) is laid out as `[numParents × numStats × totalBinFeatures]` flat (same shape), where the `p`-th slice corresponds to the smaller child of parent `p`. This is **a direct rank-1 elementwise subtract**:

```cpp
// histSmaller : [numParents * numStats * totalBinFeatures] — output of §1.3
// histParent  : [numParents * numStats * totalBinFeatures] — cached from depth d-1
auto histLarger = mx::subtract(histParent, histSmaller);    // mlx/ops.h:890
// histLarger  : [numParents * numStats * totalBinFeatures]
```

`mx::subtract` is documented at `mlx/mlx/ops.h:890`. It is a single elementwise op, lazy, fusable, single-writer-per-cell — no atomics, no scatter, no cross-lane reduction (probe-spec §2).

### §1.6  Output assembly via `take_along_axis`

The downstream consumer `FindBestSplitGPU` (`score_calcer.cpp:10-11, 187-188`) expects a histogram array of shape `[numPartitions × numStats × totalBinFeatures]` flat — i.e. `[2 * numParents × numStats × totalBinFeatures]` flat. C6 must assemble the smaller + larger slices into THIS exact shape with the partition-index ordering of §1.1 (partition `p` = left child for `p ∈ [0, numParents)`; partition `p` = right child of parent `(p - numParents)` for `p ∈ [numParents, 2*numParents)`).

**Risk 2 anti-pattern:** `mx::concatenate(...)` along the partition axis may force materialization in the current MLX lowering (probe-spec §3 unverified-claim 2). The recommended primitive per S49 T0c Q3 is `mx::take_along_axis` (`mlx/mlx/ops.h:1070-1074`), which is documented as a fusable gather op.

**Assembly via `take_along_axis`:**

```cpp
// Step 1: Stack smaller and larger histograms along a new leading "side" axis (0 = small, 1 = large).
// Shape after reshape to per-partition rows:
//   histSmaller3D : [numParents, numStats * totalBinFeatures]
//   histLarger3D  : [numParents, numStats * totalBinFeatures]
const int rowSize = static_cast<int>(numStats * totalBinFeatures);
auto smallRows = mx::reshape(histSmaller,
    {static_cast<int>(numParents), rowSize});                // mlx/ops.h:133
auto largeRows = mx::reshape(histLarger,
    {static_cast<int>(numParents), rowSize});                // mlx/ops.h:133

// Step 2: Build "sideIndices" per output partition: a [2*numParents, rowSize] int32 array
// whose value at row q indicates which of {small, large} the partition q draws from.
//   For q ∈ [0, numParents):              sideIndices[q] = (smallerIsLeft[q] ? 0 : 1)
//     because partition q is the LEFT child of parent q, which is smaller iff smallerIsLeft[q].
//   For q ∈ [numParents, 2*numParents):   sideIndices[q] = (smallerIsLeft[q - numParents] ? 1 : 0)
//     because partition q is the RIGHT child of parent (q - numParents), which is smaller iff !smallerIsLeft.
// We construct sideIndices by concatenating smallerIsLeft (as int) and its bitwise-not, then reshape.
auto smallerIsLeftI32 = mx::astype(smallerIsLeft, mx::int32);              // [numParents]
auto smallerIsRightI32 = mx::subtract(
    mx::array(1, mx::int32), smallerIsLeftI32);                            // [numParents] = !smallerIsLeft
// For LEFT children: side = (smallerIsLeft ? 0 : 1)  — i.e. smallerIsRight
// For RIGHT children: side = (smallerIsLeft ? 1 : 0) — i.e. smallerIsLeft
// So the concatenation is [smallerIsRightI32, smallerIsLeftI32] of shape [2*numParents].
auto sideIndices1D = mx::concatenate(
    {smallerIsRightI32, smallerIsLeftI32}, 0);                             // [2*numParents]
// Broadcast to [2*numParents, rowSize] for take_along_axis:
auto sideIndices = mx::broadcast_to(
    mx::reshape(sideIndices1D, {static_cast<int>(2 * numParents), 1}),
    {static_cast<int>(2 * numParents), rowSize});                          // [2*numParents, rowSize]
```

Wait — `mx::concatenate` here is structurally fine because it operates on `[numParents]` 1-D index arrays, NOT on the per-partition histogram rows; it does not force a sync on the histogram path. The Risk 2 concern is `concatenate` on the histogram data array itself. The index array is small (`2*numParents ≤ 128` at depth 6 with `numPartitions = 64`) and the concatenate fuses trivially.

```cpp
// Step 3: Stack histograms along axis 0 by partition pair:
//   For partition q, sourceArray is histSmall[q'] or histLarge[q'] where q' = q mod numParents.
// Build a [2, numParents, rowSize] array, then take_along_axis on the leading axis with
// index = sideIndices reshaped to [2*numParents, rowSize].
//
// Easier formulation: build a [2 * numParents, rowSize] gather:
//   sourceA[q] = histSmall[q mod numParents]      // candidate 0
//   sourceB[q] = histLarge[q mod numParents]      // candidate 1
// then mx::where(sideIndices == 0, sourceA, sourceB).
//
// `mx::where` lowers to a fused select; `mx::take_along_axis` also lowers cleanly.
// Both are fusable. Final decision is a T2 micro-experiment (one-line swap), but we
// REQUIRE `mx::where` as the default per Risk 2.
auto srcSmallTiled = mx::concatenate({smallRows, smallRows}, 0);  // [2*numParents, rowSize]
auto srcLargeTiled = mx::concatenate({largeRows, largeRows}, 0);  // [2*numParents, rowSize]
// More efficient than concatenate-tile: use mx::tile on axis 0:
//   auto srcSmallTiled = mx::tile(smallRows, {2, 1});  // mlx/ops.h ≈ line 720
//   auto srcLargeTiled = mx::tile(largeRows, {2, 1});
auto assembled2D = mx::where(
    mx::equal(sideIndices, mx::array(0, mx::int32)),
    srcSmallTiled,
    srcLargeTiled
);                                                              // mlx/ops.h:510
auto assembledFlat = mx::reshape(assembled2D,
    {static_cast<int>(2 * numParents) * rowSize});              // mlx/ops.h:133
// assembledFlat : [numPartitions * numStats * totalBinFeatures]  ← MATCHES production shape
```

`mx::where` (`ops.h:510-514`) is the documented fusable conditional select. `mx::tile` (search `ops.h` for `tile`) lowers to a stride view + reshape, also fusable. **No `mx::concatenate` along the histogram data axis. No `mx::eval()` boundary.** This is the assembly pattern T2 must use.

### §1.7  Sequence summary

For depth `d ≥ 1`, per dimension `k`:

| Step | Op | File:line |
|---|---|---|
| 1. Sibling-pair sizes | `mx::slice` + `mx::less` | `ops.h:461` |
| 2. Index arrays | `mx::arange` + `mx::add` + `mx::where` (×2) | `ops.h:510-514` |
| 3. Gather smaller-partition offsets/sizes | `mx::take(..., axis=0)` | `ops.h:1061` |
| 4. Dispatch smaller-child histogram kernel | `ComputeHistogramsImpl` reuse | `histogram.cpp:112-217` |
| 5. Derive larger-child histogram | `mx::subtract(histParent, histSmaller)` | `ops.h:890` |
| 6. Output assembly | `mx::where` + `mx::tile` + `mx::reshape` | `ops.h:510-514, 133` |
| 7. Cache update for depth d+1 | `parentHistograms[k] = histSmaller` (NOT assembledFlat — see §4) | structure_searcher.cpp |

Total new ops per depth per dim: ~9 lazy MLX ops + 1 reuse of existing histogram dispatch. **Zero `mx::eval()`. Zero CPU readback.**

---

## §2  Lazy-graph fusion preservation (Risk 2)

### §2.1  Definition of "sync point"

A sync point is an op that forces evaluation of the lazy MLX graph, either explicitly or because it requires CPU access to array contents. The relevant MLX entry points are:
- `mx::eval(arr)` — explicit, blocks until `arr` is materialized
- `array::item()` — implicit, blocks (PairLogit path uses this at `train.cpp:242`)
- `array::data<T>()` — implicit, blocks (used in upload-shaped paths)
- `TMLXDevice::EvalAtBoundary(...)` — wrapper for `mx::eval`, used at `mlx_boosting.cpp:92` (per-iter cursor boundary) and `mlx_boosting.cpp:72` (validation init)

### §2.2  Inventory of sync points in C6 hot path

The C6 sequence (§1.7) uses only the following ops:

- `mx::reshape`, `mx::slice`, `mx::less`, `mx::where`, `mx::arange`, `mx::add`, `mx::take`, `mx::subtract`, `mx::tile`, `mx::astype`, `mx::concatenate` (1-D index arrays only), `mx::equal`, `mx::broadcast_to`
- `ComputeHistogramsImpl` via `mx::fast::metal_kernel` registration (`histogram.cpp:63-77`) — already lazy in v0.7.0 per the comment at `histogram.cpp:206-209` ("No EvalNow here — histogram is consumed lazily as an input to the suffix_sum_histogram Metal kernel")

**None of these ops force sync.** All are documented MLX core ops that return lazy `mx::array` expressions. The full C6 fragment composes into ONE lazy graph rooted at the assembled histogram array, which is then consumed by `FindBestSplitGPU` (`score_calcer.cpp:187-188`) in the same Metal command buffer — exactly as v0.7.0 does today.

### §2.3  Verification protocol (T2 deliverable)

Before T3 fires, T2 must run with `CATBOOST_MLX_STAGE_PROFILE` enabled (`mlx_boosting.cpp:12-16, 43-50`) and count sync points per iter in the C6 build vs the v0.7.0 baseline. Acceptance criterion:

> **Sync-point count per iter in C6 ≤ sync-point count per iter in v0.7.0.**

Sync points expected in v0.7.0 per-iter:
1. `EvalAtBoundary` at `mlx_boosting.cpp:92` (cursor boundary, 1×)
2. `EvalAtBoundary` at `mlx_boosting.cpp:72` (validation init, only on first iter if validation enabled)

C6 must add **zero** new sync points inside the depth loop. If the count increases, **STOP at T2** and use `mx::where`/`take_along_axis` to replace the offending op per §1.6.

### §2.4  Parent-cache lifetime and graph boundaries

The parent cache `parentHistograms[k]` is a `mx::array` held across iterations of the depth loop. MLX `mx::array` is a reference-counted handle; holding a handle does NOT force evaluation. The cache becomes a graph root only when the depth-`d` `mx::subtract` consumes it.

**Critical invariant:** the parent histogram for depth `d-1` MUST have been evaluated by the time depth `d`'s subtract executes. This is guaranteed by the v0.7.0 sync at `FindBestSplitGPU` call (`structure_searcher.cpp:130-137`), which forces evaluation of `perDimHistograms[k].Histograms` to pick the best split. Since `FindBestSplitGPU` reads from the SAME `mx::array` we cache, the cache contents are materialized as a side-effect of the depth-`d-1` `FindBestSplitGPU` call. No additional `mx::eval()` is required.

**Boundary:** the cache is alive WITHIN one tree (one call to `SearchTreeStructure`). It is reset between iterations because `SearchTreeStructure` re-enters with a fresh `parentHistograms` local. The cache is **NOT** persisted across iterations.

---

## §3  Loss-conditional dispatch architecture (Q4 lock)

### §3.1  Fork-point: training entry, ONCE per training run

Q4 lock states: "C6 path applies to Logloss + MultiClass loss configurations only. RMSE configurations use the production src-broadcast path (unchanged). Dispatch decision is loss-type-based, made at training start. No `use_histogram_subtraction` runtime flag."

**Fork point file:line:** `catboost/mlx/train_lib/train.cpp:162` — the line that reads `auto lossFunction = updatedOptions.LossFunctionDescription->GetLossFunction();`. Immediately after this line (and the `targetPtr` switch at `:166-284`), C6 inserts a **single boolean computed once** that flags the dispatch policy:

```cpp
// After train.cpp:162 (lossFunction extraction) and before the switch.
const bool useHistogramSubtraction =
    (lossFunction == ELossFunction::Logloss)
    || (lossFunction == ELossFunction::CrossEntropy)
    || (lossFunction == ELossFunction::MultiClass);
// Note: CrossEntropy uses TLoglossTarget per train.cpp:170-176, so the gradient/hessian
// envelope is identical to Logloss. Including it here matches the existing CrossEntropy/Logloss
// merge in the targetPtr switch (single TLoglossTarget instantiation).
```

This boolean is then propagated through `TBoostingConfig` (new field, §3.3) into `SearchTreeStructure`, which forks at the depth-loop entry.

### §3.2  Why a `bool` field, not a feature flag

The Q4 lock explicitly forbids a runtime feature flag. The distinction is:
- **Forbidden:** a flag readable via env var / config option / Python kwarg that the user can flip per-call. This introduces dead-code rot per devils-advocate's argument.
- **Permitted:** an internal `bool` derived from `lossFunction` at training entry, immutable for the lifetime of the training call, NOT exposed to users. This is a code-organization choice, not a user-facing toggle.

The `bool` does NOT branch per-iter or per-depth; it gates ONE compile-shaped path-select at the depth-loop entry. Both branches are present in the codebase, but ONE is selected at training-call entry based on loss type. This is identical in shape to how `EGrowPolicy` gates `SearchLossguideTreeStructure` vs `SearchTreeStructure` at `mlx_boosting.cpp:129-167`.

### §3.3  `TBoostingConfig` extension

Add ONE field to `TBoostingConfig` (`catboost/mlx/methods/mlx_boosting.h:41-60`):

```cpp
struct TBoostingConfig {
    // ... existing fields unchanged ...
    ui64 RandomSeed = 42;

    // [S49 C6] Histogram dispatch policy — set ONCE at training entry based on loss type.
    // true  => use parent-minus-sibling subtraction (Logloss / CrossEntropy / MultiClass)
    // false => use production src-broadcast dispatch (RMSE and all other loss types)
    // This is NOT a user-facing toggle. Set by RunTrainMLXImpl based on lossFunction
    // (train.cpp:162) per S49-T0c Q4 lock.
    bool UseHistogramSubtraction = false;
};
```

`RunBoosting` (`mlx_boosting.cpp:20-26`) reads `config.UseHistogramSubtraction` and propagates it to `SearchTreeStructure` via a new function parameter (default `false` for backward compatibility):

```cpp
// catboost/mlx/methods/structure_searcher.h:55-61 — extend signature
TObliviousTreeStructure SearchTreeStructure(
    TMLXDataSet& dataset,
    ui32 maxDepth,
    float l2RegLambda,
    ui32 approxDimension = 1,
    TStageProfiler* profiler = nullptr,
    bool useHistogramSubtraction = false   // [S49 C6] — NEW
);
```

Inside `SearchTreeStructure`, the fork is:

```cpp
// catboost/mlx/methods/structure_searcher.cpp:86-107 — replace per-dim histogram block
{
    STAGE_TIMER_DEPTH(profiler, EStageId::HistogramBuild,
        static_cast<int>(depth), {});
    for (ui32 k = 0; k < approxDimension; ++k) {
        // ... slice dimGrads, dimHess as today ...

        THistogramResult histResult;
        if (useHistogramSubtraction && depth >= 1) {
            // C6 path — subtract from cached parent
            histResult = ComputeHistogramsSmallerChildAndAssemble(
                dataset, dimGrads, dimHess, layout,
                /*histParent=*/parentHistograms[k],
                numPartitions
            );
        } else {
            // Production path — direct full-shape dispatch (unchanged at depth 0 always)
            histResult = ComputeHistograms(
                dataset, dimGrads, dimHess,
                layout.DocIndices, layout.PartOffsets, layout.PartSizes,
                numPartitions
            );
        }
        // Cache for next depth (regardless of branch — both produce same shape)
        if (useHistogramSubtraction) {
            parentHistograms[k] = histResult.Histograms;
        }
        perDimHistograms.push_back(std::move(histResult));
    }
}
```

The branch IS at runtime (one `if` per depth per dim), BUT the value of `useHistogramSubtraction` is constant for the lifetime of `SearchTreeStructure`. The branch predictor handles this trivially; there is no measurable per-iter overhead. This is the same shape as the existing `if (approxDimension == 1)` branch at `structure_searcher.cpp:89-99`.

### §3.4  Why not template-specialize `SearchTreeStructure<bool useSub>`

Considered and rejected. Templating doubles compile time and emits two copies of the function into the binary, with no runtime gain (the branch predictor already handles this). The runtime `if` is exactly the structural pattern used at `mlx_boosting.cpp:129-167` for grow-policy selection. Conformance with surrounding code wins.

### §3.5  Plumbing summary

| File:line | Change |
|---|---|
| `catboost/mlx/train_lib/train.cpp:162` (after) | Add `const bool useHistogramSubtraction = ...` derived from `lossFunction` |
| `catboost/mlx/train_lib/train.cpp` (where `TBoostingConfig` is constructed) | Set `config.UseHistogramSubtraction = useHistogramSubtraction` |
| `catboost/mlx/methods/mlx_boosting.h:60` | New `bool UseHistogramSubtraction = false` field |
| `catboost/mlx/methods/mlx_boosting.cpp:161-167` | Pass `config.UseHistogramSubtraction` to `SearchTreeStructure` |
| `catboost/mlx/methods/structure_searcher.h:55-61` | Add `bool useHistogramSubtraction = false` parameter |
| `catboost/mlx/methods/structure_searcher.cpp:46-52` | Receive parameter; allocate `parentHistograms` when true |
| `catboost/mlx/methods/structure_searcher.cpp:86-107` | Insert the fork in §3.3 |

---

## §4  Parent cache lifetime

### §4.1  Storage location

The cache lives as a **local variable** of `SearchTreeStructure`:

```cpp
// catboost/mlx/methods/structure_searcher.cpp:46-52 — inside SearchTreeStructure scope
// Allocated lazily on first use; one entry per approx-dim.
std::vector<mx::array> parentHistograms;
if (useHistogramSubtraction) {
    parentHistograms.reserve(approxDimension);
    // mx::array default-construction is cheap (empty handle); actual GPU memory is
    // allocated only when the depth-0 histogram is assigned in the cache update.
}
```

This is **NOT** a member of `TPartitionLayout` (`structure_searcher.h:22`). `TPartitionLayout` describes the doc-partition mapping at the CURRENT depth; the parent cache holds histogram data from the PREVIOUS depth. The two have different shapes and lifetimes.

### §4.2  Allocation strategy

UMA-shared (probe-spec §1.4 + T2 F1). MLX `MetalAllocator::malloc` (`mlx/mlx/backend/metal/allocator.cpp:98`) backs `mx::array` storage with `MTL::ResourceStorageModeShared` (`:14-15`). MLX maintains an internal allocator cache — back-to-back same-shape allocations of the same byte size hit the cache. C6 leverages this: the parent cache is written at depth `d`'s end and READ-only at depth `d+1`. The depth-`d-1` cache (now stale) is released when `parentHistograms[k] = newHistogram` reassigns the handle; the old buffer returns to the allocator pool and is reused at depth `d+1` when `histSmaller` is computed.

**No explicit buffer-reuse logic required.** The MLX allocator handles this for us per the UMA-shared model.

**Hot-path allocation guard (probe-spec §7.2 Risk 5):** If T2 profiling shows `MetalAllocator::malloc` overhead on the C6 path (e.g. > 1% of iter time), T2 may add an explicit pre-allocation buffer hoisted to iter-loop scope, but the default design relies on the MLX allocator cache.

### §4.3  Footprint

Per probe-spec §1.4 + S48-T2 Q1 silicon-architect verdict:

| Dataset | depth=6 cache size | Two-level transient peak |
|---|---|---|
| Higgs-1M iter=1000 | ≈ 1.8 MB | ≈ 3.6 MB |
| Epsilon iter=2000 | ≈ 130 MB | ≈ 260 MB |
| Amazon iter=1000 | < 1 MB | < 2 MB |

**Worst-case peak (Epsilon, two-level transient): 260 MB.** M3 Max `recommendedMaxWorkingSetSize ≈ 27 GB` (`mlx/mlx/backend/metal/allocator.cpp:83`). C6 cache is 0.96% of available working set. **PASS.**

### §4.4  Invalidation timing

The cache `parentHistograms` is a local of `SearchTreeStructure`. It is destroyed when `SearchTreeStructure` returns (end of tree). Per-tree (per-iter) invalidation is **automatic** by C++ scope. No manual reset required.

**Pseudocode:**

```cpp
TObliviousTreeStructure SearchTreeStructure(...) {
    // ... existing init ...
    std::vector<mx::array> parentHistograms;  // EMPTY at tree start
    if (useHistogramSubtraction) parentHistograms.reserve(approxDimension);

    for (depth = 0 .. maxDepth - 1) {
        // ... layout ...
        for (k = 0 .. approxDim - 1) {
            if (useHistogramSubtraction && depth >= 1) {
                // C6: subtract from parentHistograms[k]
                histResult = ComputeHistogramsSmallerChildAndAssemble(...);
            } else {
                // depth 0 OR not-C6: direct build
                histResult = ComputeHistograms(...);
            }
            if (useHistogramSubtraction) {
                if (depth == 0) parentHistograms.push_back(histResult.Histograms);
                else            parentHistograms[k] = histResult.Histograms;
            }
            perDimHistograms.push_back(...);
        }
        // ... rest of depth loop unchanged ...
    }
    // parentHistograms destructor runs here; GPU buffers return to allocator pool.
    return result;
}
```

The cache "persists across depth-level boundaries but NOT within a depth's iteration" — verified by this structure. Within depth `d`, all `k ∈ [0, approxDim)` read the depth-(d-1) cache; the cache is updated only AFTER all `k` iterations for depth `d` complete (lines: `parentHistograms[k] = histResult.Histograms` runs inside the loop but writes to slot `k`, leaving slots `k+1..` still holding depth-(d-1) data until processed).

**Subtle correctness check:** at the moment we run `mx::subtract(parentHistograms[k], histSmaller)` for dim `k` at depth `d`, the value of `parentHistograms[k]` MUST still be the depth-(d-1) histogram, not depth-(d-2) and not depth-`d`. The structure above satisfies this: the `parentHistograms[k] = ...` write happens AFTER the subtract is enqueued in the lazy graph. The `mx::array` assignment in MLX is a handle-swap, not an in-place buffer write — the existing handle (already captured by the subtract op) remains valid until the subtract evaluates. **Safe.**

---

## §5  Smaller-child docs/thread floor (Risk 4)

### §5.1  T1 empirical baseline

Per S48-T1 child-imbalance analysis (`docs/sprint48/T1/child-imbalance/analysis.md` referenced by probe-spec §0):
- Higgs-1M geomean smaller-child fraction at depth 5: ~0.47
- Epsilon geomean smaller-child fraction at depth 5: ~0.47 (~14700 docs/256-thread-TG ≈ 57 docs/thread)

Probe-spec §0 §3 ("≥22 docs/thread at deepest level") is the structurally-conservative floor; DEC-017 cliff fires at **3 docs/thread** (`.claude/state/DECISIONS.md:208`).

### §5.2  Instrumentation requirement at T2

T2 MUST instrument the smaller-child dispatch to record `smallerSize / 256` (docs per TG-thread, since threadgroup width is 256 per `histogram.cpp:88`) for every (depth, partition) tuple in the run. Storage: append to a per-dispatch CSV or stage-profiler record keyed by (iter, depth, parent_idx).

**Where to instrument:** inside `ComputeHistogramsSmallerChildAndAssemble`, immediately after `smallerPartSizes` is computed, force a one-time evaluation via `mx::eval(smallerPartSizes)` ONLY when `CATBOOST_MLX_STAGE_PROFILE` is defined (per `mlx_boosting.cpp:12-16` pattern). The eval is debug-only; production builds skip it and preserve full lazy fusion.

```cpp
#ifdef CATBOOST_MLX_STAGE_PROFILE
    // T1 §5 docs/thread floor verification — debug-only sync, no production impact.
    mx::eval(smallerPartSizes);
    const uint32_t* sizesPtr = smallerPartSizes.data<uint32_t>();
    for (ui32 p = 0; p < numParents; ++p) {
        const float docsPerThread = static_cast<float>(sizesPtr[p]) / 256.0f;
        profiler->RecordSmallerChildDocsPerThread(depth, p, docsPerThread);
    }
#endif
```

### §5.3  Decision rule (T2 → T3 escalation)

After the T2 instrumented run on Higgs-1M + Epsilon + Amazon × 3 seeds × iter=1000/2000/1000:

| Fraction of dispatches with docs/thread < 10 | Action |
|---|---|
| ≤ 5% | PROCEED to T3 (Gate B parity). No fallback needed. |
| 5–20% | Implement per-partition fallback in T2: for the specific `(depth, parent_idx)` tuples with `smallerSize < 10 * 256 = 2560`, fall back to direct dispatch for that PARTITION (not the whole depth). Re-run T2 instrumentation. |
| > 20% | STOP at T2. Escalate to @ml-product-owner for arc-retire decision per probe-spec §4.1 Gate A. |

### §5.4  Per-partition fallback design (deferred, contingent on §5.3)

If fallback is required, T2 implements:

```cpp
// Build a mask: smallerSize > threshold ⇒ subtract; else direct build.
constexpr ui32 kDocsPerThreadFloor = 10;
constexpr ui32 kSmallerSizeFloor = kDocsPerThreadFloor * 256;  // 2560

auto smallerSizeMask = mx::greater(
    smallerPartSizes,
    mx::array(static_cast<uint32_t>(kSmallerSizeFloor), mx::uint32)
);
// ... dispatch via mx::where, gating per-partition between subtract and direct paths ...
```

This is structurally `mx::where(mask, subtractedHist, directlyComputedHist)` per partition slice. Adds one extra full-shape histogram dispatch (the "direct" branch); negates ~half the C6 savings. **Only deploy if §5.3 requires it.** Default S49 assumption: fallback not needed.

---

## §6  Code modification surface (file:line index)

The complete list of files and lines T2 must touch. Every other file is OFF-LIMITS to C6.

| # | File | Line(s) | Change type | Description |
|---|---|---|---|---|
| 1 | `catboost/mlx/methods/histogram.h` | (append) | ADD | Declare `ComputeHistogramsSmallerChildAndAssemble(...)` |
| 2 | `catboost/mlx/methods/histogram.cpp` | (append after `:329`) | ADD | New function body per §1.3 + §1.5 + §1.6 |
| 3 | `catboost/mlx/methods/histogram.cpp` | `:63-77` | UNCHANGED | `mx::fast::metal_kernel` registration (reused) |
| 4 | `catboost/mlx/methods/histogram.cpp` | `:83-88` | UNCHANGED | Grid math — auto-scales with `numPartitions` arg |
| 5 | `catboost/mlx/methods/histogram.cpp` | `:112-217` | UNCHANGED | `ComputeHistogramsImpl` (reused by C6 path) |
| 6 | `catboost/mlx/methods/histogram.cpp` | `:133-137` | UNCHANGED | `maxBlocksPerPart=1` static_assert — must stay |
| 7 | `catboost/mlx/methods/structure_searcher.h` | `:19-23` | UNCHANGED | `TPartitionLayout` struct — NO new fields |
| 8 | `catboost/mlx/methods/structure_searcher.h` | `:55-61` | EDIT | Add `bool useHistogramSubtraction = false` param to `SearchTreeStructure` decl |
| 9 | `catboost/mlx/methods/structure_searcher.cpp` | `:46-52` | EDIT | Receive param; allocate `parentHistograms` local |
| 10 | `catboost/mlx/methods/structure_searcher.cpp` | `:60-108` | EDIT | Insert C6 fork per §3.3 inside per-dim loop |
| 11 | `catboost/mlx/methods/structure_searcher.cpp` | `:120-125` | UNCHANGED | `ComputeLeafSumsGPU` consumes gradients directly, not histogram |
| 12 | `catboost/mlx/methods/structure_searcher.cpp` | `:130-137` | UNCHANGED | `FindBestSplitGPU` consumes `perDimHistograms` (full-shape) |
| 13 | `catboost/mlx/methods/structure_searcher.cpp` | `:163-195` | UNCHANGED | Partition update (bit-shift OR) |
| 14 | `catboost/mlx/methods/structure_searcher.cpp` | `:201-453` | UNCHANGED | `SearchDepthwiseTreeStructure` (out of S49 scope) |
| 15 | `catboost/mlx/methods/structure_searcher.cpp` | `:455-747` | UNCHANGED | `SearchLossguideTreeStructure` (out of S49 scope) |
| 16 | `catboost/mlx/methods/mlx_boosting.h` | `:41-60` | EDIT | Add `bool UseHistogramSubtraction = false` to `TBoostingConfig` |
| 17 | `catboost/mlx/methods/mlx_boosting.cpp` | `:161-167` | EDIT | Pass `config.UseHistogramSubtraction` to `SearchTreeStructure` |
| 18 | `catboost/mlx/train_lib/train.cpp` | after `:162` | EDIT | Derive `const bool useHistogramSubtraction = ...` from `lossFunction` |
| 19 | `catboost/mlx/train_lib/train.cpp` | (where `TBoostingConfig` is filled) | EDIT | Set `config.UseHistogramSubtraction = ...` |
| 20 | `catboost/mlx/methods/stage_profiler.h` / `.cpp` | (extend) | OPTIONAL | Add `RecordSmallerChildDocsPerThread(...)` if §5.2 instrumentation lands |
| 21 | `catboost/mlx/kernels/kernel_sources.h` | `:107` | UNCHANGED | `kHistOneByteSource` kernel — STRICTLY off-limits to C6 |
| 22 | `catboost/mlx/methods/score_calcer.cpp` | `:10-11, 187-188` | UNCHANGED | `FindBestSplitGPU` — strict downstream consumer |

**Total code surface: 9 EDIT files; 0 deletion; ~150 LOC additions; 1 ADD function declaration.**

---

## §7  Branch-B regression preservation

### §7.1  Risk R6: predict-path edits

C6 must NOT modify any file in the predict path. Per probe-spec §4.3 + §7.1, the Branch-B test (`python/tests/regression/test_branch_b_regression.py:54`) loads a v0.6.1 model and runs `predict()`; training is not exercised. C6 changes ONLY training-path code.

### §7.2  Files C6 modifies (training path)

All in `catboost/mlx/methods/` and `catboost/mlx/train_lib/` — exclusively training-path:
- `catboost/mlx/methods/histogram.{h,cpp}`
- `catboost/mlx/methods/structure_searcher.{h,cpp}`
- `catboost/mlx/methods/mlx_boosting.h`
- `catboost/mlx/methods/mlx_boosting.cpp`
- `catboost/mlx/train_lib/train.cpp`
- (Optional) `catboost/mlx/methods/stage_profiler.{h,cpp}` — debug-only

### §7.3  Predict-path files — OFF-LIMITS

T2 must NOT touch any file in:
- `python/catboost_mlx/_predict_utils.py` — explicit OFF-LIMITS (R6, hard rule)
- `python/catboost_mlx/_predict.py` (if present)
- `catboost/mlx/methods/tree_applier.{h,cpp}` — predict-time tree application
- Any file matching `*predict*` or `*apply*` in the C6 modification surface

### §7.4  Pre-condition check at T2 entry

Per `docs/sprint49/sprint-plan.md` Task T2 pre-condition: T2 must verify that the pending `_predict_utils.py` modification visible in `git status` at S49 cut is UNRELATED to C6 (likely a stranded edit from S44/S43). If related, revert before T2 commit. If unrelated, leave as-is or commit independently before C6 work begins.

### §7.5  Branch-B test guarantee

Given §7.2 + §7.3, Branch-B regression PASSES trivially: the test loads a v0.6.1 model file and runs predict; neither file load nor predict touches any code modified by C6. **No C6-specific test workaround is required.** The test acts as an automatic guard rail — if it fires during T2, the cause is a stranded predict-path edit that must be reverted.

---

## §8  T2 engineering checklist

The concrete to-do list for @ml-engineer. Each task has a file:line target and a single-line acceptance criterion.

| # | Task | File:line | Acceptance |
|---|---|---|---|
| T2.1 | Add `bool UseHistogramSubtraction = false` to `TBoostingConfig` | `catboost/mlx/methods/mlx_boosting.h:60` (append before `};`) | Field present, default `false`, comment per §3.3 |
| T2.2 | Add `bool useHistogramSubtraction = false` param to `SearchTreeStructure` decl | `catboost/mlx/methods/structure_searcher.h:55-61` | Decl compiles; default preserves v0.7.0 behavior |
| T2.3 | Add same param to `SearchTreeStructure` definition | `catboost/mlx/methods/structure_searcher.cpp:46-52` | Receives param; allocates `std::vector<mx::array> parentHistograms` |
| T2.4 | Declare `ComputeHistogramsSmallerChildAndAssemble(...)` | `catboost/mlx/methods/histogram.h` (append) | Header compiles; signature per §1.3 + assembly fused |
| T2.5 | Implement `ComputeHistogramsSmallerChildAndAssemble(...)` body | `catboost/mlx/methods/histogram.cpp:329` (append after) | Per §1.1–§1.6; reuses `ComputeHistogramsImpl`; NO `mx::eval()` |
| T2.6 | Insert C6 fork in per-dim histogram block | `catboost/mlx/methods/structure_searcher.cpp:86-107` | Branch on `useHistogramSubtraction && depth >= 1`; cache update logic per §4.4 |
| T2.7 | Pass `config.UseHistogramSubtraction` to `SearchTreeStructure` | `catboost/mlx/methods/mlx_boosting.cpp:161-167` | Plumbed; depthwise/lossguide branches NOT modified |
| T2.8 | Derive `useHistogramSubtraction` from `lossFunction` | `catboost/mlx/train_lib/train.cpp` after `:162` | Logloss + CrossEntropy + MultiClass ⇒ true; all others false |
| T2.9 | Set `config.UseHistogramSubtraction` on the config object | `catboost/mlx/train_lib/train.cpp` (where config is filled) | Propagates to `RunBoosting` |
| T2.10 | Pre-condition check: confirm `_predict_utils.py` pending diff is unrelated to C6 | `python/catboost_mlx/_predict_utils.py` | Either reverted or confirmed-unrelated and noted in T2 PR |
| T2.11 | (If §5.3 requires) Add `STAGE_PROFILE`-gated docs/thread instrumentation | `catboost/mlx/methods/histogram.cpp` (inside new function) + `stage_profiler.{h,cpp}` | `CATBOOST_MLX_STAGE_PROFILE` build records per-(depth, parent) docs/thread |
| T2.12 | Sync-point parity check vs v0.7.0 | `catboost/mlx/methods/stage_profiler.{h,cpp}` (existing) | Per-iter sync count in C6 ≤ v0.7.0 (§2.3 verification) |
| T2.13 | Branch-B regression: confirm GREEN before merge | `python/tests/regression/test_branch_b_regression.py` | Both `test_higgs_1m_predict_byte_equivalent_to_v061` and `test_epsilon_subset_predict_byte_equivalent_to_v061` PASS |
| T2.14 | Loss-coverage smoke test: RMSE training runs unchanged path | (manual run) | RMSE iter ≈ v0.7.0 iter (within CV); confirms fork at `train.cpp:162` is correct |

**Total: 14 tasks.** Estimated 2.0 agent-days per `docs/sprint49/sprint-plan.md` Task T2 budget. Item T2.11 is conditional on §5.3 outcome; if Gate A measurement at T3 shows < 5% sub-floor dispatches, T2.11 collapses to "instrumentation added but no fallback wired."

---

## §9  Risks addressed by design

| Risk | Source | Addressed by |
|---|---|---|
| R2 — MLX lazy-graph fusion broken | sprint-plan §5 R2 | §1.6 mandates `mx::where` + `mx::tile` (NOT `mx::concatenate` on histogram data); §2 inventory of sync points; §2.3 T2 verification protocol; sync-point parity is a hard gate before T3 |
| R3 — DEC-008 envelope (γ propagation) | sprint-plan §5 R3 | §3 loss-conditional dispatch — RMSE (tightest envelope) stays on v0.7.0 path; γ_13 only applies to Logloss/MultiClass which have γ_14 ≈ 8.3e-7 ceiling. T3 18-config sweep is the empirical gate. |
| R4 — DEC-017 cliff at small smaller-children | sprint-plan §5 R4 | §5 docs/thread floor instrumentation at T2; per-partition fallback design ready if needed; >20% sub-floor triggers arc retire |
| R5 — Memory pressure | sprint-plan §5 R5; probe-spec §7.2 | §4.2 leverages MLX allocator cache (no explicit reuse logic needed); §4.3 footprint table confirms <1% of working-set ceiling; §4.4 automatic scope-based invalidation |
| R6 — Branch-B regression | sprint-plan §5 R6 | §7 file-modification surface is exclusively training-path; predict-path explicitly OFF-LIMITS; T2.10 pre-condition check; Branch-B passes by construction |

---

## §10  Open questions

**None.** All design decisions are made in §1–§9. T2 has zero ambiguity. The only conditional logic in this design is §5.3 (Gate A measurement → fallback decision), which is gated on T2-empirical data and resolved before T3.

---

## §11  Cross-references

- `docs/sprint48/T3/probe-spec-c6.md` — load-bearing probe-spec (mechanism, gates, risks)
- `docs/sprint48/T2/feasibility.md` — silicon-architect MANDATORY-CODE-INSPECTION (T2 F1–F6)
- `docs/sprint49/sprint-plan.md` — S49 task structure, T0c locks, decision tree
- `.claude/state/DECISIONS.md` — DEC-008 parity envelope; DEC-017 docs/thread cliff; DEC-052 OUTCOME A
- `catboost/private/libs/algo/scoring.cpp:315-332` — CPU-CatBoost `FixUpStats` (F5 precedent)
- `catboost/private/libs/algo/calc_score_cache.cpp:533, 1155-1168` — CPU-CatBoost smaller-side selector
- `mlx/mlx/ops.h:133, 461, 510-514, 890, 1061, 1070-1074` — MLX primitive citations

---

**END OF T1 DESIGN. READY FOR T2 ENGINEERING.**
