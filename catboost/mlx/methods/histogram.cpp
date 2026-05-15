#include "histogram.h"
#include "structure_searcher.h"   // TPartitionLayout (C6 dependency — histogram.h forward-declares it)

#include <catboost/mlx/kernels/kernel_sources.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>
#include <mlx/mlx.h>
#include <mlx/fast.h>

namespace NCatboostMlx {

    namespace {
        // Threadgroup-memory budget guard for the L1a histogram kernel (Sprint 18).
        // The kernel declares `threadgroup float simdHist[NUM_SIMD_GROUPS][HIST_PER_SIMD]`
        // in kernel_sources.h; its byte size must stay within Apple Silicon's 32 KB
        // threadgroup-memory limit. MSL does not support static_assert in shader source,
        // so we mirror the constants here and assert on the host side. Any future bump
        // to the SIMD-group count or per-SIMD histogram size must re-tile the layout
        // (e.g. split the 4-tile reduction into more tiles) before tripping this assert.
        constexpr unsigned kHistSimdSize         = 32;
        constexpr unsigned kHistBlockSize        = 256;
        constexpr unsigned kHistFeaturesPerPack  = 4;
        constexpr unsigned kHistBinsPerByte      = 256;
        constexpr unsigned kHistNumSimdGroups    = kHistBlockSize / kHistSimdSize;            // 8
        constexpr unsigned kHistPerSimd          = kHistFeaturesPerPack * kHistBinsPerByte;    // 1024
        constexpr unsigned kHistThreadgroupBytes = kHistNumSimdGroups * kHistPerSimd * sizeof(float); // 32768
        constexpr unsigned kAppleSiliconTgLimit  = 32768;
        static_assert(kHistThreadgroupBytes <= kAppleSiliconTgLimit,
                      "L1a histogram kernel exceeds Apple Silicon's 32 KB threadgroup limit; "
                      "bumping NUM_SIMD_GROUPS or HIST_PER_SIMD requires re-tiling the reduction.");

        // Batched histogram dispatch — one Metal dispatch covers ALL feature groups.
        mx::array DispatchHistogramBatched(
            const mx::array& compressedData,
            const mx::array& stats,
            const mx::array& docIndices,
            const mx::array& partOffsets,
            const mx::array& partSizes,
            const mx::array& featureColumnIndices,     // [numGroups] — group g reads column g
            ui32 lineSize,
            ui32 maxBlocksPerPart,
            ui32 numGroups,
            const mx::array& foldCountsFlat,           // [numGroups * 4]
            const mx::array& firstFoldIndicesFlat,     // [numGroups * 4]
            ui32 totalBinFeatures,
            ui32 numStats,
            ui32 numPartitions,
            ui32 totalNumDocs,
            const mx::Shape& histShape
        ) {
            // Scalar uniforms → 0-dim arrays → `const constant T&` in Metal signature.
            // Flatten compressed data to 1D for linear doc*lineSize indexing in kernel.
            auto flatCompressed = mx::reshape(compressedData, {-1});
            auto lineSizeArr   = mx::array(static_cast<uint32_t>(lineSize), mx::uint32);
            auto maxBlocksArr  = mx::array(static_cast<uint32_t>(maxBlocksPerPart), mx::uint32);
            auto numGroupsArr  = mx::array(static_cast<uint32_t>(numGroups), mx::uint32);
            auto totalBinsArr  = mx::array(static_cast<uint32_t>(totalBinFeatures), mx::uint32);
            auto numStatsArr   = mx::array(static_cast<uint32_t>(numStats), mx::uint32);
            auto totalDocsArr  = mx::array(static_cast<uint32_t>(totalNumDocs), mx::uint32);

            // Input names match kHistOneByteSource variable names exactly.
            // The kernel body reads `featureColumnIndices[groupIdx]` (array) and
            // `numGroups` (scalar) — both must be present in input_names.
            auto kernel = mx::fast::metal_kernel(
                "histogram_one_byte_features",
                /*input_names=*/{
                    "compressedIndex", "stats", "docIndices",
                    "partOffsets", "partSizes",
                    "featureColumnIndices", "lineSize", "maxBlocksPerPart", "numGroups",
                    "foldCountsFlat", "firstFoldIndicesFlat",
                    "totalBinFeatures", "numStats", "totalNumDocs"
                },
                /*output_names=*/{"histogram"},
                /*source=*/KernelSources::kHistOneByteSource,
                /*header=*/KernelSources::kHistHeader,
                /*ensure_row_contiguous=*/true,
                /*atomic_outputs=*/true   // writeback uses atomic_fetch_add_explicit
            );

            // Grid: (256 * maxBlocksPerPart * numGroups, numPartitions, numStats).
            // Dividing by threadgroup (256,1,1) gives
            // (maxBlocksPerPart * numGroups, numPartitions, numStats) threadgroups.
            // Each threadgroup handles one (group, block, partition, stat) tuple.
            auto grid = std::make_tuple(
                static_cast<int>(256 * maxBlocksPerPart * numGroups),
                static_cast<int>(numPartitions),
                static_cast<int>(numStats)
            );
            auto threadgroup = std::make_tuple(256, 1, 1);

            auto results = kernel(
                /*inputs=*/{
                    flatCompressed, stats, docIndices,
                    partOffsets, partSizes,
                    featureColumnIndices, lineSizeArr, maxBlocksArr, numGroupsArr,
                    foldCountsFlat, firstFoldIndicesFlat,
                    totalBinsArr, numStatsArr, totalDocsArr
                },
                /*output_shapes=*/{histShape},
                /*output_dtypes=*/{mx::float32},
                grid,
                threadgroup,
                /*template_args=*/{},
                /*init_value=*/0.0f,
                /*verbose=*/false,
                /*stream=*/mx::Device::gpu
            );

            return results[0];
        }

        // Common histogram dispatch logic shared by both ComputeHistograms overloads.
        THistogramResult ComputeHistogramsImpl(
            const TMLXCompressedIndex& compressedIndex,
            const TVector<TCFeature>& features,
            const mx::array& statsArr,
            const mx::array& docIndices,
            const mx::array& partitionOffsets,
            const mx::array& partitionSizes,
            ui32 numDocs,
            ui32 lineSize,
            ui32 numStats,
            ui32 totalBinFeatures,
            ui32 numPartitions
        ) {
            mx::Shape histShape = {static_cast<int>(numPartitions * numStats * totalBinFeatures)};
            // Sibling S-1 race guard: kHistOneByte cross-threadgroup writeback uses
            // atomic_fetch_add on device atomic_float and is non-deterministic when
            // maxBlocksPerPart > 1. The in-TG path is clean (per-SIMD sub-histograms),
            // but the global-merge fallback races on FP non-associativity. Do NOT raise
            // this literal without first fixing the writeback. Authoritative record:
            // docs/sprint23/d0_bimodality_verification.md §C and
            // .claude/state/KNOWN_BUGS.md (Sibling S-1).
            constexpr ui32 maxBlocksPerPart = 1;
            static_assert(maxBlocksPerPart == 1,
                "Sibling S-1 latent race: kHistOneByte writeback is racy when "
                "maxBlocksPerPart > 1. Fix the cross-TG atomic_float accumulation "
                "before raising this constant (see KNOWN_BUGS.md).");

            const ui32 numFeatures = features.size();
            const ui32 numGroups   = (numFeatures + 3) / 4;

            // Build flat fold metadata for all groups (4 slots per group).
            // featureColumnIndices[g] = g (group g reads the g-th ui32 column).
            TVector<ui32> foldCountsFlatVec(numGroups * 4, 0u);
            TVector<ui32> firstFoldIndicesFlatVec(numGroups * 4, 0u);
            TVector<ui32> featureColumnIndicesVec(numGroups, 0u);

            ui32 maxFoldCount = 0;
            for (ui32 g = 0; g < numGroups; ++g) {
                featureColumnIndicesVec[g] = g;
                const ui32 featureStart    = g * 4;
                const ui32 featuresInGroup = std::min(4u, numFeatures - featureStart);
                for (ui32 slot = 0; slot < featuresInGroup; ++slot) {
                    const ui32 folds = features[featureStart + slot].Folds;
                    foldCountsFlatVec[g * 4 + slot]       = folds;
                    firstFoldIndicesFlatVec[g * 4 + slot] = features[featureStart + slot].FirstFoldIndex;
                    if (folds > maxFoldCount) maxFoldCount = folds;
                }
            }

            // DEC-016 T1 invariant: the kernel uses bit 31 of the packed uint32 as a
            // per-lane valid sentinel. Each feature occupies 8 bits, so the MSB of
            // slot-0 aliases with bin values ≥ 128. The kernel masks this bit before
            // bin extraction, which would silently rewrite bin 128..255 → 0..127.
            // Fail loudly if a caller attempts to dispatch outside the supported
            // envelope (bin ≤ 127 ⇔ Folds ≤ 127).
            CB_ENSURE(maxFoldCount <= 127u,
                      "CatBoost-MLX histogram kernel: max fold count " << maxFoldCount
                      << " exceeds DEC-016 T1 envelope (≤ 127). The MSB-sentinel "
                         "collides with bin values ≥ 128. Reduce MaxBins to ≤ 128 "
                         "(and ensure no NaN offset pushes a bin to 128) or wait for "
                         "Sprint 20's wider-envelope kernel (DEC-017).");

            auto foldCountsArr = mx::array(
                reinterpret_cast<const int32_t*>(foldCountsFlatVec.data()),
                {static_cast<int>(numGroups * 4)}, mx::uint32
            );
            auto firstFoldArr = mx::array(
                reinterpret_cast<const int32_t*>(firstFoldIndicesFlatVec.data()),
                {static_cast<int>(numGroups * 4)}, mx::uint32
            );
            auto featureColsArr = mx::array(
                reinterpret_cast<const int32_t*>(featureColumnIndicesVec.data()),
                {static_cast<int>(numGroups)}, mx::uint32
            );

            auto histogram = DispatchHistogramBatched(
                compressedIndex.GetCompressedData(),
                statsArr,
                docIndices,
                partitionOffsets,
                partitionSizes,
                featureColsArr,
                lineSize,
                maxBlocksPerPart,
                numGroups,
                foldCountsArr,
                firstFoldArr,
                totalBinFeatures,
                numStats,
                numPartitions,
                numDocs,
                histShape
            );

            // No EvalNow here — histogram is consumed lazily as an input to the
            // suffix_sum_histogram Metal kernel in FindBestSplitGPU.  MLX will
            // materialise the full graph in that same command buffer, avoiding an
            // unnecessary CPU-GPU sync point.

            return THistogramResult{
                .Histograms = histogram,
                .NumPartitions = numPartitions,
                .NumStats = numStats,
                .TotalBinFeatures = totalBinFeatures
            };
        }
    }  // anonymous namespace

    // DispatchHistogramT2 implementation is in histogram_t2_impl.cpp (minimal deps).
    // The NIT-4 maxBlocksPerPart == 1 constraint (Sibling S-1 guard) is enforced
    // as a compile-time `constexpr` + `static_assert` at the ComputeHistogramsImpl
    // call site below. Any refactor that raises the literal fails to compile with
    // a pointer to KNOWN_BUGS.md / docs/sprint23/d0_bimodality_verification.md §C.

    mx::array CreateZeroHistogram(ui32 numPartitions, ui32 numStats, ui32 totalBinFeatures) {
        // Return a lazy zero array — no EvalNow needed.
        // mx::zeros() produces a trivially lazy expression; any downstream consumer
        // (e.g. scatter-add) will materialise it as part of that operation's graph.
        return mx::zeros(
            {static_cast<int>(numPartitions * numStats * totalBinFeatures)},
            mx::float32
        );
    }

    THistogramResult ComputeHistograms(
        const TMLXDataSet& dataset,
        const mx::array& docIndices,
        const mx::array& partitionOffsets,
        const mx::array& partitionSizes,
        ui32 numPartitions,
        bool useWeights
    ) {
        const auto& compressedIndex = dataset.GetCompressedIndex();
        const auto& features = compressedIndex.GetFeatures();
        const ui32 numDocs = compressedIndex.GetNumDocs();
        const ui32 lineSize = compressedIndex.GetNumUi32PerDoc();
        const ui32 numStats = useWeights ? 2 : 1;

        // Count total bin features
        ui32 totalBinFeatures = 0;
        for (const auto& feat : features) {
            totalBinFeatures += feat.Folds;
        }

        CATBOOST_DEBUG_LOG << "CatBoost-MLX: Computing histograms for "
            << features.size() << " features, " << totalBinFeatures << " bin-features, "
            << numPartitions << " partitions, " << numStats << " stats" << Endl;

        if (totalBinFeatures == 0) {
            return THistogramResult{
                .Histograms = mx::zeros({1}, mx::float32),
                .NumPartitions = numPartitions,
                .NumStats = numStats,
                .TotalBinFeatures = 0
            };
        }

        // Build the stats array: gradient only, or gradient + hessian stacked
        auto statsArr = dataset.GetGradients();
        if (useWeights) {
            statsArr = mx::concatenate({
                mx::reshape(dataset.GetGradients(), {1, static_cast<int>(numDocs)}),
                mx::reshape(dataset.GetHessians(), {1, static_cast<int>(numDocs)})
            }, 0);
            statsArr = mx::reshape(statsArr, {static_cast<int>(numStats * numDocs)});
        } else {
            statsArr = mx::reshape(statsArr, {static_cast<int>(numDocs)});
        }

        return ComputeHistogramsImpl(
            compressedIndex, features, statsArr,
            docIndices, partitionOffsets, partitionSizes,
            numDocs, lineSize, numStats, totalBinFeatures, numPartitions
        );
    }

    THistogramResult ComputeHistograms(
        const TMLXDataSet& dataset,
        const mx::array& gradients,
        const mx::array& hessians,
        const mx::array& docIndices,
        const mx::array& partitionOffsets,
        const mx::array& partitionSizes,
        ui32 numPartitions
    ) {
        const auto& compressedIndex = dataset.GetCompressedIndex();
        const auto& features = compressedIndex.GetFeatures();
        const ui32 numDocs = compressedIndex.GetNumDocs();
        const ui32 lineSize = compressedIndex.GetNumUi32PerDoc();
        const ui32 numStats = 2;  // always grad + hess

        ui32 totalBinFeatures = 0;
        for (const auto& feat : features) {
            totalBinFeatures += feat.Folds;
        }

        if (totalBinFeatures == 0) {
            return THistogramResult{
                .Histograms = mx::zeros({1}, mx::float32),
                .NumPartitions = numPartitions,
                .NumStats = numStats,
                .TotalBinFeatures = 0
            };
        }

        // Build stats array from provided gradient + hessian
        auto statsArr = mx::concatenate({
            mx::reshape(gradients, {1, static_cast<int>(numDocs)}),
            mx::reshape(hessians, {1, static_cast<int>(numDocs)})
        }, 0);
        statsArr = mx::reshape(statsArr, {static_cast<int>(numStats * numDocs)});

        return ComputeHistogramsImpl(
            compressedIndex, features, statsArr,
            docIndices, partitionOffsets, partitionSizes,
            numDocs, lineSize, numStats, totalBinFeatures, numPartitions
        );
    }

    // [S49 C6] Parent-minus-sibling histogram subtraction (design §1.1–§1.6).
    //
    // Dispatches the histogram kernel over SMALLER CHILDREN ONLY (numParents partitions)
    // then derives larger-child histograms via mx::subtract(histParent, histSmaller),
    // and assembles the full [numPartitions × numStats × totalBinFeatures] result
    // in canonical partition-index order using mx::where + mx::tile + mx::reshape.
    //
    // SYNC INVARIANT: zero mx::eval() boundaries inserted. All ops are lazy and fusable.
    // The parent histogram (histParent) is guaranteed materialized by the prior depth's
    // FindBestSplitGPU call (§2.4). No additional eval is needed here.
    //
    // Oblivious-tree partition-index convention (design §1.1, CRITICAL):
    //   At depth d, parent p_parent produces:
    //     Left child  → partition index p_parent          ∈ [0, numParents)
    //     Right child → partition index p_parent + numParents  ∈ [numParents, numPartitions)
    //   Sibling pairs are numParents apart — NOT adjacent. The design §1.1 corrected formulation
    //   (two-half slices) is used here; the naive reshape({numParents, 2}) is explicitly rejected.
    THistogramResult ComputeHistogramsSmallerChildAndAssemble(
        const TMLXDataSet& dataset,
        const mx::array& gradients,
        const mx::array& hessians,
        const TPartitionLayout& fullLayout,
        const mx::array& histParent,
        ui32 numPartitions
    ) {
        CB_ENSURE(numPartitions >= 2,
            "CatBoost-MLX C6: ComputeHistogramsSmallerChildAndAssemble requires numPartitions >= 2 "
            "(depth >= 1). Got numPartitions=" << numPartitions);
        CB_ENSURE((numPartitions & 1u) == 0u,
            "CatBoost-MLX C6: numPartitions must be even (power-of-two at depth >= 1). "
            "Got numPartitions=" << numPartitions);

        const auto& compressedIndex = dataset.GetCompressedIndex();
        const auto& features = compressedIndex.GetFeatures();
        const ui32 numDocs   = compressedIndex.GetNumDocs();
        const ui32 lineSize  = compressedIndex.GetNumUi32PerDoc();
        const ui32 numStats  = 2;  // grad + hess (matches ComputeHistograms grad/hess overload)
        const ui32 numParents = numPartitions / 2;

        ui32 totalBinFeatures = 0;
        for (const auto& feat : features) {
            totalBinFeatures += feat.Folds;
        }

        // Early-exit: no features. Return zero histogram of full shape.
        if (totalBinFeatures == 0) {
            return THistogramResult{
                .Histograms      = mx::zeros({static_cast<int>(numPartitions)}, mx::float32),
                .NumPartitions   = numPartitions,
                .NumStats        = numStats,
                .TotalBinFeatures = 0
            };
        }

        // -------------------------------------------------------------------------
        // §1.1 Smaller-child mask — two-half slice (CORRECT convention for oblivious trees)
        //
        // layout.PartSizes at depth d:
        //   [0 .. numParents-1]          = sizes of left  children (partition indices 0..numParents-1)
        //   [numParents .. numPartitions-1] = sizes of right children (partition indices numParents..numPartitions-1)
        //
        // Sibling pairs are (p, p+numParents) — NOT adjacent. See design §1.1 correction.
        // -------------------------------------------------------------------------
        auto leftSizes  = mx::slice(fullLayout.PartSizes,
            {0},
            {static_cast<int>(numParents)});                    // [numParents] uint32 — left siblings
        auto rightSizes = mx::slice(fullLayout.PartSizes,
            {static_cast<int>(numParents)},
            {static_cast<int>(numPartitions)});                 // [numParents] uint32 — right siblings
        // smallerIsLeft[p] = true iff left child of parent p has fewer docs
        auto smallerIsLeft = mx::less(leftSizes, rightSizes);   // [numParents] bool — ops.h:461

        // -------------------------------------------------------------------------
        // §1.2 Smaller-child and larger-child partition-index arrays
        //
        //   leftIdxArr[p]  = p              (left child partition index at depth d)
        //   rightIdxArr[p] = p + numParents (right child partition index at depth d)
        // -------------------------------------------------------------------------
        auto leftIdxArr  = mx::arange(0, static_cast<int>(numParents), mx::uint32); // [numParents] uint32
        auto rightIdxArr = mx::add(
            leftIdxArr,
            mx::array(static_cast<uint32_t>(numParents), mx::uint32)
        );                                                                             // [numParents] uint32

        // smallerIndices[p] = partition index of the smaller child of parent p
        // (largerIndices is derived implicitly via the assembly sideIndices logic — not needed explicitly)
        auto smallerIndices = mx::where(smallerIsLeft, leftIdxArr, rightIdxArr);     // ops.h:510-514

        // -------------------------------------------------------------------------
        // §1.3 Gather smaller-child partition offsets and sizes from full layout
        // -------------------------------------------------------------------------
        auto smallerPartSizes   = mx::take(fullLayout.PartSizes,   smallerIndices, 0); // ops.h:1061
        auto smallerPartOffsets = mx::take(fullLayout.PartOffsets,  smallerIndices, 0); // ops.h:1061

#ifdef CATBOOST_MLX_STAGE_PROFILE
        // T2 §5.2 docs/thread floor verification — debug-only sync, zero production impact.
        // Records per-(depth, parent) docs/thread to detect DEC-017 cliff (< 3 docs/thread).
        // The profiler pointer is not accessible here; emit to stderr for T2 analysis.
        mx::eval(smallerPartSizes);
        {
            const uint32_t* sizesPtr = smallerPartSizes.data<uint32_t>();
            for (ui32 p = 0; p < numParents; ++p) {
                const float docsPerThread = static_cast<float>(sizesPtr[p]) / 256.0f;
                fprintf(stderr,
                    "[C6-PROBE] parent=%u smallerSize=%u docs/thread=%.2f\n",
                    p, sizesPtr[p], docsPerThread);
            }
        }
#endif

        // -------------------------------------------------------------------------
        // §1.3 (cont.) Dispatch histogram kernel over smaller children only.
        // numParents partitions instead of numPartitions — grid auto-scales.
        // Reuses 100% of ComputeHistogramsImpl; only partitionOffsets/Sizes change.
        // -------------------------------------------------------------------------

        // Build stats array: [numStats * numDocs] = concat(grad, hess) along stat axis
        auto statsArr = mx::concatenate({
            mx::reshape(gradients, {1, static_cast<int>(numDocs)}),
            mx::reshape(hessians,  {1, static_cast<int>(numDocs)})
        }, 0);
        statsArr = mx::reshape(statsArr, {static_cast<int>(numStats * numDocs)});

        // Dispatch over smaller children (numParents partitions) using full DocIndices.
        // The kernel reads docIndices[partOffsets[p] .. partOffsets[p]+partSizes[p]] for each p;
        // smallerPartOffsets/Sizes point into the same DocIndices buffer — routing invariant preserved.
        auto histSmaller = ComputeHistogramsImpl(
            compressedIndex, features, statsArr,
            fullLayout.DocIndices,          // unchanged — full depth-d sorted array
            smallerPartOffsets,             // [numParents] gathered
            smallerPartSizes,               // [numParents] gathered
            numDocs, lineSize, numStats, totalBinFeatures,
            numParents                      // dispatch over numParents partitions only
        );

        // -------------------------------------------------------------------------
        // §1.5 Derive larger-child histogram via rank-1 elementwise subtraction.
        //
        // Both histSmaller.Histograms and histParent have shape:
        //   [numParents * numStats * totalBinFeatures]
        // Parent slice p corresponds to larger child of parent p.
        // Subtraction is a single lazy elementwise op — zero sync, zero scatter, zero atomics.
        // -------------------------------------------------------------------------
        const int rowSize = static_cast<int>(numStats * totalBinFeatures);
        auto histLarger = mx::subtract(histParent, histSmaller.Histograms); // ops.h:890
        // histLarger: [numParents * numStats * totalBinFeatures] — same shape as histSmaller

        // -------------------------------------------------------------------------
        // §1.6 Output assembly — produce [numPartitions * numStats * totalBinFeatures]
        //      in canonical partition-index order.
        //
        // Canonical order:
        //   q ∈ [0, numParents):         partition q = left  child of parent q
        //   q ∈ [numParents, 2*numParents): partition q = right child of parent (q - numParents)
        //
        // sideIndices[q] ∈ {0=smaller, 1=larger} indicates which histogram the output at
        // row q draws from.
        //
        // Implementation: mx::where(sideIndices == 0, smallTiled, largeTiled).
        // No mx::concatenate on histogram data axis (Risk 2 compliant).
        // mx::concatenate IS used on small index arrays (≤ 128 entries) — trivially fusable.
        // -------------------------------------------------------------------------

        // Reshape histograms to [numParents, rowSize] for per-parent-row selection
        auto smallRows = mx::reshape(histSmaller.Histograms,
            {static_cast<int>(numParents), rowSize});              // ops.h:133
        auto largeRows = mx::reshape(histLarger,
            {static_cast<int>(numParents), rowSize});              // ops.h:133

        // Build sideIndices: [2*numParents] int32
        //   For LEFT children  (q ∈ [0,         numParents)):   side = smallerIsLeft ? 0 : 1
        //                                                               = !smallerIsLeft (as int) = smallerIsRight
        //   For RIGHT children (q ∈ [numParents, numPartitions)): side = smallerIsLeft ? 1 : 0
        //                                                               = smallerIsLeft (as int)
        // Concatenation: [smallerIsRight, smallerIsLeft] — index arrays only (NOT histogram data).
        auto smallerIsLeftI32  = mx::astype(smallerIsLeft, mx::int32);          // [numParents] int32
        auto smallerIsRightI32 = mx::subtract(
            mx::array(1, mx::int32), smallerIsLeftI32);                          // [numParents] int32
        // sideIndices1D[q]:
        //   q < numParents       → smallerIsRightI32[q]  (0 if left is smaller, 1 if right is smaller)
        //   q >= numParents      → smallerIsLeftI32[q-numParents] (1 if left is smaller, 0 if right)
        auto sideIndices1D = mx::concatenate(
            {smallerIsRightI32, smallerIsLeftI32}, 0);                           // [2*numParents] int32

        // Broadcast sideIndices to [2*numParents, rowSize] for row-wise selection
        auto sideIndices = mx::broadcast_to(
            mx::reshape(sideIndices1D,
                {static_cast<int>(2 * numParents), 1}),
            {static_cast<int>(2 * numParents), rowSize});                         // [2*numParents, rowSize]

        // Tile histograms to [2*numParents, rowSize] — mx::tile is a stride-view+reshape, fusable.
        auto srcSmallTiled = mx::tile(smallRows, {2, 1});                         // [2*numParents, rowSize]
        auto srcLargeTiled = mx::tile(largeRows, {2, 1});                         // [2*numParents, rowSize]

        // Fused conditional select: select small or large histogram per output row.
        // mx::where is documented fusable at ops.h:510-514.
        auto assembled2D = mx::where(
            mx::equal(sideIndices, mx::array(0, mx::int32)),
            srcSmallTiled,
            srcLargeTiled
        );                                                                          // [2*numParents, rowSize]

        // Flatten to canonical [numPartitions * numStats * totalBinFeatures]
        auto assembledFlat = mx::reshape(assembled2D,
            {static_cast<int>(numPartitions) * rowSize});                          // ops.h:133

        return THistogramResult{
            .Histograms      = assembledFlat,
            .NumPartitions   = numPartitions,
            .NumStats        = numStats,
            .TotalBinFeatures = totalBinFeatures
        };
    }

}  // namespace NCatboostMlx
