#pragma once

// Histogram computation dispatch for CatBoost-MLX.
// Orchestrates Metal kernel calls to compute per-feature, per-bin gradient/weight histograms.

#include <catboost/mlx/gpu_data/mlx_data_set.h>
#include <catboost/mlx/gpu_data/gpu_structures.h>

#include <mlx/mlx.h>

namespace NCatboostMlx {
    namespace mx = mlx::core;

    // Forward declaration — TPartitionLayout is defined in structure_searcher.h.
    // histogram.h cannot include structure_searcher.h (circular dependency:
    // structure_searcher.h already includes histogram.h).
    struct TPartitionLayout;

    // Result of histogram computation: a flat buffer of gradient (and optionally weight) sums
    // per feature-bin, per partition.
    // Layout: [numPartitions, numStats, totalBinFeatures]
    struct THistogramResult {
        mx::array Histograms;   // float32
        ui32 NumPartitions;
        ui32 NumStats;
        ui32 TotalBinFeatures;
    };

    // Compute histograms for all features across all leaf partitions.
    //
    // For each partition (leaf), for each feature, for each bin:
    //   histogram[part][stat][binFeature] = sum of stat[doc] for docs in partition where feature_bin == bin
    //
    // Where stat=0 is gradient, stat=1 is weight (for weighted GBDT).
    //
    // This dispatches the appropriate Metal kernel variant based on feature encoding:
    //   - One-byte: histogram_one_byte_features (most common)
    //   - Half-byte: histogram_half_byte_features
    //   - Binary: histogram_binary_features
    THistogramResult ComputeHistograms(
        const TMLXDataSet& dataset,
        const mx::array& docIndices,         // [numDocs] uint32 — sorted doc indices by partition
        const mx::array& partitionOffsets,   // [numPartitions] uint32 — doc offset per leaf
        const mx::array& partitionSizes,     // [numPartitions] uint32 — doc count per leaf
        ui32 numPartitions,
        bool useWeights = false               // if true, compute weight histograms too (numStats=2)
    );

    // Overload accepting explicit gradient/hessian arrays (for multi-dimensional approx).
    // Allows the caller to pass a single dimension's gradient+hessian slice
    // without modifying the dataset's internal buffers.
    THistogramResult ComputeHistograms(
        const TMLXDataSet& dataset,
        const mx::array& gradients,          // [numDocs] single dimension gradient
        const mx::array& hessians,           // [numDocs] single dimension hessian
        const mx::array& docIndices,
        const mx::array& partitionOffsets,
        const mx::array& partitionSizes,
        ui32 numPartitions
    );

    // Zero-initialize a histogram buffer
    mx::array CreateZeroHistogram(ui32 numPartitions, ui32 numStats, ui32 totalBinFeatures);

    // [S49 C6] Compute histograms for SMALLER CHILDREN only, then derive LARGER CHILDREN
    // via parent-minus-sibling subtraction, and assemble the full [numPartitions × numStats × totalBinFeatures]
    // histogram in partition-index order.
    //
    // Oblivious-tree partition convention at depth d:
    //   Left child of parent p  → partition index p
    //   Right child of parent p → partition index p + numParents   (numParents = numPartitions / 2)
    //
    // Algorithm (per design §1.1–§1.6):
    //   1. Compute smaller-child mask per parent via mx::slice + mx::less (sync-free).
    //   2. Build partition-index arrays for smaller and larger sides via mx::arange + mx::where.
    //   3. Gather smallerPartOffsets / smallerPartSizes from fullLayout via mx::take.
    //   4. Dispatch ComputeHistogramsImpl over smaller children only (numParents partitions).
    //   5. Derive larger-child histogram: histLarger = histParent - histSmaller (mx::subtract).
    //   6. Assemble full [numPartitions] output via mx::where + mx::tile + mx::reshape.
    //
    // NO mx::eval() boundary in this function — all ops are lazy and fusable.
    // Sync-point inventory: zero (per §2.2). Production builds: CATBOOST_MLX_STAGE_PROFILE
    // block is the only conditional eval, gated at compile-time.
    //
    // Parameters:
    //   dataset       — training dataset (features, compressed index)
    //   gradients     — [numDocs] float32 gradient for dimension k
    //   hessians      — [numDocs] float32 hessian for dimension k
    //   fullLayout    — depth-d partition layout (all numPartitions = 2*numParents partitions)
    //   histParent    — [numParents * numStats * totalBinFeatures] float32 — parent histogram cache
    //                   (materialized as side-effect of prior depth's FindBestSplitGPU call; §2.4)
    //   numPartitions — total partitions at depth d = 2 * numParents
    //
    // Returns THistogramResult with NumPartitions = numPartitions (full shape, assembled).
    THistogramResult ComputeHistogramsSmallerChildAndAssemble(
        const TMLXDataSet& dataset,
        const mx::array& gradients,          // [numDocs] float32
        const mx::array& hessians,           // [numDocs] float32
        const TPartitionLayout& fullLayout,  // depth-d layout (all 2*numParents partitions)
        const mx::array& histParent,         // [numParents * numStats * totalBinFeatures] float32
        ui32 numPartitions                   // = 2 * numParents
    );

    // T2 histogram dispatch (Sprint 24 D0 v5 — DEC-023 complete fix).
    //
    // Single-kernel dispatch: kT2AccumSource (T1-style SIMD-shuffle accumulation for
    // ALL features 0-3 reading from docIndices).  The T2-sort kernel (kT2SortSource)
    // is no longer called; all features produce ULP=0 vs T1 by construction.
    //
    // Pre-conditions (enforced by CB_ENSURE):
    //   - maxBlocksPerPart == 1  (T2 kernels dispatch exactly one block per partition;
    //     any value > 1 would silently waste TGs — NIT-4 guard).
    //   - maxFoldCount <= 127    (DEC-016 T1 envelope).
    //
    // Parameters mirror DispatchHistogramBatched at the raw-array level.
    // Callers that have feature metadata in CatBoost types should use
    // ComputeHistograms() which builds the fold arrays and calls this function.
    mx::array DispatchHistogramT2(
        const mx::array& compressedData,     // [numDocs * lineSize] uint32, flat
        const mx::array& stats,              // [numStats * numDocs] float32
        const mx::array& docIndices,         // [numDocs] uint32
        const mx::array& partOffsets,        // [numPartitions] uint32
        const mx::array& partSizes,          // [numPartitions] uint32
        const mx::array& featureColIndices,  // [numGroups] uint32
        const mx::array& foldCountsFlat,     // [numGroups * 4] uint32
        const mx::array& firstFoldFlat,      // [numGroups * 4] uint32
        ui32 lineSize,
        ui32 maxBlocksPerPart,
        ui32 numGroups,
        ui32 numPartitions,
        ui32 numStats,
        ui32 totalBinFeatures,
        ui32 totalNumDocs,
        const mx::Shape& histShape
    );

}  // namespace NCatboostMlx
