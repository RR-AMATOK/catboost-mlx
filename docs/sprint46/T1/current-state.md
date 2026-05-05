# S46-T1 — Current-State Characterization: Histogram Accumulation Kernel

**Date:** 2026-04-30
**Branch:** `mlx/sprint-46-simd-shuffle-research`
**Owner:** @performance-engineer
**Scope:** Read-only code inspection. No production changes. Every claim cites file:line.

---

## 1. Src-broadcast chain mechanics

**Loop location:** `catboost/mlx/kernels/kernel_sources.h:209–224`

```metal
for (uint src = 0u; src < SIMD_SIZE; ++src) {       // 32 iterations
    const uint  p_s = simd_shuffle(packed, src);     // shuffle 1
    const float s_s = simd_shuffle(stat,   src);     // shuffle 2
    if ((p_s & VALID_BIT) == 0u) continue;           // uniform branch
    const uint p_clean = p_s & 0x7FFFFFFFu;

    for (uint f = 0u; f < FEATURES_PER_PACK; ++f) { // 4 inner iters
        const uint bin = (p_clean >> (24u - 8u * f)) & 0xFFu;
        if (bin < foldCountsFlat[foldBase + f] + 1u &&
            (bin & (SIMD_SIZE - 1u)) == lane) {
            simdHist[simd_id][f * BINS_PER_BYTE + bin] += s_s;
        }
    }
}
```

**Loop range:** exactly `SIMD_SIZE = 32` outer iterations per batch-doc-window
(`kernel_sources.h:28`). `FEATURES_PER_PACK = 4` inner iterations (`kernel_sources.h:29`).

**Data dependencies — serial vs parallel:**

- The outer `src` loop is **serially dependent**: each iteration broadcasts a different
  lane's `packed` and `stat` values to all 32 lanes via `simd_shuffle`. Iteration `src+1`
  cannot begin until the two shuffles of iteration `src` retire, because the shuffle
  instruction retires on the SIMD unit and the result feeds the inner predicate.
- The inner `f` loop (4 iterations) is also serial: each iteration writes to a different
  `simdHist` slot addressed by `f * BINS_PER_BYTE + bin`, and the slot address depends
  on the packed bits decoded in that same `src` iteration.
- **No independent parallelism across `src` iterations within a single SIMD group.** The
  32 `simd_shuffle` pairs execute in strict serial order within the warp.

**Why all 32 lanes execute the inner loop every `src` iteration:**

`simd_shuffle(packed, src)` broadcasts lane `src`'s value to all 32 lanes
simultaneously (`kernel_sources.h:210`). All 32 lanes then enter the `f` loop and
evaluate the ownership predicate `(bin & (SIMD_SIZE - 1u)) == lane` at line 220. The
predicate is per-lane — it does not prevent the loop body from executing for all lanes;
it gates the `simdHist` write. Every lane evaluates the predicate for every `src`
iteration; only the lane whose index matches `bin & 31` actually writes. This is the
structural cost: 32 lanes × 32 `src` iterations × 4 features = 4096 predicate
evaluations per batch window, of which at most 128 write (32 lanes × 4 features × 1
owner per feature-bin).

**Shuffle count per `src` iteration:** 2. DEC-016 (`DECISIONS.md:179–195`) dropped the
original 3 shuffles (packed, stat, valid-flag) to 2 by packing the valid flag into bit
31 of `packed` (`VALID_BIT = 0x80000000u`, `kernel_sources.h:189`). The host-side
`CB_ENSURE(maxFoldCount <= 127u)` at `histogram.cpp:167` enforces that bin values never
set bit 31, making the sentinel safe. The 2-shuffle design is confirmed directly at
`kernel_sources.h:210–211`.

---

## 2. Ownership predicate cost

**Predicate location:** `kernel_sources.h:220`

```metal
(bin & (SIMD_SIZE - 1u)) == lane
```

For `FEATURES_PER_PACK = 4`, the ownership fraction per lane per `src` iteration is
**4/32 = 12.5%** (4 features, each owned by exactly 1 of 32 lanes). This is the
theoretical write rate assuming all bins are valid.

The actual write rate is lower: the valid-bin guard `bin < foldCountsFlat[foldBase + f] + 1u`
at `kernel_sources.h:219` discards bin 0 (missing-value sentinel in CatBoost convention,
per writeback at `kernel_sources.h:272` which accesses `bin + 1u`). Gate config uses
`folds = 127` (DEC-016 envelope), so bins 1..127 are valid and bin 0 is not. On average,
`(127/256)` of the 256 bin-space is populated, reducing effective write density further.

**Register pressure of the conditional:** the two shuffles per `src` iteration produce
`p_s` (uint) and `s_s` (float) in registers. These are live through the entire 4-feature
inner loop (both feed `bin` extraction and the `simdHist` accumulate). Together with the
outer-loop state (`batch_start`, `d`, `valid`, `packed`, `stat`, `sortedPos`, `docIdx`)
this is roughly 8–10 live registers per lane during accumulation. The D1c baseline
before DEC-014's A1 proposal estimated ~9 VGPRs; A1 was rejected precisely because
doubling the slab to 64 docs pushed register allocation over the spill threshold
(DEC-014, `DECISIONS.md:136–156`).

---

## 3. Threadgroup memory utilization

**Hard ceiling:** 32 KB per Apple Silicon threadgroup — DEC-011 (`DECISIONS.md:103–112`).
This is enforced by `static_assert` at `histogram.cpp:27`:

```cpp
static_assert(kHistThreadgroupBytes <= kAppleSiliconTgLimit, ...);  // histogram.cpp:27
```

**simdHist allocation:** `kernel_sources.h:158`

```metal
threadgroup float simdHist[NUM_SIMD_GROUPS][HIST_PER_SIMD]; // 32 KB
```

**Exact calculation from defined constants** (`kernel_sources.h:28–34`):

| Constant | Value | Source |
|---|---|---|
| `SIMD_SIZE` | 32 | `kernel_sources.h:28` |
| `BLOCK_SIZE` | 256 | `kernel_sources.h:31` |
| `NUM_SIMD_GROUPS` | `BLOCK_SIZE / SIMD_SIZE = 8` | `kernel_sources.h:32` |
| `HIST_PER_SIMD` | `FEATURES_PER_PACK * BINS_PER_BYTE = 4 × 256 = 1024` | `kernel_sources.h:33` |
| `simdHist` size | `8 × 1024 × 4 B = 32,768 B = 32 KB` | `kernel_sources.h:158` |

The buffer is at the ceiling. Zero free TG memory headroom.

**Implications for S46 candidates:**

Any candidate adding threadgroup memory must do exactly one of:
- **(a) Re-tile:** change `NUM_SIMD_GROUPS` or `HIST_PER_SIMD`, which requires a full
  layout redesign and re-derivation of the zero-init and reduction phases.
- **(b) Shrink an existing buffer:** e.g., eliminating `simdHist[g != 0][*]` slots by
  changing reduction topology — but that also changes DEC-009 structure.
- **(c) Re-negotiate DEC-011:** accept `< 1 TG/SM` occupancy. DEC-011 notes this forces
  exactly 1 TG/SM; reducing occupancy further would worsen it. Requires empirical
  validation.

This constraint ruled out Candidate B in DEC-014 (`DECISIONS.md:144`: TG-memory doc
staging "BLOCKED: 35–58 KB TG memory exceeds DEC-011 32 KB ceiling").

---

## 4. Reduction phase mechanics

**Reduction loop location:** `kernel_sources.h:240–255`

```metal
for (uint tile = 0u; tile < FEATURES_PER_PACK; tile++) {   // 4 tiles
    const uint tile_base = tile * BINS_PER_BYTE;
    if (tid < BINS_PER_BYTE) {
        float sum = 0.0f;
        for (uint g = 0u; g < NUM_SIMD_GROUPS; g++) {   // 8 terms
            sum += simdHist[g][tile_base + tid];
        }
        simdHist[0][tile_base + tid] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); // 1 barrier per tile
}
```

**Cross-SIMD fold topology (DEC-009, `DECISIONS.md:82–91`):** 8-term linear sum in fixed
`g = 0..7` order. Result written to `simdHist[0][tile_base + tid]`, aliased as
`stagingHist` at `kernel_sources.h:263`. This produces a deterministic reduction order
across all runs — no atomic scatter, no data-dependent ordering.

**Reduction depth and Higham bound:** 7 addition levels (summing 8 terms). Higham
γ_7 ≈ 4.2e-7 FP32 (`kernel_sources.h:88`, `DECISIONS.md:238`). Within DEC-008 RMSE/Logloss
ulp ≤ 4 envelope.

**Barrier count — full kernel body** (verified by reading `kernel_sources.h:107–282`):

| Barrier | Location | Purpose |
|---|---|---|
| barrier 1 | `kernel_sources.h:170` | zero-init complete |
| barrier 2 | `kernel_sources.h:226` | accumulation complete |
| barriers 3–6 | `kernel_sources.h:254` (×4, 1 per tile) | cross-SIMD fold: one barrier after each tile |

**Total: 6 barriers.** This matches the comment at `kernel_sources.h:256–257`:
`"barrier 1 (zero-init) + barrier 2 (accumulation) + 4 (cross-SIMD, 1/tile × 4) = 6"`.
Verified independently by reading the source. Down from the broken-L1a kernel's 10 and
the Sprint 17 D1c kernel's 9 (DEC-012, `DECISIONS.md:163–169`).

**DEC-008 constraint on reduction depth:** any candidate that changes the number of
reduction levels must either (a) stay within γ_N ≤ ~4.2e-7 (RMSE/Logloss) or (b)
explicitly renegotiate DEC-008 with a full 18-config parity sweep. The current γ_7 is
already tight — a hierarchical reduction to 4 levels (DEC-049 Candidate B) would
loosen to γ_12 ≈ 7.2e-7 but still within the DEC-008 envelope (`DECISIONS.md:242`,
DEC-012 §"Future trigger").

---

## 5. Writeback mechanics

**Writeback loop location:** `kernel_sources.h:267–281`

```metal
for (uint f = 0u; f < FEATURES_PER_PACK; f++) {
    const uint folds = foldCountsFlat[foldBase + f];
    const uint firstFold = firstFoldIndicesFlat[foldBase + f];

    for (uint bin = tid; bin < folds; bin += BLOCK_SIZE) {
        const float val = stagingHist[f * BINS_PER_BYTE + bin + 1u];
        if (abs(val) > 1e-20f) {
            device atomic_float* dst = (device atomic_float*)(histogram + histBase + firstFold + bin);
            atomic_fetch_add_explicit(dst, val, memory_order_relaxed);
        }
    }
}
```

**Writeback mechanism:** `atomic_fetch_add_explicit` on `device atomic_float`
(`kernel_sources.h:278`). The writeback uses the `atomic_outputs=true` flag passed to
`mx::fast::metal_kernel` at `histogram.cpp:76`.

**Race gate:** the Sibling S-1 latent race — cross-threadgroup atomic-float writeback
is non-deterministic when `maxBlocksPerPart > 1`. It is gated exclusively by the
compile-time constant at `histogram.cpp:133`:

```cpp
constexpr ui32 maxBlocksPerPart = 1;
static_assert(maxBlocksPerPart == 1, ...);  // histogram.cpp:134
```

`histogram.cpp:221–224` documents this as "NIT-4" — the `static_assert` enforces the
guard at compile time, pointer to `KNOWN_BUGS.md`. `KNOWN_BUGS.md:50–52` (Sibling S-1
section) is the authoritative race record. The guard comment at `histogram.cpp:128–132`
reads: *"Do NOT raise this literal without first fixing the writeback."*

**Lifting the gate:** any S46 candidate that requires `maxBlocksPerPart > 1` (e.g.,
Candidate D's Split-K partial histograms) must fix the cross-TG writeback race before
the candidate can be shipped. The race is currently dead code; lifting the constant
reactivates it (see KNOWN_BUGS.md Sibling S-1).

**NIT-4 / KNOWN_BUGS.md citation:** NIT-4 is the code-review label for the
`maxBlocksPerPart == 1` enforcement, referenced at `histogram.cpp:221`. The authoritative
bug record is `KNOWN_BUGS.md` Sibling S-1 section (the `BUG-T2-001` entry's sibling
race subsection). No other `KNOWN_BUGS.md` entry is relevant to this section.

---

## 6. S19-01c re-attribution numbers

**Attribution measurements** (source: `docs/sprint45/T2/probe-verdict.md:115` citing
S19-01c; also DEC-049 `DECISIONS.md:2619–2621`):

| Phase | Share of accumulation_ms | Share of histogram_ms | Config |
|---|---|---|---|
| simd_shuffle src-broadcast chain | **86%** | **~80%** | 50k/RMSE/d6/128b |

These figures are from S19-01c probe-D: running the production kernel with
`compressedIndex` loads replaced by constants showed the gather cost is ~0% (DEC-019,
`DECISIONS.md:259–280`), confirming the shuffle chain as the dominant cost. The
86%/~80% split was measured at the 50k/RMSE/d6/128b gate config (DEC-008 envelope).

**Epsilon extrapolation caveat:** S19-01c measurements were taken at N=50k (the DEC-008
gate config). Epsilon has N=400k and 2000 features. DEC-049 (`DECISIONS.md:2649–2668`)
notes that `~200M batch-TG-ops on Epsilon vs ~438k on Higgs` — a 456× ratio that tracks
the cross-class wall-clock gap. However, at N=400k/2000-features, the shuffle chain's
fraction of `histogram_ms` may differ from the 80% measured at 50k/128-features if
the outer-batch iteration count changes the relative cost of accumulation vs writeback or
reduction. **T3 must measure the shuffle fraction directly on Epsilon at production
dispatch shape — the 50k figure cannot be assumed to transfer.**

---

## 7. DEC-008 envelope constraints

**Source:** `DECISIONS.md:71–81` (DEC-008)

| Loss function | Max ulp drift | Higham basis |
|---|---|---|
| RMSE | ≤ 4 ulp | γ_8 at FP32 ≈ 1e-6 relative |
| Logloss | ≤ 4 ulp | γ_8 at FP32 ≈ 1e-6 relative |
| MultiClass | ≤ 8 ulp | 3× compounding for K=3 dims |

**Scope qualifier:** bounded to `approxDim ∈ {1, 3}`, `N ≤ 50k`, 50 iterations, depth
6. Beyond this envelope, error bounds require re-validation.

**Constraint on S46 candidates:** any candidate must either (a) preserve reduction depth
≤ γ_7 ≈ 4.2e-7 (current) or at least ≤ γ_8 ≈ 4.77e-7 (the DEC-008 ulp-4 RMSE ceiling),
OR (b) explicitly renegotiate DEC-008 with a full 18-config parity sweep before committing.
No exceptions. DEC-049 Candidate B (intra-SIMD butterfly reintroduction) would push to
γ_12 ≈ 7.2e-7 — beyond the current floor but still within DEC-008's MultiClass ulp ≤ 8
ceiling (~1.3e-6). RMSE/Logloss at γ_12 would require remeasurement to verify ulp ≤ 4.

---

## 8. Per-iter dispatch context (S45 finding confirmation)

**DispatchHistogramBatched signature and grid:** `histogram.cpp:31–88`

```cpp
auto grid = std::make_tuple(
    static_cast<int>(256 * maxBlocksPerPart * numGroups),  // X
    static_cast<int>(numPartitions),                        // Y
    static_cast<int>(numStats)                              // Z
);
```

`numGroups` is baked into the X dimension of the 3D Metal dispatch grid
(`histogram.cpp:84`). All feature groups execute in a single Metal kernel invocation.
`DispatchHistogramBatched` is called once per `ComputeHistograms` invocation
(`histogram.cpp:187`).

**Dispatch loop in SearchTreeStructure:** `structure_searcher.cpp:60–108`

```cpp
for (ui32 depth = 0; depth < maxDepth; ++depth) {          // maxDepth = 6
    for (ui32 k = 0; k < approxDimension; ++k) {            // approxDim = 1 (binary)
        ComputeHistograms(...)                               // → 1 Metal dispatch
    }
}
```

**Dispatch count verification:**

| Dataset | n_features | numGroups = ceil(n_features/4) | maxDepth | approxDim | Dispatches/iter |
|---|---|---|---|---|---|
| Epsilon | 2000 | 500 | 6 | 1 | **6** |
| Higgs-1M | 28 | 7 | 6 | 1 | **6** |

6 dispatches/iter for both — confirmed identical, regardless of feature count. This
matches DEC-048 (`DECISIONS.md:2522–2526`) and the S45-T2 probe verdict
(`docs/sprint45/T2/probe-verdict.md:44–48`). Dispatch overhead = 6 × ~30 µs = 0.18 ms
= **0.008%** of Epsilon's 2241 ms/iter (`probe-verdict.md:68–75`). This is a sanity
check, not a novel claim — DEC-048 is the authoritative finding.

---

## Appendix: constant table

All constants defined at `kernel_sources.h:28–34`:

| Constant | Value | Expression |
|---|---|---|
| `SIMD_SIZE` | 32 | literal |
| `FEATURES_PER_PACK` | 4 | literal |
| `BINS_PER_BYTE` | 256 | literal |
| `BLOCK_SIZE` | 256 | literal |
| `NUM_SIMD_GROUPS` | 8 | `BLOCK_SIZE / SIMD_SIZE` |
| `HIST_PER_SIMD` | 1024 | `FEATURES_PER_PACK × BINS_PER_BYTE` |
| `TOTAL_HIST_SIZE` | 8192 | `NUM_SIMD_GROUPS × HIST_PER_SIMD` |
| `simdHist` bytes | 32,768 (32 KB) | `TOTAL_HIST_SIZE × 4 B` |
