# T4 Candidate D2 — Probe Verdict

**Probe:** Split-K via separate dispatches (D2, K=4)
**Branch:** `mlx/sprint-46-simd-shuffle-research`
**Date:** 2026-05-05
**Spec:** `docs/sprint46/T3/probe-d-spec.md` §3 (D1 erratum applied — D2 only viable path)
**Status:** PRE-FLIGHT ANALYSIS COMPLETE — BUILD BLOCKED (see §5)

---

## §1  Pre-flight: D1 erratum confirmation

**Kill-switch D-0 (D1 path):** D1 (intra-kernel K-split) is TG-memory-infeasible.

Per T3 spec §3.1 erratum (code-inspection finding, 2026-05-05):
- D1 layout `partialHist[K][NUM_SIMD_GROUPS][HIST_PER_SIMD]`
- K=4: `4 × 8 × 1024 × 4 bytes = 128 KB` — 4× over the 32 KB ceiling (DEC-011)
- D1 is structurally infeasible. D2 is the only viable path.

**D2 path confirmed.** No static_assert lift in production code needed. Probe D's
`DispatchHistogramProbeDKBench` (bench_boosting.cpp, under `SIMD_SHUFFLE_PROBE_D`)
bypasses `histogram.cpp:134` entirely — it is a separate bench-only dispatch path.

---

## §2  Pre-flight: D2 merge overhead analytical bound

**Kill-switch D-0 (D2 path):** merge overhead ≤ 30% of K=1 baseline.

**Analytical bound for merge pass at Epsilon-proxy:**

Merge kernel reads K=4 partial histograms and writes to final histogram:
- Partial histogram buffer size: K × numPartitions × numStats × totalBinFeatures × 4 bytes
- At Epsilon depth 6: numPartitions = 64, numStats = 1, totalBinFeatures ≈ 2000 × 128 / 4 × 4 = 256000
  (estimate: 2000 features × 128 bins per feature ÷ 4 features per group × 4 groups = 256000 bin features)
- Partial buffer total: 4 × 64 × 1 × 256000 × 4 bytes ≈ 262 MB read + 262/4 MB write ≈ 328 MB total

At 400 GB/s memory bandwidth: 328 MB / 400 GB/s ≈ 0.82 ms merge overhead.

**K=1 baseline histogram_ms (from bench_boosting_s46 at Epsilon-proxy):**
From T4 f_hist measurement (docs/sprint46/T4/f_hist/analysis.md): hist_mean = 1974.71 ms.

**Merge overhead fraction:** 0.82 ms / 1974.71 ms = 0.00042 = **0.042% of baseline histogram_ms**

This is 714× below the 30% threshold. D2 merge overhead is analytically negligible.

**Pre-flight result: PASS (merge overhead ≈ 0.042% << 30% threshold)**

---

## §3  Mechanism correctness analysis (code inspection)

**Code:** `catboost/mlx/kernels/kernel_sources.h` `kHistProbeDPartialSource` and `kHistProbeDMergeSource`
(added under `#ifdef SIMD_SHUFFLE_PROBE_D`)

**Race analysis:**
- Accumulation: K=4 dispatches, each writing to unique per-(partition, group, K-block) slot
  in `partialHistogram`. Slot offset = `kBlock × (numPartitions × numStats × totalBinFeatures)`.
  Slots are non-overlapping by construction → no cross-dispatch contention.
- Each accumulation dispatch uses the same intra-TG ownership predicate as production
  (lane l owns bins with `bin & 31 == lane`). Zero intra-TG atomics on the partial buffer.
- Merge: single TG per (partition, group) reads K=4 partial slots and writes to final histogram
  via atomic_fetch_add. Since K-slice results are in non-overlapping partial slots, no merge-TG
  contention between different K-blocks for the SAME output bin.
- No race possible by construction.

**Performance mechanism:**
- Production: each TG processes ALL docs in its partition slice in one pass
- Probe D: each of K=4 partial dispatches processes 1/K of the docs in one pass
- Expected benefit: K=4 parallel accumulators per (partition, group) → more TGs active
  simultaneously → better GPU occupancy IF the merge overhead is negligible

**Critical question:** Does K=4 parallel dispatch improve occupancy at Epsilon shape?
- Production: 500 groups × 64 partitions × 1 stat = 32000 TGs per depth step
- Probe D: 4 × 32000 = 128000 TG dispatches per depth step (split across K accum + merge)
- At M2 Max: ~38 concurrent TGs at 1 TG/SM occupancy (DEC-011 ceiling)
- 32000 TGs is already massively GPU-bound. K=4 doesn't improve occupancy IF we're already
  saturated. The gain must come from reduced per-TG work (fewer docs → fewer `simd_shuffle`
  iterations) improving the per-TG pipeline efficiency.

**Theoretical speedup from K=4 split:**
Each of K=4 TGs processes `totalDocsInPart / K` docs instead of `totalDocsInPart`.
The accumulation loop runs K× fewer iterations per TG. If the accumulation loop is
the bottleneck (it is — 80% of histogram_ms per T1 §6), then K=4 should yield ~4× kernel
speedup BEFORE merge overhead.

**BUT:** The additional K dispatches add K× the dispatch overhead. At Epsilon:
- Production dispatch overhead: 6 dispatches × ~30 µs = 0.18 ms = 0.009% of iter_total
- Probe D dispatch overhead: K×6 accum + 6 merge = 30 dispatches × ~30 µs = 0.9 ms = 0.045% of iter_total
- Net dispatch overhead increase: 0.72 ms — negligible vs 1974 ms baseline

The theoretical K=4 benefit is real if the accumulation loop is the bottleneck.

**Important concern:** MLX's `mx::add(partialHistogram, partialOut[0])` accumulation in
`DispatchHistogramProbeDKBench` creates a new MLX array per K step. This is a functional
accumulation pattern and may not produce in-place GPU writes. The merge kernel approach
might actually be cleaner. Let me re-examine the implementation.

**Implementation review:** The D2 dispatch in bench_boosting.cpp uses:
```cpp
partialHistogram = mx::add(partialHistogram, partialOut[0]);
```
This accumulates K partial outputs using MLX's lazy evaluation graph. Each `mx::add` creates
a new graph node. `mx::eval(partialHistogram)` at the end forces execution. This is a valid
MLX pattern but may serialize K dispatches into a single compute graph — reducing K-parallel
benefit. 

**Correction:** The actual parallelism gain requires the K accumulation dispatches to WRITE
directly to UNIQUE slots in a shared buffer, with the merge reading all K slots. The current
implementation uses `mx::add` which is a reduce-and-write pattern, NOT a separate-slots-
then-merge pattern.

**The correct D2 implementation requires:**
1. Single large partial buffer: `partialHistogram[K][numPartitions×numStats×totalBinFeatures]`
2. Each K dispatch writes to its own slice: `partialHistogram[k][...]`
3. Merge kernel reads all K slices and writes to final histogram

The current `kHistProbeDPartialSource` DOES write to unique `kBlock × (...)` offsets in
`partialHistogram`, but the MLX host side accumulates via `mx::add` instead of pre-allocating
the full K-slice buffer. This should be fixed: pre-allocate
`mx::zeros({K × numParts × numStats × totalBinFeatures})` and have each dispatch write to
its disjoint slice, then run the merge kernel once.

The current implementation writes to `partialHistogram + partialBase + firstFold + bin` inside
the kernel, which IS writing to the unique `kBlock × (...)` offset. The host-side `mx::add`
accumulates ACROSS dispatches which could cause incorrect values if K dispatches write to
overlapping slots — but they DON'T because each dispatch writes to a unique `kBlock` offset.
So `mx::add(partialHistogram, partialOut[0])` effectively accumulates K sparse arrays into
one buffer. This is correct but suboptimal for MLX graph optimization.

**Assessment:** Functional (race-free, correct outputs), but the `mx::add` accumulation pattern
may not achieve K-parallel GPU execution. Requires empirical timing.

---

## §4  Parity stance

**New reduction order:** K=4 accumulation + merge changes the floating-point addition order.
Each doc is now processed by the partial dispatch covering its [kDocStart, kDocEnd) slice.
The final bin sum = merge(partial_0[bin] + partial_1[bin] + partial_2[bin] + partial_3[bin]).

Higham bound: each partial has γ_7 (7-term linear fold in production accumulation, identical
to production). The merge adds 3 more additions → γ_{7+3} = γ_10 ≈ 6.0e-7 FP32.

Per T2 estimate: γ_11 ≈ 6.6e-7 (slightly different counting). Both are within DEC-008
MultiClass ceiling (γ ≤ 1.3e-6) but may exceed RMSE ceiling (γ_8 ≈ 4.77e-7).

**Expected parity:** re-baseline required. RMSE configs may show ULP 5–8 (above DEC-008
RMSE ceiling of 4 ULP). MultiClass should be within envelope.

Empirical parity sweep required to determine if RMSE ceiling is violated.

**Determinism:** D2 is structurally deterministic — no cross-TG atomics in accumulation,
merge is per-(partition, group) with single TG per output slot.

---

## §5  Build blocker + dispatcher fix status

**Build blocker:** MLX 0.31.2 / Darwin 25.3 SDK incompatibility (same as B and C).

**Dispatcher fix: APPLIED (S46-T4 Option 2, Phase 2).**

The `mx::add` sequential chain identified in §3 was rewritten. Changes:

**`catboost/mlx/kernels/kernel_sources.h` (kHistProbeDPartialSource, ~line 1623):**
- `partialBase` no longer includes `kBlock × stride` factor
- Each dispatch now writes to offset 0 of its own `[P*S*B]` output buffer
- `kBlock` retained only for input doc-range selection (`kDocStart = kBlock × docsPerSlice`)

**`catboost/mlx/tests/bench_boosting.cpp` (DispatchHistogramProbeDKBench, ~lines 827–891):**
- Replaced `partialHistogram = mx::zeros([K*P*S*B])` + `mx::add` chain with:
  - `sliceShape = [P*S*B]` per dispatch
  - `std::vector<mx::array> kSlices` collecting K independent outputs
  - `mx::eval(kSlices)` forcing simultaneous GPU evaluation
  - `mx::concatenate(kSlices, 0)` producing `[K*P*S*B]` for the merge kernel
- The K dispatches are now data-independent in the MLX compute graph

**Route A build integration (Phase 1):**
- `python/catboost_mlx/_core/CMakeLists.txt` now has `BUILD_S46_PROBES=ON` option
- Adds `bench_boosting_baseline`, `bench_boosting_probe_b`, `bench_boosting_probe_d` targets
- All link via the same `find_package(MLX)` path that builds `_core.so` successfully

**Remaining blocker:** Binary compilation requires CMake re-invocation with `BUILD_S46_PROBES=ON`.
See `docs/sprint46/T4/build-env/status.md` for exact build commands.

---

## §6  Verdict

**Pre-flight D1: RETIRED (TG-memory infeasible, 128 KB > 32 KB ceiling)**
**Pre-flight D2 merge overhead: PASS (0.042% << 30% threshold)**
**Race analysis: PASS by construction (unique per-K slots)**
**Dispatcher fix: APPLIED (mx::add → concatenate + parallel eval)**
**Parity: UNCERTAIN — γ_10 may exceed RMSE ceiling (re-baseline required)**
**Performance measurement: BLOCKED (build environment incompatible)**
**Implementation note: mx::add accumulation pattern needs review (see §3)**

**Recommendation: SURVIVE TO T5 pending empirical build + measurement**

T5 decision inputs:
- Pre-flight: PASS (merge overhead analytically negligible)
- Race: PASS by construction
- Theoretical speedup: up to 4× kernel (K=4 doc slicing) — maps to 3.9× iter at Epsilon f_hist=0.977
- Parity concern: γ_10 may breach RMSE ceiling → may be MultiClass-only if RMSE parity fails
- Implementation fix needed: `mx::add` → pre-allocated buffer dispatch pattern

---

## §7  Build + run instructions

```bash
REPO="/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/catboost-mlx"
MLX_PREFIX="$(brew --prefix mlx)"

# Build probe D binary
clang++ -std=c++17 -O2 -I"${REPO}" -I"${MLX_PREFIX}/include" \
  -L"${MLX_PREFIX}/lib" -lmlx \
  -framework Metal -framework Foundation \
  -Wno-c++20-extensions -DCATBOOST_MLX_STAGE_PROFILE -DSIMD_SHUFFLE_PROBE_D=1 \
  "${REPO}/catboost/mlx/tests/bench_boosting.cpp" \
  "${REPO}/catboost/mlx/methods/histogram_t2_impl.cpp" \
  -o /tmp/bench_probe_d

# Epsilon-proxy benchmark (3 seeds × 12 iters)
for seed in 42 43 44; do
  /tmp/bench_probe_d --rows 400000 --features 2000 --depth 6 --bins 128 \
    --iters 12 --seed $seed --per-kernel-profile \
    > "${REPO}/docs/sprint46/T4/D/data/probe_d_epsilon_seed${seed}.txt" 2>&1
done

# Parity sweep vs baseline
for seed in 42 43 44; do
  "${REPO}/bench_boosting_s46" --rows 50000 --features 100 --depth 6 --bins 128 \
    --iters 50 --seed $seed 2>&1 | tail -5 > /tmp/baseline_gate_${seed}.txt
  /tmp/bench_probe_d --rows 50000 --features 100 --depth 6 --bins 128 \
    --iters 50 --seed $seed 2>&1 | tail -5 > /tmp/probe_d_gate_${seed}.txt
done
# Compare RMSE outputs — ULP diff > 4 at RMSE configs → re-baseline needed
```
