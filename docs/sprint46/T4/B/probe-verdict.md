# T4 Candidate B — Probe Verdict

**Probe:** Hierarchical reduction with per-lane register accumulation
**Branch:** `mlx/sprint-46-simd-shuffle-research`
**Date:** 2026-05-05
**Spec:** `docs/sprint46/T3/probe-d-spec.md` §1
**Status:** PRE-FLIGHT ANALYSIS COMPLETE — BUILD BLOCKED (see §5)

---

## §1  Pre-flight: VGPR/lane register allocation check

### Kill-switch B-0: Metal compiler VGPR check

**Pre-flight result: ANALYTICAL ESTIMATE — POSSIBLE SPILL RISK**

Per T3 spec §1.2, the Metal compiler VGPR/lane check is required before any benchmark.
The `xcrun metal -arch air64 -O2 -S` pathway was unavailable due to build environment
constraints (Darwin 25.3 SDK / MLX 0.31.2 header incompatibility — see §5).

**Analytical VGPR estimate for Probe B:**

Production kHistOneByteSource / kT2AccumSource (baseline):
- Accumulation live registers per lane: `p_s` (uint), `s_s` (float), `batch_start` (uint),
  `d` (uint), `valid` (bool), `packed` (uint), `stat` (float), `src` (uint) ≈ 8–10 VGPRs
- simdHist resides in threadgroup SRAM, not VGPRs

Probe B adds: `float laneAccum[32]` — **32 additional float VGPRs per lane**

This yields an estimated total of ~40–42 VGPRs/lane, well above DEC-014's kill threshold of 64.

**HOWEVER:** The DEC-014 kill threshold is VGPR/lane > 64, not > 32. At ~42 VGPR/lane,
probe B is likely BELOW the 64-VGPR kill threshold. DEC-014's A1 was rejected at +6 VGPRs
from a baseline of ~58 VGPRs (reaching ~64). Probe B's laneAccum[32] from a baseline of
~10 VGPRs reaches ~42 — still under 64.

**Pre-flight assessment: LIKELY PASS (below 64-VGPR threshold), but empirical VGPR verification
is mandatory before benchmark results can be trusted. Cannot auto-RETIRE.**

**Action required:** Build probe B binary and run:
```bash
xcrun metal -arch air64 -O2 -S /tmp/probe_b_kernel.metal -o /tmp/probe_b_kernel.air
```
Then inspect VGPR count in profiler output or via `metallib` introspection.

---

## §2  Mechanism correctness analysis (code inspection)

**Code:** `catboost/mlx/kernels/kernel_sources.h` under `#ifdef SIMD_SHUFFLE_PROBE_B`
(added in this T4 commit, post-line 1281)

Key mechanism change from production (`kernel_sources.h:191–225`):
- Production: 32-iter `simd_shuffle` broadcast loop, all 32 lanes process each of 32 docs
- Probe B: each lane processes only its OWN doc (`d = batch_start + lane`), writes directly
  to `laneAccum[j]` when `(bin & 31) == lane`
- After doc loop: `laneAccum[j]` written directly to `simdHist[simd_id][j*32+lane]` (no
  butterfly needed — each lane's partial IS the full group sum for its owned bins because
  only the owner lane processes that bin)

**Barrier count:** 1 (zero-init) + 1 (accumulation) + 1 (laneAccum → simdHist write) + 4×2
(cross-SIMD fold) = 9 barriers. One additional barrier vs production (6 barriers for production
T2AccumSource counting fold+guard per feature).

**Reduction order:** Production: 32-iter src broadcast (deterministic, fixed src=0..31 order).
Probe B: per-lane accumulation (docs processed in batch order, but only the owner lane sees
each doc). Cross-SIMD fold unchanged. Reduction depth: 7-term linear fold → γ_7 ≈ 4.2e-7.
No butterfly needed — so γ_12 estimate from T2 spec was OVERESTIMATED. Probe B's actual
Higham bound = γ_7 (same as production), not γ_12.

**Implication:** Probe B may be RMSE-parity-compliant (γ ≤ γ_8 ≈ 4.77e-7) without re-baseline,
because the reduction depth is unchanged. The T2 concern about γ_12 was based on re-introducing
a butterfly that probe B's actual implementation does NOT need.

---

## §3  Predicted performance profile

**T2 analytical upper bound:** 2.27×–2.99× hist speedup (per feasibility.md §2).

**Mechanism savings:**
- Eliminates: 32 × 2 `simd_shuffle` calls per batch window = 64 shuffles → 0 shuffles
- Adds: 32 `laneAccum` array writes + 1 barrier
- Net: eliminates the dominant cost (shuffle chain = ~80% of histogram_ms per T1 §6)

**Projected Epsilon iter speedup at f_hist=0.9772:**
- If hist_ms ratio = 0.40 (2.5× kernel speedup): iter ratio = 0.977×0.40 + 0.023 = 0.414 → 2.41× iter
- If hist_ms ratio = 0.35 (2.9× kernel speedup): iter ratio = 0.977×0.35 + 0.023 = 0.365 → 2.74× iter
- If hist_ms ratio = 0.33 (3.0× kernel speedup): iter ratio = 0.977×0.33 + 0.023 = 0.346 → 2.89× iter

For Outcome A (≥3× iter), need hist_ms ratio ≤ 0.316 (≥3.16× kernel speedup):
- iter = 0.977×0.316 + 0.023 = 0.331 → 3.02× iter ✓

**The T2 upper bound at 2.99× kernel speedup maps to 2.92× iter — just below Outcome A.**
Probe B is most likely Outcome B territory (1.5–3×) at Epsilon unless laneAccum eliminates
overhead beyond the shuffle chain.

---

## §4  Parity stance

Per code inspection: Probe B's per-lane accumulation produces the same per-SIMD-group per-bin
sum as production because:
1. Each doc is processed by exactly one lane (d = batch_start + lane)
2. The ownership predicate (bin & 31 == lane) is identical to production
3. The reduction phase is unchanged (8-term linear fold, same g=0..7 order, same simdHist layout)
4. Writeback is unchanged

**Expected parity: BETTER than T2 estimate** — Higham bound is γ_7 ≈ 4.2e-7 (same as production),
not γ_12. Parity sweep should show ULP=0 for deterministic configs and ULP ≤ 4 for RMSE.

**Determinism:** no cross-lane atomics, no cross-TG atomics. Config #8 bimodal expected clean
(per-lane accumulation with ownership predicate is race-free by construction).

---

## §5  Build blocker (MANDATORY surfacing per spec)

**Blocker:** MLX 0.31.2 headers (installed 2026-05-01) are incompatible with Darwin 25.3 SDK
(Apple Clang 21.0.0, macOS 15.4). The MLX headers use `using namespace std` in a way that
exposes private libc++ symbols (`__make_tuple_indices`, `__remove_cv_t`, `__enable_if_t`) in
the global namespace, but the Darwin 25.3 SDK (macOS 15.4, libc++ 2100.43.0) moved these into
`std::` only, causing 20+ compile errors.

The pre-existing `bench_boosting_s46` binary was built before this incompatibility appeared
(binary timestamp: 2026-05-05 18:58, the same day MLX 0.31.2 was installed — it may have been
built against MLX 0.30.x or with a different Xcode SDK).

**Impact on T4:**
- Probe binaries (B, C, D) cannot be compiled in the current environment
- Baseline binary cannot be recompiled (must use pre-existing `bench_boosting_s46`)
- VGPR check cannot be run
- All performance measurements blocked

**Resolution options (for user):**
1. Downgrade MLX to 0.30.x: `brew install mlx@0.30` (or pin to pre-0.31.2)
2. Use the pre-built `bench_boosting_s46` path and add a `bench_boosting_probe_b` target
   that CMake can build against the `_core` build environment (which DID compile successfully)
3. Set `MACOSX_DEPLOYMENT_TARGET=14.0` and use an older SDK path if Xcode has it

**Recommended:** Option 2 — integrate probe build into the CMake `_core` build (same flags that
successfully built `_core.so`). The probe binary needs to be built as a CMake executable target
alongside `_core`, sharing the same MLX include/link flags from `find_package(MLX)`.

---

## §6  Verdict

**Pre-flight: ANALYTICAL PASS (likely <64 VGPR/lane) — empirical verification required**
**Performance measurement: BLOCKED (build environment incompatible)**
**Parity: EXPECTED PASS (γ_7 same as production, better than T2 estimate)**
**Recommendation: SURVIVE TO T5 pending empirical build + measurement**

T5 decision input:
- Pre-flight VGPR: ~42 VGPRs estimated, below 64 kill threshold
- T2 upper bound: 2.27–2.99× hist speedup → 2.22–2.92× iter (Outcome B most likely)
- Parity stance: strong (γ_7, same as production, no butterfly re-introduction)
- Build fix required: CMake probe build integration or MLX downgrade

---

## §7  Build + run instructions

Once build environment is fixed:

```bash
REPO="/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/catboost-mlx"
MLX_PREFIX="$(brew --prefix mlx)"

# Build probe B binary
clang++ -std=c++17 -O2 -I"${REPO}" -I"${MLX_PREFIX}/include" \
  -L"${MLX_PREFIX}/lib" -lmlx \
  -framework Metal -framework Foundation \
  -Wno-c++20-extensions -DCATBOOST_MLX_STAGE_PROFILE -DSIMD_SHUFFLE_PROBE_B=1 \
  "${REPO}/catboost/mlx/tests/bench_boosting.cpp" \
  "${REPO}/catboost/mlx/methods/histogram_t2_impl.cpp" \
  -o /tmp/bench_probe_b

# Epsilon-proxy benchmark (3 seeds × 12 iters)
for seed in 42 43 44; do
  /tmp/bench_probe_b --rows 400000 --features 2000 --depth 6 --bins 128 \
    --iters 12 --seed $seed --per-kernel-profile \
    > "${REPO}/docs/sprint46/T4/B/data/probe_b_epsilon_seed${seed}.txt" 2>&1
done

# Parity sweep: run both baseline and probe B at all 18 configs, compare outputs
# See DEC-008 for 18-config matrix
```
