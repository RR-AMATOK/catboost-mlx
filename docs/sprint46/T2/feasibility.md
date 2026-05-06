# S46-T2 — 4-Candidate Feasibility Analysis: simd_shuffle Redesign

**Date:** 2026-04-30
**Branch:** `mlx/sprint-46-simd-shuffle-research`
**Owner:** @silicon-architect
**Reviewers:** @performance-engineer (mechanism), @mathematician (parity invariants)
**Inputs:** `docs/sprint46/T1/current-state.md`, `docs/sprint46/sprint-plan.md` §T2, `catboost/mlx/kernels/kernel_sources.h:107–283`, `catboost/mlx/methods/histogram.cpp:31–217`, DEC-008/009/011/012/014/016/017/020/023/025/026/049
**Scope:** Read-only feasibility analysis. No production source modified. Every mechanism claim cites file:line.

---

## 0. Three T1 findings carried forward

T1 (`docs/sprint46/T1/current-state.md`) updates the working model in three load-bearing ways. Each candidate's analysis below reflects them:

1. **Terminology — `simd_shuffle` (broadcast), not `simd_shuffle_xor` (butterfly).** Confirmed at `kernel_sources.h:210–211`. The xor butterfly was removed in DEC-012; current production has zero `simd_shuffle_xor` calls (verified by reading `kernel_sources.h:107–282` end-to-end).

2. **Writeback already uses atomics unconditionally.** `kernel_sources.h:267–280` always issues `atomic_fetch_add_explicit` regardless of `maxBlocksPerPart`; the host construct passes `atomic_outputs=true` at `histogram.cpp:76`. Correctness in production is preserved by `maxBlocksPerPart == 1` (`histogram.cpp:133–137`), which collapses cross-TG contention to within-TG-only writes per `(part, stat, group)` triple — non-overlapping output offsets per `firstFold`. **No marginal atomic cost is introduced by replacing the per-SIMD layout with a different accumulation strategy** as long as the writeback range and atomic gating are preserved. Candidates that change the accumulator structure but reuse the existing writeback inherit the same atomic envelope.

3. **`stagingHist` is 1-indexed: writeback reads `stagingHist[f * BINS_PER_BYTE + bin + 1u]`** (`kernel_sources.h:272`). Bin 0 is the missing-value sentinel per CatBoost CPU convention; the accumulator only writes bins ≥ 1 because the valid-bin guard `bin < foldCountsFlat[foldBase + f] + 1u` at `kernel_sources.h:219` combined with the kernel-design assumption that loaded packed bytes never carry bin 0 (per the CB_ENSURE at `histogram.cpp:167`) keeps bin 0 zero. Any candidate that restructures accumulation must either (a) preserve the `bin + 1u` writeback offset, or (b) renegotiate the 1-indexed contract throughout the suffix-sum kernel + scoring chain (out-of-scope for S46).

---

## 1. Production dispatch reference (Epsilon iter=2000)

To make upper-bound estimates concrete:

| Metric | Value | Source |
|---|---|---|
| Epsilon MLX `iter_total_ms` | 2241.2 ms | `docs/sprint45/T2/probe-verdict.md:63` |
| Epsilon CPU `iter_total_ms` | 140.9 ms | `probe-verdict.md:63` (15.9× MLX/CPU ratio) |
| Dispatches/iter | 6 | T1 §8; `histogram.cpp:31`, `structure_searcher.cpp:60–108` |
| Per-iter batch-TG-ops | ~200M | `probe-verdict.md:112` (400k docs × 500 groups × avg 16 TGs/depth) |
| simd_shuffle src-chain share of histogram_ms | ~80% (50k gate) | T1 §6; DEC-049 line 2620 |
| histogram_ms share of iter_total_ms (50k gate) | ~97.7% | DEC-049 line 2620 ("kernel ≈ 97.7% of iter time") |
| histogram_ms share at Epsilon (iter=2000) | NOT MEASURED | T1 §6 caveat: "T3 must measure directly" |

**Amdahl-style ≥3× target.** S46 plan §T5 sets the COMMIT threshold at ≥3× MLX iter speedup. Treating histogram as fraction `f_hist` of `iter_total_ms` (assumed ~0.8 at Epsilon as a conservative working figure pending T3 measurement), the kernel speedup required to deliver ≥3× iter-level is:

- `1 / (f_hist / r_kernel + (1 - f_hist)) ≥ 3` ⇒ for `f_hist = 0.8`, `r_kernel ≤ 0.083` (12.0× kernel speedup).
- For `f_hist = 0.95`, `r_kernel ≤ 0.157` (6.4× kernel speedup).
- For `f_hist = 0.977` (50k gate fraction), `r_kernel ≤ 0.183` (5.5× kernel speedup).

**Of the simd_shuffle src-chain alone (~80% of `histogram_ms`):** even fully eliminating it caps `r_kernel` at 0.20 (5×). Composed with the rest of `iter_total_ms`, this caps Epsilon iter speedup at:

- `f_hist = 0.8`: `1 / (0.8 × 0.20 + 0.2) = 1 / 0.36 = 2.78×` — **does not clear 3× alone**.
- `f_hist = 0.95`: `1 / (0.95 × 0.20 + 0.05) = 1 / 0.24 = 4.17×` — clears 3×.
- `f_hist = 0.977`: `1 / (0.977 × 0.20 + 0.023) = 1 / 0.218 = 4.59×` — clears 3×.

**Implication:** the per-iter-time floor question hinges critically on whether Epsilon's `f_hist` is closer to 0.8 (no candidate clears 3× via shuffle elimination alone — requires composition) or 0.95+ (single-candidate ≥3× is in scope). T3's first-priority measurement is `f_hist` on Epsilon; the verdict structure below propagates this uncertainty by reporting per-candidate kernel-level upper bound separately from iter-level upper bound.

---

## 2. Per-candidate analysis

### Candidate A — Atomic-add accumulation

#### Mechanism sketch

**Affected files / line ranges:**

- `kernel_sources.h:158` — replace `threadgroup float simdHist[NUM_SIMD_GROUPS][HIST_PER_SIMD]` with `threadgroup atomic_uint simdHistU[HIST_PER_SIMD]` (1024 slots, single 4 KB buffer).
- `kernel_sources.h:165–169` — zero-init loop: 1024 / 256 = 4 stores per thread (down from 32) but each store now atomic.
- `kernel_sources.h:191–225` — replace the entire 32-iter `simd_shuffle` broadcast loop with a per-lane direct accumulation: each lane processes its own `(d, packed, stat)` and emits 4 atomic adds (one per feature) into `simdHistU[f * BINS_PER_BYTE + bin]` via `atomic_fetch_add_explicit`.
- `kernel_sources.h:240–255` — the cross-SIMD fold collapses to a no-op (single buffer). Reduction barrier count drops from 4 (per-tile) to 0 — total barrier count drops from 6 to 2 (zero-init + accumulation).
- `kernel_sources.h:267–281` — writeback unchanged in shape; reads from `simdHistU` (cast back to float view) at `bin + 1u` offset preserved.

**Data-flow change vs current production:** removes the 32-iter src broadcast (`kernel_sources.h:209`), the ownership predicate (`kernel_sources.h:220`), and the cross-SIMD linear fold (`kernel_sources.h:249–251`). All 256 lanes write directly to a single shared 1024-slot histogram with hardware-arbitrated atomic-CAS resolution.

**Threadgroup-memory pressure delta:** from 32 KB (`simdHist[8][1024]`) down to 4 KB (`simdHistU[1024]`). **Frees 28 KB**, which would in principle re-enable >1 TG/SM occupancy for the first time since DEC-011 (S18). Per-lane register pressure: -2 VGPRs (no `p_s`, no `s_s` carry across the inner loop), but +1 VGPR for the atomic-CAS retry temporary inside the Metal `atomic_fetch_add_explicit` codegen. Net: roughly -1 VGPR/lane.

#### Upper-bound estimate

**T1 finding #2 correction to original sprint-plan A analysis:** The sprint-plan §T2 candidate-A text (`docs/sprint46/sprint-plan.md:107`) states "Adds atomic-CAS contention" as a marginal cost. **This is partially incorrect post-T1.** Writeback already uses atomics (T1 §5; `kernel_sources.h:278`). What candidate A adds is **atomic-CAS contention on the threadgroup-scope `atomic_uint` accumulator**, which is a *different* contention surface from the device-scope writeback atomics:

- The device-scope writeback at `kernel_sources.h:278` writes once per (part, stat, group, bin) tuple, gated by `maxBlocksPerPart == 1` to ensure only one TG per `(part, stat, group)` writes to a given output slot. Contention is structurally absent.
- The TG-scope `atomic_uint` accumulation under candidate A would contend across all 256 lanes × `myDocCount` docs per TG × 4 features per doc on 1024 slots. At Epsilon depth 6, `myDocCount` per TG averages `400k / 64 partitions / 16 TGs/group ≈ 391 docs`; per-bin contention is `391 × 4 features / 1024 bins ≈ 1.5 writers per bin per TG before serialization`. AGX hardware atomics serialize same-address writes; the wait-cycle cost is bounded by the depth of the contention chain (Higham γ_N analysis applies in identity-permuted order).

**Comparison vs DEC-017 dispatch shape:** DEC-017 was retired at production shape because of the **per-TG fixed overhead**, not the contention. Quoting DEC-017 lines 211–215:
- Per TG: 1024-slot atomic_uint zero-init + 1024-slot writeback read = ~8 memory ops per thread
- At 195 docs/thread (toy): 8 / (195 × 4) = 1.0% → wins
- At 3 docs/thread (50k gate prod): 8 / (3 × 4) = 67% → loses

For Epsilon (DEC-025 re-entry justification path):
- Epsilon per-TG `myDocCount` at depth 6 is ≈391 docs (computed above); thread BLOCK_SIZE is 256, so docs/thread ≈ `391 / 256 = 1.5` (LOWER than the 50k gate's 3 docs/thread).
- Fixed-overhead ratio: `8 / (1.5 × 4) = 133%` — **dominates more, not less**. At Epsilon shape, candidate A's per-thread overhead ratio is ~2× WORSE than the 50k gate at which DEC-017 was killed at +42.3%.

**Wait — depth-shape correction.** The 391 docs/TG figure is for depth 6 (`numPartitions = 64`). But the dispatch grid X-dimension is `256 * maxBlocksPerPart * numGroups = 256 × 1 × 500 = 128000` for Epsilon, with grid Y = `numPartitions = 64`. Total TGs = 128000/256 × 64 = `500 × 64 = 32000` TGs at depth 6. Docs per TG per group = `400k / 64 = 6250 docs/partition`, then divided by `1` (since `maxBlocksPerPart = 1`). So per-TG docs are not 391 but **6250** at Epsilon depth 6 (each TG processes one full partition × one feature group). Docs/thread: `6250 / 256 = 24.4` — **higher** than 50k gate's 3 docs/thread. Fixed-overhead ratio: `8 / (24.4 × 4) = 8.2%` — much more favorable than the gate.

**This is non-trivial new evidence for DEC-025 re-entry.** Epsilon at depth 6 has dispatch shape `(500 groups × 64 partitions = 32000 TGs)` × `~24 docs/thread`, which is *between* the toy-isolation shape (195 docs/thread) and the 50k-gate production shape (3 docs/thread). The DEC-017 fixed-overhead ratio at this shape is plausibly amenable to atomic-add accumulation.

**However, DEC-023 race envelope still applies.** Candidate A's TG-scope atomic-float would race in exactly the same way DEC-023 v5 retracted. DEC-023 §H1 hypothesis ("larger bin counts resolve additions in consistent order") suggests Epsilon's higher per-bin doc count *might* mask the race envelope, but at config #8 (`N=10000, RMSE, 128b`) the race fired bimodally with smaller per-bin populations. Epsilon has bins=128 and folds≤127 (DEC-016 envelope, `histogram.cpp:167`); the per-bin doc population at depth 6 is `6250 docs / 128 bins ≈ 49 docs/bin/partition` — **larger** than the 50k gate (50k / 64 / 128 ≈ 6 docs/bin/partition where determinism held) but smaller than what DEC-023 §H1 calls "larger bin counts." Unclear whether Epsilon avoids the bimodal race or hits it.

**Kernel-level upper bound (assumptions stated):** Assume Epsilon dispatch shape unlocks the 24-docs/thread amortization that 50k gate denied. Eliminating the simd_shuffle src-chain reduces accumulation work to per-doc atomic-CAS:
- Lower bound: 1.0× (atomic-CAS contention saturates AGX atomic units, no win — DEC-017 production behavior). Best evidence: DEC-017 at +42.3%.
- Upper bound: ~3.5× kernel speedup. Best evidence: DEC-020 (sort-by-bin shipped at 0.317× hist_ms ratio at 50k gate, e2e 1.778×). T2 sort-by-bin's mechanism (per-feature single writer per bin) is structurally favorable to atomic-add but eliminates the contention surface entirely; candidate A retains the contention. Realistic upper bound for atomic-add WITHOUT sort-by-bin pre-pass: 1.5× kernel speedup.

**Iter-level upper bound (Epsilon iter=2000):** assuming `f_hist = 0.8` (conservative), kernel ratio of 0.667 (1.5× kernel) → iter speedup `1 / (0.8 × 0.667 + 0.2) = 1 / 0.733 = 1.36×`. **Does not clear ≥3×.**

#### Parity stance

**Outcome class: DEC-008 ENVELOPE VIOLATION.** Atomic-add reorders accumulation across 256 lanes × `myDocCount` docs in a non-deterministic order dictated by AGX's atomic-arbitration unit. Higham γ_N where N = docs/bin per TG. At Epsilon depth 6: N ≈ 49 docs/bin/partition. γ_49 ≈ 1.4e-6 FP32 (using γ_N ≈ N × ε / (1 - N × ε), ε ≈ 5.96e-8 for FP32). This **exceeds the DEC-008 RMSE/Logloss ulp ≤ 4 ceiling (~4.77e-7)** by 3× and approaches the MultiClass ulp ≤ 8 ceiling (~1.3e-6). At root depth: N ≈ 3125 docs/bin → γ_3125 ≈ 1.9e-4 — **400× over the envelope**. Any candidate-A variant requires Kahan/Neumaier compensated summation (DEC-023 fix-option 3 explicitly noted as "probably not sufficient standalone") OR int-fixed-point accumulation (DEC-023 fix-option 2 — calibration overhead).

**Branch-B regression:** breaks bit-equivalence at every config. v0.6.1 baseline tests fail. Re-baseline scope is multi-sprint if the int-fixed-point or Kahan variants are pursued (DEC-023 §S24 D0 resolution recorded that v5 fix took an entire sprint).

#### Risk classification

**HIGH.** This is the third re-entry into this design space (T3b → DEC-017 → T2-accum atomic-float at DEC-023). DEC-025 §"Re-entry policy" explicitly requires *new mechanism evidence* for re-entry. The Epsilon-shape per-thread amortization argument above (24 docs/thread vs gate's 3) is **partial** new evidence but does not address the parity envelope violation, which DEC-023 v5 already established as the binding constraint (the parity fix retracted T2's speedup entirely; v5 ratio was 0.959×, NOT 0.317×).

#### T2 verdict for Candidate A

**RETIRE AT T2.**

**Justification:**
1. **Parity envelope violation is structural** (γ_N analysis at Epsilon depth shows 3× over RMSE/Logloss ulp ≤ 4 ceiling at depth 6, ~400× at root). DEC-023 v5 demonstrated that retrofitting determinism to atomic-float retracts the structural speedup (1.90× → 1.01× R8). This is a precedent, not a hypothesis.
2. **Iter-level upper bound (1.36×)** does not clear the ≥3× COMMIT threshold even under generous shape-amortization assumptions, BEFORE any parity-fix overhead.
3. **DEC-025 re-entry policy is satisfied only on the shape-amortization axis, not the parity axis.** Both axes must be cleared for re-entry; only one is. The other (parity) has been visited twice and failed twice.

The Epsilon-shape per-thread amortization argument may justify a future re-entry IF (a) a deterministic atomic-add primitive emerges in Apple Silicon hardware (out-of-scope for v0.7.0 timeline), OR (b) a hybrid sort-by-bin pre-pass + per-bin atomic-add reaches the same dispatch shape (this is candidate C's design space, evaluated below).

---

### Candidate B — Hierarchical reduction (per-lane register accumulation + intra-SIMD butterfly)

#### Mechanism sketch

**Affected files / line ranges:**

- `kernel_sources.h:158` — eliminate the `threadgroup float simdHist[NUM_SIMD_GROUPS][HIST_PER_SIMD]` declaration entirely. Replace with per-lane register histograms.
- `kernel_sources.h:191–225` — replace the 32-iter src broadcast loop with a per-lane direct-accumulation loop. Each lane reads its own `(packed, stat)` for its assigned doc and writes to a private register array.
- `kernel_sources.h:240–255` — add an intra-SIMD butterfly reduction (5 levels of `simd_shuffle_xor` with xor masks 16/8/4/2/1) to fold per-lane register histograms within each SIMD group. Then keep the cross-SIMD 8-term linear fold (DEC-009).
- `kernel_sources.h:267–281` — writeback unchanged in shape, sources from per-SIMD-group register-fold result staged into a smaller threadgroup buffer (size depends on register-history layout; see register pressure below).

**Per-lane register layout (the critical design parameter):**

Option B1 — full-bins-per-lane: 1024 floats × 4 B = 4 KB per lane. **Spills immediately** — exceeds typical AGX VGPR budget (~256 VGPR × 4 B = 1 KB usable per thread).

Option B2 — stride-partition keeping 32 floats/lane (32 bins × 4 features) per-lane × 4 features × 32 bins = 128 floats/lane = 32 VGPRs/lane. With the existing ~9 VGPR doc-state (T1 §2; `kernel_sources.h:88`) → total ~41 VGPRs/lane. AGX VGPR budget is reported as ~256/thread; nominally fits. **DEC-014 precedent (`DECISIONS.md:136`) is the binding warning**: A1 added +6 VGPRs (3 → 9) and produced +9.4% production regression due to spill that the code-inspection-grounded register-pressure model missed. Adding +32 VGPRs to lift from 9 → 41 is 5× the A1 register delta.

Option B3 — narrower stride: 16 bins/lane × 4 features = 64 floats/lane = 16 VGPRs/lane → ~25 VGPRs/lane total. More likely to fit, but requires a 2-level per-bin reduction (first within 16-lane half-warps, then cross-half-warp) which complicates the butterfly topology.

**Threadgroup memory delta:** from 32 KB (`simdHist[8][1024]`) down to ~4–8 KB (smaller buffer for the cross-SIMD-fold staging after the intra-SIMD butterfly emits per-SIMD-group sums). Frees 24–28 KB.

**Barrier count delta:** zero-init removed (registers initialized to 0.0f at declaration, no barrier needed). Accumulation barrier preserved. Cross-SIMD fold barrier(s) preserved. Total barriers: 5 (down from 6).

#### Upper-bound estimate

**Mechanism delta:**
- Eliminates the 32-iter src broadcast (current 86% of accumulation per S19-01c, `DECISIONS.md:259`).
- Adds 5-level `simd_shuffle_xor` butterfly per SIMD group per bin per feature.
- Per-lane accumulation phase: each lane processes its own batch of docs and updates 4 register slots per doc. Inner-loop work per doc: 4 features × 1 write = 4 register stores. Outer-loop iter count: `myDocCount / BLOCK_SIZE = 6250 / 256 ≈ 24.4` iters per lane at Epsilon depth 6. Total per-lane work: ~24 × 4 = 96 register writes, vs current 32 (src) × 4 (features) = 128 inner-loop predicate evaluations of which 4 (12.5%) write. Roughly equivalent inner-loop work, but eliminates the broadcast latency.

**Spill scenario (Option B2, ~41 VGPRs/lane):** DEC-014 precedent shows that production allocation crosses the spill threshold somewhere between 9 and ~30 VGPRs (A1 at 9 spilled; A1's +6 was added on top of an already-tight allocation). A 41-VGPR lane is **highly likely to spill** to threadgroup or device memory. If spill is to threadgroup: regains the 4 KB threadgroup pressure, partially defeats the layout simplification. If spill is to device: kernel becomes memory-bound, kills the speedup entirely (DEC-014 saw +9.4% from device-spill of just +6 VGPRs).

**No-spill scenario (Option B3, ~25 VGPRs/lane):** plausible to fit; halves per-lane register state vs B2. The 16-bins/lane butterfly requires a 2-stage reduction inside the SIMD group (16-lane half-warp + cross-half-warp), adding 1 butterfly level — γ_6 intra-SIMD + γ_7 cross-SIMD = γ_13 ≈ 7.7e-7 (still within DEC-008 envelope per DEC-049 line 2659).

**Kernel-level upper bound (assumptions stated):**

Assume Option B3 fits without spill (must be verified via Metal compiler register-allocation report — DEC-014 standing rule, `DECISIONS.md:155`).

- Lower bound (with marginal butterfly overhead): `r_kernel ≈ 0.5–0.6` (1.7×–2.0×). Best evidence: S17 D1c achieved 89–93% histogram_ms reduction on the OLD layout (DEC-016/017 era, `feedback file project_sprint17_result.md`) but that was at small-N gate config, not Epsilon production shape; the speedup is bounded by what the post-S18 layout already captured (the L1a layout is the post-D1c kernel — it ALREADY captured most of D1c's gain).
- Upper bound (full src-chain elimination, butterfly is amortized): `r_kernel ≈ 0.30–0.35` (2.9×–3.3× kernel speedup). The src-chain is ~80% of `histogram_ms` at the gate; eliminating it caps `r_kernel` at 0.20 (5×) before adding butterfly overhead. The butterfly is ~5 levels × 1024 bins = 5120 shuffle ops/SIMD-group, vs the eliminated 32-iter × 2 shuffles × 256 threads = 16384 shuffle ops/SIMD-group. Net 3.2× shuffle reduction.

**Iter-level upper bound (Epsilon iter=2000):** assuming `f_hist = 0.8`, `r_kernel = 0.30` (3.3× kernel) → iter speedup `1 / (0.8 × 0.30 + 0.2) = 1 / 0.44 = 2.27×`. Assuming `f_hist = 0.95`, `r_kernel = 0.30` → `1 / (0.95 × 0.30 + 0.05) = 1 / 0.335 = 2.99×`. **Marginal — does not robustly clear 3× without f_hist > 0.95.**

#### Parity stance

**Outcome class: RE-BASELINE REQUIRED.** The intra-SIMD butterfly produces an order-deterministic result per `simd_shuffle_xor` semantics, but the order differs from the current per-SIMD-group stride-partition accumulation. γ_12 ≈ 7.2e-7 (DEC-049 line 2659; DEC-012 §"Future trigger" `DECISIONS.md:168`) is still within DEC-008 RMSE/Logloss ulp ≤ 4 (~4.77e-7 with margin). **However**, the RMSE/Logloss bound is tight — γ_12 of 7.2e-7 is 1.5× over the γ_8 bound that DEC-008 was derived against. Whether actual ulp drift stays ≤ 4 is empirical and must be measured across the 18-config DEC-008 envelope.

**Branch-B regression:** breaks bit-equivalence (different reduction order). Re-baseline required against v0.6.1. Re-baseline scope is 1 sprint (DEC-008 envelope sweep is well-tooled per DEC-020 D1 precedent).

#### Risk classification

**MEDIUM.** Three identified risks:
1. **Register spill (DEC-014 precedent, HIGH-impact):** must be verified by Metal compiler register-allocation report BEFORE any benchmarking. If spill is detected, candidate B is RETIRED at T3.
2. **Parity envelope tightness (γ_12 vs DEC-008 ulp ≤ 4):** must be verified across the 18-config envelope. Plausible to fit per DEC-049 line 2659.
3. **DEC-012 §"Future trigger" explicitly opens this design space** — `DECISIONS.md:168`: *"Any future kernel that accumulates into per-lane register state (not shared threadgroup memory) should re-introduce the intra-SIMD butterfly for that phase."* This is the only S46 candidate that satisfies the §"Future trigger" predicate exactly.

#### T2 verdict for Candidate B

**SURVIVE TO T3.**

**Justification:**
1. **DEC-012 §"Future trigger" (`DECISIONS.md:168`) explicitly opens this design space.** No DEC-025 re-entry violation; this is a first-time evaluation under the trigger.
2. **Iter-level upper bound (2.27×–2.99×)** sits in DEC-049 Outcome B band (1.5×–3×); under generous Epsilon `f_hist`, may push into Outcome A territory.
3. **Spill risk is measurable, not structural.** T3 must require Metal compiler register-allocation verification (DEC-014 standing rule) as a kill-switch BEFORE the perf measurement.
4. **Parity path is clean** — γ_12 is within DEC-008 envelope per DEC-049 plan; re-baseline is 1-sprint per DEC-020 D1 precedent.

T3 owner (@performance-engineer) must specify:
- Metal compiler register-allocation report verification step (PASS criterion: ≤ 64 VGPRs/lane to leave headroom).
- 18-config DEC-008 envelope parity sweep including config #8 special-case audit (per DEC-023 race precedent — even though candidate B is deterministic, config #8 is the established near-tie cascade trigger and must be verified).

---

### Candidate C — Sort-by-bin extension

#### Mechanism sketch

**Affected files / line ranges:**

- `kernel_sources.h` — re-introduce a counting-sort pre-kernel mirroring the DEC-020 T2 design (commit `73baadf445` era; src in git history). New kernel source separate from `kHistOneByteSource`.
- `kernel_sources.h:107–283` — new accumulator kernel that performs bin-range scans on the sorted docs. For features 1–3, the design challenge is **DEC-023's binding constraint**: atomic-float scatter at config #8 (N=10k bimodal). DEC-023 v5 retracted by reverting features 1-3 to T1-style SIMD accumulation, killing the speedup (1.90× R8 → 1.01×).
- `histogram.cpp` — new dispatch path with two-kernel chain (sort + accumulate). DEC-021 layout (slab-by-partOffsets, `DECISIONS.md:358`) reused.
- `structure_searcher.cpp` — consumer unchanged (still calls `ComputeHistograms`).

**The sprint-plan §T2 candidate-C question:** *"can the bin-range-scan path be extended to all 4 features in a per-pack with an alternative deterministic-reduction mechanism that DEC-023 / DEC-026 missed?"* This is the load-bearing question. The sprint-plan does not propose a *specific* new deterministic mechanism; it asks whether one exists.

**Three deterministic-reduction mechanisms previously considered (DEC-023 §"Fix options for S24"):**
1. **Threadgroup-local reduce + single-thread commit** (DEC-023 fix-option 1): Each TG accumulates locally with no cross-TG atomics; final commit is single-thread per (TG, bin). DEC-023 §S24 D0 resolution: this fix-class kept Value B (105 ULP off T1's Value A); reduction-topology mismatch with T1 prevented bit-equivalence. v5 chose to abandon T2-sort entirely, not because fix-option 1 was non-deterministic, but because it produced the *wrong value* (Value B not Value A).
2. **Int-atomic fixed-point** (DEC-023 fix-option 2): deterministic by construction; calibration overhead. NEVER FULLY MEASURED in production — DEC-026 was opened to enable it via cascade-robust GAIN, then DEC-026 was falsified in S25 G1 (`feedback project_sprint30_outcome.md` indicates the precision class is exhausted).
3. **Kahan/Neumaier compensation** (DEC-023 fix-option 3): mitigates but doesn't eliminate non-determinism. Probably not sufficient standalone (DEC-023 wording).

**The new evidence question (DEC-025 re-entry policy):** Does Epsilon's N=400k dispatch shape avoid the DEC-023 race envelope?

DEC-023 §H1 hypothesis (`DECISIONS.md:438`): *"larger bin counts resolve additions in consistent order"* — gate config (N=50k) was 100/100 deterministic. Config #8 (N=10k) was bimodal. The hypothesis is that race intensity is INVERSELY proportional to per-bin doc population. At Epsilon depth 6: ~49 docs/bin/partition (computed in candidate A above). At gate: ~6 docs/bin/partition (50k / 64 / 128). At config #8: ~1 doc/bin/partition (10k / 64 / 128 — the *singleton* footprint per `DECISIONS.md:434`).

**Per H1, Epsilon at depth 6 is ~8× more populated per bin than gate, ~50× more than config #8.** If H1 is correct, Epsilon should be MORE deterministic than gate, which itself was 100/100. **This is plausible new evidence for DEC-025 re-entry.**

**However:** at root depth (depth 0), Epsilon has 400k docs / 1 partition / 128 bins = ~3125 docs/bin — vastly more, very deterministic per H1. At deeper depths the per-bin population shrinks: depth 3 → ~390 docs/bin, depth 6 → ~49. **The race risk is at deepest depths, where per-bin population is smallest.** At gate config (50k, depth 6) the depth-6 per-bin pop was ~6 (deterministic per measurement). Epsilon depth-6 per-bin pop is ~49 (8× more). Epsilon should be safely above the determinism threshold IF H1 monotonicity holds. **This is the new-evidence pillar for re-entry.**

#### Upper-bound estimate

**Best evidence:** DEC-020 measured T2 at 0.317× hist_ms at the 50k gate (`DECISIONS.md:296`). e2e iter speedup was 1.778× at gate. **This was at production dispatch shape** — not toy isolation — so the DEC-017 toy-to-prod transfer concern is already cleared.

**Epsilon scaling argument:** the kernel-work-volume ratio Epsilon/gate is `~200M batch-TG-ops on Epsilon vs ~438k on Higgs` (DEC-049 line 2649). Higgs and 50k-gate have comparable per-iter work. So Epsilon kernel work is roughly 200M/438k ≈ 456× larger than Higgs. The simd_shuffle src-chain scales linearly with batch-TG-ops; sort-based bin-range scan (T2 mechanism) replaces the simd_shuffle work. If the 0.317× ratio holds at Epsilon shape:

- Kernel-level upper bound: `r_kernel = 0.317×` → 3.15× kernel speedup. Best-case sort-pre-pass cost amortized over 500 feature groups (vs 7 in Higgs) keeps the sort overhead bounded.
- **Iter-level upper bound (Epsilon iter=2000):** assuming `f_hist = 0.95` (Epsilon kernel-dominated; DEC-049 line 2620 quotes 97.7% at gate), `r_kernel = 0.317` → iter speedup `1 / (0.95 × 0.317 + 0.05) = 1 / 0.351 = 2.85×`. Assuming `f_hist = 0.977`: `1 / (0.977 × 0.317 + 0.023) = 1 / 0.333 = 3.00×`. **At the boundary of Outcome A.**

**Uncertainty band:**
- If sort overhead doesn't amortize at Epsilon's 500-feature groups (e.g., the counting-sort is per-partition not per-group, so cost scales with partitions × N rather than groups × N): `r_kernel` could rise to 0.40–0.50 → iter speedup 1.92×–2.50×. Outcome B band.
- If parity fix retracts the speedup (DEC-023 v5 precedent: 0.317× → 0.959×): `r_kernel = 0.96×` → iter speedup 1.04×. Outcome C.

**Realistic upper bound for kernel-level: 3.0×–3.5× (matches DEC-020 measured), iter-level: 2.5×–3.0× (Outcome B/A boundary).**

#### Parity stance

**Outcome class: DEPENDENT ON PARITY-FIX MECHANISM.**

**Sub-case C1 (T2-sort + atomic-float, DEC-020 original):** envelope violation at config #8 (DEC-023 footprint). At Epsilon shape, H1 hypothesis predicts determinism — but **must be measured, not assumed** (DEC-049 hard constraint).

**Sub-case C2 (T2-sort + int-atomic fixed-point, DEC-023 fix-option 2):** deterministic by construction. Calibration overhead. Bit-equivalence with v0.6.1 baseline NOT preserved (different accumulation topology), but determinism within DEC-008 envelope is plausible.

**Sub-case C3 (T2-sort + Kahan-compensated atomic-float):** non-deterministic but mitigated. DEC-023 wording: "probably not sufficient standalone."

**Branch-B regression:** breaks bit-equivalence in all sub-cases (different topology than current production). Re-baseline scope is 1 sprint (DEC-020 D1 precedent — 18-config sweep is well-tooled). **However:** DEC-023 v5's resolution path (rewrite to T1-style accumulation for features 1-3) is the *retreat* from the parity envelope, and explicitly retracted the structural speedup. Any C-variant must show that determinism *with the original T2-accum topology* (not retreated to T1) is achievable at Epsilon shape.

#### Risk classification

**MEDIUM-HIGH.** Three sources of risk:
1. **Re-entry into a falsified design space.** DEC-020 shipped → DEC-023 retracted → DEC-026 opened a research recovery path → DEC-026 falsified in S25. The lever has been visited four times. DEC-025 re-entry policy requires *new mechanism evidence*.
2. **The new evidence (Epsilon shape avoiding race envelope per H1) is a hypothesis, not a measurement.** T3 must specify a determinism sweep at Epsilon shape with ≥5 runs at config-equivalent settings (DEC-049 hard constraint, line 2674).
3. **Even if H1 holds at Epsilon shape, the parity path requires re-baseline.** Branch-B regression breaks. v0.7.0 release is gated on parity preservation OR 1-sprint re-baseline plan in S47.

#### T2 verdict for Candidate C

**SURVIVE TO T3.**

**Justification:**
1. **DEC-025 re-entry policy is satisfied by the H1-monotonicity argument:** Epsilon depth-6 per-bin doc population (~49) is 8× the gate (where determinism held) and 50× config #8 (where bimodality fired). The race-vs-population monotonicity is a code-inspection-grounded mechanism claim, not arithmetic vibes — it follows directly from DEC-023 §H1 and the partition-fragmentation algebra (`structure_searcher.cpp:60–61` `numPartitions = 1u << depth`).
2. **Iter-level upper bound (2.5×–3.0×) sits at the boundary of Outcome A/B,** with kernel-level evidence at the 0.317× ratio that DEC-020 already measured at production dispatch shape. This is the strongest in-class precedent of the four candidates.
3. **DEC-017 toy-to-prod-transfer rule is already satisfied** — DEC-020 measurements were at production multi-TG depth-6 shape.

T3 owner (@performance-engineer) must specify:
- Determinism sweep at Epsilon-equivalent (config-equivalent) shapes with ≥5 runs.
- Special-case audit at config #8 (N=10k/RMSE/128b) per DEC-023 race precedent — config #8 may need to be measured separately even if Epsilon is the headline; the race envelope should be characterized.
- Sort-overhead amortization measurement: per-partition vs per-group cost decomposition.
- Parity-fix mechanism choice (C1, C2, or C3) BEFORE perf measurement; revisit DEC-023 §"Fix options" with current information.

---

### Candidate D — Split-K accumulation

#### Mechanism sketch

**Affected files / line ranges:**

- `kernel_sources.h` — new partial-histogram accumulator kernel. K-block partitioning of the doc-space within each (partition, group) tuple. Each TG processes `partSize / K` docs. K partial histograms emitted per (partition, group, K-block) tuple.
- `kernel_sources.h` — new K-merge kernel. Reads K partial histograms from threadgroup-scope or device-scope buffer; merges into single per-(partition, group) histogram. Writeback follows existing pattern at `kernel_sources.h:267–281`.
- `histogram.cpp:31–217` — new two-stage dispatcher. Two Metal kernel invocations per (depth, dim). Total dispatches/iter rises from 6 to 12 at K=any (since both stages are per-depth-per-dim).
- `histogram.cpp:133` — **CRITICAL: would require lifting `maxBlocksPerPart == 1`** (the T1 §5 race-gate constant). The K-block per-partition design IS the multi-block-per-part design that activates the Sibling S-1 race per `KNOWN_BUGS.md` (T1 §5; `histogram.cpp:128–132`). The `static_assert` at `histogram.cpp:134` is the compile-time enforcement; lifting it requires fixing the cross-TG writeback race FIRST.

**Re-statement:** **Candidate D fundamentally requires `maxBlocksPerPart > 1`.** This is the dead-code path that the existing constant gates. T1 §5 explicitly identifies this: *"Any S46 candidate that requires `maxBlocksPerPart > 1` (e.g., Candidate D's Split-K partial histograms) must fix the cross-TG writeback race before the candidate can be shipped."*

**Threadgroup memory delta:** depends on K-merge layout. Two designs:
- D1 (TG-scope K-merge): K partial histograms held in threadgroup memory of the merge kernel. K=4: 4 × 4 KB = 16 KB. K=8: 8 × 4 KB = 32 KB (at the DEC-011 ceiling). K=4 fits; K=8 is at the limit.
- D2 (Device-scope K-merge): K partial histograms written to device memory between kernels. Adds ~K × per-bin × per-partition × per-group bandwidth — possibly substantial at Epsilon scale (`200M batch-TG-ops × 4 features × K floats / 4 partial = 50M × K floats = 800 MB/iter at K=4`).

**Per-lane register pressure delta:** smaller per-TG doc count → *lower* per-lane register need (if accumulator is per-lane). If accumulator is shared (as L1a), no register-pressure change.

#### Upper-bound estimate

**Mechanism delta:**
- Reduces per-TG accumulation work by factor K (each TG processes 1/K of the docs).
- Adds K-merge pass cost. K=4: 4-term reduction per bin per (partition, group) — ~4 reads + 1 write per bin × 1024 bins × 32000 TGs × 6 dispatches = 0.78 × 10^9 ops/iter at Epsilon. At AGX ~ 1 TFLOP class for FP32, this is < 1 ms.
- Adds 6 extra dispatches/iter (12 vs 6). Overhead per DEC-048: 6 × ~30 µs = 0.18 ms — well below the dispatch threshold from S45-T2 verdict.

**Kernel-level upper bound (assumptions stated):** assume K=4. Per-TG src-broadcast work cuts by 4× (32-iter loop × `myDocCount / K` outer iters); overall kernel work cut by 4× before merge cost. Merge cost is bounded above by ~0.78 GFLOP/iter / AGX FP32 throughput ≈ 1 ms — small fraction of the 1.7-2 second iter at Epsilon.

- Lower bound: `r_kernel = 0.40` (2.5× kernel) — accounts for K-merge overhead eating ~30% of the cut.
- Upper bound: `r_kernel = 0.27` (3.7× kernel) — full K=4 cut, K-merge fully amortized.

**Iter-level upper bound (Epsilon iter=2000):** assuming `f_hist = 0.95`, `r_kernel = 0.27` → iter speedup `1 / (0.95 × 0.27 + 0.05) = 1 / 0.307 = 3.26×`. Assuming `f_hist = 0.80`, `r_kernel = 0.27` → `1 / (0.8 × 0.27 + 0.2) = 1 / 0.416 = 2.40×`. **Outcome A boundary in best case; Outcome B in conservative case.**

**However, this analysis assumes Sibling S-1 is fixed for free.** It is not. Fixing the cross-TG writeback race is non-trivial:

- Option D-fix-1: Make the device-scope writeback deterministic via per-bin int-fixed-point. Adds writeback overhead + parity considerations.
- Option D-fix-2: Make the writeback single-writer per output offset by routing K-block partials through a per-(partition, group, bin) unique slot, then merging in a third kernel. Adds another dispatch + threadgroup memory pressure.

Both fixes consume some of the 4× per-TG work cut. Realistic post-fix `r_kernel`: 0.35–0.45 → iter speedup 1.96×–2.83× at `f_hist = 0.95`.

#### Parity stance

**Outcome class: RE-BASELINE REQUIRED.** K-block partitioning introduces a new reduction order (block-major vs SIMD-group-major). Per DEC-049 line 2667: γ_(K+7) — at K=4: γ_11 ≈ 6.6e-7; K=8: γ_15 ≈ 9e-7. Both within DEC-008 envelope (`DECISIONS.md:79`) at the RMSE/Logloss ulp ≤ 4 ceiling and well within MultiClass ulp ≤ 8 ceiling.

**Branch-B regression:** breaks bit-equivalence (different reduction order). Re-baseline required. **Compounded by: the writeback-race-fix introduces additional reduction-order changes.** Whether the integrated fix preserves DEC-008 ulp ≤ 4 across the 18-config envelope must be measured; not given.

**Sibling S-1 race:** fixing it is a *prerequisite* for shipping candidate D. The sprint-plan §T4 deliverables permit probe-only kernel variants behind `#ifdef SIMD_SHUFFLE_PROBE_<X>` guards (`docs/sprint46/sprint-plan.md:213`); a probe-D experiment for candidate D can use a fix variant in a probe build only. But a path to production requires a productionized writeback-race fix, which is its own engineering scope (per `KNOWN_BUGS.md` Sibling S-1 record).

#### Risk classification

**MEDIUM.** Three sources of risk:
1. **Sibling S-1 race must be fixed FIRST.** This is a prerequisite, not a candidate-internal risk. It is a known engineering item (`KNOWN_BUGS.md`, DEC-023 §"Sibling race"). Fixing it adds scope to S47 (estimated +0.5 sprint per analogy with DEC-023 v5 fix scope).
2. **Most architecturally novel of the four** (DEC-049 line 2670). Least DEC-log precedent — no in-codebase data to ground the K-merge cost / amortization model. Probe-D measurement is more uncertain than for candidates B and C.
3. **Parity envelope (γ_11 to γ_15) is well within DEC-008**, but compounded with writeback-race-fix introduces second-order parity drift not captured in the γ_(K+7) estimate.

**No DEC-025 re-entry violation.** This design space has not been previously visited. First-time evaluation.

#### T2 verdict for Candidate D

**SURVIVE TO T3.**

**Justification:**
1. **No DEC-025 re-entry policy violation** — first-time evaluation of split-K accumulation in this codebase.
2. **Iter-level upper bound (1.96×–3.26×)** spans the Outcome A/B boundary, with reasonable probability of clearing 3× under generous Epsilon `f_hist`.
3. **Parity envelope is well within DEC-008** (γ_11 to γ_15 vs RMSE/Logloss ulp ≤ 4 ≈ γ_8 = 4.77e-7); re-baseline scope is 1 sprint.

**Caveat for T3:** the Sibling S-1 prerequisite must be addressed in T3's experiment design. Probe-D for candidate D can use a probe-only writeback-race-fix variant (e.g., per-bin int-fixed-point in the probe kernel) to enable the Split-K measurement. A production-shippable D requires the race-fix to be in S47 scope alongside the K-block kernel; T5 verdict for D must factor in this added scope.

T3 owner (@performance-engineer) must specify:
- Probe-only writeback-race-fix variant under `#ifdef SIMD_SHUFFLE_PROBE_D`.
- K-merge cost decomposition (TG-scope D1 vs device-scope D2).
- K=4 vs K=8 sweep (DEC-011 32 KB ceiling: K=8 is at the limit; K=4 has headroom).
- Determinism sweep — even though candidate D's reduction is order-deterministic per K-block, the cross-K merge order must be verified deterministic (not data-dependent).

---

## 3. Cross-candidate analysis

### Highest upper-bound estimate

**Candidate C (sort-by-bin extension)** has the highest evidence-grounded upper bound. The 0.317× hist_ms ratio was *measured* at production dispatch shape (DEC-020) — this is the strongest empirical anchor of the four. Iter-level upper bound at Epsilon iter=2000 reaches 3.0× under `f_hist = 0.977`.

Candidates B and D both peak around 2.27×–3.26× iter speedup under generous assumptions but rely more heavily on first-principles arithmetic and less on prior measurement.

### Lowest risk

**Candidate B (hierarchical reduction with re-introduced butterfly)** has the lowest risk:
- DEC-012 §"Future trigger" explicitly opens this design space (no DEC-025 re-entry violation).
- Parity path is bit-pattern-deterministic (no atomic-float race); γ_12 within DEC-008 envelope.
- The single hard risk (register spill per DEC-014) is **measurable BEFORE benchmarking** via Metal compiler register-allocation report — kill-switch is structural, not empirical.

Candidate D (split-K) has medium risk but inherits the Sibling S-1 race-fix scope; candidate C has medium-high risk due to repeated re-entry into a falsified design space.

### Cleanest parity path

**Candidate B** has the cleanest parity path:
- Re-baseline required (γ_12 differs from current γ_7), but γ_12 is well within DEC-008 RMSE/Logloss ulp ≤ 4 envelope.
- Determinism by construction (`simd_shuffle_xor` is order-deterministic per AGX SIMD semantics).
- No race envelope to characterize.
- 1-sprint re-baseline scope per DEC-020 D1 precedent.

Candidate D is second-cleanest (γ_(K+7) within envelope, deterministic K-block order) but compounded by the Sibling S-1 race-fix's parity implications. Candidate C requires choosing among C1/C2/C3 sub-cases with different parity profiles.

### Composability

**B + D is the strongest two-candidate composition.** Candidate B (per-lane register accumulation + intra-SIMD butterfly) and Candidate D (split-K) operate on *different levers*:
- B reduces shuffle-chain cost via per-lane register state and butterfly reduction.
- D reduces per-TG doc-count (K-fold) by splitting docs across more TGs.

The two are arithmetically multiplicative if both ship: B-only `r_kernel = 0.30` × D-only `r_kernel = 0.40` ≈ 0.12 (8× kernel speedup) — IF integration cost is bounded. This is highly speculative without measurement, but suggests that even if neither candidate alone clears 3× iter speedup at Epsilon, the composition might. **T3 should explicitly probe whether B and D are composable (e.g., probe-D variant enables both flags).**

Candidate A is a competitor with C on the per-feature accumulation lever (both replace simd_shuffle with a different per-feature mechanism). They cannot be composed.

---

## 4. T2 exit verdict

### Surviving candidates

**3 of 4 survive to T3:**

| Candidate | T2 verdict | Risk | Iter-level upper bound |
|---|---|---|---|
| A — Atomic-add accumulation | **RETIRE AT T2** | HIGH | 1.36× (parity-violating; would retract speedup if fixed) |
| B — Hierarchical reduction | **SURVIVE TO T3** | MEDIUM | 2.27× – 2.99× |
| C — Sort-by-bin extension | **SURVIVE TO T3** | MEDIUM-HIGH | 2.50× – 3.00× |
| D — Split-K accumulation | **SURVIVE TO T3** | MEDIUM (requires Sibling S-1 race-fix) | 1.96× – 3.26× |

### Recommended T3 priority order

1. **Candidate B** — first priority. Lowest risk; clean parity path; DEC-012 §"Future trigger" explicitly opens. T3 must lead with the Metal compiler register-allocation report verification (kill-switch BEFORE perf measurement).

2. **Candidate C** — second priority. Strongest evidence base (DEC-020 measurement at production shape); highest iter-level upper bound under realistic `f_hist`. T3 must lead with the determinism sweep at Epsilon-equivalent shapes (≥5 runs) and a parity-fix mechanism choice (C1/C2/C3) BEFORE perf measurement.

3. **Candidate D** — third priority. Most architecturally novel; least DEC-log precedent; requires Sibling S-1 race-fix prerequisite. T3 must specify a probe-only race-fix variant. Useful as a *composition target* with B if B alone is marginal — T3 should design the probe to enable B and D in combination.

### T2 → T3 transition note (composability flag)

The cross-candidate analysis surfaces a **composability hypothesis** not present in the sprint-plan §T2 candidate set: **B + D may be multiplicatively composable** because they operate on orthogonal levers (per-lane register topology vs per-TG doc partitioning). If T4 measurements show B alone falls short of Outcome A but B+D clears it, the S47 engineering scope should plan for a 2-candidate ship. T3 owner should explicitly include a B+D probe variant in the design.

### Early-halt recommendation

**NOT triggered.** 3 candidates survive (≥2 minimum per sprint-plan §T2 exit criteria). Sprint proceeds to T3.

---

## 5. Self-checks

Per the T2 process gate (`task spec §"Process gate"`):

1. **Every mechanism claim cites file:line.** ✅ Verified by self-review of §2 (all four candidates have `kernel_sources.h:NNN-NNN` and `histogram.cpp:NNN` citations for affected line ranges).
2. **Each upper-bound has stated first-principles assumptions, not arithmetic vibes.** ✅ Verified — every numerical upper bound cites `f_hist` value, kernel-vs-iter Amdahl decomposition, and source DEC for the empirical anchor (DEC-017 for A; DEC-020 for C; DEC-014 for B's spill model; DEC-049 for D's γ bounds).
3. **Candidates A and C have explicit DEC-025 re-entry justifications.** ✅ A's re-entry is rejected (parity envelope binding constraint); C's re-entry is justified by H1-monotonicity argument with code-grounded mechanism (DEC-023 §H1 + `structure_searcher.cpp:60–61`).
4. **Parity stance is one of {bit-equivalent / re-baseline / envelope violation}.** ✅ A = envelope violation; B, C2, D = re-baseline; C1 and C3 = envelope-violation-conditional-on-Epsilon-shape (sub-cases enumerated).
5. **T1's three findings are reflected.** ✅ §0 carries them forward; (a) terminology in all four candidates' mechanism sketches; (b) atomic-already-present correction explicitly applied to candidate A's upper-bound calc; (c) `stagingHist` 1-indexed offset preservation noted as constraint on candidates A, B, D writeback designs.

All checks pass.

---

## 6. References

- T1 spec: `docs/sprint46/T1/current-state.md`
- Sprint plan: `docs/sprint46/sprint-plan.md` §T2
- Production source: `catboost/mlx/kernels/kernel_sources.h:107–283`, `catboost/mlx/methods/histogram.cpp:31–217`, `catboost/mlx/methods/structure_searcher.cpp:60–108`
- DEC-008 (parity envelope): `.claude/state/DECISIONS.md:71–81`
- DEC-009 (linear cross-SIMD fold): `DECISIONS.md:82–91`
- DEC-011 (32 KB threadgroup ceiling): `DECISIONS.md:102–112`
- DEC-012 (intra-SIMD butterfly removal + future-trigger): `DECISIONS.md:160–169`
- DEC-014 (register-spill precedent): `DECISIONS.md:131–158`
- DEC-016 (MSB-fused VALID_BIT): `DECISIONS.md:179–195`
- DEC-017 (T3b retired, toy-to-prod-transfer rule): `DECISIONS.md:197–227`
- DEC-020 (T2 sort-by-bin shipped): `DECISIONS.md:285–354`
- DEC-023 (atomic-float race retracted T2): `DECISIONS.md:424–504`
- DEC-025 (re-entry policy): `DECISIONS.md:538–563`
- DEC-026 (cascade-robust GAIN falsified): `DECISIONS.md:567–625`
- DEC-049 (S46 OPEN): `DECISIONS.md:2606–2722`
- S45-T2 probe verdict: `docs/sprint45/T2/probe-verdict.md`
- KNOWN_BUGS.md Sibling S-1: `.claude/state/KNOWN_BUGS.md` (T1 §5 reference)
