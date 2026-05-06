# S46-T3 — Probe-D Experiment Specification
# Candidates B, C, D: Ablation Design, Dispatch, Parity, Kill Thresholds

**Branch:** `mlx/sprint-46-simd-shuffle-research`
**Status:** OPEN — T3 deliverable (experiment design only; implementation is T4)
**Inputs:** T1 current-state (`docs/sprint46/T1/current-state.md` commit af3ad27dfc),
            T2 feasibility (`docs/sprint46/T2/feasibility.md` commit cf2ac74437)
**Surviving candidates:** B (hierarchical reduction), C (sort-by-bin revival), D (split-K)
**Retired:** A (γ envelope violation + DEC-025 re-entry blocked)

---

## §0  Priority-1: f_hist Measurement Protocol

**f_hist** = `histogram_mean_ms / iter_total_mean_ms` at production dispatch shape.
This is the single load-bearing unknown that determines whether any candidate can reach
Outcome A (≥3× iter speedup) unilaterally. All subsequent candidate decisions are conditioned
on this measurement. Measure f_hist BEFORE any probe build.

### §0.1  Tool selection

`bench_boosting --per-kernel-profile` is the correct tool.

- `bench_boosting.cpp:25–26` — flag description ("Insert mx::eval() sync points...UPPER BOUNDS")
- `bench_boosting.cpp:955` — `std::vector<double> HistogramMs` field in TPerKernelTimings
- `bench_boosting.cpp:1103–1112` — histogram dispatch + `mx::eval()` sync + `pk_histAccum` accumulation per depth
- `bench_boosting.cpp:1175` — `pkOut->HistogramMs.push_back(pk_histAccum)` appending per-iter histogram time
- `bench_boosting.cpp:1471–1473` — warm-iters 10%-trimmed mean/stdev for histogram bucket
- `bench_boosting.cpp:1486–1516` — per-kernel profile report to stdout

No new patch is required. `CATBOOST_MLX_STAGE_PROFILE` also provides `histogram_ms` and
`iter_total_ms` (see `stage_profiler.h:56,70,95`; activated via `-DCATBOOST_MLX_STAGE_PROFILE`
at CMake time), but `--per-kernel-profile` is available without any special build and is
sufficient for the f_hist ratio (ratio is robust even though individual timings are upper bounds
from sync points).

### §0.2  Dataset shapes

Three shapes must be measured. All use the production dispatch config (multi-TG, depth-6,
partition-fragmented — per DEC-017 standing rule).

| Shape label | rows | features | iters | depth | bins | seeds |
|---|---|---|---|---|---|---|
| Higgs-1M | 1,000,000 | 28 | 12 | 6 | 128 | 42, 43, 44 |
| Epsilon-proxy | 400,000 | 2,000 | 12 | 6 | 128 | 42, 43, 44 |
| Gate-config | 50,000 | 100 | 12 | 6 | 128 | 42, 43, 44 |

`--iters 12`: iter-0 is cold (JIT). Iters 1–11 are warm; bench_boosting reports 10%-trimmed
mean over warm iters automatically. Use that mean as the per-shape f_hist numerator and
denominator.

**Note:** f_hist is iter-count-independent (ratio, not absolute). The proxy at 400k/2000 with
iters=12 is valid for the Epsilon dispatch shape. Running the full 2000 iters is unnecessary
for T3.

### §0.3  Extraction

From bench_boosting stdout, capture the per-kernel profile block:

```
histogram  mean= <H> ms  stdev= ... ms
iter_total mean= <I> ms  stdev= ... ms
```

Then: `f_hist = H / I`

Extraction script:

```bash
python3 - << 'PY'
import re, sys, glob

for path in sorted(glob.glob("/tmp/fhist_*.txt")):
    text = open(path).read()
    hm = re.search(r'histogram\s+mean=\s*([\d.]+)', text)
    im = re.search(r'iter_total\s+mean=\s*([\d.]+)', text)
    if hm and im:
        h, it = float(hm.group(1)), float(im.group(1))
        print(f"{path}: f_hist={h/it:.4f}  hist={h:.3f}ms  iter={it:.3f}ms")
    else:
        print(f"{path}: PARSE ERROR")
PY
```

### §0.4  Decision table

Average f_hist across 3 seeds per shape. Use the Epsilon-proxy shape as the primary
decision anchor (most representative of production Epsilon).

| f_hist (Epsilon-proxy) | Interpretation | Action |
|---|---|---|
| ≥ 0.95 | Histogram dominates; all three candidates individually viable for Outcome A | Proceed with B, C, D probes in order |
| 0.80 ≤ f_hist < 0.95 | Marginal; single-candidate Outcome A uncertain | Proceed; B+D composability probe is mandatory if both survive |
| < 0.80 | Non-histogram time dominates; no single candidate can reach 3× iter | Proceed for Outcome B evidence; note in T4 that Outcome A requires composition |

If f_hist < 0.60: flag to user before any probe build — the architectural premise of S46 is
weakened, escalate to @silicon-architect before T4.

### §0.5  Reproduction commands (f_hist)

```bash
REPO="/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/catboost-mlx"
BIN="$REPO/python/build/temp.macosx-14.0-arm64-cpython-311/catboost_mlx._core/bench_boosting"

# If existing bench_boosting binary is stale (MLX version mismatch), rebuild first:
# cmake --build "$REPO/python/build/..." --target bench_boosting
# See T4 for rebuild commands.

for seed in 42 43 44; do
  "$BIN" --rows 1000000 --features 28  --classes 2 --depth 6 --bins 128 \
         --iters 12 --seed $seed --per-kernel-profile \
    2>&1 | tee /tmp/fhist_higgs_seed${seed}.txt

  "$BIN" --rows 400000  --features 2000 --classes 2 --depth 6 --bins 128 \
         --iters 12 --seed $seed --per-kernel-profile \
    2>&1 | tee /tmp/fhist_epsilon_proxy_seed${seed}.txt

  "$BIN" --rows 50000   --features 100  --classes 2 --depth 6 --bins 128 \
         --iters 12 --seed $seed --per-kernel-profile \
    2>&1 | tee /tmp/fhist_gate_seed${seed}.txt
done
```

---

## §1  Candidate B — Hierarchical Reduction (Per-Lane Register Accumulation + Butterfly)

### §1.1  Mechanism

Replace the src-broadcast chain (`kernel_sources.h:191–225`) with per-lane register
accumulation. Each lane accumulates into a private register array, then intra-SIMD butterfly
reduction collects the per-lane histograms. This re-introduces the butterfly removed under
L1a (DEC-012 §"Future trigger" clause: "any future kernel accumulating into per-lane register
state should re-introduce the intra-SIMD butterfly").

Current hot loop (target for replacement):
- `kernel_sources.h:191–225` — 32-iter simd_shuffle loop; 2 shuffles per src per feature pack
  (`kernel_sources.h:209–211`)
- `kernel_sources.h:158` — `threadgroup float simdHist[NUM_SIMD_GROUPS][HIST_PER_SIMD]`
  (32 KB at ceiling; may be reducible if per-lane registers replace the threadgroup staging)
- `kernel_sources.h:240–255` — 4-tile cross-SIMD linear fold (survives; butterfly is
  intra-SIMD, fold is cross-SIMD)

Butterfly re-entry point: at the boundary currently occupied by barrier 2
(`kernel_sources.h:226`), before the cross-SIMD fold begins.

### §1.2  Pre-flight kill-switch (STRUCTURAL — no benchmark if triggered)

**Kill-switch B-0: Metal compiler register allocation**

Before any benchmark, compile the probe Metal source through `xcrun metal` and inspect
VGPR allocation:

```bash
xcrun metal -arch air64 -O2 -S \
  /tmp/probe_b_kernel.metal \
  -o /tmp/probe_b_kernel.air 2>&1
# Then check VGPR count in the .air IR or use:
xcrun metallib -arch air64 /tmp/probe_b_kernel.air -o /tmp/probe_b.metallib
# Inspect VGPR via shader profiler or compiler diagnostic output
```

Threshold: VGPR/lane > 64 → **RETIRE B at T4 pre-flight without benchmark**.

DEC-014 precedent: A1 was rejected on +9.4% regression from register spill at +6 VGPR/lane.
DEC-014 standing rule: "S19-03 must verify no spill via Metal compiler register-allocation
report before any benchmark."

Per-lane register array for 128-bin histogram (32 floats per lane minimum) risks exceeding the
64-VGPR safe ceiling. If the compiler assigns >64 VGPR/lane, spill to device memory
eliminates any gain. Do not proceed to measurement.

### §1.3  Ablation design

Guard: `#ifdef SIMD_SHUFFLE_PROBE_B` in both `kernel_sources.h` and `bench_boosting.cpp`.

Changes:
1. In `kernel_sources.h:158` — conditional: probe uses per-lane register array instead of
   `simdHist[8][1024]`. If register pressure forces retention of threadgroup memory, record
   the actual allocation in T4.
2. In `kernel_sources.h:191–225` — replace src-broadcast loop with per-lane accumulation
   (each lane writes directly to its register slot when `(bin & 31) == lane`; no shuffle
   needed during accumulation phase).
3. At `kernel_sources.h:226` (barrier 2) — add intra-SIMD butterfly reduction to consolidate
   per-lane register state into per-SIMD threadgroup staging before the cross-SIMD fold.
4. `kernel_sources.h:240–255` — cross-SIMD linear fold unchanged.
5. `kernel_sources.h:267–281` — writeback unchanged.

In `bench_boosting.cpp`: `#ifdef SIMD_SHUFFLE_PROBE_B` guard around the B-variant kernel
string selection. Probe binary calls `DispatchHistogramT2Bench` with B-variant Metal source.

### §1.4  Dispatch shape

Production config per DEC-017: multi-TG, depth-6, partition-fragmented.
- Epsilon-proxy: 400k rows, 2000 features, depth 6, bins 128
- Higgs-1M: 1M rows, 28 features, depth 6, bins 128
- Gate-config: 50k rows, 100 features, depth 6, bins 128 (DEC-049 gate)

Do NOT use toy isolation (single-TG). DEC-017 precedent: toy-isolation produced +42.3%
regression at production shape due to different docs/thread ratio (195 vs 3).

Epsilon dispatch geometry (`kernel_sources.h` and `histogram.cpp:83–88`):
- 400k/2000 features → 6250 docs/TG at depth 6
- 24.4 docs/thread (not 391 — the corrected figure from T1 §4.3)
- `histogram.cpp:133–134`: `maxBlocksPerPart = 1` must remain 1 for probe B (Sibling S-1
  gate unchanged; B does not require lifting it)

### §1.5  Parity sweep

1. **18-config DEC-008 envelope sweep**: all 18 standard configurations (same suite used for
   S17–S22 gates). Max ULP ≤ 4 for RMSE/Logloss (γ_8 ≈ 4.77e-7), ≤ 8 for MultiClass.
   Reference: `DEC-008` lines 71–81 in `DECISIONS.md`.
   B is deterministic (no atomic scatter) — config #8 bimodal expected clean.

2. **100-run determinism at gate config** (50k/RMSE/d6/128b, seed 42): all 100 outputs
   bit-exact against each other. Report max ULP across 100 runs.

3. **Config #8 special audit** (N=10k/RMSE/128b bimodal): run 20 seeds; confirm bit-exact
   within seed. B's per-lane accumulation is deterministic (no cross-lane atomics), but
   the butterfly reduction introduces a new reduction order — re-baseline against v5 is
   required, not just determinism within-seed.

Parity failure → RETIRE regardless of perf.

### §1.6  Kill thresholds

Measurement: `histogram_mean_ms` ratio = (probe B) / (v5 baseline), Epsilon-proxy shape.
Iter measurement: `iter_total_mean_ms` ratio, same shape.

| Metric | Threshold | Decision |
|---|---|---|
| VGPR/lane > 64 (pre-flight) | — | RETIRE (structural, no measurement) |
| Any parity failure | — | RETIRE |
| hist_ms ratio < 0.667 (>1.5× kernel speedup) AND iter ratio < 0.667 | — | RETIRE (worse) |
| hist_ms ratio ≥ 0.667 but iter ratio ≥ 0.333 (1.5–3× iter speedup) | 1.5× ≤ iter < 3× | Outcome B — user call |
| iter ratio ≤ 0.333 (≥3× iter speedup) | ≥3× | Outcome A — COMMIT |

Note: kill criterion direction. A lower ratio = faster. "iter ratio ≤ 0.333" = probe is at
most 1/3 the time of baseline = 3× speedup. Kill = RETIRE means performance is insufficient
(<1.5× total), not that it's a numerical failure.

**If γ_12 ≈ 7.2e-7 (T2 estimate) exceeds DEC-008 RMSE ceiling (γ_8 ≈ 4.77e-7):**
Parity sweep will catch this. The T2 estimate is within MultiClass ceiling (γ ≤ 1.3e-6) but
not RMSE. If RMSE configs fail, candidate B is limited to MultiClass workloads only — note in
T4, do not auto-RETIRE unless user confirms RMSE is required.

---

## §2  Candidate C — Sort-by-Bin Revival (Pre-v5 kT2AccumSource)

### §2.1  Mechanism

Revive the pre-v5 `kT2AccumSource` from commit `73baadf445` (DEC-020: "T2 sort-by-bin shipped;
0.317× hist_ms at gate"). The current v5 replaces this with T1-style SIMD-shuffle accumulation
to fix the DEC-023 atomic-float race (config #8: 105 ULP at N=10k bimodal). The probe
re-introduces the pre-v5 accumulation under H1-monotonicity justification (DEC-023 §H1):
"larger bin counts resolve additions in consistent order."

Epsilon depth-6 per-bin population:
- Epsilon-proxy (400k/2000/d6): ~49 docs/bin (400k / 2^6 / (2000 * 4)) — see T1 §4.3
- Gate (50k/100/d6): ~6 docs/bin
- Config #8 (10k/bimodal/128b): ~1 doc/bin (the race trigger)

H1 hypothesis: race intensity inversely proportional to per-bin doc population. Epsilon is
~8× more populated than gate → plausibly deterministic. Config #8 remains the race trigger
regardless.

DEC-025 re-entry policy: new mechanism evidence required. The H1-monotonicity reasoning
constitutes new mechanism evidence (population-dependent race intensity was not the basis
for the DEC-023 fix; the fix was a blanket T1-style replacement). Re-entry is valid.

### §2.2  Pre-flight kill-switch (ANALYTICAL — verify before code change)

**Kill-switch C-0: H1-monotonicity analytical verification**

Compute per-bin population for all three shapes analytically:

```
per_bin_pop(shape) = rows / (2^depth) / (num_features * bins_per_feature)
```

| Shape | per_bin_pop | H1 status |
|---|---|---|
| Epsilon-proxy (400k/2000/d6/128b) | 400000 / 64 / (2000 * 32) | ~0.098 docs/bin |
| Gate (50k/100/d6/128b) | 50000 / 64 / (100 * 32) | ~0.244 docs/bin |
| Config #8 (10k bimodal/128b) | ~1 doc/bin (T2 measured) | H1 violated |

**Correction note:** The formula above yields sub-1 values because it averages over all
partition-level bins globally. The relevant population is per-leaf per-feature per-bin at
depth 6 (only 1 of 64 leaves is the active partition). Actual per-leaf per-bin pop:

```
per_leaf_per_bin_pop(shape) = rows / (num_features * bins_per_feature)
```

| Shape | per_leaf_per_bin_pop | H1 status |
|---|---|---|
| Epsilon-proxy | 400000 / (2000 * 32) ≈ 6.25 | Marginal |
| Gate | 50000 / (100 * 32) ≈ 15.6 | Plausible |
| Config #8 | 10000 / (N_features * 128) — bimodal → ~1 | H1 violated |

**If any target shape shows per_leaf_per_bin_pop < 2.0: flag H1 risk in T4.** The race
is empirical — H1 only reduces probability, it does not eliminate it. Config #8 will race
regardless; the parity sweep at §2.5 will catch it definitively.

No code change needed for this pre-flight. It is a paper check.

### §2.3  Ablation design

Source to revive: `git show 73baadf445:catboost/mlx/kernels/kernel_sources.h`
Relevant function: `kT2AccumSource` (the pre-v5 version with feature-0 bin-range scan +
features 1-3 atomic scatter).

Current production kernel_sources.h:
- `kernel_sources.h:1008` — `kT2SortSource` (unchanged; sorting stage is correct in v5)
- `kernel_sources.h:1156` — `kT2AccumSource` (current v5, T1-style SIMD-shuffle accumulation)

Probe changes:
1. Extract pre-v5 `kT2AccumSource` from `73baadf445` and name it `kT2AccumProbeSource`.
2. Install `kT2AccumProbeSource` at `kernel_sources.h:1156` area under
   `#ifdef SIMD_SHUFFLE_PROBE_C`. Production `kT2AccumSource` unchanged in `#else` branch.
3. In `bench_boosting.cpp`: `#ifdef SIMD_SHUFFLE_PROBE_C` guard to select probe dispatch
   function (`DispatchHistogramT2ProbeBench`) that calls `kT2SortSource` + `kT2AccumProbeSource`.

Retrieve pre-v5 source:
```bash
REPO="/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/catboost-mlx"
git -C "$REPO" show 73baadf445:catboost/mlx/kernels/kernel_sources.h \
  | awk '/kT2AccumSource/,/^};/' \
  | head -200 > /tmp/t2_accum_source_73baadf445.metal.txt
```

### §2.4  Dispatch shape

Same three shapes as §1.4 (production multi-TG, depth-6, partition-fragmented per DEC-017).
`histogram.cpp:133–134` `maxBlocksPerPart = 1` remains unchanged (C does not require lifting
the race gate; probe C still uses single-block-per-partition dispatch).

### §2.5  Parity sweep — race audit is primary

**Race audit overrides all performance results. Any race firing = automatic RETIRE.**

1. **Config #8 race audit — 100 runs** (N=10k bimodal, RMSE, 128b, seed 42):
   Report ULP distribution across 100 runs vs v5 baseline. Expected outcome: races fire
   (per DEC-023: 105 ULP at this config). If deterministic here → H1 is stronger than
   expected; note but do not trust without further evidence.

2. **Epsilon-equivalent race audit — 100 runs** (N=400k, 2000 features, RMSE, 128b, seed 42):
   Report ULP distribution. Per H1, this should be near-deterministic. If max ULP > 4 →
   RETIRE (RMSE ceiling violation, DEC-008). If max ULP ≤ 4 → CONDITIONAL PASS (document;
   Config #8 racing is still a problem for general use).

3. **18-config DEC-008 envelope sweep** (excluding config #8 from pass/fail if it races, as
   expected; document the race as known):
   For all non-bimodal configs: max ULP ≤ 4 (RMSE/Logloss), ≤ 8 (MultiClass).

4. **100-run determinism at gate config** (50k/RMSE/d6/128b, seed 42): expected clean per
   per_leaf_per_bin_pop ≈ 15.6. Confirm bit-exact across 100 runs.

**Decision rule:** If Epsilon-equivalent audit passes (ULP ≤ 4) AND gate-config determinism
passes AND 18-config sweep passes (excluding bimodal config #8): C may proceed to T4
implementation with restriction "production use requires per-dataset population check; bimodal
config #8 is a known race." If Epsilon-equivalent fails: RETIRE C unconditionally.

### §2.6  Kill thresholds

| Metric | Threshold | Decision |
|---|---|---|
| Epsilon-equivalent ULP > 4 in race audit | — | RETIRE (unconditional) |
| Gate-config 100-run fails determinism | — | RETIRE |
| 18-config non-bimodal sweep: any ULP > threshold | — | RETIRE |
| hist_ms ratio > 0.667 (< 1.5× kernel speedup) | — | RETIRE |
| 1.5× ≤ iter speedup < 3× | — | Outcome B — user call |
| ≥ 3× iter speedup AND parity passes | — | Outcome A — COMMIT |

DEC-020 baseline: 0.317× hist_ms at gate config (v5 sort+accum vs v4; the pre-v5 accum
contributes most of this). If probe C reverts only the accumulation stage, the expected
ratio should be near 0.317× at gate. Verify this matches — if probe C is significantly worse
than 0.317×, the pre-v5 source extraction may have missed a sorting interaction.

---

## §3  Candidate D — Split-K Accumulation (K=4 Partial Histograms + On-Chip Merge)

### §3.1  Mechanism

K=4 parallel partial-histogram accumulators per threadgroup, each processing a 1/K doc slice,
followed by an on-chip merge pass. This trades accumulation serialization for merge overhead.
With K=4:
- Each TG processes 4 doc slices instead of 1 → 4× more parallel accumulators per TG
- Merge pass: 4 partial histograms → 1 final histogram per TG
- ~~TG memory for K=4: `K * HIST_PER_SIMD * sizeof(float) = 4 * 1024 * 4 = 16 KB` (fits
  DEC-011 32 KB ceiling)~~
- ~~K=8: 32 KB (at ceiling; no margin) → K=4 is preferred~~

**ERRATUM (T4 code-inspection finding, 2026-05-05):** The TG-memory arithmetic above is wrong by 8×. The actual D1 layout requires the `NUM_SIMD_GROUPS=8` dimension as well (per `kernel_sources.h:28–34`, current production `simdHist[8][1024]` consumes 32 KB exactly). For D1 to maintain per-SIMD-group locality, the partial histogram layout is `partialHist[K][NUM_SIMD_GROUPS][HIST_PER_SIMD]`:
- For K=4: `4 × 8 × 1024 × 4 bytes = 131,072 bytes = 128 KB` — **4× over the 32 KB ceiling**
- For K=8: `8 × 8 × 1024 × 4 bytes = 256 KB` — **8× over ceiling**

The "16 KB" figure in the original spec corresponds to a flat `K × HIST_PER_SIMD × sizeof(float)` layout that omits the NUM_SIMD_GROUPS dimension, which would require atomic accumulation within each K-slice (re-introducing the DEC-023 race envelope).

**D1 as specified is structurally TG-memory-infeasible.** D MUST proceed via the D2 path (separate accumulation + merge dispatches). See §3.4 below for the corrected D2 specification.

### §3.2  Pre-flight kill-switch (STRUCTURAL — Sibling S-1 race gate)

**Kill-switch D-0: Sibling S-1 static_assert lift feasibility**

`histogram.cpp:133–134`:
```cpp
constexpr size_t maxBlocksPerPart = 1;
static_assert(maxBlocksPerPart == 1, "Multi-block histogram requires race-free writeback");
```

Lifting this constraint under `#ifdef SIMD_SHUFFLE_PROBE_D` requires a race-free writeback
mechanism. Two options:

- ~~**D1 (TG-scope merge):** K partial histograms merged inside the TG before global writeback.
  `maxBlocksPerPart` remains 1 from the dispatch perspective (the K-split is internal to the
  kernel, not external multi-block dispatch). No lift of the static_assert needed for D1.
  This is the preferred path.~~ **(See ERRATUM in §3.1 — D1 is TG-memory-infeasible.)**

- **D2 (device-scope multi-block):** K separate TG dispatches, each writing to a unique device
  buffer slot, then a separate merge kernel. **D2 is now the only viable D path.** Code-inspection-grounded race analysis (T4, 2026-05-05): D2 is structurally race-free if (a) accumulation writes to unique per-`(partition, group, K-block)` device slots, and (b) the merge kernel uses `maxBlocksPerPart=1`. Both are design choices, not code-path constraints. The static_assert at `histogram.cpp:134` is in `ComputeHistogramsImpl` — the probe build's `DispatchHistogramProbeDKBench` (under `#ifdef SIMD_SHUFFLE_PROBE_D`) bypasses it entirely. **No production-code static_assert lift needed for the probe.** D2 merge overhead analytical bound: ~1.3 ms at Epsilon K=4 (memory-bandwidth-limited; partial reads = 4 × 32000 TGs × 4 KB = 512 MB at ~400 GB/s) — 0.058% of 2241 ms iter total, far below the 30% retirement threshold.

**T4 pre-flight decision:** If D1 (intra-kernel K-split) is achievable, proceed without
touching `histogram.cpp:133–134`. If D2 is required, the race-fix mechanism overhead must
be measured first:

```
Race-fix overhead threshold: if (merge_pass_time / K1_baseline_time) > 0.30 → RETIRE D2
```

DEC-017 standing rule applies: production dispatch shape mandatory for this measurement.

### §3.3  Ablation design

Guard: `#ifdef SIMD_SHUFFLE_PROBE_D` in `kernel_sources.h` and `bench_boosting.cpp`.

**D1 design (preferred — no static_assert lift):**

1. In `kernel_sources.h` under `#ifdef SIMD_SHUFFLE_PROBE_D`: new kernel string
   `kHistOneByteProbeKSource` with K=4 internal split:
   - Allocate `threadgroup float partialHist[K][NUM_SIMD_GROUPS][HIST_PER_SIMD]` (16 KB for K=4)
   - Each SIMD group processes docs `[simd_id * docs_per_simd / K .. (simd_id + 1) * docs_per_simd / K]`
     for its assigned K-slice
   - After K-slice accumulation: reduce across K slices (threadgroup barrier + K-fold add)
   - Writeback unchanged (`kernel_sources.h:267–281`)

2. In `bench_boosting.cpp`: K=4 constant under `#ifdef SIMD_SHUFFLE_PROBE_D`; dispatch via
   `DispatchHistogramProbeDKBench` using the K-variant kernel.

3. `histogram.cpp:133–134` static_assert: unchanged for D1 path. Only lift under separate
   `#ifdef SIMD_SHUFFLE_PROBE_D2` if D2 path is needed.

### §3.4  Dispatch shape

Same three shapes (production config, DEC-017). Dispatch count changes:
- Production (K=1): 6 dispatches/iter (confirmed by `structure_searcher.cpp:60–108` and S45-01)
- Probe D (D1 K=4 intra-kernel): 6 dispatches/iter (K-split is internal; dispatch count unchanged)
- Probe D (D2 device-scope): 12 dispatches/iter (6 accum + 6 merge) — verify in bench output

Verify dispatch count in bench_boosting output. If D1: confirm 6 dispatches. If D2: confirm 12.

### §3.5  Parity sweep

1. **New reduction order baseline:** K=4 reduces in different order than K=1. Re-baseline
   against v5 (not bit-for-bit match expected). Establish new ULP bound for probe D itself.

2. **γ_11 ≈ 6.6e-7 (K=4 from T2 estimate):** within DEC-008 RMSE ceiling (γ_8 ≈ 4.77e-7)
   per T2 — but this is a model estimate. Empirical sweep required.

3. **18-config DEC-008 envelope sweep**: same suite. RMSE/Logloss ≤ 4 ULP, MultiClass ≤ 8 ULP.

4. **100-run determinism at gate config** (50k/RMSE/d6/128b, seed 42): K=4 intra-kernel merge
   is deterministic (no cross-TG atomics in D1 path). Confirm bit-exact across 100 runs.

5. **Config #8 audit** (N=10k bimodal): D1 intra-kernel merge is deterministic — race only
   exists on D2 path if global atomics re-enter. D1 should be clean.

### §3.6  Kill thresholds

| Metric | Threshold | Decision |
|---|---|---|
| D1 intra-kernel K-split infeasible AND D2 merge overhead > 30% | — | RETIRE D |
| Any parity failure (DEC-008 ceiling breach) | — | RETIRE |
| hist_ms ratio > 0.667 (< 1.5× kernel speedup) | — | RETIRE |
| 1.5× ≤ iter speedup < 3× | — | Outcome B — user call |
| ≥ 3× iter speedup AND parity passes | — | Outcome A — COMMIT |

If D2 merge overhead ≤ 30% AND D2 race-fix mechanism is clean: proceed with D2 path, document
the static_assert lift scope in T4.

---

## §4  B+D Composability Probe

### §4.1  Trigger condition

Composability probe is mandatory if:
- Both B and D individually survive (parity pass, ≥ 1.5× iter speedup)
- f_hist is in the marginal range (0.80 ≤ f_hist < 0.95) OR both B and D are individually
  Outcome B only (< 3×)

Composability probe is optional if either B or D individually reaches Outcome A (≥ 3×) —
run it anyway to characterize the interaction, but it is not blocking for T4 commit.

### §4.2  Orthogonality argument

B and D operate on orthogonal dimensions:
- B: per-lane register topology in the accumulation phase (`kernel_sources.h:191–225`)
- D: doc partitioning per TG (K-split of the doc stream)

Both changes are to the same kernel region but at different levels (accumulation strategy vs
doc slicing). Interference risk: register pressure from B's per-lane array + D's K=4 partial
histograms may push VGPR/lane past 64 together even if B alone passes the pre-flight.

### §4.3  Probe design

Guard: `#ifdef SIMD_SHUFFLE_PROBE_BD` in `kernel_sources.h` and `bench_boosting.cpp`.

Combined kernel: per-lane register accumulation (B) × K=4 doc slicing (D). Each of K=4
doc slices is processed with per-lane register state; K-slice merge at end; butterfly
reduction before merge (B's butterfly is still needed for intra-SIMD consolidation).

**Pre-flight for BD combined:** Run Metal compiler register allocation check on the combined
kernel. Kill-switch: VGPR/lane > 64 → RETIRE BD composability (proceed with B and D
individually if they individually pass).

### §4.4  Measurement

Shape: Epsilon-proxy (400k/2000/d6/128b), 3 seeds, `--iters 12`, `--per-kernel-profile`.

Compute:
- `r_B` = hist_ms ratio for probe B alone vs baseline
- `r_D` = hist_ms ratio for probe D alone vs baseline
- `r_BD` = hist_ms ratio for BD combined vs baseline
- Non-interference ratio: `r_BD / (r_B * r_D)` — should be ≈ 1.0 if orthogonal

| Non-interference ratio | Interpretation |
|---|---|
| 0.9 – 1.1 | Orthogonal — composability confirmed |
| < 0.9 | Synergistic — BD better than product (report as positive surprise) |
| > 1.1 | Interference — B and D interact; composability degraded |
| > 1.5 | Severe interference — BD worse than either individually; do not compose |

### §4.5  Parity

Same 18-config sweep + config #8 audit. BD combination may have different ULP characteristics
than either individually — re-baseline separately.

### §4.6  Kill threshold

If `r_BD < 0.333` (≥ 3× hist_ms speedup) AND parity passes → **BD Outcome A path** even if
B alone and D alone are both Outcome B. Flag to user as the f_hist-marginal Outcome A route.

---

## §5  #ifdef Guard Conventions

All probe guards follow the `#ifdef SIMD_SHUFFLE_PROBE_<X>` pattern. Never nest probes.
Production code is always in the `#else` branch. Guards apply in:

| File | Guard | Scope |
|---|---|---|
| `catboost/mlx/kernels/kernel_sources.h` | `SIMD_SHUFFLE_PROBE_B` | B accumulation loop + threadgroup layout |
| `catboost/mlx/kernels/kernel_sources.h` | `SIMD_SHUFFLE_PROBE_C` | C pre-v5 accum source string |
| `catboost/mlx/kernels/kernel_sources.h` | `SIMD_SHUFFLE_PROBE_D` | D K=4 kernel source string |
| `catboost/mlx/kernels/kernel_sources.h` | `SIMD_SHUFFLE_PROBE_BD` | BD combined kernel source string |
| `catboost/mlx/tests/bench_boosting.cpp` | `SIMD_SHUFFLE_PROBE_B` | dispatch function selector |
| `catboost/mlx/tests/bench_boosting.cpp` | `SIMD_SHUFFLE_PROBE_C` | dispatch function selector |
| `catboost/mlx/tests/bench_boosting.cpp` | `SIMD_SHUFFLE_PROBE_D` | dispatch function selector + K constant |
| `catboost/mlx/tests/bench_boosting.cpp` | `SIMD_SHUFFLE_PROBE_BD` | dispatch function selector |
| `catboost/mlx/methods/histogram.cpp` | `SIMD_SHUFFLE_PROBE_D2` | static_assert lift (D2 path only) |

Build one probe binary per candidate. Never combine multiple probe guards in a single binary.

Example build flag usage (T4 implementation):
```bash
clang++ -DSIMD_SHUFFLE_PROBE_B ...   # Probe B binary
clang++ -DSIMD_SHUFFLE_PROBE_C ...   # Probe C binary
clang++ -DSIMD_SHUFFLE_PROBE_D ...   # Probe D binary
clang++ -DSIMD_SHUFFLE_PROBE_BD ...  # Composability binary
```

---

## §6  Regression Guard

Branch-B regression test must remain GREEN on master throughout T3 and T4 probe builds.

File: `python/tests/regression/test_branch_b_regression.py`

This test uses load-and-predict (not re-train) due to Metal GPU atomic cross-process
non-determinism (~1-3 ULP weight drift; see `project_mlx_training_nondeterminism.md` memory).
Probe builds exist on a separate binary path (`bench_boosting` + probe flags) and do not
affect the Python path `_core.so`. The regression test is not at risk from probe builds.

Verify before T4 implementation begins:
```bash
cd "/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/catboost-mlx"
python -m pytest python/tests/regression/test_branch_b_regression.py -v
```

Expected: all tests PASS. If any fail before probe build: investigate immediately —
do not proceed to T4 implementation with a failing regression suite.

---

## §7  T3 Done Definition

T3 is complete when:

1. `docs/sprint46/T3/probe-d-spec.md` committed to `mlx/sprint-46-simd-shuffle-research`
2. §0 f_hist measurement protocol fully specified (tool, shapes, seeds, extraction script,
   decision table) — priority-1, unblocks all candidate decisions
3. §1 Candidate B: pre-flight (VGPR kill-switch), ablation (kernel_sources.h:191–225 and
   158 under SIMD_SHUFFLE_PROBE_B), dispatch, parity (18-config + determinism + config #8),
   kill thresholds — all specified
4. §2 Candidate C: pre-flight (H1 analytical), ablation (pre-v5 kT2AccumSource from
   73baadf445 under SIMD_SHUFFLE_PROBE_C), dispatch, race audit (config #8 + Epsilon-equivalent
   100-run), kill thresholds — all specified
5. §3 Candidate D: pre-flight (D1 vs D2 path, Sibling S-1 gate), ablation (K=4 intra-kernel
   under SIMD_SHUFFLE_PROBE_D), dispatch, parity, kill thresholds — all specified
6. §4 B+D composability probe: trigger condition, orthogonality argument, design
   (SIMD_SHUFFLE_PROBE_BD), measurement (non-interference ratio), kill threshold — specified
7. §5 #ifdef guard convention documented (file-scope table, build flag examples)
8. §6 Regression guard documented (test file, verify command, non-interference with probe builds)

**T4 entry criteria (ordered):**

1. f_hist measured at all 3 shapes (Gate, Higgs-1M, Epsilon-proxy) — priority-1; drives
   Outcome A / B / composition routing
2. B pre-flight: Metal compiler VGPR check (no benchmark if VGPR/lane > 64)
3. C pre-flight: H1 analytical per-bin population check (flag if < 2.0 docs/leaf/bin)
4. D pre-flight: D1 vs D2 path feasibility — if D1 achievable, no static_assert lift;
   if D2 required, measure merge overhead (retire if > 30%)
5. Probe execution order: B → C → D (can run in parallel if separate hardware sessions)
6. B+D composability: run if both B and D survive AND f_hist marginal OR both individually
   Outcome B

**Estimated T4 time budget:** 2–3 days (pre-flight checks: 1 day; probe builds + measurement:
1 day; parity sweeps: 1 day; composability: same session as B+D individual if both pass)

---

## §A  Appendix: Key File:Line Reference

| Symbol | File:Line | Role in T3 |
|---|---|---|
| src-broadcast chain | `kernel_sources.h:191–225` | B's ablation target |
| simdHist threadgroup | `kernel_sources.h:158` | B's layout replacement target |
| 2 shuffles per src | `kernel_sources.h:209–211` | B's eliminated operations |
| cross-SIMD linear fold | `kernel_sources.h:240–255` | Unchanged in B and D |
| writeback loop | `kernel_sources.h:267–281` | Unchanged in B, D (D1 path) |
| kT2SortSource | `kernel_sources.h:1008` | Unchanged in C |
| kT2AccumSource (v5) | `kernel_sources.h:1156` | C's replacement target |
| static_assert race gate | `histogram.cpp:133–134` | D2 path pre-flight |
| dispatch grid | `histogram.cpp:83–88` | D2 path grid change scope |
| atomic_outputs | `histogram.cpp:76` | Context for writeback design |
| per-kernel profile flag | `bench_boosting.cpp:25–26` | f_hist measurement tool |
| HistogramMs field | `bench_boosting.cpp:955` | f_hist numerator source |
| histogram sync + accum | `bench_boosting.cpp:1103–1112` | f_hist measurement path |
| 10%-trimmed mean | `bench_boosting.cpp:1471–1473` | Output stat used for f_hist |
| profile report | `bench_boosting.cpp:1486–1516` | stdout parsing target |
| STAGE_PROFILE flag | `stage_profiler.h:6` | Alternative measurement path |
| histogram_ms name | `stage_profiler.h:70` | Alternative extraction key |
| IterTotalMs | `stage_profiler.h:95` | Alternative denominator |
| dispatch loop | `structure_searcher.cpp:60–108` | 6 dispatches/iter confirmed |
| STAGE_TIMER_DEPTH | `structure_searcher.cpp:84,238,557` | Existing instrumentation |
| pre-v5 kT2AccumSource | `git show 73baadf445:catboost/mlx/kernels/kernel_sources.h` | C's revival source |
| Branch-B regression | `python/tests/regression/test_branch_b_regression.py` | Must stay GREEN |
| DEC-008 parity envelope | `DECISIONS.md:71–81` | ULP ceilings for all sweeps |
| DEC-012 butterfly trigger | `DECISIONS.md:160–169` | B's re-entry justification |
| DEC-014 register spill | `DECISIONS.md:131–158` | B's kill-switch precedent |
| DEC-017 production shape | `DECISIONS.md:197–227` | Dispatch shape mandate |
| DEC-020 0.317× baseline | `DECISIONS.md:285–354` | C's expected gain |
| DEC-023 H1 hypothesis | `DECISIONS.md:424–485` | C's re-entry justification |
| DEC-025 re-entry policy | `DECISIONS.md:538–563` | C and D re-entry condition |
| DEC-049 kill thresholds | `DECISIONS.md:2606–2722` | < 1.5× RETIRE, ≥ 3× Outcome A |
