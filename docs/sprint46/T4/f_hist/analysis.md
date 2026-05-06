# T4 — f_hist Measurement Analysis

**Date:** 2026-05-05
**Branch:** `mlx/sprint-46-simd-shuffle-research`
**Tool:** `bench_boosting_s46` built with `-DCATBOOST_MLX_STAGE_PROFILE`, `--per-kernel-profile` flag (per `bench_boosting.cpp:25–26, 1103–1112, 1471–1473`)
**Methodology:** 3 seeds × 12 iters (1 cold + 11 warm) × 3 shapes; 10%-trimmed warm mean per `stage_profiler.h`
**Decision:** ≥0.95 → all candidates Outcome-A viable; 0.80–0.95 → BD composability mandatory; <0.80 → composition required; <0.60 → escalate

## Verdict

**f_hist Epsilon-proxy = 0.9772 ≥ 0.95.** All 3 surviving candidates from T2 (B, C, D2) are individually viable for Outcome A.

## Results

| Shape | rows × features | iter_mean (ms) | hist_mean (ms) | **f_hist** | f_hist std |
|---|---|---|---|---|---|
| Gate-config | 50,000 × 100 | 50.07 | 36.77 | **0.7345** | 0.0068 |
| Higgs-1M-proxy | 1,000,000 × 28 | 451.13 | 404.70 | **0.8970** | 0.0066 |
| Epsilon-proxy ★ | 400,000 × 2,000 | 2020.83 | 1974.71 | **0.9772** | 0.0008 |

★ = load-bearing measurement for v0.7.0 perf gate.

## Findings

### 1. Epsilon f_hist confirms the architectural premise

f_hist = 0.9772 means histogram phase consumes **97.7% of Epsilon iter wall-clock** at production dispatch shape (400k × 2000 features × depth 6 × 128 bins). This is the ceiling on what any histogram-internal optimization can buy: a 3× histogram speedup yields up to ~2.93× iter speedup; a 4× histogram speedup yields up to ~3.91× iter speedup.

Mapping each T2 candidate's analytical upper bound to projected iter speedup at f_hist=0.977:

| Candidate | T2 hist upper-bound | Projected iter speedup | Outcome |
|---|---|---|---|
| B (hierarchical) | 2.27×–2.99× | 2.22×–2.92× | Outcome B (marginal — 1.5–3×) likely |
| C (sort-by-bin) | 2.50×–3.00× | 2.44×–2.93× | Outcome A/B borderline |
| D2 (split-K) | 1.96×–3.26× | 1.92×–3.18× | Outcome A possible |

Note these are analytical upper bounds; actual probe-D measurement at production shape is the load-bearing test (per DEC-017 standing rule).

### 2. Gate-config f_hist DIFFERS from S19-01c's claimed 0.977

**f_hist gate = 0.7345**, not 0.977 as claimed by S19-01c (cited in DEC-049:2620 and `docs/sprint46/sprint-plan.md` §"Strategic context"). Possible explanations:

- S19-01c's gate config may have used different `depth` or `bins` parameters than this measurement (depth=6, bins=128).
- Post-S22 `sort-by-bin` shipping (DEC-020) shifted the gate-config attribution.
- The 0.977 figure may have included additional kernel work (split_score, leaf_estimation) that this measurement separates.

**This does NOT affect the T5 decision** — Epsilon is the v0.7.0 perf gate. But the historical 0.977 figure should be treated as approximate; the current measurement at production shape is the authoritative number.

LESSONS-LEARNED follow-up: any DEC entry citing a perf number across sprint boundaries should carry a date + measurement-config stamp. The S19-01c 0.977 was stale by 2-3 sprints and shifted ~25 percentage points without anyone noticing.

### 3. f_hist scales with feature dimensionality (consistent with src-broadcast cost driver)

The pattern gate (0.73) → Higgs (0.90) → Epsilon (0.98) is consistent with the T1 finding that the src-broadcast chain at `kernel_sources.h:209–224` is the cost driver: more features = more 4-byte packed iterations = more inner-loop work in the chain.

Specifically, the `FEATURES_PER_PACK=4` inner loop at `kernel_sources.h:217–223` runs once per src iteration regardless of feature count, but the OUTER feature-group dispatch count grows linearly with feature dim (Epsilon's 2000 features = 500 groups × 32-iter src loop = 16000 iterations of the inner predicate per partition; Higgs's 28 features = 7 groups × 32 = 224 iterations).

This corroborates T2's identification of the src-broadcast chain as the load-bearing lever — and validates Epsilon as the right empirical anchor for v0.7.0 perf gate calibration.

## Reproduction commands

```bash
cd "/path/to/catboost-mlx"
MLX_PREFIX="$(brew --prefix mlx)"

# Build bench_boosting with stage profile flag
clang++ -std=c++17 -O2 -I. -I"${MLX_PREFIX}/include" -L"${MLX_PREFIX}/lib" -lmlx \
  -framework Metal -framework Foundation -Wno-c++20-extensions \
  -DCATBOOST_MLX_STAGE_PROFILE \
  catboost/mlx/tests/bench_boosting.cpp \
  catboost/mlx/methods/histogram_t2_impl.cpp \
  -o bench_boosting_s46

# Run f_hist sweep (3 shapes × 3 seeds × 12 iters)
mkdir -p docs/sprint46/T4/f_hist/data
for seed in 42 43 44; do
  ./bench_boosting_s46 --rows 50000   --features 100  --classes 2 --depth 6 --bins 128 \
    --iters 12 --seed $seed --per-kernel-profile > docs/sprint46/T4/f_hist/data/gate_config_seed${seed}.txt 2>&1
  ./bench_boosting_s46 --rows 1000000 --features 28   --classes 2 --depth 6 --bins 128 \
    --iters 12 --seed $seed --per-kernel-profile > docs/sprint46/T4/f_hist/data/higgs_proxy_seed${seed}.txt 2>&1
  ./bench_boosting_s46 --rows 400000  --features 2000 --classes 2 --depth 6 --bins 128 \
    --iters 12 --seed $seed --per-kernel-profile > docs/sprint46/T4/f_hist/data/epsilon_proxy_seed${seed}.txt 2>&1
done
```

Parsing: `histogram mean=` and `warm mean (...iters):` lines from each output. f_hist = histogram_ms / iter_total_ms.

Results aggregated to `docs/sprint46/T4/f_hist/results.json`.

## Implications for T5 decision

Per `docs/sprint46/sprint-plan.md` §T5 outcome thresholds, f_hist clearing 0.95 means each surviving T2 candidate (B, C, D2) has the headroom to clear Outcome A on its own IF its actual probe-D measurement matches the T2 analytical upper bound. The remaining empirical questions for T5:

- **B**: Does VGPR/lane stay ≤ 64 with the per-lane register histogram? (Pre-flight blocker; structural — Metal compiler dump verifies before any benchmark.)
- **C**: Does H1-monotonicity hold at Epsilon shape (per-leaf per-bin pop ~6.25, marginal but above 2.0 threshold)? Does any race fire at config #8 equivalent during 100-run audit?
- **D2**: Does merge overhead stay below 30% of K=1 baseline? (Analytical bound: 0.058% — far below threshold.)

If all three pre-flights pass, T5 maps probe-D measurements to outcomes. If any pre-flight fails, that candidate retires; remaining candidates still have headroom under the f_hist green light.
