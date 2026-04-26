# Sprint 42 Close — Upstream Benchmark Adoption

**Sprint:** 42  |  **Status:** READY-TO-CLOSE  |  **Branch:** `mlx/sprint-42-benchmarks`
**Cut from:** master `659ab3d17c` (post v0.5.1)
**Authoritative record:** this file
**Theme:** Run upstream `catboost/benchmarks` datasets head-to-head against catboost-mlx and 3 reference GBDT libraries on the same Apple Silicon machine; produce a Pareto-frontier deliverable; rebuild the CI perf-regression gate against runner-matched data (S41 carry-over).

---

## Mission

Per the @ml-product-owner advisory of 2026-04-26 (recorded in S41 close-out): every downstream commitment in the strategist's roadmap (E3 launch, the staged upstream RFC at `docs/upstream_issue_draft.md`, any "characterized variant" claim users can verify) is gated on having defensible head-to-head numbers on the **upstream-canonical GBDT benchmark datasets**, run on the same M-series machine across all 4 reference frameworks (LightGBM, XGBoost, CatBoost-CPU, CatBoost-MLX).

S42 produced that artifact. Two of five datasets shipped with full sweeps; the remaining three have working adapters but require user-action data acquisition (Kaggle credentials / registration forms / ~12 GB libsvm download). The **DEC-046 "numeric-only ⇒ bounded gap" claim is now empirically validated on a second independent dataset** (Higgs, 1M rows, 28 numeric features) — exactly the kind of cross-validation the S40 advisory-board synthesis required.

S42 also retired the S41 bridge mode on the perf-regression gate by replacing absolute wall-clock comparison with a hardware-invariant speedup-ratio gate.

---

## Items Completed

| Item | Description | Outcome | Commit |
|------|-------------|---------|--------|
| **T0** | Sprint scaffold (branch, plan, dirs, state updates) | Sprint plan, `benchmarks/upstream/` skeleton, HANDOFF/TODOS updated | `1ba0d55edf` |
| **T1** | Dataset adapters for 5 upstream benchmark targets | `_common.py` shared utilities + 5 per-dataset adapters (Adult auto-downloaded; Amazon/Higgs/Epsilon/MSLR with download instructions); each writes `meta.json` for downstream tooling. Verified: `python -m benchmarks.upstream.adapters.adult` end-to-end. | `13088f517a` + `fd00d1a321` |
| **T2** | 4-framework runner scripts + driver | LightGBM, XGBoost, CatBoost-CPU, CatBoost-MLX runners + `_runner_common.py` (BENCH_HP contract, BenchResult dataclass, metric helpers) + `run_subset.sh` driver. Validated end-to-end on Adult (12/12 runs across 3 seeds). | `4823b024e6` |
| **T3** | Pareto-frontier writeup + aggregator + plot generator | `aggregate_results.py`, `make_pareto_plots.py`, `docs/benchmarks/v0.5.x-pareto.md`. Adult + Higgs-1M sweeps shipped with cross-dataset summary table; DEC-046 decomposition applied to both datasets. | `bee28d02c4` + `71ef53c54b` |
| **T4** | Perf-regression gate redesigned as hardware-invariant speedup-ratio | Wall-clock-vs-baseline comparison replaced with `cpu_time_s / mlx_time_s` ratio comparison. Verified across 6 simulated scenarios including the actual S41-trigger case. `continue-on-error: false` restored on the wall-clock gate. Histogram-stage gate remains informational pending its own redesign. | `4391cee586` |
| **T5** | Sprint close-out + (optional) v0.5.2 tag | This file + HANDOFF/TODOS/CHANGELOG-DEV updates. v0.5.2 candidate stages on this branch. | (this commit) |

---

## Headline Findings

### 1. DEC-046 "numeric-only ⇒ bounded gap" claim cross-validated

The S40 advisory-board synthesis demanded that the DEC-046 decomposition (architectural-floor + categorical-encoding-asymmetry) be validated on a second independent dataset before any "characterized variant" public-facing claim. Adult and Higgs-1M now provide that validation:

| Dataset | n_train × n_features | cat features | MLX-vs-CPU logloss gap | Mechanism |
|---|---|---|---|---|
| **Adult** | 32k × 14 | 8 (57%) | **+0.1695** | architectural floor (39%) + categorical asymmetry (61%) |
| **Higgs (1M subset)** | 1M × 28 | 0 (0%) | **+0.0012** | architectural floor only — categoricals don't apply |

The gap depends almost entirely on whether the dataset has categorical features. With zero categoricals on a 30× larger workload, MLX agrees with CatBoost-CPU to within 0.0012 logloss (0.2% of CPU baseline). On Adult (8 cats out of 14 features), the categorical-attributable share is 61%. Mechanism unchanged across the two datasets; magnitude scales with cat-feature density, exactly as DEC-046 predicted.

### 2. XGBoost dominates the Pareto frontier on both datasets

On Adult and on Higgs-1M, **XGBoost (CPU hist) is the only framework on the Pareto frontier** — fastest train AND best logloss. LightGBM is virtually identical in metric on both datasets but ~2–3× slower in train. CatBoost-CPU is ~0.012 logloss behind LightGBM/XGBoost on Higgs at 200 iterations (a known CatBoost characteristic at low iteration counts; **CPU CatBoost shows the same gap as MLX**, so this is not MLX-specific).

CatBoost-MLX is consistently the slowest framework at our 200-iteration setting. Throughput per row improves materially with scale (~6k rows/s on Adult vs ~38k rows/s on Higgs-1M); GPU launch overhead amortizes at larger N. The full 11M Higgs run is pending and would directly measure whether MLX continues to scale or hits a different bottleneck.

### 3. CI perf-regression gate is now hardware-invariant

The S41 bridge mode (continue-on-error: true) is retired. The wall-clock gate now compares `cpu_time_s / mlx_time_s` ratios instead of absolute MLX wall-clock. Verified against 6 scenarios:

- **Runner 4× uniformly slower** (the S41 trigger case that needed the bridge): now correctly passes (+0.0% Δ ratio).
- **Real MLX +10% slower vs CPU on same machine**: correctly fires (+9.1% Δ).
- **MLX +3% relative**: correctly passes within 5% tolerance.

`continue-on-error: false` restored. Threshold remains 5% relative degradation in the speedup ratio.

---

## What's NOT in S42

| Item | Status | Why deferred |
|------|--------|--------------|
| Full 11M Higgs run | Data on disk (~8 GB HIGGS.csv); only the 1M subset was sweeped | Compute-window: full 4-framework × 3-seed at 11M is hour-scale; 1M subset is materially representative for the writeup |
| Epsilon (training_speed/) | Adapter ready; manual ~12 GB libsvm download | Not auto-fetchable; user's call |
| Amazon (quality_benchmarks/) | Adapter ready; Kaggle CLI auth required | DEC-046 footnote dataset; included in writeup as "to be added once data lands" |
| MSLR-WEB10K (ranking/) | Adapter ready; Microsoft registration form required | Cannot auto-fetch |
| Histogram-stage CI gate redesign | Currently informational (continue-on-error: true) | Hardware-invariant histogram gate is a deeper refactor (compare histogram_ms as fraction of total iter time, or relative-Δ across the 18-config grid); out of S42 scope |
| Investigating MLX-Adult absolute gap mechanism | Documented via DEC-046 framework only | Closing the gap is the optional narrow Lane D investigation per DEC-046; not yet started |
| Updating the staged upstream RFC with these numbers | Two-dataset preview is too narrow | RFC update happens after the full 5-dataset suite lands, per DEC-046 trigger conditions |

---

## CI Status

| Workflow | Status | Notes |
|---|---|---|
| Compile csv_train (Apple Silicon) | green | unchanged |
| MLX Python Test Suite (macos-14, py3.13) | green | unchanged |
| `mlx-perf-regression.yaml` (wall-clock gate) | redesigned to hardware-invariant ratio gate; **hard-fail restored** | T4 deliverable; verified via 6 simulated scenarios |
| `mlx-perf-regression.yaml` (histogram-stage gate) | informational pending redesign | continue-on-error: true; tracked as S42 follow-up |

The S42 branch did not touch any path-filter-matched files for the perf-regression workflow (it only touched the YAML itself), so the workflow does not re-trigger on this branch's pushes. First real validation of the new gate logic in CI happens on the next PR that touches `catboost/mlx/**`, `benchmarks/bench_mlx_vs_cpu.py`, or `python/catboost_mlx/**` after S42 merges to master.

---

## Files Changed

```
benchmarks/upstream/                                              (new)
  __init__.py
  README.md
  adapters/                                                       (new)
    __init__.py
    _common.py             # shared utilities (~220 lines)
    adult.py               # 32k × 14, 8 cats, auto-download
    amazon.py              # 32k × 9 cats, Kaggle CLI required
    higgs.py               # 11M × 28 numeric, auto-download (~2.7 GB)
    epsilon.py             # 500k × 2k numeric, manual libsvm download
    mslr.py                # 1.2M × 136 numeric, MS registration form
  scripts/                                                        (new)
    __init__.py
    _runner_common.py      # BENCH_HP, BenchResult, metrics
    run_lightgbm.py
    run_xgboost.py
    run_catboost_cpu.py
    run_catboost_mlx.py
    run_subset.sh          # driver
    aggregate_results.py
    make_pareto_plots.py
  results/                                                        (new)
    adult_*_42|43|44.json  # 12 Adult + 3 MLX-no-cat = 15 JSONs
    higgs_*_42|43|44.json  # 12 Higgs JSONs
docs/sprint42/                                                    (new)
  sprint-plan.md
  sprint-close.md          # this file
docs/benchmarks/                                                  (new)
  v0.5.x-pareto.md         # the writeup
  results_summary.csv      # auto-generated
  results_table.md         # auto-generated
  plots/
    adult_pareto.png       # auto-generated
    higgs_pareto.png       # auto-generated
.github/workflows/mlx-perf-regression.yaml                        (T4 redesign)
.claude/state/HANDOFF.md, TODOS.md, CHANGELOG-DEV.md             (close-out)
```

No `catboost/mlx/**.{cpp,h,metal}` source changes. No kernel changes. Production kernel v5 (`784f82a891`) byte-identical from S30 → S42 (md5 `9edaef45b99b9db3e2717da93800e76f`).

---

## Branch Lifecycle

`mlx/sprint-42-benchmarks`: 7 commits + this close-out (8 total). PR pending.

---

## Optional v0.5.2 tag

Post-merge, an optional `v0.5.2` tag covers:
- The benchmark suite shipped under `benchmarks/upstream/` (a user-facing artifact)
- The `docs/benchmarks/v0.5.x-pareto.md` Pareto-frontier writeup (a user-facing claim)
- The CI perf-gate redesign (a CI-visible behavior change)

Decision deferred to user: the additions are user-discoverable but not strictly user-API changes (no new pip-install behavior, no Python API additions). Could ship as v0.5.2 patch or fold into v0.6.0 with Ordered Boosting.

---

## Next Sprint Entry Points

S42 closes with master at `659ab3d17c` (pre-merge); after PR merge, master advances. No active feature sprint until the next decision.

Plausible S43 candidates (in rough priority order):

1. **Run the remaining 3 datasets** (Epsilon / Amazon / MSLR) once data lands. Same mechanical sweep as Adult/Higgs; extends the writeup table. Half-sprint to a sprint depending on data availability.
2. **Full 11M Higgs sweep** for the canonical upstream comparison. Hour-scale compute window; data already on disk.
3. **Ordered Boosting (E2)** — strategist's hero feature for v0.6.0; ~5 sprints, major Metal kernel work.
4. **Narrow Lane D (CTR RNG ordering closure)** — 3-day kill-switched investigation per DEC-046; would close the categorical-attributable share of the MLX-vs-CPU gap (61% on Adult, 0% on Higgs since no cats).
5. **Histogram-stage CI gate redesign** — make it hardware-invariant like the wall-clock gate now is.

None of these block the others; all are individually fundable.
