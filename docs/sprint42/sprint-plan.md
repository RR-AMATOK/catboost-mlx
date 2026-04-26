# Sprint 42 Plan — Upstream Benchmark Adoption

**Sprint:** 42  |  **Status:** IN PROGRESS  |  **Branch:** `mlx/sprint-42-benchmarks`
**Cut from:** master `659ab3d17c` (post v0.5.1)
**Theme:** Run the upstream `catboost/benchmarks` suite (the same datasets the GBDT community evaluates CatBoost on) against CatBoost-MLX, produce a Pareto-frontier deliverable, and rebuild the CI perf-regression gate against runner-matched baselines (S41 carry-over).

## Strategic context

Per the S40 advisory-board synthesis (DEC-046) and the @ml-product-owner advisory of 2026-04-26, S42 produces the **defensible head-to-head numbers** that gate every downstream commitment:

- **The eventual E3 launch** (HN/Twitter/MLX-Slack post) is gated on having credible benchmarks, not just synthetic-anchor parity. Without the head-to-head numbers, "characterized variant" is a phrase; with them, it's a Pareto frontier readers can pick a point on.
- **The staged upstream RFC** (`docs/upstream_issue_draft.md`) is materially stronger when it cites real numbers on the upstream's own datasets — it shows the work meets the community's evaluation standard, not just a curated synthetic anchor.
- **The S41 perf-regression gate bridge mode** (continue-on-error: true since the Sprint 16 baseline is hardware-mismatched) needs a proper fix this sprint — runner-matched baselines or a switch to relative speedup measures.

## Authoritative dataset inventory (verified via gh API 2026-04-26)

The upstream `catboost/benchmarks` repo (172 stars, last updated 2026-03-15) has these directories:

```
quality_benchmarks/   <- 9 classification datasets, Logloss metric, head-to-head
training_speed/       <- abalone, airline, epsilon, higgs, letters, msrank, synthetic
ranking/              <- MSLR-WEB10K, Yahoo (NDCG)
kaggle/               <- rossmann-store-sales
gpu_vs_cpu_training_speed/   <- speed harness
model_evaluation_speed/, monotone_constraints/, pool_creation_benchmark/, shap_speed/
```

`quality_benchmarks/` README confirms the published datasets:
- **Adult** (UCI, ~48k, classification)
- **Amazon** (Kaggle Amazon Employee Access, 32k × 9 cats — *includes our DEC-046 characterized gap*)
- **Appet** (KDD Cup 2009, ~50k, cats)
- **Click** (KDD Cup 2012 Track 2, ~40M+ rows — **out of reach** per our 16M ceiling)
- **Internet** (1999 KDD Cup)
- **Kdd98**
- **Kddchurn** (KDD Cup 2009)
- **Kick** (Kaggle DontGetKicked)
- **Upsel** (KDD Cup 2009)

`training_speed/` README (datasets across grids of depth 6 / 10 / 12 × lr 0.02–0.15 × 1000 iterations):
- **abalone, airline, epsilon, higgs, letters, msrank, msrank-classification, synthetic, synthetic-5k-features**

## Target subset — 5 datasets

Defensible runnable scope for v0.5.x evidence pack (see Strategist + Devils-advocate advice in the S40 synthesis: include the unflattering one with a footnote, don't cherry-pick):

| Dataset | Source dir | Task | Why included |
|---|---|---|---|
| **Higgs (depth 6)** | `training_speed/` | Binary classification, 11M × 28 | Flagship runnable target; under our 16M cap; widely-cited GBDT baseline |
| **Epsilon** | `training_speed/` | Binary classification, 500k × 2k | Pure numeric; ideal showcase for Apple Silicon GPU on dense feature work |
| **Adult** | `quality_benchmarks/` | Classification, ~48k | Tiny; confirms small-N path works; lets us match upstream's tuned-CatBoost number on real data |
| **Amazon** | `quality_benchmarks/` | Classification, 32k × 9 cats | **Triggers the DEC-046 CTR rare-class gap — include with footnote, do not cherry-pick around** |
| **MSLR-WEB10K** | `ranking/` | Ranking (NDCG), YetiRank | Confirms the ranking path ships honest numbers on the upstream-canonical ranking dataset |

**Out of reach** (documented in the writeup, not run):
- Click (>16M rows, our `ComputePartitionLayout` ceiling)
- Anything requiring `boosting_type='Ordered'` (not implemented; Plain only)
- Depth > 6 grids (kernel constraint)
- Anything requiring `NewtonL2` / `NewtonCosine` score functions (rejected at API)

## Honest-publishing constraints (carried from S40 synthesis)

1. **Run all 4 frameworks on the same M-series machine.** Do NOT compare our wall-clock vs upstream's published A100 numbers. The Pareto frontier is "what's the best framework on Apple Silicon today", not "is MLX faster than CUDA on a different chip."
2. **Include Amazon (the unflattering one) with a DEC-046 footnote**, not cherry-pick around it. The framing is "characterized variant" — the gap is the artifact, not a problem to hide.
3. **Show our depth-6 number against upstream's depth-6 number**, not against their tuned depth-8/10 number. We're not capping their depth-8 run; we just can't run depth>6 ourselves yet.
4. **Re-frame the `gpu_vs_cpu_training_speed/` comparison.** Their harness assumes CUDA-vs-CPU. We re-frame as "MLX-GPU vs CatBoost-CPU on M-series" — different question, same value to the reader.

## Items

| # | Description | Effort | Status |
|---|---|---|---|
| **T0** | Sprint scaffold (branch, plan, dirs, state updates) | 0.5d | DONE (this commit) |
| **T1** | Dataset adapters: load + preprocess each of 5 datasets into MLX-consumable form | 1.5d | TBD |
| **T2** | Run benchmarks across 4 frameworks (LightGBM, XGBoost, CatBoost-CPU, CatBoost-MLX), 3 seeds × 5 datasets | 2d wall-clock (Higgs ≈ 4–8h; Epsilon 1–2h; MSLR ≈ 1h; Adult/Amazon < 30 min each) | TBD |
| **T3** | `docs/benchmarks/v0.5.x-pareto.md` — per-dataset Pareto-frontier scatter + tables + machine-readable JSON | 1.5d | TBD |
| **T4** | Rebuild CI perf-regression gate against runner-matched baselines OR switch to relative speedup (carry-over from S41 bridge mode) | 1d | TBD |
| **T5** | Sprint close-out + (optional) v0.5.2 tag | 0.5d | TBD |

**Total: ~1 sprint** (allowing 2 days wall-clock compute that runs unattended).

## Deliverables

```
benchmarks/upstream/
  ├── adapters/
  │   ├── higgs.py          # libsvm/tsv -> CSV + cat_features=[] config
  │   ├── epsilon.py
  │   ├── adult.py
  │   ├── amazon.py         # cat_features=indices + DEC-046 disclosure in adapter
  │   └── mslr.py           # ranking with group_id
  ├── scripts/
  │   ├── run_subset.sh     # one-shot reproducer (full 4-framework × 5-dataset run)
  │   ├── run_lightgbm.py   # LightGBM-CPU runner
  │   ├── run_xgboost.py    # XGBoost-CPU-hist runner
  │   ├── run_catboost.py   # CatBoost-CPU runner
  │   └── run_catboost_mlx.py  # MLX runner (same hyperparams)
  └── results/
      └── {dataset}_{framework}_{seed}.json    # per-run wall-clock, peak RSS, metric
docs/benchmarks/
  └── v0.5.x-pareto.md      # writeup with scatter plots + honest framing + DEC-046 footnote
```

## Pareto-frontier output spec

Per dataset, a single table + a single scatter plot:

| Framework | Test metric | Train wall-clock | Peak RSS | Hardware |
|---|---|---|---|---|
| LightGBM CPU | 0.XXX | XXs | XGB | M-series (specific chip) |
| XGBoost CPU hist | 0.XXX | XXs | XGB | same |
| CatBoost-CPU | 0.XXX | XXs | XGB | same |
| **CatBoost-MLX** | 0.XXX | XXs | XGB | same, GPU |

Scatter plot: x=wall-clock (log scale), y=test metric (linear), point per framework. "Pareto-optimal" frameworks lie on the convex hull. The writeup explicitly names which framework is Pareto-optimal at each (dataset, configuration) pair.

## CI perf-regression gate rebuild (T4)

Per the S41 close-out, two options to retire the bridge-mode `continue-on-error: true`:

**Option A — runner-matched baseline**: regenerate `.cache/benchmarks/sprint16_baseline.json` on a fresh GitHub-hosted macos-14 run. Restore `continue-on-error: false`. Pros: minimal infrastructure change. Cons: GitHub-hosted runners drift over time; may need periodic refresh.

**Option B — relative speedup gate**: change the gate to compare CPU vs MLX *speedup ratio* (which is invariant to absolute machine speed), not MLX wall-clock. A "regression" then means MLX slowed down *relative to* CPU on the same machine. Pros: hardware-independent; survives runner replacement. Cons: requires both CPU and MLX to run on the same job; doubles CI minutes.

Recommend Option B for robustness; choose during S42-T4.

## Out of scope explicitly

- **Ordered Boosting implementation** (E2, hero feature for v0.6.0)
- **CTR RNG ordering closure** (deferred per characterized > unimplemented)
- **PyPI publish** (T4 audit done in S41; gated on `MACOSX_DEPLOYMENT_TARGET=14.0` build env)
- **Posting the upstream RFC** (gated on DEC-046 trigger conditions; benchmarks here are *one* such trigger getting closer, but Ordered Boosting still gates)
- **HN/Twitter/MLX-Slack launch** (E3, post-Ordered-Boosting)
- **Click dataset** (>16M rows; out of reach until row-cap is lifted)

## Files in scope

- `docs/sprint42/**`
- `benchmarks/upstream/**`
- `docs/benchmarks/v0.5.x-pareto.md`
- `.github/workflows/mlx-perf-regression.yaml` (T4)
- `.cache/benchmarks/sprint16_baseline.json` and/or `.cache/profiling/sprint18/baseline/*.json` (T4)
- `.claude/state/{HANDOFF,TODOS,CHANGELOG-DEV}.md` (close-out)

## Files explicitly NOT in scope

- `catboost/mlx/**.{cpp,h,metal}` — no kernel changes; no source-of-truth changes
- `python/catboost_mlx/**` — no Python API changes
- Anything outside the benchmarks + CI domain
