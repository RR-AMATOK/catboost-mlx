# Pareto Lane Dashboard — Specification

**Sprint:** 45  |  **Status:** SPEC — do not implement until ml-engineer + technical-writer are assigned
**Author:** data-scientist  |  **Date:** 2026-04-30
**Builds on:** S44-T1 Epsilon gate, S43-T2 Higgs-1M iter grid, S43-T4 Branch B lock
**Dashboard ships in:** S45 (minimum viable); S46-T1 (Plotly HTML final form)

---

## TL;DR (200 words)

The Pareto Lane dashboard is the primary v0.6.0 launch artifact. It makes one argument visible: on Apple Silicon, at fair convergence, CatBoost-MLX delivers accuracy statistically indistinguishable from CatBoost-CPU on numeric workloads — and both CatBoost backends beat LightGBM and XGBoost on high-dimensional workloads (Epsilon) by ~0.006 logloss at iter=2000.

The data does not support a speed claim. MLX is 5–16× slower than CatBoost-CPU across all measured datasets. The dashboard must not hide this. The positioning is not "fastest GBDT on Apple Silicon" — it is "GPU-native CatBoost semantics on Apple Silicon, accuracy-equivalent to CPU, requiring no CUDA hardware."

The Pareto Lane is structurally honest: MLX is dominated by CatBoost-CPU on every single (dataset, iter) cell (CPU is both faster and more accurate). The dashboard's narrative axis is therefore not standard Pareto-optimality (accuracy × wall-clock) but a reframing: the relevant frontier for an Apple-Silicon-only user is accuracy × hardware-class, where MLX enables the CatBoost algorithm on GPU without a CUDA box.

Dollar-cost-per-model framing was tested and rejected (see Section 7). The dashboard uses wall-clock as the x-axis with an explicit "Apple Silicon M3 Max" label. No cloud-cost comparisons.

The minimum viable dashboard is four static matplotlib PNGs (one per measured dataset with MLX data) plus a convergence-trajectory chart for Epsilon. Estimated build: 3 person-days.

---

## 1. Goal and Non-Goals

**Goal.** The Pareto Lane dashboard is a set of benchmark charts embedded in `docs/benchmarks/v0.6.0/` and linked from the v0.6.0 README. Its audience is:

- **HN reader** (primary): a data engineer or ML practitioner evaluating Apple Silicon ML tooling, who will spend 30 seconds on the README and needs one visual that answers "does this actually work vs the alternatives?"
- **Data engineer evaluating Apple Silicon ML** (secondary): someone who owns an M-series Mac and runs CatBoost workflows locally, deciding whether catboost-mlx is a credible replacement for catboost-cpu or whether to just keep using CatBoost-CPU on the same machine.
- **Internal team**: sprint close-out artifact; grounds the HN/RFC launch post.

The dashboard supports exactly one decision: "Is catboost-mlx producing credible CatBoost-equivalent accuracy on my Apple Silicon hardware?" The answer the data supports is yes on numeric workloads at fair convergence, no on categorical workloads, and slower on wall-clock in all cases.

**Non-goals.**
- NOT a claim that MLX is faster than any other framework on any dataset.
- NOT a leaderboard ranking all 4 frameworks by "best" — this is a characterization, not a competition.
- NOT live-updating from production or CI. Static files committed to the repo.
- NOT a throughput headline. Speed numbers appear only as context for the accuracy claim.
- NOT a cloud-cost comparison. Dollar-cost framing was evaluated and rejected (see Section 7).

---

## 2. Axes and Cells

### Primary axis choice

**x-axis: train wall-clock (seconds, log scale, M3 Max).** Familiar, concrete, reproducible by any reader with Apple Silicon. Lower is faster (left).

**y-axis: test logloss (or NDCG@10 for MSLR).** Lower is better (down). The metric the S44 convergence-trajectory tables already use.

This is a standard accuracy-vs-time Pareto scatter. One point per (framework, iter-level) pair, error bars across 3 seeds. Color-coded by framework using the existing palette in `make_pareto_plots.py`.

### Plot structure: one convergence overlay per dataset, not one static scatter

The current plotter produces one scatter per dataset at a fixed iter count. For the Pareto Lane, the critical insight is the convergence trajectory — especially the Epsilon crossover where CatBoost overtakes LightGBM at iter=2000. The dashboard therefore uses a **dual-panel layout per dataset**:

- **Panel A (left):** Accuracy-vs-wall-clock scatter at the dataset's `argmin_iter` (best iter on the measured grid). Standard Pareto plot. Pareto-optimal points highlighted; dominated points faded.
- **Panel B (right):** Logloss-vs-iter convergence curves, one line per framework (x-axis: iter count; y-axis: mean logloss). This is the chart that shows the Epsilon crossover and MLX-vs-CPU gap narrowing.

One combined HTML page with a dataset selector (dropdown) is preferred over separate PNGs, because: (a) the convergence panel is meaningless without the dataset context; (b) a single URL is easier to share on HN; (c) Plotly HTML is self-contained and survives 12 months without a running server. See Section 5.

### Dollar-cost framing: proposed, tested, rejected

The mathematician proposed $/converge using M3 Max all-in cost (amortized machine + power = $0.6485/hr at $4k machine, 3yr, 8hr/day, 50W).

Computed values:

| Cell | Train time | $/model (M3 Max local) |
|---|---|---|
| Epsilon iter=2000, catboost_mlx | 4,482 s | $0.807 |
| Epsilon iter=2000, catboost_cpu | 282 s | $0.051 |
| Epsilon iter=2000, lightgbm | 501 s | $0.090 |
| Higgs-1M iter=1000, catboost_mlx | 129 s | $0.023 |
| Higgs-1M iter=1000, catboost_cpu | 25 s | $0.004 |

MLX costs 16× more per model than catboost-cpu on the same machine for Epsilon. The dollar-cost framing makes MLX look worse, not better, because the time gap is not covered by any machine-cost differential when both frameworks run on the same hardware. For the framing to favor MLX, you would need to compare a local M3 Max against a cloud GPU instance — which introduces counterfactual hardware (the user may not have a CUDA box), non-trivial cloud overhead assumptions, and minimum-billing-unit distortions ($0.60/hr on a 1-minute minimum = $0.01/run minimum, which is cheap for any workload). The comparison becomes too speculative for a launch document. **Use wall-clock only.** Note the hardware in the chart footer ("Apple M3 Max, macOS 26.3").

### "Converged" definition

For the convergence panels, "converged" is defined as: the iter level where the logloss trajectory flattens — operationally, the `argmin_iter` on the measured grid {200, 500, 1000, 2000} per (framework, dataset). Where only a subset of iter levels were run (Adult: {200, 1000}; Higgs-1M: {200, 1000}), report all available points and note the limited grid in the chart caption.

---

## 3. Where catboost-mlx Wins (and Where It Does Not)

**Structural honesty first:** on every measured (dataset, iter) cell, catboost-mlx is dominated on the standard Pareto frontier by catboost-cpu — which is both faster and more accurate or equally accurate on the same machine. The dashboard cannot claim MLX is Pareto-optimal in the classic sense. This must be stated, not hidden.

The actual catboost-mlx wins are:

**Epsilon (400k × 2000 numeric) — the strongest cell.**
At iter=2000 (3 seeds, M3 Max): catboost-mlx logloss = 0.26818 ± 0.0002, LightGBM = 0.27361 ± 0.0003, XGBoost = 0.27428 ± 0.0000. MLX beats LightGBM by 0.00543 and XGBoost by 0.00610 on accuracy alone. MLX is also 9–16× slower than LGB/XGB in wall-clock. MLX is 15.9× slower than catboost-cpu (4,482 s vs 282 s). The win claim: "for users who specifically want CatBoost semantics on Apple Silicon, MLX delivers the same accuracy as catboost-cpu within fp32 noise (+0.00055), and both CatBoost backends beat LightGBM and XGBoost on this 2000-feature workload at fair convergence." Source: `benchmarks/upstream/results/epsilon_iter2000_*.json`.

**Higgs-1M (1M × 28 numeric) — accuracy parity, slower.**
At iter=1000: catboost-mlx logloss = 0.50596 ± 0.0001, catboost-cpu = 0.50580 ± 0.0002 (gap +0.0002, within fp32 noise). XGBoost = 0.50089, LightGBM = 0.50045. MLX is parity with catboost-cpu but 5.25× slower, and still behind XGBoost/LGB by ~0.005. Win: "GPU-native CatBoost delivers fp32-noise-equivalent accuracy to CatBoost-CPU on Higgs at fair convergence." Source: `benchmarks/upstream/results/higgs_iter1000_*.json`.

**Higgs-11M (10.5M × 28 numeric) — accuracy holds at scale, slowdown unchanged.**
At iter=200: catboost-mlx logloss = 0.53036 ± 0.0002, catboost-cpu = 0.52909 ± 0.0002 (gap +0.0013). MLX/CPU ratio = 5.16× — identical to Higgs-1M (5.41×), confirming the slowdown is structural. No MLX-specific win. Source: `benchmarks/upstream/results/higgs_11m_*.json`.

**Adult (32k × 14, 8 cats) — MLX LOSES, do not claim a win.**
catboost-mlx logloss = 0.44640 vs catboost-cpu = 0.27692 — a 0.1695 gap. 61% is categorical-encoding asymmetry (DEC-046). MLX at iter=1000 is unchanged (0.44640) — the gap does not close with more iters. Source: `benchmarks/upstream/results/adult_*.json`. Framing: "categorical workloads have a documented wider gap due to CTR encoding asymmetry; the bounded-gap claim is restricted to numeric workloads."

**Amazon (32k × 9 cats) — MLX data missing from results.**
Results directory contains catboost-cpu, LightGBM, XGBoost at {200, 500, 1000, 2000} iters, but NO catboost-mlx runs. T2 sweep was launched in S44 but is not yet in the results. Amazon is all-categorical (9/9 features); per DEC-046, expect a wide gap. No MLX win claim possible; document as "pending or excluded per DEC-046 categorical characterization."

**MSLR-WEB10K — no data.**
No results files. T3 sweep not yet run. Dashboard must launch without MSLR or defer its panel to v0.6.x.

**Pareto frontier sketch (Epsilon iter=2000, the headline cell):**

```
logloss (lower = better)
0.268 |  [mlx]
      |  [cpu]*         <- Pareto frontier (1 point)
0.270 |
0.272 |
0.274 |  [lgb]  [xgb]
      +---------+---------> train_seconds (log)
         100      1000   10000
```

`[cpu]` is the single Pareto-optimal point (fastest + best accuracy). `[mlx]` is above and to the right — dominated on both axes. `[lgb]` and `[xgb]` are faster than `[mlx]` but have worse accuracy.

---

## 4. Data Pipeline

**Input:** `benchmarks/upstream/results/*.json` — one file per (dataset, framework, seed) run. Schema (from live files): `framework`, `framework_version`, `dataset`, `task`, `metric_name`, `metric_value`, `seed`, `train_seconds`, `predict_seconds`, `peak_rss_bytes`, `n_train`, `n_test`, `n_features`, `cat_indices`, `hyperparameters`, `hardware`, `python_version`, `notes`.

**Aggregator:** `benchmarks/upstream/scripts/aggregate_results.py` already groups by `(dataset, framework, metric_name)` and computes mean ± std for metric, train_s, predict_s, RSS. Output: CSV and a markdown table fragment.

**What the aggregator is missing for the Pareto Lane:**

1. `n_cat` (derived as `len(cat_indices)`) — needed for dataset metadata cards.
2. `iter` parsed from the dataset name field (e.g., `epsilon_iter2000` → iter=2000) — needed for the convergence-curve panels. The aggregator currently treats `epsilon_iter2000` and `epsilon_iter1000` as separate datasets rather than the same dataset at different iter levels.
3. A `pareto_optimal` boolean flag per (dataset, iter, framework) row.
4. Dataset metadata table (hardcoded, not in JSONs): n_rows, n_features, n_cat, task_type, metric, notes.

**Recommended output format:** a single `docs/benchmarks/v0.6.0/pareto_data.json` with two sections:

```json
{
  "metadata": {
    "generated": "<ISO timestamp>",
    "hardware": "Apple M3 Max | macOS 26.3",
    "datasets": { "<name>": { "n_rows": ..., "n_features": ..., "n_cat": ..., "task": "...", "metric": "..." } }
  },
  "runs": [
    { "dataset_base": "epsilon", "iter": 2000, "framework": "catboost_mlx",
      "metric_mean": 0.26818, "metric_std": 0.0002,
      "train_s_mean": 4482.3, "train_s_std": 154.2,
      "n_seeds": 3, "pareto_optimal": false }
  ]
}
```

This JSON is what the Plotly HTML page loads. The `aggregate_results.py` script should be extended (or a new `build_pareto_data.py` script written) to produce it.

---

## 5. Rendering

**Recommendation: one self-contained Plotly HTML file.**

Rationale: (a) Plotly's `fig.write_html(..., include_plotlyjs='cdn')` produces a single ~50 KB HTML file that any browser opens without a server — survives indefinitely. (b) A dataset dropdown (`updatemenus`) and dual-panel layout (scatter + convergence curve side by side) fit Plotly's figure API without a running backend. (c) The HN reader links to the HTML file in the repo or a GitHub Pages URL; 30-second grok is achievable with clear axis labels and a two-sentence caption per panel.

Fallback: if Plotly is unavailable or too much build complexity, four static matplotlib PNGs (Adult, Higgs-1M, Epsilon, Higgs-11M) in `docs/benchmarks/v0.6.0/plots/` plus a fifth PNG for the Epsilon convergence trajectory. These can be auto-generated by extending `make_pareto_plots.py`. The PNG fallback is the minimum viable launch artifact.

**Proposed charts (MVP, static PNG path):**

- **Chart 1** — Epsilon convergence trajectory. x: iter {200, 500, 1000, 2000}; y: mean logloss; one line per framework; error band = ±1 std across seeds. Caption: "At iter=2000 both CatBoost backends overtake LightGBM and XGBoost by ~0.006 logloss." This is the single most important chart in the dashboard.
- **Chart 2** — Epsilon Pareto scatter at iter=2000. x: train wall-clock (log); y: logloss; one point per framework. catboost-cpu is the Pareto point. Caption notes MLX/CPU gap = +0.00055, within fp32 noise.
- **Chart 3** — Higgs-1M Pareto scatter at iter=1000. Shows fp32-noise parity between MLX and CPU; XGBoost/LGB remain ~0.005 ahead.
- **Chart 4** — MLX-vs-CPU logloss gap as a function of iter (bar chart). Datasets: Epsilon and Higgs-1M. Shows the gap narrowing from 0.0036 at iter=200 to 0.00055 at iter=2000. Reinforces the "architectural floor" narrative.

**Interactive Plotly HTML (preferred for HN link):** wraps all four charts with a dataset selector. The convergence-trajectory and Pareto-scatter panels display side by side for the selected dataset.

---

## 6. Implementation Cost and Owner

| Task | Owner | Estimate | Scope |
|---|---|---|---|
| Extend aggregator: parse iter from dataset name, add pareto flag, emit pareto_data.json | ml-engineer | 0.5 days | v0.6.0 |
| Chart 1: Epsilon convergence trajectory (matplotlib PNG, then Plotly port) | ml-engineer | 0.5 days | v0.6.0 |
| Charts 2–4: Pareto scatters + gap-vs-iter bar (extend make_pareto_plots.py) | ml-engineer | 0.5 days | v0.6.0 |
| Plotly HTML wrapper with dataset selector | ml-engineer | 0.5 days | v0.6.0 |
| README + launch post integration (embed Chart 1 PNG; link to HTML) | technical-writer | 1 day | v0.6.0 |
| **Total** | | **3 person-days** | |

**Minimum viable Pareto Lane (day 1 of S45):** Chart 1 only (Epsilon convergence trajectory PNG) embedded in the README. This alone supports the headline launch claim. Charts 2–4 and the Plotly HTML can follow in S45 days 2–3.

**v0.6.x follow-up:** MSLR-WEB10K panel (once T3 sweep completes); Amazon MLX runs (once DEC-046 CTR characterization is complete or MLX runs are added); interactive tooltip showing raw seed values on hover.

---

## 7. Risks and Failure Modes

**Risk 1: The data does not show a clean Pareto win for MLX.**
This is already true. catboost-mlx is Pareto-dominated by catboost-cpu on every measured cell. The mitigation is to reframe the dashboard as a characterization, not a competition: "here is what CatBoost-MLX accuracy looks like relative to all alternatives on Apple Silicon." The Epsilon convergence trajectory (Chart 1) is the strongest honest chart — it shows CatBoost FAMILY winning at fair convergence without requiring MLX to beat catboost-cpu specifically.

**Risk 2: Dollar-cost framing comes off as contrived.**
Confirmed by the numbers: MLX costs $0.81 vs CPU's $0.05 per Epsilon iter=2000 model on the same hardware. Any HN reader who runs the math will notice. The dollar-cost framing is dropped from the dashboard entirely. If the question "why pay for GPU if CPU is faster?" is raised on HN, the honest answer is: "on this hardware, CPU is currently faster; MLX is a proof that CatBoost-equivalent accuracy is achievable on Apple Silicon GPU, and the gap closes at fair convergence vs LightGBM/XGBoost." That is the defensible position.

**Risk 3: The audience conflates "MLX vs catboost-cpu" with "MLX vs GPU-accelerated CatBoost on CUDA."**
The hardware footer on every chart ("Apple M3 Max | macOS 26.3") and a one-sentence methodology note ("all comparisons on the same Apple Silicon machine; no CUDA hardware used") mitigate this. The README should NOT include wall-clock comparisons against published x86/CUDA numbers (per the S40 advisory constraint).

**Risk 4: Amazon and MSLR data are absent at launch.**
Amazon MLX runs are missing from the results directory. MSLR has no data. The v0.6.0 dashboard launches with 3 datasets (Epsilon, Higgs-1M, Higgs-11M at iter=200 only). Adult is included with an explicit "categorical workloads have a documented wider gap" callout. No MSLR panel until data exists.

**Risk 5: The Plotly HTML approach over-engineers the MVP.**
Mitigation: build the matplotlib PNG version first (0.5 days). Ship the PNG in the README on day 1. Port to Plotly HTML in S45 day 2–3 if time permits. The spec is written so either path ships a defensible dashboard.

**v0.7.x evolution path.** The natural next step is to add a GPU-class dimension to the Pareto frontier — comparing MLX against CatBoost-CUDA on equivalent hardware (e.g., M3 Max vs A10G cloud instance) once the train-binary-IPC sprint closes the subprocess overhead. At that point the dollar-cost framing becomes legitimate because the two machines have materially different acquisition costs and the time gap may narrow. For v0.6.0, that comparison does not exist in the data.
