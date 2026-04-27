# Sprint 43 Plan — Falsification and Highest-ROI Polish

**Sprint:** 43  |  **Status:** IN PROGRESS  |  **Branch:** `mlx/sprint-43-falsification-and-roi`
**Cut from:** master `659ab3d17c` (S42 PR #40 in flight in parallel)
**Theme:** Resolve the strategic ambiguity around v0.6.0 by running cheap experiments before committing to scope. Three moves, all individually justifiable, collectively decisive.

## Strategic context

After the post-S42 advisory-board synthesis (recorded in this session's transcript), four voices identified sharp disagreement on the v0.6.0 narrative:

- **Strategist** — keep Ordered Boosting as v0.6.0 hero, demoted from "gap-closure" to "feature-completeness."
- **Devils-advocate** — shipping anti-leakage tech (Ordered Boosting) on the slowest backend is misleading; v0.6.0 launch landing odds ~25%; the project's "deterministic moat" framing is structurally compromised by being slowest.
- **Silicon-architect** — RED on "fastest GBDT on Apple Silicon at v0.6.0"; YELLOW on "competitive with CatBoost-CPU on numeric workloads" (5–8 sprints to match).
- **Mathematician** — the empirical "MLX matches CatBoost-CPU within ~0.001 logloss" claim is achievable in 1.5 sprints; closure is largely a methodology + alignment problem.

**The user's decision** (recorded in this session): **v0.6.0 is delayed until catboost-mlx can credibly claim "competitive on at least one axis users care about."** S43 produces the empirical data that resolves whether such a claim is achievable in the near term.

## The three cheap moves

| # | Action | Decisive output |
|---|---|---|
| **T1** | **Full 11M Higgs sweep** (4 frameworks × 3 seeds, 200 iters). Devils-advocate's falsification test: at 11M rows where GPU launch overhead is fully amortized, does MLX become competitive with CatBoost-CPU on wall-clock? | Yes → throughput story has legs; v0.6.0 = Ordered Boosting hero. **No** → throughput story structurally broken; v0.6.0 needs strategic pivot. |
| **T2** | **Bump benchmark iters to 1000**, re-run Adult + Higgs-1M (and 11M if T1 lands). Mathematician's claim: at fair convergence, the +0.012 logloss CatBoost-vs-XGBoost gap collapses to ~0.002. Removes a methodology-induced artifact from every number we've published. | Reframes the head-to-head story. If CatBoost-family matches LightGBM/XGBoost at 1000 iters, the published "MLX is +0.0012 vs CPU" claim becomes meaningful in the broader frame. |
| **T3** | **Predict-path binary IPC** (port CTR application to nanobind, expose `_core.predict()`). Silicon-architect's #1 ROI fix in the codebase. <1 sprint. | User-visible win regardless of T1/T2 outcomes. ~95% latency reduction on the cat-feature predict path. Removes the documented 41× subprocess slowdown. The same plumbing applies to the train path in a future sprint. |

T3 ships value regardless of T1/T2; T1/T2 gate the v0.6.0 direction decision (T4).

## Items

| # | Description | Effort | Status |
|---|---|---|---|
| **T0** | Sprint scaffold (branch, plan, dirs, state updates) | 0.5 day | DONE (this commit) |
| **T1** | Full 11M Higgs sweep | 1 day plumbing + ~1 hour compute | TBD |
| **T2** | 1000-iter re-run of Adult + Higgs-1M | 0.5 sprint + ~30 min compute | TBD |
| **T3** | Predict-path binary IPC (CTR via nanobind) | <1 sprint (3–5 days) | TBD |
| **T4** | Synthesis: v0.6.0 direction decision | 0.5 day after T1/T2 land | TBD |
| **T5** | Sprint close-out + (optional) v0.5.3 tag | 0.5 sprint | TBD |

**Total: ~1 sprint** (T1+T2 mostly compute-bound; T3 is the engineering piece).

## v0.6.0 decision branches (T4)

After T1/T2/T3 land, three plausible directions for v0.6.0:

### Branch A — Throughput story holds (T1 shows MLX ≤ CatBoost-CPU at 11M)
v0.6.0 = Ordered Boosting + 5-dataset characterization + predict-fix + PyPI publish. The strategist's prior plan, validated. ~6 sprints out.

### Branch B — Accuracy narrative pivot (T1 inconclusive but T2 shows CatBoost-family converges)
v0.6.0 = "competitive accuracy at fair convergence" + CTR-RNG closure (Lane D) + predict-fix. Reframes the launch story away from "fastest" to "accurate, deterministic, Apple Silicon-native." ~4 sprints.

### Branch C — Strategic pivot (T1 fails AND T2 doesn't help)
v0.6.0 deferred indefinitely. Project enters Lane D mechanism investigation (CTR RNG + architectural floor closure) OR a more fundamental pivot (e.g., narrowing the product to a niche where MLX has a genuine advantage — see visionary's earlier "RS=0 deterministic moat / Pareto frontier as the product / MLX-ecosystem citizen" reframings). Devils-advocate gets a follow-up round.

The choice is data-driven, not aspirational. T4 records which branch the data points to.

## Out of scope (deferred to v0.6.x or later)

- Ordered Boosting implementation (E2 hero, ~5 sprints, gated on T1 outcome)
- CTR RNG ordering closure (Lane D) — feasible per mathematician (~1 sprint Option 1) but only relevant if v0.6.0 has a launch story to wrap it in
- Histogram-stage CI gate redesign (S42 carry-over)
- Full deterministic-tiebreak fix for Adult numeric-only floor (mathematician: "2+ sprints, low value")
- `max_depth > 6`, 16M-row cap, NewtonL2/Cosine score functions
- HN/Twitter/MLX-Slack launch (E3) — explicitly delayed until v0.6.0 has a defensible "competitive on" claim

## Files in scope

- `docs/sprint43/**`
- `benchmarks/upstream/adapters/higgs.py` (T1 — re-materialize full scale)
- `benchmarks/upstream/scripts/_runner_common.py` (T2 — iters parameter)
- `benchmarks/upstream/results/higgs_*` (T1 + T2 output)
- `benchmarks/upstream/results/adult_*_iter1000_*` (T2)
- `python/catboost_mlx/_core.cpp` or equivalent + `python/catboost_mlx/core.py` (T3)
- `python/tests/` (T3 verification)
- `docs/benchmarks/v0.5.x-pareto.md` (extend with 11M Higgs and 1000-iter sections)
- `.claude/state/{HANDOFF,TODOS,CHANGELOG-DEV}.md` (close-out)

## Files explicitly NOT in scope

- `catboost/mlx/**.{cpp,h,metal}` — no kernel changes
- `python/catboost_mlx/core.py` Python API surface — only the predict-dispatch path changes (T3)
- `.github/workflows/` — perf-gate redesign is in flight on PR #40 (S42-T4)

## Branch + PR plan

Single branch `mlx/sprint-43-falsification-and-roi`. Single PR. Optional v0.5.3 patch tag post-merge if T3 ships and produces measurable predict-perf improvement.
