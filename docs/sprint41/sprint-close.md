# Sprint 41 Close — Polish-to-Trust (E1)

**Sprint:** 41  |  **Status:** READY-TO-CLOSE  |  **Date:** 2026-04-26
**Branch:** `mlx/sprint-41-polish`  |  **Base:** master `aac00046a1` (post v0.5.0)
**Authoritative record:** this file
**Theme:** Close every paper-cut a v0.5.0 user could hit on the happy path. Make the README the single document a new user needs. Stage Track B (upstream RFC) without posting. Honestly profile and document the predict() subprocess slowdown. Audit PyPI publish-readiness without publishing.

---

## Mission

Following DEC-046's lane lock (Lane B / "RS=0 deterministic moat" framing, PyPI + community evangelism as primary distribution), the immediate work is no longer correctness — it is **adoption**. S41 closes the smallest set of paper cuts that make v0.5.0 trustworthy in the hands of a new user, plus stages the upstream-coordination courtesy issue (refreshed but not posted) and audits the package for future PyPI publication.

No source-of-truth code changes. No kernel changes. Production kernel v5 (`784f82a891`) byte-identical from S30 through S41 (md5 `9edaef45b99b9db3e2717da93800e76f`).

---

## Items Completed

| Item | Description | Outcome | Commit |
|------|-------------|---------|--------|
| **T1** | `bootstrap_type` validator case-insensitive | Validator at `core.py:632` now accepts `'No'`, `'NO'`, `'no'`, `'Bayesian'`, `'BAYESIAN'`, etc. and normalizes to lowercase internally. New test `test_bootstrap_type_case_insensitive` covers 7 case variants. Existing `test_invalid_bootstrap_type` still passes. | `d0bc7a1a87` |
| **T2** | README "Installation & Quick Start" + 30-second smoke test | New top-of-document section between "When to use" and Feature Status. Covers prerequisites, source install, 30-second smoke test (synthetic data, train, predict, print "OK"), optional CPU parity verification, CLI quick test. Smoke test verified end-to-end on dev install. | `b570d5c154` |
| **T3** | `predict()` subprocess slowdown — profile + document | Mechanism identified at `core.py:1769`: numeric-only models already use in-process NumPy tree evaluator (~940k rows/s). Subprocess path triggers only for categorical models. Phase breakdown: 58% CSV serialization of input, 35% binary work, <1% other. README §"Python API uses subprocess" updated with two-row mechanism table, per-path throughput numbers, and three workarounds (pre-encode categoricals, batch calls, embed via library path). | `50e2c7f9d4` |
| **T4** | PyPI publish-readiness audit | Built sdist (382 KB) + wheel (421 KB) cleanly via `python -m build`. Sdist content audit: 47 files; no `.git`, `.env`, `.claude`, `.cache`, secrets. Wheel installs into a fresh venv with only numpy + mlx as deps; smoke test passes against the venv-installed package. One must-fix gate identified (F1): wheel macOS deployment target stamps as `macosx_26_0_arm64` because the build host is on macOS 26.3 — needs `MACOSX_DEPLOYMENT_TARGET=14.0` for production. Non-blocking; applied at PyPI publish time. | `118d63246e` |
| **T5** | Refresh upstream RFC draft (STAGED, not posted) | Replaced the Sprint 15 draft with a v0.5.0-aligned version. Reframed from "feature proposal / want to upstream" to "informational discussion issue / coordination-of-interest check" per DEC-046. Added explicit "What it does NOT do" section (Ordered Boosting, depth ≤ 6, 16M cap, NewtonL2/Cosine, CTR characterized gap). Added the five trigger conditions before any actual PR submission. Tone shift: lowered commitment level. Status header updated to "STAGED — NOT YET POSTED" with a pointer to DEC-046's trigger conditions. | `c08fa10cda` |
| **T6** | Sprint close-out + (optional) v0.5.1 patch tag | This file + state updates + close-out PR. v0.5.1 tag if T1-T5 land cleanly. | (this commit) |

---

## Out of Scope (deferred)

| Item | Status | Reason |
|------|--------|--------|
| **Ordered Boosting (E2, 5 sprints)** | Hero feature for v0.6.0 | Major work; per-permutation gradient path requires significant Metal kernel work |
| **CTR RNG ordering closure (narrow Lane D)** | Indefinitely deferred | Per DEC-046: characterized > unimplemented; the gap is bounded and documented; closure is optional post-release |
| **PyPI publish (the actual `twine upload`)** | Pending | Audit complete (T4); F1 (deployment target) must be addressed at publish time; release infrastructure pending |
| **HN/Twitter/MLX-Slack launch (E3)** | Post-Ordered-Boosting | Per strategist's roadmap; launch needs the marquee feature first |
| **Posting the upstream RFC** | Staged, not posted | Per DEC-046 trigger conditions; gated on at least Ordered Boosting + S42 benchmarks |
| **Upstream-comparable benchmarks (S42)** | Planned | Plan recorded by ml-product-owner; opens after S41 close-out merges |

---

## Mechanism Findings (T3)

The 41× MLX-vs-CPU `predict()` slowdown observed in the irrigation Kaggle notebook (270k × 53 features, 6 chunks of 50k) was the trigger for this investigation. Profile output (`docs/sprint41/profile_output.txt`) shows:

- **In-process path** (numeric-only models, `cat_features=[]`): ~53 ms / 50k rows = ~940k rows/s. Within ~1.5× of CatBoost-CPU's `predict()` throughput.
- **Subprocess path** (any categorical features): ~450 ms / 50k rows = ~111k rows/s.
- **Subprocess phase breakdown**:
  - write data.csv: 262 ms (58% — *dominant cost*)
  - subprocess csv_predict run: 153 ms (35% — binary load + Metal init + actual predict)
  - write model.json + read predictions.csv: ~0.6 ms total

CSV serialization scales linearly with `n_rows × n_features`, which is why the irrigation footprint produced 41× while the smaller test config produced 8.5×.

**No code change shipped for T3.** Two real fixes exist (binary IPC for csv_predict, or port C++ CTR-application logic to Python for in-process predict), both non-trivial. T3 closed as "documented + workaround steered" — the strategist's predicted outcome and the right call given engineering cost.

**User workarounds documented in README**:
1. Pre-encode categoricals (one-hot, target-encode at training time) and train with `cat_features=[]` → routes predict through the in-process path.
2. Batch predict calls — overhead is per-call, not per-row.
3. Embed via `mlx_boosting.h` C++ library to bypass Python entirely.

---

## CI Status

| Workflow | PR Run (expected) | Push Run | Notes |
|---|---|---|---|
| Compile csv_train (Apple Silicon) | pass | pass | |
| MLX Python Test Suite (macos-14, py3.13) | pass (incl. new `test_bootstrap_type_case_insensitive`) | pass | |
| `mlx-perf-regression.yaml` | not triggered (path filter excludes docs/state changes) | not triggered | Fixed in S40 close-out PR #38; correctly skips when paths don't match `catboost/mlx/**`, `benchmarks/bench_mlx_vs_cpu.py`, `python/catboost_mlx/**` |

The S40 perf-regression fix is now visibly working — the workflow correctly skips on doc-only branches like this one.

---

## Files Changed

```
docs/sprint41/                                      (new)
  sprint-plan.md                                    (new — sprint scaffold)
  profile_predict.py                                (new — T3 profiler)
  profile_output.txt                                (new — T3 profile result)
  T4-pypi-readiness.md                              (new — T4 audit)
  sprint-close.md                                   (new — this file)
docs/upstream_issue_draft.md                        (T5 — refreshed for v0.5.0; STAGED)
catboost/mlx/README.md                              (T2 + T3 — install/test path + predict() doc)
python/catboost_mlx/core.py                         (T1 — validator)
python/tests/test_basic.py                          (T1 — new test)
.claude/state/HANDOFF.md                            (close-out)
.claude/state/TODOS.md                              (close-out)
.claude/state/CHANGELOG-DEV.md                      (close-out)
```

No `catboost/mlx/**.{cpp,h,metal}` source changes. No kernel changes.

---

## Branch Lifecycle

- `mlx/sprint-41-polish`: 6 commits (scaffold + T1 + T2 + T3 + T4 + T5 + this close-out, single PR).

---

## Next Sprint Entry Point

S41 closes with a clear next sprint defined: **S42 — Upstream Benchmark Adoption** (per ml-product-owner plan, 2026-04-26 advisory).

- Subset: Higgs (depth 6), Epsilon, MSLR-WEB10K, Yahoo-LTR, Amazon (5 datasets)
- Effort: ~1 sprint (1.5d adapters + 2d compute + 1.5d Pareto-frontier writeup)
- Deliverable: `benchmarks/upstream/` adapters + `docs/benchmarks/v0.5.x-pareto.md` Pareto frontier
- Honest framing: M-series only (no comparison vs A100), include Amazon (the unflattering one) with DEC-046 footnote
- Output ships as v0.5.x evidence pack; cited backbone for the eventual T5 upstream RFC and the E3 launch when v0.6.0 ships Ordered Boosting

**Optional v0.5.1 tag**: post-merge, tag `v0.5.1` covering T1 (bootstrap_type) + T3 documentation + T5 RFC stage. Notes drawn from this close-out and the T1/T3 commits.
