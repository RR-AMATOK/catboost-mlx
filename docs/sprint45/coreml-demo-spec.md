# CoreML Pipeline Demo — Scoping Spec

**Sprint:** 45  |  **Status:** SPEC — awaiting go/no-go decision  
**Author:** technical-writer  |  **Date:** 2026-04-30  
**Context:** Advisory board concluded catboost-mlx is Pareto-dominated on every (dataset, iter) cell.  
**Proposed pivot:** category-creation framing — "the missing GBDT in your CoreML pipeline."

---

## TL;DR (200 words)

CoreML export exists and is structurally sound. `export_coreml.py` correctly unfolds oblivious trees into CoreML's `TreeEnsemble` spec, handles both regression and classification, and is tested for spec validity. The optional `coremltools>=7.0` dependency is declared in `python/pyproject.toml`. The `export_coreml()` method is on the public API at `core.py:1866`.

However, three risks undermine the pitch before it ships: (1) **upstream CatBoost already supports CoreML** — `catboost.CatBoost.save_model(format="coreml")` is documented in the installed package — so the "only GBDT with CoreML" claim is false on arrival; (2) **CoreML `TreeEnsemble` models do not run on the Apple Neural Engine** — ANE compiles only neural-network-shaped compute graphs (Conv, MatMul, softmax); decision trees fall back to CPU in the CoreML runtime; the "run on iPhone's Neural Engine" hook in the pitch is technically wrong; (3) **the tests validate spec structure only, not prediction correctness** — neither `test_export_coreml_regression` nor `test_export_coreml_classification` loads the model and runs inference, so round-trip numerical accuracy is unverified.

**Recommendation: (c) Kill the CoreML demo as the v0.6.0 launch artifact.** The pitch's two load-bearing claims — "only GBDT with CoreML" and "runs on ANE" — are both false. The Pareto Lane spec (already drafted in `docs/sprint45/pareto-lane-spec.md`) is the correct launch artifact.

---

## Q1 — Does catboost-mlx already support CoreML export?

**Yes, and it is structurally complete.**

| Item | Location | State |
|---|---|---|
| `export_coreml()` public method | `python/catboost_mlx/core.py:1866–1874` | Implemented, delegates to `export_coreml.py` |
| Core export logic | `python/catboost_mlx/export_coreml.py` | 123 lines; handles regression + binary + multiclass |
| Oblivious-tree unfolding | `python/catboost_mlx/_tree_utils.py` | BFS expansion; NaN bin offset; one-hot category branches |
| Optional dependency | `python/pyproject.toml:54` | `coremltools>=7.0` under `[project.optional-dependencies.coreml]` |
| Spec-validity tests | `python/tests/test_basic.py:981–1000` | Two tests: assert `HasField("treeEnsembleRegressor")` / `treeEnsembleClassifier` |
| Pre-fit guard test | `python/tests/test_qa_adversarial.py:835–838` | Asserts `RuntimeError` before fit |

The implementation correctly:
- Uses `ct.models.tree_ensemble.TreeEnsembleClassifier` / `TreeEnsembleRegressor` from `coremltools`
- Maps numeric splits to `BranchOnValueGreaterThan` (line 104)
- Maps one-hot categorical splits to `BranchOnValueEqual` (line 96)
- Applies `Classification_SoftMax` post-evaluation transform for classifiers (line 71)

**Critical gap: no end-to-end inference test.** Both CoreML tests stop at `spec.HasField(...)`. Neither loads the model, runs `model.predict()`, and compares output against the catboost-mlx native prediction. Round-trip numerical correctness is unverified in the test suite.

**`coremltools` is not installed** in the project's active conda environments (`base`, `machine_learning`, `python_cli`). The tests would be skipped via `pytest.importorskip("coremltools")` in any current CI run.

---

## Q2 — What does the demo actually show?

Proposed 6-minute storyboard, with per-segment assessment:

| Time | Frame | Code required | What exists | Risk |
|---|---|---|---|---|
| 0:00–0:30 | Hook: "No GBDT has a clean train→CoreML→ANE story." | None — narration only | — | **FATAL**: the hook's central claim is wrong (see Q4). Upstream CatBoost already exports CoreML. ANE does not run tree ensembles. |
| 0:30–2:00 | `pip install catboost-mlx[coreml]`; fit on a Kaggle dataset (e.g., Titanic or California Housing, ~1k rows, ~60s train) | Training script (~30 LoC) | Nothing — needs to be written | Low on its own; coremltools install not yet confirmed in dev env |
| 2:00–3:30 | `model.export_coreml("model.mlpackage")`; show file size; show schema in Netron or Quick Look | Export already exists (`core.py:1866`) | `export_coreml()` is implemented | Medium: `coremltools` not installed; output is `.mlmodel` (proto), not `.mlpackage` (directory bundle) — the API uses `save_spec` not `ct.convert`, so the package format depends on the path suffix handling in `coremltools.models.utils.save_spec` — needs verification |
| 3:30–5:00 | Drop `.mlpackage` into SwiftUI app; run on-device inference on iPhone | Minimal SwiftUI app (~50 LoC + Xcode project) | Nothing — needs to be written | Medium-high: requires Xcode, provisioning, a physical device or simulator. CoreML inference on simulator is CPU-only (not ANE). Physical device required for any hardware utilization claim. |
| 5:00–6:00 | Show ANE utilization in Instruments; "no other GBDT has this"; GitHub link | Instruments capture; narration | — | **FATAL**: `TreeEnsemble` models do not compile to ANE. Instruments will show CPU utilization, not ANE. The "ANE utilization" segment cannot be recorded. |

---

## Q3 — What needs to be built?

| Work item | Estimate | Blocker? |
|---|---|---|
| (a) Verify CoreML export end-to-end: install `coremltools`, run existing tests, confirm `.mlpackage` output, add inference round-trip test | 0.5 days | No — but currently untested at inference level |
| (b) Demo training script (dataset download, fit, export — ~30 LoC) | 0.25 days | No |
| (c) Minimal SwiftUI iOS app consuming `.mlpackage` | 1.5 days | Requires Xcode, Apple Developer account, physical device for ANE segment |
| (d) Instruments capture of ANE utilization | 0.5 days | **FATAL BLOCKER**: `TreeEnsemble` does not run on ANE; this segment cannot be produced |
| (e) Screencast recording + editing | 1.0 days | Depends on all segments being producible |
| **Total if (d) were possible** | **~3.75 days** | — |
| **Total excluding ANE segment** | **~3.25 days** | The demo loses its closing hook |

**The load-bearing work item is (a), and it reveals (d) is impossible.** Even if the export is functionally correct, the 5:00–6:00 ANE segment — which is the differentiator that makes the demo memorable — cannot be filmed.

---

## Q4 — What could undermine the pitch?

### Upstream CatBoost already supports CoreML (FATAL)

The installed `catboost` package exposes `save_model(format="coreml")`. This is documented in the method's help string. The claim "no other GBDT has a clean CoreML story" is false. Upstream CatBoost has had CoreML export for several years (it predates MLX). LightGBM and XGBoost do not have native CoreML export, but the canonical GBDT with CoreML is already CatBoost-CPU, not catboost-mlx.

The most defensible claim catboost-mlx could make would be: "GPU-native CatBoost on Apple Silicon with CoreML export in one toolchain." But this is a minor ergonomic convenience over running CatBoost-CPU (which also runs natively on Apple Silicon and also exports CoreML), not a category creation.

### ANE does not run TreeEnsemble models (FATAL)

CoreML's `TreeEnsemble` spec type maps to a CPU implementation in Core ML's runtime. The Apple Neural Engine compiles only neural-network compute graphs (convolution, matrix multiply, normalization, activation). Decision trees traverse conditional branches — a control-flow pattern that ANE hardware cannot express. Apple's own CoreML documentation and the `coremltools` performance tooling confirm that `MLModel` objects backed by `TreeEnsemble` specs are dispatched to CPU, not ANE or GPU.

The demo's hook ("run on iPhone's Neural Engine") and its closing segment ("ANE utilization in Instruments") both depend on ANE execution. Neither is achievable with a tree model.

### Model size for app bundling

A trained catboost-mlx model with 500 trees of depth 6 has 500 × 64 = 32,000 leaf nodes. At ~16 bytes per node in the CoreML proto, this is roughly 0.5–1 MB — acceptable for app bundling. This is not a blocking risk.

### On-device predict latency vs LightGBM alternatives

Single-row CPU inference for a 500-tree depth-6 ensemble is O(500 × 6) = 3,000 comparisons — sub-millisecond on any device. This is not a competitive differentiator but also not a problem. The latency story is fine; the ANE story is not.

### `.mlpackage` vs `.mlmodel` format

`export_coreml.py:122` calls `ct.models.utils.save_spec(builder.spec, path)`. Whether this produces a flat `.mlmodel` or a `.mlpackage` directory bundle depends on the path suffix and the installed `coremltools` version. The demo storyboard shows a `.mlpackage`. This needs a one-line test to confirm; it is not blocking but is unverified.

---

## Q5 — Recommendation

**Option (c): Kill the CoreML demo as the v0.6.0 launch artifact.**

The two claims the demo is built around — "only GBDT with CoreML" and "runs on ANE" — are independently fatal:

1. Upstream CatBoost has shipped CoreML export for years. The category does not need to be created; it already exists, and the incumbent is CatBoost itself. Any HN commenter who checks the CatBoost docs will immediately surface this, and the framing collapses.

2. CoreML `TreeEnsemble` models execute on CPU. The ANE utilization segment — which is the demo's emotional close and the "why this matters" payoff — cannot be produced. Running the demo without it means the 5:00–6:00 segment has no content.

The CoreML export implementation itself is solid and worth shipping as a documented feature (it is already in the README and `python/README.md`). The gap — no inference round-trip test, `coremltools` not in the dev environment — is a 0.5-day cleanup worth doing for completeness. But it is a feature footnote, not a launch narrative.

**The correct v0.6.0 launch artifact is the Pareto Lane dashboard** as specified in `docs/sprint45/pareto-lane-spec.md`. The Branch B data is real and defensible. The demo adds no incremental evidence to that claim.

If @visionary wants a "category creation" hook that survives scrutiny, the honest version is: "GPU-native CatBoost accuracy on Apple Silicon, no CUDA required" — which is exactly the Pareto Lane's message. The CoreML angle is not the differentiator.

---

## Appendix: File and Line Citations

| Claim | File | Lines |
|---|---|---|
| `export_coreml()` public method | `python/catboost_mlx/core.py` | 1866–1874 |
| Export implementation (oblivious tree → TreeEnsemble) | `python/catboost_mlx/export_coreml.py` | 1–123 |
| Oblivious tree BFS unfolding | `python/catboost_mlx/_tree_utils.py` | 28–114 |
| CoreML optional dependency declaration | `python/pyproject.toml` | 54, 56 |
| Spec-validity tests (no inference check) | `python/tests/test_basic.py` | 981–1000 |
| Adversarial pre-fit guard | `python/tests/test_qa_adversarial.py` | 835–838 |
| CoreML mentioned in user-facing README | `python/README.md` | 24, 309, 368–369 |
| Upstream CatBoost `save_model(format="coreml")` | system `catboost` package | `save_model` docstring |
| Branch B lock (v0.6.0 scope) | `docs/sprint43/T4-synthesis.md` | 60–79 |
| Pareto Lane dashboard spec | `docs/sprint45/pareto-lane-spec.md` | entire document |
