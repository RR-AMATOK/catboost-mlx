# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.1] - 2026-05-02

Documentation + release-hygiene patch on top of v0.6.0. No code
changes; no API changes; no benchmark methodology changes.

### Fixed
- **Version metadata correctness**: `python/pyproject.toml` and
  `python/catboost_mlx/__init__.py` were not bumped before the
  v0.6.0 tag was cut, so a wheel built off the v0.6.0 commit
  self-identified as `0.5.4`. v0.6.1 corrects the metadata. Future
  release scripts should bump versions before tagging.
- **`docs/benchmarks/v0.5.x-pareto.md`** marked as SUPERSEDED with
  a pointer to `v0.6.0-pareto.md`. The v0.5.x writeup is preserved
  for historical reference.
- **`docs/sprint44/sprint-plan.md`** status updated from
  "IN PROGRESS" to "COMPLETE (2026-05-02; PR #44 merged, v0.6.0
  tagged)".
- **`catboost/mlx/README.md`** install instructions updated to
  reference v0.6.1 GitHub Release tarball (was v0.6.0; both work,
  but v0.6.1 has correct version metadata).
- **`python/CHANGELOG.md`** v0.6.0 entry added (was missing — the
  v0.6.0 release shipped without a CHANGELOG entry).

### Recommended install
- Use `git checkout v0.6.1` rather than `v0.6.0` if you want the
  installed package to correctly self-identify its version. The
  underlying code is bit-identical between the two tags.

## [0.6.0] - 2026-05-02

Reproducibility-grade release. Branch B locked (per S43-T4 advisory):
"deterministic, bit-equivalent Apple Silicon-native CatBoost-Plain
port" — accuracy-led, not throughput-led. The launch artifact is
`docs/benchmarks/v0.6.0-pareto.md`.

### Headline claim

On numeric workloads at fair convergence, CatBoost-MLX produces
statistically indistinguishable results from CatBoost-CPU. Bit-
equivalence holds within fp32 numerical noise on Higgs-1M (gap
+0.0002 at iter=1000) and within the architectural floor on Epsilon
(gap +0.00055 at iter=2000). We are not faster than CatBoost-CPU
on the same chip — training is 5–16× slower across measured
datasets. We are reproducible, deterministic, and GPU-native on
Apple Silicon.

### Added

- **5-dataset Pareto sweep at fair convergence** (S44-T2 + earlier
  sprints): Adult, Higgs-1M, Higgs-11M, Epsilon, Amazon. Iter-grid
  methodology with three guardrails (pre-registered Δlogloss
  thresholds, explicit `argmin_iter` reporting, conditional
  downgrade of bit-equivalence wording). MSLR-WEB10K (ranking)
  deferred to v0.6.x.
- **Axis C cross-over experiment** on Epsilon (S44 follow-up to
  the original mathematician hypothesis): 5 seeds × 5 iter levels
  × 2 backends = 50 runs with full per-iter trajectory. The
  MLX-vs-CPU gap monotonically decays from +0.00346 (iter=200) to
  -0.00013 (iter=4000). Paired-t-test verdict: t = -0.968 (n=5),
  not significant. Conservative interpretation: bit-equivalent at
  fair convergence; the variance-reduction hypothesis is
  consistent with the data but not confirmed by it.
- **Per-iter trajectory infrastructure** in
  `benchmarks/upstream/scripts/run_catboost_cpu.py`: new
  `--metric-period` flag captures `evals_result_["validation"]
  ["Logloss"]` into the result JSON's `trajectory` field. Zero
  wall-time overhead — CatBoost computes per-iter eval logloss
  internally during the same training pass.
- **`benchmarks/axisC/`** subdirectory with the launcher, results,
  and analysis pipeline.
- **Launch writeup** at `docs/benchmarks/v0.6.0-pareto.md` (524
  lines, 9 sections): TL;DR, when-to-use, methodology, per-dataset
  results, Axis C finding, what-works, honest limitations,
  reproducibility receipts, summary.

### Changed

- **Upstream RFC draft** (`docs/upstream_issue_draft.md`)
  refreshed for v0.6.0 framing (was staged at v0.5.x). Status:
  **STAGED — NOT POSTED**.
- **Documented architectural floor scaling**: floor magnitude
  scales predictably with feature dimensionality (Higgs 28 features
  → +0.0002 floor; Epsilon 2000 features → +0.0006 floor). More
  histogram reduction steps per iter → more accumulated fp32
  rounding per leaf-Newton step.
- **Sprint 45 spec docs** added (`docs/sprint45/`): Pareto Lane
  dashboard scoping (kept as v0.6.x exploration); CoreML demo spec
  (kill recommendation captured — upstream CatBoost already has
  CoreML, ANE doesn't run trees).

### Known limitations (carry-forward + new)

- 5-16× slower than CatBoost-CPU on training across measured
  datasets. The gap is structural compute-throughput, not GPU
  launch overhead. Closing it requires train-binary-IPC + histogram
  kernel work, est. 5-8 sprints, deferred indefinitely.
- Predict latency widens with model size (3.3× on Adult post-S43-T3
  to 140× on Epsilon iter=2000). GPU launch overhead dominates at
  low batch sizes.
- **High-cardinality categorical features (cardinality > 255)
  alias at training time** in the C++ training path's `uint8`
  bin assumption (`csv_train.cpp:static_cast<uint8_t>`). v0.5.4
  patched the predict-time crash; the underlying training-time
  aliasing fix is tracked as DEC-047 (v0.6.x).
- MSLR-WEB10K (ranking) iter-grid deferred to v0.6.x.
- Ordered Boosting (`boosting_type='Ordered'`) still not
  implemented (only `Plain`); v0.7.x scope.
- CTR-encoded models (`ctr=True`) still use the subprocess predict
  path; port to Python is a v0.6.x or later follow-up.

### Internal

- Sprint 44 closed; tasks T0-T2, T4, T5 done; T3 (MSLR) deferred.
- v0.5.4 patch shipped mid-sprint as a focused release for the
  Amazon cat-overflow fix (see `[0.5.4]` entry).
- DEC-047 added to `.claude/state/DECISIONS.md`: Axis C cross-over
  verdict + Amazon uint8 cat-aliasing as v0.6.x DEC item.

## [0.5.4] - 2026-04-30

Patch release. Fixes a hard-fail in the in-process predict path for
high-cardinality categorical features (cardinality > 255). Surfaced by
the S44-T2 Amazon iter-grid sweep, where every catboost-mlx run on
Amazon (RESOURCE column, cardinality 799) raised
`OverflowError: Python integer 799 out of bounds for uint8` from
`_predict_utils.quantize_features`.

### Fixed
- **High-cardinality cat predict overflow** (S44-T2):
  `_predict_utils.quantize_features` now masks `cat_hash_map[val] & 0xFF`
  before assigning into the uint8 binned buffer. This mirrors the
  `static_cast<uint8_t>` truncation that `csv_train.cpp` already applies
  at fit time, keeping the Python in-process predict path consistent
  with the trained splits. The C++ training path's 8-bit-per-bin
  architectural assumption (and the resulting cat-aliasing for
  cardinality > 255) is unchanged in this release; widening the bin
  width is tracked as a separate v0.6.x DEC item.
- **Regression test** at
  `python/tests/test_qa_round4.py::TestDispatchCorrectness::test_high_cardinality_cat_predict_no_overflow`
  trains on a 300-cardinality cat feature and asserts predict_proba
  returns valid probabilities without raising.

### Note
This release is a focused patch on top of v0.5.3; no API changes, no
benchmark methodology changes, and no v0.6.0 framing implications.
Sprint 44's full 5-dataset Pareto sweep continues separately.

## [0.5.3] - 2026-04-26

Polish-and-perf release covering Sprints 41–43. The headline win is an
**8.5× faster `predict()` on OneHot-cat workloads** (Sprint 43-T3) by
routing those models through the existing in-process tree evaluator
instead of the C++ subprocess. Also: empirical bit-equivalence claim on
numeric workloads at fair convergence, full upstream-benchmark suite
infrastructure, and standard polish (case-insensitive `bootstrap_type`
validator, refreshed README, hardware-invariant CI perf-regression
gate).

### Added
- **Predict-path in-process dispatch for OneHot-cat models** (S43-T3,
  silicon-architect's #1 ROI fix in the codebase): `_run_predict` now
  routes any model whose `model_data.ctr_features` is empty through the
  in-process NumPy tree evaluator. The default `ctr=False` path covers
  the vast majority of cat-feature workloads. CTR-encoded models still
  use the C++ subprocess until the CTR table application is ported to
  Python (tracked as a follow-up). User-visible: ~8.5× faster
  `predict()` on Adult-class workloads, no API change. Bit-identical
  output verified by new `tests/test_basic.py::TestPredictDispatch`.
- **Upstream benchmark suite infrastructure** (S42–S43): `benchmarks/upstream/`
  with 5 dataset adapters (Adult auto-downloads; Higgs, Epsilon, Amazon,
  MSLR with download instructions), 4 framework runners (LightGBM,
  XGBoost, CatBoost-CPU, CatBoost-MLX), driver script, results aggregator,
  and Pareto-frontier plot generator. `--iterations` flag supported across
  all 4 runners with `_iter<N>` output tagging for non-default iter counts.
- **Pareto-frontier writeup** (`docs/benchmarks/v0.5.x-pareto.md`):
  head-to-head numbers across 4 frameworks on the same M-series machine
  for Adult, Higgs-1M, Higgs-11M (full upstream scale), Adult-iter1000,
  Higgs-1M-iter1000.
- **Bit-equivalence-at-fair-convergence claim** (S43-T2): on Higgs-1M at
  iter=1000, MLX agrees with CatBoost-CPU to within 0.0002 logloss —
  fp32 numerical noise. The DEC-046 +0.0012 "architectural floor" claim
  was itself partly a 200-iter under-convergence artifact. The publishable
  claim strengthens to "deterministic, bit-equivalent CatBoost-Plain on
  Apple Silicon" for numeric workloads at fair convergence.
- **README "When to use this backend"** section + canonical Installation
  & Quick Start with a 30-second smoke test (S41).
- **Sprint 41 polish bundle**: `bootstrap_type` validator now
  case-insensitive (`'No'`/`'NO'`/`'no'`/`'Bayesian'`/etc. all accepted,
  normalized to lowercase). Matches CatBoost-CPU UX. Resolves a paper-cut
  hit on real-world workloads. (S41-T1)
- **PyPI publish-readiness audit** complete (S41-T4). Sdist + wheel
  build cleanly; sdist is hygienic; fresh-venv install passes smoke test.
  One must-fix at publish time: `MACOSX_DEPLOYMENT_TARGET=14.0`.
- **Upstream RFC draft** (`docs/upstream_issue_draft.md`) refreshed for
  v0.5.x reality (S41-T5). Status: **STAGED — NOT POSTED**.

### Fixed
- **CI perf-regression gate** rebuilt as hardware-invariant CPU/MLX
  speedup-ratio comparison (S42-T4). The S41 bridge mode
  (`continue-on-error: true`) is retired; `continue-on-error: false`
  restored on the wall-clock gate. Verified across 6 simulated scenarios
  including the actual S41-trigger case (4× uniformly slower runner now
  correctly passes at +0.0% Δ ratio; real MLX +10% slowdown still fires).
- **`mlx-perf-regression.yaml` chronic 0s startup failure** (S40-S41
  follow-up): job-level `if: runner.os == 'macOS'` removed (current
  GitHub Actions schema rejects the `runner` context at the job level;
  `runs-on: macos-14` already constrains the platform).
- **`bootstrap_type` validator** (S41-T1) — see Added above.

### Changed
- **Methodology note** added to the Pareto-frontier writeup (S43-T2):
  Adult overfits at iter=1000 across all 4 frameworks. Future sweeps
  should use early-stopping or per-dataset iter-count tuning rather
  than a fixed iter count.
- **Sprint 41 documentation pass**: README §"Python API uses subprocess"
  rewritten with a profile-derived two-row mechanism table (in-process
  vs subprocess at ~940k vs ~111k rows/s) and three workarounds (S41-T3).
- **Strategic direction recorded** (S43-T4): v0.6.0 scope locked to
  "deterministic, bit-equivalent Apple Silicon-native CatBoost-Plain
  port" (Branch B). Ordered Boosting demoted from v0.6.0 hero to optional
  v0.7.x. Throughput optimization deferred indefinitely (5–8 sprints to
  match CatBoost-CPU per silicon-architect; not load-bearing for the
  launch story). See `docs/sprint43/T4-synthesis.md` for the full
  decision rationale.

### Known limitations (carry-forward)
- `boosting_type='Ordered'` still not implemented (only `Plain`).
- `NewtonL2`/`NewtonCosine` still rejected at the Python API.
- `max_depth` capped at 6, 16M-row dataset ceiling.
- CTR-encoded models (`ctr=True`) still use the subprocess predict path;
  port to Python is a v0.6.x or later follow-up.
- Throughput: MLX is ~5× slower than CatBoost-CPU on numeric workloads
  at all measured scales (1M, 11M); the gap is structural compute-
  throughput, not GPU launch overhead. Closing it requires kernel-level
  work (subprocess removal + dispatch fusion + histogram tightening),
  estimated 5–8 sprints, deferred indefinitely.

### Internal
- Sprints 41, 42, 43 landed since 0.5.0. Authoritative records under
  `.claude/state/HANDOFF.md`, `.claude/state/DECISIONS.md` (DEC-046
  reaffirmed by S43-T2 bit-equivalence finding), and per-sprint
  close-out docs in `docs/sprint{41,42,43}/`.
- Note: GitHub releases v0.5.1 and v0.5.2 were tag-only (or never cut).
  Source version was at 0.5.0 from the S41 release through S43 close.
  This 0.5.3 release brings source-version, GitHub-release, and PyPI-
  ready state back into alignment.

## [0.5.0] - 2026-04-26

This release closes the cross-runtime correctness arc (Sprints 15–40) and ships
CatBoost-MLX as a *characterized-difference Apple Silicon CatBoost-Plain port*.
Synthetic-anchor parity is now bit-identical to CatBoost-CPU at `RandomStrength=0`,
and the real-world residual gap on multiclass-with-categoricals workloads is fully
decomposed into seed-noise + architectural floor + categorical-encoding asymmetry
(see `catboost/mlx/README.md` § Known Limitations and `docs/sprint40/pre_lane_check/FINDING.md`).

### Added
- **Cosine score function across all grow policies** (Sprint 33, DEC-042):
  `score_function=Cosine` now ships for SymmetricTree, Depthwise, and Lossguide. The
  prior ST+Cosine and LG+Cosine guards were removed once the underlying mechanism
  (degenerate-child skip in `FindBestSplitPerPartition`) was identified and fixed
  with a per-side mask. One-hot Cosine retains joint-skip (correct for parentless gain).
- **Real-world cross-runtime characterization** (Sprint 40, DEC-046): a 3-experiment
  decomposition methodology (arithmetic reconcile + `cat_features=[]` discriminator +
  CPU 5-seed noise floor) that quantifies the prediction-disagreement floor on real
  multiclass-with-categoricals data. Reproducibility scripts shipped under
  `docs/sprint40/pre_lane_check/scripts/`.
- **Numeric-only parity guarantee**: workloads with `cat_features=[]` converge to within
  the architectural floor — 99.948% prediction agreement, mean absolute probability
  difference 2.2e-3, no rare-class skew (DEC-046).
- **Cross-runtime parity guidance** in README: explicit `RandomStrength=0` matching is
  required for bit-identical synthetic-anchor agreement; default `RS=1.0` produces a
  bounded (mean −4.08%, 95% CI [−4.78%, −3.39%]) RNG-implementation bias on a 10-seed
  small-N anchor (DEC-045).
- **README "When to use this backend" section**: positioning copy clarifying the
  decision matrix (when to choose CatBoost-MLX vs CatBoost-CPU/CUDA), the
  deterministic-greedy framing for Apple Silicon, and the characterized-difference
  contract.

### Fixed
- **DEC-036**: ST+Cosine 52.6% → 0.027% iter=50 drift (1941× improvement). Root cause
  was a `continue` statement at `csv_train.cpp:1980` skipping the entire partition's
  contribution when one side was degenerate; the per-side mask fix adds the non-empty
  side and zeros the empty side, matching CatBoost-CPU's `UpdateScoreBinKernelPlain`
  semantics. Mechanism diagnosed in PROBE-E (Sprint 33), fix shipped in S33-L4-FIX.
- **DEC-042**: per-side mask ported to FindBestSplitPerPartition (Sprint 38, commit
  `a481972529`), closing the partition-state divergence class for both ordinal and
  one-hot branches.
- **DEC-045**: small-N "drift residual" from Sprints 30–37 root-caused as a harness
  configuration mismatch — comparison scripts invoked CPU with `random_strength=0`
  while leaving MLX `csv_train` at its default `RS=1.0`. With matched configuration,
  drift collapses to near-zero. PROBE-G/H/F2/Q-1 verdicts retracted as causal
  interpretations of the same artifact; their captured data remains valid for what it
  observed.
- **DEC-038/039** (Sprint 32): `GreedyLogSumBestSplit` now operates on all-doc values
  (not deduplicated); histogram kernel `fold_count` capped at 127 to avoid
  `VALID_BIT` aliasing.
- **BUG-007 mitigated** (commit `71aabaa842`): `group_id` sorting via `np.argsort`
  ensures rows are grouped before training; ranking without `group_id` raises a clear
  `ValueError` instead of producing silent garbage.

### Changed
- **Known Limitations rewrite**: the README section now documents the Ordered Boosting
  absence (`boosting_type='Ordered'` not implemented; only `Plain` supported) and the
  real-world cross-runtime characterization with the 3-row decomposition table. Prior
  references to "13–44% small-N drift" are retired as historical-only.
- **Probe methodology lessons** captured in `.claude/state/LESSONS-LEARNED.md`:
  counterfactual-vs-observational distinction, cross-runtime configuration symmetry,
  RNG-bias multi-seed verification, and the new 3-experiment decomposition triage for
  cross-runtime ML port release-readiness.

### Known limitations (carry-forward and new)
- `boosting_type='Ordered'` not implemented (only `Plain`). Ordered CTR (online target
  encoding for categoricals) IS implemented — these are different features.
- `NewtonL2`/`NewtonCosine` score functions explicitly rejected at the Python API.
- `max_depth` capped at 6 (kernel constraint, DEC-003 era).
- 16M-row dataset limit (`ComputePartitionLayout` int32 ceiling).
- 41× MLX vs CPU `predict()` slowdown via subprocess path; the in-process nanobind
  path (`_HAS_NANOBIND=True`) does not have this overhead.
- On real multiclass-with-categoricals workloads, expect ~99.92% prediction agreement
  with CatBoost-CPU at matched `RS=0`; rare-class asymmetry concentrated near the
  CTR-encoded categorical features (DEC-046).

### Internal
- ~26 sprints of work landed since 0.4.0. Source-of-truth sprint records under
  `.claude/state/HANDOFF.md` (current state) and `.claude/state/DECISIONS.md`
  (DEC-005 through DEC-046).

## [0.4.0] - 2026-04-12

### Added
- **CI/CD on GitHub Actions** (macos-14 Apple Silicon M1 runners)
  - `mlx-build.yaml`: C++ compile gate — verifies csv_train.cpp links against MLX
  - `mlx-test.yaml`: Full Python test suite (1010 tests) with Metal GPU and nanobind extension build
- **Performance benchmarks**: `bench_mlx_vs_cpu.py` comparing MLX (Metal GPU) vs CatBoost CPU across 3 dataset scales × 3 loss functions
- Benchmark results for M3 Max 128GB published in `benchmarks/results/`
- **Library path feature parity** (Sprint 13): `IModelTrainer` implementation now supports all 12 loss functions, all 3 grow policies on GPU
  - Depthwise and Lossguide grow policies on GPU — a feature the CUDA backend does not offer
  - Non-symmetric tree export via `TNonSymmetricTreeModelBuilder`
  - Full parameter wiring: GrowPolicy, MaxLeaves, RandomSeed, SubsampleRatio, ColsampleByTree, EarlyStoppingPatience, MetricPeriod
  - PairLogit and YetiRank ranking losses in the library path via `pairwise_target.h`
- **Documentation refresh** (Sprint 12): ARCHITECTURE.md rewrite with nanobind architecture section, ranking losses, two-code-paths explanation
- **Ranking hardening** (Sprint 12): BUG-007 `group_id` sorting fix, 52 ranking-specific tests
- Upstream issue draft for `catboost/catboost` with honest performance data

### Fixed
- BUG-007: `group_id` sorting now uses `np.argsort` to ensure rows are grouped correctly before training
- Ranking without `group_id` now raises a clear `ValueError` instead of producing silent garbage

### Changed
- Upstream issue draft performance section replaced with real M3 Max benchmark data (previously said "competitive with CPU")
- `mlx_test.yaml` renamed from `mlx_test.yaml` (underscore) for consistent naming

## [0.3.0] - 2026-04-11

### Added
- **Nanobind in-process GPU training** (Sprint 11): zero-copy numpy→C++ via `nb::ndarray`, GIL released during training
- `_core` nanobind extension module with `train()` and `predict()` C++ bindings
- CMake-based extension build via `mlx.extension.CMakeBuild`
- Automatic fallback to subprocess path when nanobind extension is not available
- `_HAS_NANOBIND` flag for runtime detection of in-process vs subprocess mode

### Fixed
- `cross_validate()`: CV output parser now matches actual binary output format (`Fold N: test_loss=...` and `Test  loss: ... +/- ...`)
- `CatBoostMLXClassifier.fit()`: `y` parameter now defaults to `None` so `fit(pool)` works without passing `y` explicitly
- `load_model()`: restores `self.loss` from model JSON so the instance reflects the trained loss, not the constructor default
- `_array_to_csv`: return type annotation tightened from `tuple` to `Tuple[int, int, int]`

### Changed
- Added `Tuple` to typing imports; added return type hint on `_unpack_predict_input`
- `MANIFEST.in`: added `include LICENSE` for proper sdist packaging

## [0.2.0] - 2026-04-02

### Added
- `CatBoostMLX.load()` classmethod for convenient model loading
- `predict()`, `predict_proba()`, and other predict methods now accept Pool objects
- Pickle / joblib serialization support (`__getstate__` / `__setstate__`)
- `conftest.py` with shared test fixtures
- `py.typed` marker for PEP 561 compliance
- `logging` module usage for non-user-facing output
- Ruff and pytest configuration in `pyproject.toml`
- CI test matrix expanded to Python 3.10, 3.11, 3.12, 3.13
- `train_timeout` / `predict_timeout` parameters to prevent subprocess hangs
- CI ruff linting and pytest-cov coverage reporting
- CI concurrency groups to cancel redundant workflow runs
- Error path tests (corrupted JSON, missing files, empty datasets, timeout validation)
- Input validation: `cat_features` bounds check, `monotone_constraints` length, `eval_period >= 1`
- `load_model()` validates required JSON keys (`model_info`, `trees`, `features`)
- `cross_validate()` now calls `_validate_params()` and `_validate_fit_inputs()` before running
- Feature name sanitization (rejects commas, newlines, null bytes)
- Executable bit check in binary discovery with `chmod +x` hint
- Model JSON serialization cache for faster repeated `predict()` calls
- NaN handling tests, MAE/Huber/Quantile loss tests
- sklearn integration tests (`cross_val_score`, `Pipeline`)
- Multiclass staged_predict and classes_ attribute tests
- Makefile with `test`, `lint`, `coverage`, `build-binaries`, `install` targets
- `.pre-commit-config.yaml` for ruff hooks
- `MANIFEST.in` for proper sdist packaging

### Fixed
- `_array_to_csv`: `isinstance(val, float)` now also matches `np.floating` (numpy 2.x compat)
- `_array_to_csv`: numeric NaN check uses `float(val)` cast to avoid `TypeError` on non-float dtypes
- `quantize_features`: `np.clip(bins, 0, 255)` prevents silent uint8 overflow at 256 bins
- `fit(y=None)` on raw arrays now raises clear `ValueError` instead of cryptic `IndexError`
- `get_shap_values()`: checks SHAP output file exists before reading
- `group_col` mutation in `fit()` now wrapped in `try/finally` to prevent state leakage on error
- `CatBoostMLXClassifier.fit(pool)`: correctly extracts `classes_` from Pool labels instead of `None`
- `staged_predict`: uses `_get_loss_type()` to split parameterized loss strings (e.g. `tweedie:1.5` → `tweedie`) so `apply_link` applies the correct transform
- `load_model()`: restores `n_features_in_` and `feature_names_in_` from model JSON so `feature_importances_` and sklearn validation work after loading
- PTY verbose mode: reads stderr before `proc.wait()` to prevent deadlock on large error output
- Validation for `bagging_temperature`, `mvs_reg`, `max_onehot_size`, `ctr_prior` parameters

### Changed
- `_HAS_SKLEARN` removed from `__all__` (still accessible via `catboost_mlx.core._HAS_SKLEARN`)
- `__version__` now reads from package metadata instead of hardcoded duplication
- `cross_validate()` reuses `_build_train_args()` to reduce code duplication
- `_to_numpy()` consolidated into `_utils.py` (imported by both `core.py` and `pool.py`)
- `Pool` no longer copies data unnecessarily (uses `np.ascontiguousarray` instead of `.copy()`)
- `cross_validate()` docstring expanded with full parameter and return value documentation
- `pyproject.toml`: ruff target-version aligned to `py39`, added Python 3.9/3.11 classifiers, `scikit-learn` added to dev deps
- Fixed 36 ruff lint violations across all Python modules
- Disabled inherited upstream CatBoost CI workflows that always fail in this fork

### Known Limitations
- Binary bundling: `pip install` does not include pre-compiled csv_train/csv_predict.
  Users must compile binaries separately via `python build_binaries.py`.

## [0.1.0] - 2026-04-02

### Added
- Initial release
- CatBoostMLX base class with 27 hyperparameters
- CatBoostMLXRegressor and CatBoostMLXClassifier subclasses
- Pool data container with pandas DataFrame auto-detection
- Loss functions: RMSE, MAE, Quantile, Huber, Poisson, Tweedie, MAPE, Logloss, Multiclass, PairLogit, YetiRank
- Staged predict and staged predict_proba for learning curves
- TreeSHAP values via csv_predict --shap
- Feature importance (gain-based) with text bar chart visualization
- Model save/load (JSON format)
- Export to CoreML and ONNX formats
- N-fold cross-validation
- Bootstrap types: Bayesian, Bernoulli, MVS
- Monotone constraints, min_data_in_leaf, snapshot resume
- Auto class weights (Balanced, SqrtBalanced)
- scikit-learn 1.8+ compatibility (fit/predict/score, get_params/set_params, clone, pipelines)
- 120 tests across 27 test classes
