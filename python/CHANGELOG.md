# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
