"""
test_branch_b_regression.py -- Branch-B bit-equivalence CI gate (S45-T1).

Purpose
-------
Lock v0.6.1's predict() output against any future engineering change.  Any
optimization that alters _predict_inprocess's float32 output (even by a single
ULP) will fail this gate.  That is the intent: the test is the correctness
contract for the reproducibility-grade frame shipped in v0.6.1.

Design: load-and-predict, not re-train
---------------------------------------
MLX Metal training is non-deterministic across separate Python process invocations.
Atomic float operations in the histogram accumulation kernel do not guarantee
identical thread-block ordering across GPU launches, producing ~1-3 ULP weight
drift between separate fit() calls with identical seed and data.  Re-training and
comparing against a pickled baseline would produce spurious failures on every CI
run.

Solution: the baseline pickle stores BOTH the reference predictions AND the trained
model weights (model._model_data, the JSON-serializable dict).  The test:
    1. Loads the stored model weights from the pickle.
    2. Reconstructs a CatBoostMLXClassifier via load_model() (using a tempfile).
    3. Calls predict_proba on the same test rows used during baseline generation.
    4. Compares the result byte-for-byte against the stored reference predictions.

Training is NOT re-run during the test.  Only _predict_inprocess is exercised.
This is the correct gate: if _predict_inprocess changes its float32 output for the
same model weights, the gate fires.

What is tested
--------------
1. test_higgs_1m_predict_byte_equivalent_to_v061
   - Loads v0.6.1 Higgs-1M model (iter=200, seed=42, 1M train / 100k test).
   - Predicts on 100,000 test rows (full test.csv).
   - Wall-clock on M3 Max MLX: ~1s (predict only, no training).

2. test_epsilon_subset_predict_byte_equivalent_to_v061
   - Loads v0.6.1 Epsilon-subset model (50k-train/10k-test, iter=200, seed=42).
   - Predicts on 10,000 test rows (rows 0..9999 of test.csv, 2000 features).
   - Wall-clock on M3 Max MLX: ~5s (predict only, 2000 features, 200 trees).

Total CI wall-clock: ~10s (predict-only; training is offline in the generator).
CI timeout for this job: 20 minutes (see mlx-branch-b-regression.yaml).

Baseline reference
------------------
python/tests/regression/v0.6.1_predict_baselines.pkl -- checked-in pickle
generated on master at commit d3bc0e1d02 (v0.6.1 release merge) by
python/tests/regression/generate_v061_baselines.py.

Assertion semantics
-------------------
    np.array_equal(actual.astype(np.float32), reference)

Byte-identity check (atol=0, rtol=0) on float32.  NOT allclose.
Any bit divergence fails the gate.

If a future optimization must change predict output, the developer must:
    1. Bump the version in generate_v061_baselines.py,
    2. Re-train + regenerate the pickle on the new master tip,
    3. Commit the updated pickle alongside the version-bump commit.

Skip conditions
---------------
- catboost_mlx not importable (wheel not built) -> pytest.skip, non-blocking.
- Dataset cache missing -> pytest.skip, non-blocking.
- Baseline pickle missing -> pytest.skip, non-blocking.
The test must NEVER hard-fail due to infrastructure missing.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ── Paths ─────────────────────────────────────────────────────────────────────

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[2]        # python/tests/regression -> repo root
_BENCH_SCRIPTS = _REPO_ROOT / "benchmarks" / "upstream" / "scripts"
_BASELINE_PKL = _THIS_DIR / "v0.6.1_predict_baselines.pkl"

sys.path.insert(0, str(_REPO_ROOT / "python"))
sys.path.insert(0, str(_REPO_ROOT))

# ── Epsilon subset index contract (must match generate_v061_baselines.py) ─────

_EPSILON_TEST_ROWS = 10_000

# ── Import helpers ────────────────────────────────────────────────────────────

def _has_catboost_mlx() -> bool:
    try:
        import catboost_mlx  # noqa: F401
        from catboost_mlx import CatBoostMLXClassifier  # noqa: F401
        return True
    except ImportError:
        return False


def _load_runner_common():
    """Load _runner_common, registering in sys.modules so @dataclass works."""
    import importlib.util
    _MOD_NAME = "benchmarks.upstream.scripts._runner_common"
    if _MOD_NAME in sys.modules:
        return sys.modules[_MOD_NAME]
    mod_path = str(_BENCH_SCRIPTS / "_runner_common.py")
    spec = importlib.util.spec_from_file_location(_MOD_NAME, mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MOD_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


def _has_dataset_cache(dataset: str) -> bool:
    try:
        rc = _load_runner_common()
        rc.load_csv_pair(dataset)
        return True
    except Exception:
        return False


def _reconstruct_model(model_data: dict):
    """Reconstruct a CatBoostMLXClassifier from stored _model_data.

    Uses load_model() via a tempfile so that all sklearn-compatibility
    attributes (n_features_in_, _is_fitted, etc.) are properly initialized.
    save_model() wraps _model_data with format_version:2; we reproduce that
    wrapper here so load_model() can parse the file correctly.
    """
    from catboost_mlx import CatBoostMLXClassifier
    payload = {"format_version": 2, **model_data}
    fd, tmp_path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        model = CatBoostMLXClassifier()
        model.load_model(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return model


# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.regression
def test_higgs_1m_predict_byte_equivalent_to_v061():
    """
    Higgs-1M predict_proba from the v0.6.1 stored model must be byte-identical
    to the v0.6.1 reference predictions in the baseline pickle.

    Does NOT re-train. Loads stored model weights, reconstructs via load_model(),
    and calls predict_proba on 100,000 test rows. Wall-clock: ~1s.
    """
    if not _has_catboost_mlx():
        pytest.skip("catboost_mlx wheel not built -- skipping regression gate")

    if not _has_dataset_cache("higgs"):
        pytest.skip(
            "Higgs dataset cache not populated. "
            "Run: python -m benchmarks.upstream.adapters.higgs_1m"
        )

    if not _BASELINE_PKL.exists():
        pytest.skip(
            f"Baseline pickle not found at {_BASELINE_PKL}. "
            "Run: python python/tests/regression/generate_v061_baselines.py"
        )

    with open(_BASELINE_PKL, "rb") as f:
        baseline = pickle.load(f)

    # Load test data (same rows the generator used -- full test.csv, 100k rows).
    rc = _load_runner_common()
    _, test_csv, meta = rc.load_csv_pair("higgs")
    X_te, _, _ = rc.load_xy(test_csv, meta)
    X_te = X_te.astype(np.float32)

    # Reconstruct model from stored weights and predict.
    model = _reconstruct_model(baseline["higgs_1m_model_data"])
    actual = model.predict_proba(X_te)[:, 1].astype(np.float32)

    reference = baseline["higgs_1m_iter200_seed42_predict_proba"]
    assert reference.dtype == np.float32, (
        f"Reference dtype mismatch: expected float32, got {reference.dtype}"
    )
    assert actual.shape == reference.shape, (
        f"Shape mismatch: actual={actual.shape}, reference={reference.shape}"
    )

    if not np.array_equal(actual, reference):
        n_differ = int(np.sum(actual != reference))
        abs_diff = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
        max_diff = float(abs_diff.max())
        mean_diff = float(abs_diff.mean())
        first_idx = int(np.argmax(actual != reference))
        pytest.fail(
            f"Higgs-1M predict_proba byte-equivalence FAILED (v0.6.1 gate)\n"
            f"  Differing elements : {n_differ} / {len(actual)}\n"
            f"  Max abs diff       : {max_diff:.6e}\n"
            f"  Mean abs diff      : {mean_diff:.6e}\n"
            f"  First divergent idx: {first_idx}  "
            f"(actual={actual[first_idx]:.8f}, ref={reference[first_idx]:.8f})\n"
            f"\n"
            f"_predict_inprocess returned different float32 output for the same\n"
            f"stored model weights. A perf optimization altered the predict path.\n"
            f"Either fix the optimization to preserve bit-equivalence, or -- if\n"
            f"the change is intentional -- regenerate the baseline:\n"
            f"  python python/tests/regression/generate_v061_baselines.py\n"
            f"  git add python/tests/regression/v0.6.1_predict_baselines.pkl"
        )


@pytest.mark.regression
def test_epsilon_subset_predict_byte_equivalent_to_v061():
    """
    Epsilon-subset predict_proba from the v0.6.1 stored model must be byte-identical
    to the v0.6.1 reference predictions in the baseline pickle.

    Does NOT re-train. Loads stored model weights, reconstructs via load_model(),
    and calls predict_proba on 10,000 test rows (rows 0..9999 of test.csv).
    Wall-clock: ~5s (2000 features, 200 trees, 10k rows).
    """
    if not _has_catboost_mlx():
        pytest.skip("catboost_mlx wheel not built -- skipping regression gate")

    if not _has_dataset_cache("epsilon"):
        pytest.skip(
            "Epsilon dataset cache not populated. "
            "Run: python -m benchmarks.upstream.adapters.epsilon"
        )

    if not _BASELINE_PKL.exists():
        pytest.skip(
            f"Baseline pickle not found at {_BASELINE_PKL}. "
            "Run: python python/tests/regression/generate_v061_baselines.py"
        )

    with open(_BASELINE_PKL, "rb") as f:
        baseline = pickle.load(f)

    # Verify the test index contract.
    stored_test_idx = baseline["test_indices_used"]["epsilon_subset"]
    expected_test_idx = list(range(_EPSILON_TEST_ROWS))
    assert stored_test_idx == expected_test_idx, (
        f"Test index contract mismatch: baseline used rows {stored_test_idx[:5]}..., "
        f"test expects rows {expected_test_idx[:5]}... -- regenerate the baseline."
    )

    # Load test data (same rows the generator used -- rows 0..9999 of test.csv).
    rc = _load_runner_common()
    _, test_csv, meta = rc.load_csv_pair("epsilon")
    X_te_full, _, _ = rc.load_xy(test_csv, meta)
    X_te = X_te_full[:_EPSILON_TEST_ROWS].astype(np.float32)

    # Reconstruct model from stored weights and predict.
    model = _reconstruct_model(baseline["epsilon_subset_model_data"])
    actual = model.predict_proba(X_te)[:, 1].astype(np.float32)

    reference = baseline["epsilon_iter200_seed42_subset_predict_proba"]
    assert reference.dtype == np.float32, (
        f"Reference dtype mismatch: expected float32, got {reference.dtype}"
    )
    assert actual.shape == reference.shape, (
        f"Shape mismatch: actual={actual.shape}, reference={reference.shape}"
    )

    if not np.array_equal(actual, reference):
        n_differ = int(np.sum(actual != reference))
        abs_diff = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
        max_diff = float(abs_diff.max())
        mean_diff = float(abs_diff.mean())
        first_idx = int(np.argmax(actual != reference))
        pytest.fail(
            f"Epsilon-subset predict_proba byte-equivalence FAILED (v0.6.1 gate)\n"
            f"  Test rows          : {_EPSILON_TEST_ROWS} (rows 0..{_EPSILON_TEST_ROWS-1})\n"
            f"  Differing elements : {n_differ} / {len(actual)}\n"
            f"  Max abs diff       : {max_diff:.6e}\n"
            f"  Mean abs diff      : {mean_diff:.6e}\n"
            f"  First divergent idx: {first_idx}  "
            f"(actual={actual[first_idx]:.8f}, ref={reference[first_idx]:.8f})\n"
            f"\n"
            f"_predict_inprocess returned different float32 output for the same\n"
            f"stored model weights. A perf optimization altered the predict path.\n"
            f"Either fix the optimization to preserve bit-equivalence, or -- if\n"
            f"the change is intentional -- regenerate the baseline:\n"
            f"  python python/tests/regression/generate_v061_baselines.py\n"
            f"  git add python/tests/regression/v0.6.1_predict_baselines.pkl"
        )
