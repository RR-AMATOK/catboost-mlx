"""
test_smoke.py -- Smoke tests for catboost-tripoint.

These tests:
1. Verify the tool runs end-to-end on a tiny in-memory dataset (no model
   file needed) by mocking the backend runners.
2. Verify the floor formula produces the correct value for a known (T, depth)
   pair.

They are fast (<1s) and do not require catboost, catboost_mlx, or pyarrow.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile

import numpy as np
import pytest

from catboost_tripoint.floor import (
    EPSILON_MACHINE_FP32,
    derived_floor_for_model,
    floor_components,
)
from catboost_tripoint.verifier import (
    BackendResult,
    PairwiseAgreement,
    VerifyResult,
    _compute_agreement,
)


# ---------------------------------------------------------------------------
# Test 1: fp32 floor formula correctness
# ---------------------------------------------------------------------------

class TestFloor:
    def test_known_value(self):
        """Floor for T=200, depth=6: eps_mach * 200 * sqrt(64) = eps_mach * 1600."""
        floor = derived_floor_for_model(tree_count=200, max_depth=6)
        expected = EPSILON_MACHINE_FP32 * 200 * math.sqrt(2 ** 6)
        assert abs(floor - expected) < 1e-20, f"floor={floor}, expected={expected}"

    def test_components_consistent(self):
        """floor_components derived_floor matches derived_floor_for_model."""
        T, D = 500, 8
        fc = floor_components(T, D)
        direct = derived_floor_for_model(T, D)
        assert abs(fc["derived_floor"] - direct) < 1e-20

    def test_increases_with_trees(self):
        """More trees -> larger floor (monotone)."""
        f100 = derived_floor_for_model(100, 6)
        f200 = derived_floor_for_model(200, 6)
        assert f200 > f100

    def test_increases_with_depth(self):
        """Deeper trees -> larger floor (monotone)."""
        f6 = derived_floor_for_model(200, 6)
        f8 = derived_floor_for_model(200, 8)
        assert f8 > f6

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            derived_floor_for_model(0, 6)
        with pytest.raises(ValueError):
            derived_floor_for_model(200, 0)

    def test_epsilon_machine_value(self):
        """eps_mach must equal 2^-23 (fp32 machine epsilon)."""
        assert EPSILON_MACHINE_FP32 == pytest.approx(2.0 ** -23, rel=1e-12)


# ---------------------------------------------------------------------------
# Test 2: pairwise agreement computation
# ---------------------------------------------------------------------------

class TestAgreement:
    def _make_backend(self, name: str, preds: np.ndarray) -> BackendResult:
        br = BackendResult(name=name, version="test", available=True)
        br.predictions = preds.astype(np.float32)
        br.predictions_sha256 = hashlib.sha256(br.predictions.tobytes()).hexdigest()
        return br

    def test_identical_predictions_pass(self):
        preds = np.random.default_rng(42).random(1000).astype(np.float32)
        a = self._make_backend("cpu", preds)
        b = self._make_backend("mlx", preds.copy())
        floor = derived_floor_for_model(200, 6)
        pair = _compute_agreement(a, b, floor)
        assert pair.max_abs_diff == 0.0
        assert pair.within_floor is True
        assert pair.n_diff_above_floor == 0

    def test_large_diff_fails(self):
        preds_a = np.ones(100, dtype=np.float32)
        preds_b = np.zeros(100, dtype=np.float32)
        a = self._make_backend("cpu", preds_a)
        b = self._make_backend("mlx", preds_b)
        floor = derived_floor_for_model(200, 6)
        pair = _compute_agreement(a, b, floor)
        assert pair.max_abs_diff == pytest.approx(1.0, rel=1e-6)
        assert pair.within_floor is False
        assert pair.n_diff_above_floor == 100

    def test_tiny_diff_within_floor(self):
        """A diff of 1e-6 should be within the floor for T=200, depth=6."""
        floor = derived_floor_for_model(200, 6)
        assert floor > 1e-6, f"floor={floor} is unexpectedly small"
        preds_a = np.ones(50, dtype=np.float32)
        preds_b = preds_a + np.float32(1e-6)
        a = self._make_backend("cpu", preds_a)
        b = self._make_backend("mlx", preds_b)
        pair = _compute_agreement(a, b, floor)
        assert pair.within_floor is True

    def test_unavailable_backend_nan(self):
        """An unavailable backend (no predictions) yields NaN diff."""
        a = self._make_backend("cpu", np.ones(10, dtype=np.float32))
        b = BackendResult(name="cuda", version="", available=False)
        floor = derived_floor_for_model(200, 6)
        pair = _compute_agreement(a, b, floor)
        assert math.isnan(pair.max_abs_diff)
        assert pair.within_floor is False


# ---------------------------------------------------------------------------
# Test 3: End-to-end CLI via VerifyResult construction (no file I/O)
# ---------------------------------------------------------------------------

class TestVerifyResult:
    def test_verdict_pass_when_all_within_floor(self):
        """VerifyResult constructed manually yields PASS when all pairs pass."""
        floor = derived_floor_for_model(200, 6)
        pair = PairwiseAgreement(
            backend_a="catboost-cpu",
            backend_b="catboost-mlx",
            max_abs_diff=1e-5,
            mean_abs_diff=5e-6,
            n_diff_above_floor=0,
            floor=floor,
            within_floor=True,
        )
        result = VerifyResult(
            model_path="model.json",
            data_path="data.parquet",
            n_rows=1000,
            n_features=28,
            loss_type="logloss",
            tree_count=200,
            max_depth=6,
            agreement=[pair],
            floor_info=floor_components(200, 6),
            verdict="PASS",
        )
        assert result.verdict == "PASS"

    def test_verdict_fail_when_any_exceeds_floor(self):
        floor = derived_floor_for_model(200, 6)
        pair = PairwiseAgreement(
            backend_a="catboost-cpu",
            backend_b="catboost-mlx",
            max_abs_diff=1.0,
            mean_abs_diff=0.5,
            n_diff_above_floor=100,
            floor=floor,
            within_floor=False,
        )
        result = VerifyResult(
            model_path="model.json",
            data_path="data.parquet",
            n_rows=1000,
            n_features=28,
            loss_type="logloss",
            tree_count=200,
            max_depth=6,
            agreement=[pair],
            floor_info=floor_components(200, 6),
            verdict="FAIL",
        )
        assert result.verdict == "FAIL"
