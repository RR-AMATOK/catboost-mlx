"""
test_integration.py -- End-to-end integration test for catboost-tripoint.

Trains a tiny CatBoost-MLX model on 500 synthetic samples, writes it to a
temp file, then runs verify() and checks that:
  - both backends return predictions
  - CPU vs MLX diff is within the theoretical fp32 floor
  - verdict is PASS
  - JSON report fields are populated correctly

Requires catboost and catboost_mlx to be installed.
Marks as 'integration' so it can be excluded in unit-only runs:
    pytest -m "not integration"
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("catboost")
pytest.importorskip("catboost_mlx")

from catboost_tripoint.cli import _build_json_report, _print_summary
from catboost_tripoint.verifier import verify


@pytest.fixture(scope="module")
def tiny_model_and_data(tmp_path_factory):
    """Train a 50-tree depth-4 model on 500 synthetic samples.

    Returns (model_path, eval_csv_path).
    """
    import io
    import catboost_mlx
    from catboost_mlx import CatBoostMLXClassifier

    rng = np.random.default_rng(42)
    n_train, n_eval, n_feat = 500, 200, 10

    X_tr = rng.standard_normal((n_train, n_feat)).astype(np.float32)
    y_tr = (X_tr[:, 0] + X_tr[:, 1] > 0).astype(np.int64)

    X_ev = rng.standard_normal((n_eval, n_feat)).astype(np.float32)
    y_ev = (X_ev[:, 0] + X_ev[:, 1] > 0).astype(np.int64)

    model = CatBoostMLXClassifier(
        iterations=50,
        depth=4,
        learning_rate=0.1,
        l2_reg_lambda=3.0,
        random_seed=42,
        verbose=False,
    )
    model.fit(X_tr, y_tr)

    tmp = tmp_path_factory.mktemp("tripoint")
    model_path = str(tmp / "tiny_model.json")
    model.save_model(model_path)

    # Write eval CSV with a 'target' column
    import pandas as pd
    feat_cols = [f"f{i}" for i in range(n_feat)]
    df = pd.DataFrame(X_ev, columns=feat_cols)
    df["target"] = y_ev
    eval_csv = str(tmp / "eval.csv")
    df.to_csv(eval_csv, index=False)

    return model_path, eval_csv


@pytest.mark.integration
def test_verify_cpu_mlx_pass(tiny_model_and_data):
    """Verify that CPU vs MLX predictions are within fp32 floor for a tiny model."""
    model_path, eval_csv = tiny_model_and_data

    result = verify(
        model_path=model_path,
        data_path=eval_csv,
        target_col="target",
        backends=["cpu", "mlx"],
    )

    # Both backends should be available
    cpu_br = result.backends.get("cpu")
    mlx_br = result.backends.get("mlx")
    assert cpu_br is not None and cpu_br.available, f"CPU backend failed: {cpu_br.error if cpu_br else 'missing'}"
    assert mlx_br is not None and mlx_br.available, f"MLX backend failed: {mlx_br.error if mlx_br else 'missing'}"

    # Predictions should have the right shape
    assert cpu_br.predictions is not None
    assert mlx_br.predictions is not None
    assert len(cpu_br.predictions) == 200
    assert len(mlx_br.predictions) == 200

    # There should be exactly one pairwise agreement entry
    assert len(result.agreement) == 1
    pair = result.agreement[0]

    # Verdict must be PASS (diff within floor)
    assert result.verdict == "PASS", (
        f"Expected PASS but got {result.verdict}. "
        f"max_abs_diff={pair.max_abs_diff:.3e}, floor={pair.floor:.3e}"
    )

    # Sanity: floor formula must produce a positive value
    assert result.floor_info["derived_floor"] > 0

    # Predictions must be finite
    assert np.all(np.isfinite(cpu_br.predictions))
    assert np.all(np.isfinite(mlx_br.predictions))


@pytest.mark.integration
def test_json_report_structure(tiny_model_and_data):
    """JSON report contains all required top-level keys and correct types."""
    model_path, eval_csv = tiny_model_and_data

    result = verify(
        model_path=model_path,
        data_path=eval_csv,
        target_col="target",
        backends=["cpu", "mlx"],
    )
    report = _build_json_report(result)

    required_keys = {
        "version", "tool", "timestamp", "model_metadata",
        "data_metadata", "backends", "agreement",
        "theoretical_floor", "verdict",
    }
    assert required_keys <= set(report.keys()), f"Missing keys: {required_keys - set(report.keys())}"
    assert report["tool"] == "catboost-tripoint"
    assert report["verdict"] in ("PASS", "FAIL", "ERROR")
    assert report["theoretical_floor"]["epsilon_machine"] == pytest.approx(2.0 ** -23, rel=1e-12)
    assert report["model_metadata"]["tree_count"] == 50
    assert report["model_metadata"]["max_depth"] == 4
    assert report["data_metadata"]["n_rows"] == 200
    assert report["data_metadata"]["n_features"] == 10

    # SHA256 must be 64 hex chars
    for _key, br in report["backends"].items():
        if br["available"]:
            assert len(br["predictions_sha256"]) == 64


@pytest.mark.integration
def test_cuda_not_available_graceful(tiny_model_and_data):
    """CUDA backend reports NOT AVAILABLE gracefully on Apple Silicon (no CUDA)."""
    model_path, eval_csv = tiny_model_and_data

    result = verify(
        model_path=model_path,
        data_path=eval_csv,
        target_col="target",
        backends=["cpu", "mlx", "cuda"],
    )

    cuda_br = result.backends.get("cuda")
    assert cuda_br is not None
    # On Apple Silicon, CUDA is unavailable; the tool should degrade gracefully
    if not cuda_br.available:
        assert cuda_br.error != ""  # some error message was recorded
        # Verdict is still determined by cpu/mlx pair
        assert result.verdict in ("PASS", "FAIL")
    # If somehow CUDA is available (not expected on M3), just pass
