"""
verifier.py -- Core parity-verification logic for catboost-tripoint.

Loads a model in each requested backend (cpu, mlx, cuda), runs prediction
on the supplied dataset, and computes pairwise numerical agreement statistics
against the theoretical fp32 floor.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .floor import derived_floor_for_model, floor_components


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BackendResult:
    name: str
    version: str
    available: bool
    wall_seconds: float = 0.0
    predictions: Optional[np.ndarray] = None
    predictions_sha256: str = ""
    error: str = ""


@dataclass
class PairwiseAgreement:
    backend_a: str
    backend_b: str
    max_abs_diff: float
    mean_abs_diff: float
    n_diff_above_floor: int
    floor: float
    within_floor: bool


@dataclass
class VerifyResult:
    model_path: str
    data_path: str
    n_rows: int
    n_features: int
    loss_type: str
    tree_count: int
    max_depth: int
    backends: dict[str, BackendResult] = field(default_factory=dict)
    agreement: list[PairwiseAgreement] = field(default_factory=list)
    floor_info: dict = field(default_factory=dict)
    verdict: str = "ERROR"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data(data_path: str, target_col: Optional[str]) -> tuple[np.ndarray, Optional[np.ndarray], list[str]]:
    """Load evaluation data from parquet (preferred) or CSV (fallback).

    Returns (X, y, feature_names).  y is None when target_col is not found.
    """
    import os
    ext = os.path.splitext(data_path)[1].lower()

    if ext == ".parquet":
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(data_path)
            df = table.to_pandas()
        except ImportError:
            raise ImportError(
                "pyarrow is required to read .parquet files.  "
                "Install with: pip install pyarrow\n"
                "Or convert your data to CSV and pass a .csv file."
            )
    elif ext == ".csv":
        import pandas as pd
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported data format: {ext!r}. Use .parquet or .csv.")

    if target_col and target_col in df.columns:
        y = df[target_col].to_numpy()
        feature_cols = [c for c in df.columns if c != target_col]
    else:
        y = None
        feature_cols = list(df.columns)

    X = df[feature_cols].to_numpy(dtype=np.float32)
    return X, y, feature_cols


# ---------------------------------------------------------------------------
# Backend runners
# ---------------------------------------------------------------------------

def _run_cpu_backend(model_path: str, X: np.ndarray) -> BackendResult:
    """Run prediction via catboost CPU (upstream catboost package)."""
    result = BackendResult(name="catboost-cpu", version="", available=False)
    try:
        import catboost
        result.version = catboost.__version__
        result.available = True
    except ImportError:
        result.error = "catboost package not installed"
        return result

    try:
        model = catboost.CatBoost()
        model.load_model(model_path)
        t0 = time.perf_counter()
        preds = model.predict(X, prediction_type="RawFormulaVal")
        result.wall_seconds = time.perf_counter() - t0
        preds = np.asarray(preds, dtype=np.float32).ravel()
        # For binary classification raw scores: convert to probability space
        # to match mlx output so diffs are comparable.
        # Keep raw here; callers compare same-space outputs.
        result.predictions = preds
        result.predictions_sha256 = hashlib.sha256(preds.tobytes()).hexdigest()
    except Exception as exc:
        result.error = str(exc)
        result.available = False

    return result


def _run_mlx_backend(model_path: str, X: np.ndarray) -> BackendResult:
    """Run prediction via catboost-mlx (Apple Silicon Metal backend)."""
    result = BackendResult(name="catboost-mlx", version="", available=False)
    try:
        import catboost_mlx
        result.version = getattr(catboost_mlx, "__version__", "<unknown>")
        result.available = True
    except ImportError:
        result.error = "catboost_mlx package not installed"
        return result

    try:
        model = catboost_mlx.CatBoostMLX()
        model.load_model(model_path)
        loss_type = model._get_loss_type()
        t0 = time.perf_counter()
        if loss_type in ("logloss", "multiclass"):
            preds = model.predict_proba(X)
            # For binary logloss predict_proba returns (n, 2); take positive class
            if preds.ndim == 2 and preds.shape[1] == 2:
                preds = preds[:, 1]
        else:
            preds = model.predict(X)
        result.wall_seconds = time.perf_counter() - t0
        preds = np.asarray(preds, dtype=np.float32).ravel()
        result.predictions = preds
        result.predictions_sha256 = hashlib.sha256(preds.tobytes()).hexdigest()
    except Exception as exc:
        result.error = str(exc)
        result.available = False

    return result


def _run_cuda_backend(model_path: str, X: np.ndarray) -> BackendResult:
    """Run prediction via catboost CUDA (GPU task_type='GPU').

    Uses the same upstream catboost package as the CPU backend but with
    task_type='GPU'.  If CUDA/GPU is unavailable, marks as not available.
    """
    result = BackendResult(name="catboost-cuda", version="", available=False)
    try:
        import catboost
        result.version = catboost.__version__
    except ImportError:
        result.error = "catboost package not installed"
        return result

    try:
        # Probe GPU availability by constructing a GPU-task model; catboost
        # raises CatBoostError if no CUDA device is present.
        model = catboost.CatBoost({"task_type": "GPU"})
        model.load_model(model_path)
        result.available = True
        t0 = time.perf_counter()
        preds = model.predict(X, prediction_type="RawFormulaVal")
        result.wall_seconds = time.perf_counter() - t0
        preds = np.asarray(preds, dtype=np.float32).ravel()
        result.predictions = preds
        result.predictions_sha256 = hashlib.sha256(preds.tobytes()).hexdigest()
    except Exception as exc:
        # Any exception — CatBoostError("No CUDA"), CUDA not found, etc.
        err_str = str(exc)
        if "GPU" in err_str or "CUDA" in err_str or "cuda" in err_str.lower():
            result.error = "CUDA GPU not available on this machine"
        else:
            result.error = err_str
        result.available = False

    return result


# ---------------------------------------------------------------------------
# Model metadata extraction
# ---------------------------------------------------------------------------

def _extract_mlx_metadata(model_path: str) -> dict:
    """Extract tree_count, max_depth, loss_type from a catboost-mlx JSON model."""
    import json
    with open(model_path) as f:
        data = json.load(f)
    info = data.get("model_info", {})
    trees = data.get("trees", [])
    tree_count = len(trees)
    loss_type = info.get("loss_type", "unknown")
    # Infer max_depth: each oblivious tree has splits list with len = depth
    max_depth = 6  # fallback default
    if trees:
        depths = []
        for t in trees:
            splits = t.get("splits", [])
            if splits:
                depths.append(len(splits))
        if depths:
            max_depth = max(depths)
    return {
        "loss_type": loss_type,
        "tree_count": tree_count,
        "max_depth": max_depth,
        "format": "catboost_mlx_json",
    }


def _extract_catboost_metadata(model_path: str) -> dict:
    """Extract tree_count, max_depth, loss_type from an upstream catboost model.

    Supports .cbm (binary) and .json formats that catboost.CatBoost can load.
    """
    try:
        import catboost
        model = catboost.CatBoost()
        model.load_model(model_path)
        params = model.get_all_params()
        tree_count = model.tree_count_
        max_depth = int(params.get("depth", 6))
        loss_type = params.get("loss_function", "unknown")
        return {
            "loss_type": loss_type,
            "tree_count": tree_count,
            "max_depth": max_depth,
            "format": "catboost_binary",
        }
    except Exception:
        return {}


def extract_model_metadata(model_path: str) -> dict:
    """Auto-detect model format and extract metadata.

    Tries catboost-mlx JSON first (since this is a catboost-mlx project),
    then falls back to upstream catboost binary format.
    """
    import os
    ext = os.path.splitext(model_path)[1].lower()
    if ext == ".json":
        try:
            return _extract_mlx_metadata(model_path)
        except Exception:
            pass
    # Try upstream catboost (handles .cbm and other formats)
    meta = _extract_catboost_metadata(model_path)
    if meta:
        return meta
    # Last resort: try JSON regardless of extension
    try:
        return _extract_mlx_metadata(model_path)
    except Exception as exc:
        raise ValueError(
            f"Cannot read model metadata from {model_path!r}: {exc}\n"
            "Supported formats: catboost-mlx JSON (.json) or catboost binary (.cbm)"
        )


# ---------------------------------------------------------------------------
# Pairwise agreement computation
# ---------------------------------------------------------------------------

def _compute_agreement(
    a: BackendResult,
    b: BackendResult,
    floor: float,
) -> PairwiseAgreement:
    """Compute pairwise numerical agreement between two backend predictions."""
    if a.predictions is None or b.predictions is None:
        return PairwiseAgreement(
            backend_a=a.name,
            backend_b=b.name,
            max_abs_diff=float("nan"),
            mean_abs_diff=float("nan"),
            n_diff_above_floor=0,
            floor=floor,
            within_floor=False,
        )

    # Align lengths in case predictions are ragged (defensive)
    n = min(len(a.predictions), len(b.predictions))
    diff = np.abs(a.predictions[:n].astype(np.float64) - b.predictions[:n].astype(np.float64))
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    n_above = int(np.sum(diff > floor))

    return PairwiseAgreement(
        backend_a=a.name,
        backend_b=b.name,
        max_abs_diff=max_diff,
        mean_abs_diff=mean_diff,
        n_diff_above_floor=n_above,
        floor=floor,
        within_floor=(max_diff <= floor),
    )


# ---------------------------------------------------------------------------
# Top-level verify
# ---------------------------------------------------------------------------

def verify(
    model_path: str,
    data_path: str,
    target_col: Optional[str] = None,
    backends: list[str] | None = None,
    batch_size: int = 1000,
) -> VerifyResult:
    """Run the tripoint verification.

    Parameters
    ----------
    model_path : str
    data_path : str
    target_col : str or None
        Column name to drop from features (used as label only for display).
    backends : list of str or None
        Subset of {"cpu", "mlx", "cuda"}.  Defaults to ["cpu", "mlx"].
    batch_size : int
        Unused in sketch (full-batch prediction). Reserved for future batching.

    Returns
    -------
    VerifyResult
    """
    if backends is None:
        backends = ["cpu", "mlx"]

    # 1. Load data
    X, _y, feature_names = _load_data(data_path, target_col)
    n_rows, n_features = X.shape

    # 2. Extract model metadata
    meta = extract_model_metadata(model_path)
    tree_count = meta.get("tree_count", 0)
    max_depth = meta.get("max_depth", 6)
    loss_type = meta.get("loss_type", "unknown")

    # 3. Compute theoretical floor
    fp_floor = derived_floor_for_model(tree_count, max_depth)
    floor_info = floor_components(tree_count, max_depth)

    result = VerifyResult(
        model_path=model_path,
        data_path=data_path,
        n_rows=n_rows,
        n_features=n_features,
        loss_type=loss_type,
        tree_count=tree_count,
        max_depth=max_depth,
        floor_info=floor_info,
    )

    # 4. Run each requested backend
    runners = {
        "cpu": _run_cpu_backend,
        "mlx": _run_mlx_backend,
        "cuda": _run_cuda_backend,
    }
    backend_results: dict[str, BackendResult] = {}
    for name in backends:
        if name not in runners:
            raise ValueError(f"Unknown backend {name!r}. Choose from: {list(runners)}")
        br = runners[name](model_path, X)
        backend_results[name] = br
    result.backends = backend_results

    # 5. Compute pairwise agreement for all pairs that have predictions
    available = [
        br for br in backend_results.values()
        if br.available and br.predictions is not None
    ]
    pairs = []
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            pairs.append(_compute_agreement(available[i], available[j], fp_floor))
    result.agreement = pairs

    # 6. Verdict: PASS if all pairwise diffs are within floor; ERROR if no
    #    pairs were tested; FAIL otherwise.
    if not pairs:
        result.verdict = "ERROR"
    elif all(p.within_floor for p in pairs):
        result.verdict = "PASS"
    else:
        result.verdict = "FAIL"

    return result
