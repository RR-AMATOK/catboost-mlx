"""Shared utilities for the 4 framework runners.

Each runner under benchmarks/upstream/scripts/run_<framework>.py:
  1. Loads meta.json + train.csv + test.csv from a dataset's cache dir
  2. Configures the framework with consistent hyperparameters (per `BENCH_HP`)
  3. Trains, predicts, and times the train phase
  4. Computes the appropriate metric (logloss / rmse / ndcg@10)
  5. Captures peak RSS via resource.getrusage
  6. Writes a single results JSON to <results_dir>/<dataset>_<framework>_<seed>.json

S42-T2. See docs/sprint42/sprint-plan.md.
"""
from __future__ import annotations

import json
import platform
import resource
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ── Hyperparameter contract (held constant across all 4 frameworks) ─────────
#
# Same depth, lr, l2 across libraries. RandomStrength=0 + bootstrap=No on
# CatBoost-CPU and CatBoost-MLX to remove RNG-injected noise as a confound
# (per DEC-045/046). LightGBM and XGBoost don't have RandomStrength but their
# default subsample+bagging behavior is already deterministic per seed.
#
# `iterations` is set per task (200 for classification, 500 for ranking) to
# keep wall-clock manageable while still being indicative of relative perf;
# upstream's training_speed/ uses 1000+ for some grids but we choose a
# tractable single point.

BENCH_HP = {
    "iterations": 200,
    "depth": 6,
    "learning_rate": 0.1,
    "l2_reg": 3.0,
    # CatBoost-only knobs:
    "random_strength": 0.0,
    "bootstrap_type": "no",
}

# Default iteration count — runs that override via `--iterations` get a
# dataset-name suffix so the aggregator groups them separately from the
# canonical 200-iter sweep (see apply_iterations_override below).
_DEFAULT_ITERATIONS = BENCH_HP["iterations"]


def apply_iterations_override(args) -> str:
    """Each runner calls this once after argparse. If `--iterations` was
    passed and differs from the default, mutate BENCH_HP['iterations'] in
    place and return a tagged dataset name (e.g. 'adult_iter1000') so the
    result JSON files group separately from the default sweep. Otherwise
    return args.dataset unchanged.

    Used by S43-T2 to re-run benchmarks at iters=1000 (the mathematician's
    fair-convergence reframe) without losing the canonical 200-iter data.
    """
    iters = getattr(args, "iterations", None)
    if iters and iters != _DEFAULT_ITERATIONS:
        BENCH_HP["iterations"] = iters
        return f"{args.dataset}_iter{iters}"
    return args.dataset


# ── Result record ───────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    framework: str
    framework_version: str
    dataset: str
    task: str                    # classification | regression | ranking
    metric_name: str             # logloss | rmse | ndcg@10
    metric_value: float
    seed: int
    train_seconds: float
    predict_seconds: float
    peak_rss_bytes: int
    n_train: int
    n_test: int
    n_features: int
    cat_indices: List[int]
    hyperparameters: Dict[str, Any]
    hardware: str = ""
    python_version: str = ""
    notes: str = ""

    def write(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.dataset}_{self.framework}_{self.seed}.json"
        path.write_text(json.dumps(asdict(self), indent=2, default=str))
        return path


# ── Cache + meta loaders ────────────────────────────────────────────────────

def load_meta(dataset: str, *, cache_root: Optional[Path] = None) -> Dict[str, Any]:
    base = cache_root or (Path.home() / ".cache" / "catboost-mlx-benchmarks")
    meta_path = base / dataset / "meta.json"
    if not meta_path.exists():
        raise RuntimeError(
            f"meta.json missing for dataset {dataset!r} at {meta_path}. "
            f"Run the adapter first: "
            f"python -m benchmarks.upstream.adapters.{dataset}"
        )
    return json.loads(meta_path.read_text())


def load_csv_pair(dataset: str, *, cache_root: Optional[Path] = None
                  ) -> Tuple[Path, Path, Dict[str, Any]]:
    """Return (train_csv, test_csv, meta_dict)."""
    meta = load_meta(dataset, cache_root=cache_root)
    base = (cache_root or (Path.home() / ".cache" / "catboost-mlx-benchmarks")) / dataset
    train_csv = base / "train.csv"
    test_csv = base / "test.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise RuntimeError(
            f"train.csv or test.csv missing in {base}. The adapter may have "
            f"prepared only meta.json + split.json for the streaming path; "
            f"re-run with --materialize for frameworks that require dense CSVs."
        )
    return train_csv, test_csv, meta


def load_xy(csv_path: Path, meta: Dict[str, Any]
            ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Read a prepared CSV into (X, y, group_id-or-None) using meta target_col / group_col.

    For non-ranking datasets, group_col is None and the third element is None.
    """
    df = pd.read_csv(csv_path)
    target_col = int(meta["target_col"])
    group_col = meta.get("group_col")

    cols = list(df.columns)
    target_name = cols[target_col]
    feature_cols = [c for i, c in enumerate(cols) if i != target_col and i != group_col]

    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df[target_name].to_numpy()
    g = df[cols[int(group_col)]].to_numpy() if group_col is not None else None
    return X, y, g


# ── Timer + RSS helpers ─────────────────────────────────────────────────────

@contextmanager
def timer():
    """Context manager that yields a callable returning elapsed seconds."""
    t0 = time.perf_counter()
    elapsed = [0.0]
    def get_elapsed():
        return elapsed[0]
    try:
        yield get_elapsed
    finally:
        elapsed[0] = time.perf_counter() - t0


def peak_rss_bytes() -> int:
    """Return peak resident set size in bytes (process-lifetime).

    macOS reports `ru_maxrss` in bytes; Linux reports in kilobytes. We
    detect platform and normalize to bytes.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(raw)
    return int(raw) * 1024


# ── Metrics ──────────────────────────────────────────────────────────────────

def logloss(y_true: np.ndarray, y_pred_proba: np.ndarray, *, eps: float = 1e-15
            ) -> float:
    """Binary or multiclass logloss. y_pred_proba is (n,) for binary or (n, K) for multiclass."""
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    if y_pred_proba.ndim == 1:
        # binary; clip for numerical stability
        p = np.clip(y_pred_proba, eps, 1 - eps)
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
    # multiclass: y_true is class indices, y_pred_proba is per-class probabilities
    p = np.clip(y_pred_proba, eps, 1 - eps)
    n = y_true.shape[0]
    return float(-np.mean(np.log(p[np.arange(n), y_true.astype(int)])))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, group_id: np.ndarray, *, k: int = 10
              ) -> float:
    """Mean NDCG@k across groups. y_true is graded relevance (0..4), y_pred is the
    raw ranking score from the model (higher = more relevant).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    group_id = np.asarray(group_id)
    unique_groups, inv = np.unique(group_id, return_inverse=True)
    ndcgs = []
    for gi, _ in enumerate(unique_groups):
        mask = inv == gi
        gt = y_true[mask]
        sc = y_pred[mask]
        if gt.size == 0:
            continue
        order = np.argsort(-sc)
        gains = (2.0 ** gt[order] - 1.0)
        discounts = np.log2(np.arange(2, gains.size + 2))
        dcg = np.sum(gains[:k] / discounts[:k])
        ideal_order = np.argsort(-gt)
        ideal_gains = (2.0 ** gt[ideal_order] - 1.0)
        idcg = np.sum(ideal_gains[:k] / discounts[:k])
        if idcg > 0:
            ndcgs.append(dcg / idcg)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


# ── Hardware detection ──────────────────────────────────────────────────────

def hardware_string() -> str:
    """Return 'Apple M3 Max | macOS 26.3' style string for results JSON."""
    try:
        import subprocess
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True, check=False).stdout.strip()
        os_v = platform.mac_ver()[0]
        return f"{chip} | macOS {os_v}" if chip else f"{platform.platform()}"
    except Exception:
        return platform.platform()
