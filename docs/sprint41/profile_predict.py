"""S41-T3 — profile predict() subprocess overhead vs in-process for the
two code paths in core.py:_run_predict (numeric-only -> in-process,
categorical -> subprocess). Goal: locate the 41× slowdown's actual mechanism.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from catboost_mlx import CatBoostMLXRegressor


def make_synthetic(n: int = 50_000, n_num: int = 8, n_cat: int = 4, seed: int = 42):
    rng = np.random.default_rng(seed)
    X_num = rng.standard_normal((n, n_num)).astype(np.float64)
    X_cat = rng.integers(0, 8, size=(n, n_cat)).astype(np.int64)
    X = np.concatenate([X_num, X_cat], axis=1)
    cat_signal_idx = n_num if n_cat > 0 else 0
    y = X[:, 0] + 0.3 * X[:, 1] + 0.1 * X[:, cat_signal_idx] + rng.standard_normal(n) * 0.1
    cat_idx = list(range(n_num, n_num + n_cat))
    return X, y, cat_idx


def time_call(label: str, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"  {label:35s}  {dt*1000:8.1f} ms")
    return dt, out


def main():
    print("=== S41-T3 — predict() path profiling ===")

    # 1. Numeric-only -> in-process path
    print("\n[1] Numeric-only model (cat_features=[]; in-process predict path)")
    X_num_only, y_num, _ = make_synthetic(n=50_000, n_num=12, n_cat=0)
    m_num = CatBoostMLXRegressor(
        iterations=100, depth=6, learning_rate=0.1,
        random_strength=0, bootstrap_type="No", verbose=False,
    )
    m_num.fit(X_num_only, y_num)
    times_num = []
    for i in range(3):
        dt, _ = time_call(f"  predict trial {i+1} (50k rows)", lambda: m_num.predict(X_num_only))
        times_num.append(dt)
    in_process_median = np.median(times_num)

    # 2. Categorical -> subprocess path
    print("\n[2] Model with 4 categorical features (subprocess predict path)")
    X_with_cat, y_cat, cat_idx = make_synthetic(n=50_000, n_num=8, n_cat=4)
    m_cat = CatBoostMLXRegressor(
        iterations=100, depth=6, learning_rate=0.1,
        random_strength=0, bootstrap_type="No", verbose=False,
        cat_features=cat_idx,
    )
    m_cat.fit(X_with_cat, y_cat)
    times_cat = []
    for i in range(3):
        dt, _ = time_call(f"  predict trial {i+1} (50k rows)", lambda: m_cat.predict(X_with_cat))
        times_cat.append(dt)
    subprocess_median = np.median(times_cat)

    # 3. Decompose the subprocess path: time individual phases.
    print("\n[3] Subprocess-path phase breakdown (50k rows × 12 features)")
    import os, tempfile, json, subprocess, shutil
    from catboost_mlx.core import _array_to_csv, _find_binary

    tmp = tempfile.mkdtemp(prefix="prof_")
    try:
        model_json = json.dumps(m_cat._model_data)
        model_path = os.path.join(tmp, "model.json")
        data_path = os.path.join(tmp, "data.csv")
        out_path = os.path.join(tmp, "predictions.csv")

        time_call("write model.json",
                  lambda: open(model_path, "w").write(model_json))
        time_call("write data.csv (50k×12)",
                  lambda: _array_to_csv(data_path, X_with_cat, cat_features=cat_idx))

        binary = _find_binary("csv_predict", None)
        args = [binary, model_path, data_path, "--output", out_path]
        time_call("subprocess csv_predict run",
                  lambda: subprocess.run(args, capture_output=True, text=True))

        time_call("read predictions.csv",
                  lambda: open(out_path).read())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 4. Summary
    print("\n=== Summary ===")
    print(f"in-process median: {in_process_median*1000:.1f} ms / 50k rows")
    print(f"subprocess median: {subprocess_median*1000:.1f} ms / 50k rows")
    print(f"slowdown ratio   : {subprocess_median / in_process_median:.1f}×")
    print(f"\nrows/s in-process: {50_000 / in_process_median:,.0f}")
    print(f"rows/s subprocess: {50_000 / subprocess_median:,.0f}")


if __name__ == "__main__":
    main()
