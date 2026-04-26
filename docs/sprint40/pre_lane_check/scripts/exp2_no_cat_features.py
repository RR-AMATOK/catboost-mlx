"""S40 pre-lane check — Experiment 2: cat_features=[] discriminator.

Re-runs the irrigation parity test with all 8 categorical columns DROPPED
(numeric-only feature matrix; cat_features=[] on both runtimes).

Predicted outcomes:
- Agreement >= 99.99% AND probability MAD <= 5e-5 -> M2 (CTR RNG ordering) dominates.
- Agreement still ~99.92% AND MAD ~1e-3                -> M1 (multiclass dispatch) drives.
- Agreement >= 99.99% AND MAD ~1e-4                    -> M3 (fp32 vs fp64 floor).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import catboost
import catboost_mlx
from catboost import CatBoostClassifier
from catboost_mlx import CatBoostMLXClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "experiments" / "s40_pre_lane_check" / "results"
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "Irrigation_Need"
CLASS_ORDER = ["High", "Low", "Medium"]
ITERATIONS = 500
DEPTH = 6
LEARNING_RATE = 0.05
L2 = 3.0
SEED = 42
RANDOM_STRENGTH = 0.0


def main() -> None:
    print(f"catboost {catboost.__version__}  catboost_mlx {getattr(catboost_mlx, '__version__', '<unset>')}")

    train_df = pd.read_parquet(DATA / "train_fe_v8.parquet")
    test_df = pd.read_parquet(DATA / "test_fe_v8.parquet")
    print(f"train: {train_df.shape}  test: {test_df.shape}")

    drop = {"id", TARGET}
    feat_cols = [c for c in train_df.columns if c not in drop]
    cat_cols = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(train_df[c])]
    numeric_cols = [c for c in feat_cols if c not in cat_cols]
    print(f"all features: {len(feat_cols)}  categorical: {len(cat_cols)}  numeric (kept): {len(numeric_cols)}")

    X_train = train_df[numeric_cols].astype(np.float64).copy()
    X_test = test_df[numeric_cols].astype(np.float64).copy()

    le_y = LabelEncoder().fit(CLASS_ORDER)
    y_train = np.asarray(le_y.transform(train_df[TARGET])).astype(np.int64)

    print(f"X_train: {X_train.shape}  X_test: {X_test.shape}  cat_features=[]")

    # CPU
    print("\n=== CatBoost CPU (no cat_features) ===")
    cpu_model = CatBoostClassifier(
        iterations=ITERATIONS, depth=DEPTH, learning_rate=LEARNING_RATE,
        l2_leaf_reg=L2, loss_function="MultiClass", eval_metric="MultiClass",
        random_seed=SEED, random_strength=RANDOM_STRENGTH, bootstrap_type="No",
        task_type="CPU", verbose=100, allow_writing_files=False,
    )
    t0 = time.perf_counter()
    cpu_model.fit(X_train, y_train, cat_features=[])
    cpu_fit_s = time.perf_counter() - t0
    print(f"CPU fit: {cpu_fit_s:.2f}s")

    t0 = time.perf_counter()
    cpu_proba = np.asarray(cpu_model.predict_proba(X_test))
    cpu_predict_s = time.perf_counter() - t0
    cpu_preds = cpu_proba.argmax(axis=1).astype(int)
    print(f"CPU predict: {cpu_predict_s:.2f}s")

    # MLX
    print("\n=== CatBoost-MLX (no cat_features) ===")
    mlx_model = CatBoostMLXClassifier(
        iterations=ITERATIONS, depth=DEPTH, learning_rate=LEARNING_RATE,
        l2_reg_lambda=L2, loss="multiclass", cat_features=[],
        random_seed=SEED, random_strength=RANDOM_STRENGTH, bootstrap_type="no",
        verbose=True,
    )
    t0 = time.perf_counter()
    mlx_model.fit(X_train, y_train)
    mlx_fit_s = time.perf_counter() - t0
    print(f"\nMLX fit: {mlx_fit_s:.2f}s")

    t0 = time.perf_counter()
    mlx_proba = np.asarray(mlx_model.predict_proba(X_test))
    mlx_predict_s = time.perf_counter() - t0
    row_sum = mlx_proba.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    mlx_proba = mlx_proba / row_sum
    mlx_preds = mlx_proba.argmax(axis=1).astype(int)
    print(f"MLX predict: {mlx_predict_s:.2f}s")

    # Compare
    agreement = float((cpu_preds == mlx_preds).mean())
    n_disagree = int((cpu_preds != mlx_preds).sum())
    proba_mad = float(np.abs(cpu_proba - mlx_proba).mean())
    proba_max_abs = float(np.abs(cpu_proba - mlx_proba).max())

    cpu_dist = pd.Series(le_y.inverse_transform(cpu_preds)).value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)
    mlx_dist = pd.Series(le_y.inverse_transform(mlx_preds)).value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)

    conf = pd.crosstab(
        pd.Series(le_y.inverse_transform(cpu_preds), name="CPU"),
        pd.Series(le_y.inverse_transform(mlx_preds), name="MLX"),
    ).reindex(index=CLASS_ORDER, columns=CLASS_ORDER, fill_value=0)

    print("\n=== RESULT (no cat_features) ===")
    print(f"Agreement: {agreement:.6%}  ({len(cpu_preds) - n_disagree}/{len(cpu_preds)})")
    print(f"Disagreement rows: {n_disagree}")
    print(f"Probability MAD: {proba_mad:.6e}")
    print(f"Probability max-abs diff: {proba_max_abs:.6e}")
    print("Class distribution:")
    print(pd.DataFrame({"cpu": cpu_dist, "mlx": mlx_dist, "delta": mlx_dist - cpu_dist}).to_string())
    print("\nDisagreement matrix (rows=CPU, cols=MLX):")
    print(conf.to_string())

    # Verdict
    if agreement >= 0.9999 and proba_mad <= 5e-5:
        verdict = "M2_dominant_CTR_RNG"
    elif agreement < 0.9995 and proba_mad > 5e-4:
        verdict = "M1_dominant_multiclass_dispatch_or_other"
    elif agreement >= 0.9999 and proba_mad > 5e-5:
        verdict = "M3_dominant_fp_precision_floor"
    else:
        verdict = "ambiguous_check_against_with_cat"

    print(f"\nVerdict: {verdict}")

    summary = {
        "experiment": "s40_exp2_no_cat_features",
        "config": {
            "iterations": ITERATIONS, "depth": DEPTH, "lr": LEARNING_RATE, "l2": L2,
            "seed": SEED, "random_strength": RANDOM_STRENGTH,
            "n_features_used": len(numeric_cols), "n_cat_dropped": len(cat_cols),
        },
        "timing_seconds": {
            "cpu_fit": cpu_fit_s, "cpu_predict": cpu_predict_s,
            "mlx_fit": mlx_fit_s, "mlx_predict": mlx_predict_s,
        },
        "result": {
            "agreement": agreement,
            "n_disagree": n_disagree,
            "proba_mad": proba_mad,
            "proba_max_abs_diff": proba_max_abs,
            "class_dist_cpu": cpu_dist.to_dict(),
            "class_dist_mlx": mlx_dist.to_dict(),
            "disagreement_matrix": conf.to_dict(),
        },
        "verdict": verdict,
        "baseline_with_cat_features": {
            "agreement": 0.999174,
            "n_disagree": 223,
            "proba_mad": 0.00378,
            "high_class_concentration_pct": 74.4,
        },
    }
    out_path = OUT / "exp2_no_cat_features.json"
    out_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
