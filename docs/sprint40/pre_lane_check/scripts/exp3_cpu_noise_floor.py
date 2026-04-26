"""S40 pre-lane check — Experiment 3: CPU-vs-CPU noise floor across seeds.

Trains CatBoost CPU at seeds 42..46 (RS=0, bootstrap='No', otherwise identical)
and measures pairwise prediction agreement + class-distribution variance.

Decision rule:
- If CPU(seed_i) vs CPU(seed_j) agreement is also ~99.92% AND class shifts on
  the rare High class are comparable to the 64-row MLX-vs-CPU shift, then the
  observed CPU-vs-MLX gap is *within* CPU's own seed variance, which means
  the 'characterized variant' framing is well-supported.
- If CPU-vs-CPU agreement is much higher (>=99.99%) AND class shifts are tiny
  (<10), then the 0.28pp balanced-accuracy gap is mostly architectural drift,
  not seed noise, and Lane D mechanism investigation is more justified.
"""
from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import catboost
from catboost import CatBoostClassifier

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
RANDOM_STRENGTH = 0.0
SEEDS = [42, 43, 44, 45, 46]


def main() -> None:
    print(f"catboost {catboost.__version__}")

    train_df = pd.read_parquet(DATA / "train_fe_v8.parquet")
    test_df = pd.read_parquet(DATA / "test_fe_v8.parquet")
    print(f"train: {train_df.shape}  test: {test_df.shape}")

    drop = {"id", TARGET}
    feat_cols = [c for c in train_df.columns if c not in drop]
    cat_cols = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(train_df[c])]
    cat_idx = [feat_cols.index(c) for c in cat_cols]

    X_train = train_df[feat_cols].copy()
    X_test = test_df[feat_cols].copy()

    for c in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train[c], X_test[c]], axis=0).astype(str).fillna("NA")
        le.fit(combined)
        X_train[c] = np.asarray(le.transform(X_train[c].astype(str).fillna("NA"))).astype(np.int64)
        X_test[c] = np.asarray(le.transform(X_test[c].astype(str).fillna("NA"))).astype(np.int64)

    for c in feat_cols:
        if c not in cat_cols:
            X_train[c] = X_train[c].astype(np.float64)
            X_test[c] = X_test[c].astype(np.float64)

    le_y = LabelEncoder().fit(CLASS_ORDER)
    y_train = np.asarray(le_y.transform(train_df[TARGET])).astype(np.int64)

    seed_results: dict[int, dict] = {}

    for seed in SEEDS:
        print(f"\n=== CPU seed {seed} ===")
        model = CatBoostClassifier(
            iterations=ITERATIONS, depth=DEPTH, learning_rate=LEARNING_RATE,
            l2_leaf_reg=L2, loss_function="MultiClass", eval_metric="MultiClass",
            random_seed=seed, random_strength=RANDOM_STRENGTH, bootstrap_type="No",
            task_type="CPU", verbose=False, allow_writing_files=False,
        )
        t0 = time.perf_counter()
        model.fit(X_train, y_train, cat_features=cat_idx)
        fit_s = time.perf_counter() - t0
        proba = np.asarray(model.predict_proba(X_test))
        preds = proba.argmax(axis=1).astype(int)
        labels = le_y.inverse_transform(preds)
        dist = pd.Series(labels).value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)
        seed_results[seed] = {
            "fit_s": fit_s,
            "preds": preds,
            "proba": proba,
            "dist": dist.to_dict(),
        }
        print(f"  fit={fit_s:.1f}s  dist=High:{dist['High']} Low:{dist['Low']} Medium:{dist['Medium']}")

    # Pairwise comparison
    print("\n=== Pairwise CPU-vs-CPU agreement (seed_i, seed_j) ===")
    pairs = []
    for s_i, s_j in combinations(SEEDS, 2):
        p_i = seed_results[s_i]["preds"]
        p_j = seed_results[s_j]["preds"]
        pr_i = seed_results[s_i]["proba"]
        pr_j = seed_results[s_j]["proba"]
        agree = float((p_i == p_j).mean())
        n_disagree = int((p_i != p_j).sum())
        mad = float(np.abs(pr_i - pr_j).mean())
        d_i = seed_results[s_i]["dist"]
        d_j = seed_results[s_j]["dist"]
        high_shift = abs(d_j["High"] - d_i["High"])
        pairs.append({
            "seed_i": s_i, "seed_j": s_j,
            "agreement": agree, "n_disagree": n_disagree,
            "proba_mad": mad, "high_class_shift": high_shift,
        })
        print(f"  seed {s_i} vs {s_j}: agree={agree:.6%}  disagree={n_disagree}  mad={mad:.4e}  highShift={high_shift}")

    # Compare against MLX-vs-CPU baseline at seed 42
    baseline = {
        "label": "MLX_vs_CPU_seed42",
        "agreement": 0.999174,
        "n_disagree": 223,
        "proba_mad": 0.00378,
        "high_class_shift": 64,
    }
    avg_cpu_agree = float(np.mean([p["agreement"] for p in pairs]))
    avg_cpu_n_disagree = float(np.mean([p["n_disagree"] for p in pairs]))
    avg_cpu_mad = float(np.mean([p["proba_mad"] for p in pairs]))
    avg_cpu_high_shift = float(np.mean([p["high_class_shift"] for p in pairs]))

    print("\n=== Summary ===")
    print(f"CPU-vs-CPU mean agreement: {avg_cpu_agree:.6%}")
    print(f"CPU-vs-CPU mean disagree:  {avg_cpu_n_disagree:.1f}")
    print(f"CPU-vs-CPU mean proba MAD: {avg_cpu_mad:.4e}")
    print(f"CPU-vs-CPU mean High shift:{avg_cpu_high_shift:.1f}")
    print(f"Baseline MLX-vs-CPU @42:   agree={baseline['agreement']:.6%} disagree={baseline['n_disagree']} mad={baseline['proba_mad']:.4e} highShift={baseline['high_class_shift']}")

    # Verdict
    if avg_cpu_agree >= 0.9990 and avg_cpu_high_shift >= 30:
        verdict = "WITHIN_NOISE_FLOOR_lane_B_strong"
    elif avg_cpu_agree >= 0.9999 and avg_cpu_high_shift < 10:
        verdict = "OUTSIDE_NOISE_FLOOR_lane_D_justified"
    else:
        verdict = "INTERMEDIATE_lane_B_with_caveat"

    print(f"\nVerdict: {verdict}")

    summary = {
        "experiment": "s40_exp3_cpu_noise_floor",
        "config": {
            "iterations": ITERATIONS, "depth": DEPTH, "lr": LEARNING_RATE, "l2": L2,
            "random_strength": RANDOM_STRENGTH, "seeds": SEEDS,
        },
        "per_seed_dist": {str(s): r["dist"] for s, r in seed_results.items()},
        "per_seed_fit_s": {str(s): r["fit_s"] for s, r in seed_results.items()},
        "pairwise": pairs,
        "summary": {
            "cpu_vs_cpu_mean_agreement": avg_cpu_agree,
            "cpu_vs_cpu_mean_disagree": avg_cpu_n_disagree,
            "cpu_vs_cpu_mean_proba_mad": avg_cpu_mad,
            "cpu_vs_cpu_mean_high_shift": avg_cpu_high_shift,
        },
        "baseline_mlx_vs_cpu_seed42": baseline,
        "verdict": verdict,
    }
    out_path = OUT / "exp3_cpu_noise_floor.json"
    out_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
