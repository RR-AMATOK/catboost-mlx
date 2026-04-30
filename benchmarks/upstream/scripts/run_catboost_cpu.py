"""CatBoost-CPU runner for the upstream benchmark suite.

Trains the upstream CatBoost CPU implementation with the shared BENCH_HP
hyperparameters (and CatBoost-specific RandomStrength=0 + bootstrap_type=No
to match catboost-mlx's deterministic-greedy contract from DEC-045/046).

Run: python -m benchmarks.upstream.scripts.run_catboost_cpu --dataset adult --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import catboost
from catboost import CatBoostClassifier, CatBoostRanker

from ._runner_common import (
    BENCH_HP,
    BenchResult,
    apply_iterations_override,
    hardware_string,
    load_csv_pair,
    load_xy,
    logloss,
    ndcg_at_k,
    peak_rss_bytes,
    timer,
)

FRAMEWORK = "catboost_cpu"


def _coerce_cats_to_int(X: np.ndarray, cat_indices) -> np.ndarray:
    """CatBoost rejects float64 X when cat_features is non-empty. Build a
    pandas DataFrame where cat cols are int64 (or NaN), numeric cols stay
    float. Pandas DataFrames with mixed dtypes are accepted.
    """
    if not cat_indices:
        return X
    import pandas as pd
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    for j in cat_indices:
        col = df[f"f{j}"]
        # NaN rows in cats become a sentinel string; CatBoost handles missing cats
        df[f"f{j}"] = col.fillna(-1).astype(np.int64)
    return df


def _train_classifier(X_tr, y_tr, X_te, cat_indices, seed,
                      metric_period=0, y_te_eval=None):
    """Train CatBoost classifier; optionally emit per-iter logloss trajectory.

    When metric_period == 1 and y_te_eval is provided, the model is fit with
    eval_set=[(X_te, y_te_eval)] and metric_period=1; the resulting
    `evals_result_["validation"][...]` list is returned as the trajectory
    (Axis C experiment, S44 follow-up). The trajectory adds zero wall-time:
    CatBoost computes per-iter eval logloss internally during the same
    training pass.
    """
    n_classes = int(np.max(y_tr)) + 1 if y_tr.dtype.kind in "iu" else 2
    loss = "MultiClass" if n_classes > 2 else "Logloss"
    X_tr = _coerce_cats_to_int(X_tr, cat_indices)
    X_te = _coerce_cats_to_int(X_te, cat_indices)
    model = CatBoostClassifier(
        iterations=BENCH_HP["iterations"],
        depth=BENCH_HP["depth"],
        learning_rate=BENCH_HP["learning_rate"],
        l2_leaf_reg=BENCH_HP["l2_reg"],
        loss_function=loss,
        random_seed=seed,
        random_strength=BENCH_HP["random_strength"],
        bootstrap_type="No",
        task_type="CPU",
        verbose=False,
        allow_writing_files=False,
        metric_period=1 if metric_period == 1 else 1,
    )
    fit_kwargs = {"cat_features": cat_indices}
    if metric_period == 1 and y_te_eval is not None:
        fit_kwargs["eval_set"] = [(X_te, y_te_eval)]
    with timer() as elapsed_train:
        model.fit(X_tr, y_tr, **fit_kwargs)
    train_seconds = elapsed_train()
    with timer() as elapsed_pred:
        y_proba = model.predict_proba(X_te)
    predict_seconds = elapsed_pred()
    if n_classes == 2:
        # CatBoost returns shape (n, 2); we use prob of class=1 for logloss
        y_proba = y_proba[:, 1]
    trajectory = None
    if metric_period == 1 and y_te_eval is not None:
        evals = getattr(model, "evals_result_", {}) or {}
        validation = evals.get("validation", {}) or {}
        # Binary uses "Logloss"; multiclass uses "MultiClass".
        traj_logloss = validation.get("Logloss") or validation.get("MultiClass")
        if traj_logloss:
            trajectory = {
                "iter": list(range(1, len(traj_logloss) + 1)),
                "logloss": [float(v) for v in traj_logloss],
            }
    return y_proba, train_seconds, predict_seconds, n_classes, trajectory


def _train_ranker(X_tr, y_tr, g_tr, X_te, y_te, g_te, seed):
    # CatBoost ranker requires the data to be sorted by group
    order_tr = np.argsort(g_tr, kind="stable")
    X_tr_s = X_tr[order_tr]; y_tr_s = y_tr[order_tr]; g_tr_s = g_tr[order_tr]
    order_te = np.argsort(g_te, kind="stable")
    X_te_s = X_te[order_te]; y_te_s = y_te[order_te]; g_te_s = g_te[order_te]

    model = CatBoostRanker(
        iterations=BENCH_HP["iterations"],
        depth=BENCH_HP["depth"],
        learning_rate=BENCH_HP["learning_rate"],
        l2_leaf_reg=BENCH_HP["l2_reg"],
        loss_function="YetiRank",
        random_seed=seed,
        random_strength=BENCH_HP["random_strength"],
        bootstrap_type="No",
        task_type="CPU",
        verbose=False,
        allow_writing_files=False,
    )
    with timer() as elapsed_train:
        model.fit(X_tr_s, y_tr_s, group_id=g_tr_s)
    train_seconds = elapsed_train()
    with timer() as elapsed_pred:
        y_score = model.predict(X_te_s)
    predict_seconds = elapsed_pred()
    return y_score, y_te_s, g_te_s, train_seconds, predict_seconds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--results-dir",
        default=str(Path(__file__).resolve().parents[1] / "results"),
    )
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--iterations", type=int, default=None,
                    help="Override BENCH_HP['iterations']; non-default runs are tagged in the output.")
    ap.add_argument("--metric-period", type=int, default=0,
                    help="When set to 1, capture per-iter eval logloss trajectory "
                         "(Axis C experiment). Tags output filename with '_axisC'. "
                         "Zero wall-time overhead; CatBoost computes the trajectory "
                         "internally via metric_period=1 + eval_set.")
    args = ap.parse_args()
    dataset_tag = apply_iterations_override(args)
    if args.metric_period == 1:
        # Insert the _axisC tag right after the dataset stem (before any _iter<N>
        # suffix), so the aggregator can group axisC runs separately.
        if "_iter" in dataset_tag:
            stem, _, rest = dataset_tag.partition("_iter")
            dataset_tag = f"{stem}_axisC_iter{rest}"
        else:
            dataset_tag = f"{dataset_tag}_axisC"

    cache_root = Path(args.cache_root) if args.cache_root else None
    train_csv, test_csv, meta = load_csv_pair(args.dataset, cache_root=cache_root)
    X_tr, y_tr, g_tr = load_xy(train_csv, meta)
    X_te, y_te, g_te = load_xy(test_csv, meta)

    task = meta["task"]
    cat_indices = meta.get("cat_indices", []) or []

    print(f"[catboost_cpu/{args.dataset}/seed={args.seed}] "
          f"train={X_tr.shape}  test={X_te.shape}  task={task}", file=sys.stderr)

    trajectory = None
    if task == "classification":
        y_tr_int = y_tr.astype(np.int64)
        y_te_int = y_te.astype(np.int64)
        y_proba, train_s, pred_s, n_classes, trajectory = _train_classifier(
            X_tr, y_tr_int, X_te, cat_indices, args.seed,
            metric_period=args.metric_period,
            y_te_eval=y_te_int if args.metric_period == 1 else None,
        )
        metric_value = logloss(y_te_int, y_proba)
        metric_name = "logloss"
    elif task == "ranking":
        y_score, y_te_sorted, g_te_sorted, train_s, pred_s = _train_ranker(
            X_tr, y_tr, g_tr, X_te, y_te, g_te, args.seed,
        )
        metric_value = ndcg_at_k(y_te_sorted, y_score, g_te_sorted, k=10)
        metric_name = "ndcg@10"
    else:
        raise ValueError(f"Unsupported task: {task!r}")

    result = BenchResult(
        framework=FRAMEWORK,
        framework_version=catboost.__version__,
        dataset=dataset_tag,
        task=task,
        metric_name=metric_name,
        metric_value=float(metric_value),
        seed=args.seed,
        train_seconds=float(train_s),
        predict_seconds=float(pred_s),
        peak_rss_bytes=peak_rss_bytes(),
        n_train=int(X_tr.shape[0]),
        n_test=int(X_te.shape[0]),
        n_features=int(X_tr.shape[1]),
        cat_indices=cat_indices,
        hyperparameters=dict(BENCH_HP),
        hardware=hardware_string(),
        python_version=sys.version.split()[0],
        notes=meta.get("notes", ""),
        trajectory=trajectory,
    )
    out = result.write(Path(args.results_dir))
    print(f"  {metric_name}={metric_value:.6f}  "
          f"train={train_s:.2f}s  predict={pred_s:.3f}s  "
          f"rss={peak_rss_bytes()/1e6:.0f} MB  -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
