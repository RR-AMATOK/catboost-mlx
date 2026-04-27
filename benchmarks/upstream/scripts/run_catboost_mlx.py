"""CatBoost-MLX runner for the upstream benchmark suite.

Trains catboost-mlx (Apple Silicon Metal/MLX backend) on the same hyperparameters
the other 3 runners use. Same as run_catboost_cpu.py except the framework
identity, the model class, and one cat-coercion detail (MLX's validator
accepts numeric cat-coded ints as well; we still coerce for consistency).

Run: python -m benchmarks.upstream.scripts.run_catboost_mlx --dataset adult --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import catboost_mlx
from catboost_mlx import CatBoostMLXClassifier, CatBoostMLX

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

FRAMEWORK = "catboost_mlx"


def _coerce_cats_to_int(X: np.ndarray, cat_indices):
    """Same shape as run_catboost_cpu's _coerce; returns numpy array with cats
    as int64 columns and numerics as float. catboost-mlx accepts pandas DFs
    with mixed dtypes via Pool, but ndarray with int+float columns isn't
    directly representable. Build a pandas DF for cleanliness.
    """
    if not cat_indices:
        return X
    import pandas as pd
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    for j in cat_indices:
        col = df[f"f{j}"]
        df[f"f{j}"] = col.fillna(-1).astype(np.int64)
    return df


def _train_classifier(X_tr, y_tr, X_te, cat_indices, seed):
    n_classes = int(np.max(y_tr)) + 1 if y_tr.dtype.kind in "iu" else 2
    loss = "multiclass" if n_classes > 2 else "logloss"
    X_tr = _coerce_cats_to_int(X_tr, cat_indices)
    X_te = _coerce_cats_to_int(X_te, cat_indices)

    model = CatBoostMLXClassifier(
        iterations=BENCH_HP["iterations"],
        depth=BENCH_HP["depth"],
        learning_rate=BENCH_HP["learning_rate"],
        l2_reg_lambda=BENCH_HP["l2_reg"],
        loss=loss,
        cat_features=cat_indices,
        random_seed=seed,
        random_strength=BENCH_HP["random_strength"],
        bootstrap_type=BENCH_HP["bootstrap_type"],
        verbose=False,
    )
    with timer() as elapsed_train:
        model.fit(X_tr, y_tr)
    train_seconds = elapsed_train()
    with timer() as elapsed_pred:
        y_proba = model.predict_proba(X_te)
    predict_seconds = elapsed_pred()
    if n_classes == 2:
        y_proba = y_proba[:, 1]
    return y_proba, train_seconds, predict_seconds, n_classes


def _train_ranker(X_tr, y_tr, g_tr, X_te, y_te, g_te, seed):
    # CatBoost-MLX's CatBoostMLX class with loss="yetirank" handles ranking;
    # data must be sorted by group_id.
    order_tr = np.argsort(g_tr, kind="stable")
    X_tr_s = X_tr[order_tr]; y_tr_s = y_tr[order_tr]; g_tr_s = g_tr[order_tr]
    order_te = np.argsort(g_te, kind="stable")
    X_te_s = X_te[order_te]; y_te_s = y_te[order_te]; g_te_s = g_te[order_te]

    model = CatBoostMLX(
        iterations=BENCH_HP["iterations"],
        depth=BENCH_HP["depth"],
        learning_rate=BENCH_HP["learning_rate"],
        l2_reg_lambda=BENCH_HP["l2_reg"],
        loss="yetirank",
        random_seed=seed,
        random_strength=BENCH_HP["random_strength"],
        bootstrap_type=BENCH_HP["bootstrap_type"],
        verbose=False,
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
    args = ap.parse_args()
    dataset_tag = apply_iterations_override(args)

    cache_root = Path(args.cache_root) if args.cache_root else None
    train_csv, test_csv, meta = load_csv_pair(args.dataset, cache_root=cache_root)
    X_tr, y_tr, g_tr = load_xy(train_csv, meta)
    X_te, y_te, g_te = load_xy(test_csv, meta)

    task = meta["task"]
    cat_indices = meta.get("cat_indices", []) or []

    print(f"[catboost_mlx/{args.dataset}/seed={args.seed}] "
          f"train={X_tr.shape}  test={X_te.shape}  task={task}", file=sys.stderr)

    if task == "classification":
        y_tr_int = y_tr.astype(np.int64)
        y_te_int = y_te.astype(np.int64)
        y_proba, train_s, pred_s, n_classes = _train_classifier(
            X_tr, y_tr_int, X_te, cat_indices, args.seed,
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
        framework_version=getattr(catboost_mlx, "__version__", "<unset>"),
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
    )
    out = result.write(Path(args.results_dir))
    print(f"  {metric_name}={metric_value:.6f}  "
          f"train={train_s:.2f}s  predict={pred_s:.3f}s  "
          f"rss={peak_rss_bytes()/1e6:.0f} MB  -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
