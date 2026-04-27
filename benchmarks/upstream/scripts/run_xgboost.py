"""XGBoost-CPU (hist) runner for the upstream benchmark suite.

Loads a prepared dataset, trains XGBoost with `tree_method='hist'` (the CPU
default in 2.x and the closest analog to LightGBM's histogram path and
CatBoost's GPU histogram), predicts on the test split, computes the
canonical metric, and writes a results JSON.

XGBoost categorical handling caveat: pandas 3.x + numpy 2.x +
xgboost 3.x have a known interaction issue where the `enable_categorical`
DMatrix path fails inside numpy's vectorize. To keep this runner stable
across the Apple Silicon Python stack we feed cat columns as numeric
(ordinal) values, the same treatment XGBoost gave them before native
category support landed in 1.5. This is the canonical XGBoost-CPU-hist
configuration in most published benchmarks; the "tuned" upstream numbers
relied on heavy preprocessing (target-encoding etc.) which is out of
scope for a head-to-head sprint. Document this caveat in the writeup.

Run: python -m benchmarks.upstream.scripts.run_xgboost --dataset adult --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

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

FRAMEWORK = "xgboost"


def _train_classifier(X_tr, y_tr, X_te, y_te, cat_indices, seed):
    n_classes = int(np.max(y_tr)) + 1 if y_tr.dtype.kind in "iu" else 2

    if n_classes > 2:
        params = {
            "objective": "multi:softprob",
            "num_class": n_classes,
            "eval_metric": "mlogloss",
        }
    else:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        }

    params.update({
        "tree_method": "hist",
        "max_depth": BENCH_HP["depth"],
        "learning_rate": BENCH_HP["learning_rate"],
        "lambda": BENCH_HP["l2_reg"],
        "seed": seed,
        "verbosity": 0,
    })

    # Pass numeric arrays directly. cat_indices are treated as ordinal codes
    # (XGBoost's pre-1.5 default behavior; see module docstring).
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    with timer() as elapsed_train:
        booster = xgb.train(params, dtrain, num_boost_round=BENCH_HP["iterations"])
    train_seconds = elapsed_train()

    dtest = xgb.DMatrix(X_te)
    with timer() as elapsed_pred:
        y_proba = booster.predict(dtest)
    predict_seconds = elapsed_pred()
    return y_proba, train_seconds, predict_seconds, n_classes


def _train_ranker(X_tr, y_tr, g_tr, X_te, y_te, g_te, seed):
    def group_sizes(qid):
        _, counts = np.unique(qid, return_counts=True)
        return counts

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dtrain.set_group(group_sizes(g_tr))
    dtest = xgb.DMatrix(X_te)

    params = {
        "objective": "rank:ndcg",
        "tree_method": "hist",
        "max_depth": BENCH_HP["depth"],
        "learning_rate": BENCH_HP["learning_rate"],
        "lambda": BENCH_HP["l2_reg"],
        "seed": seed,
        "verbosity": 0,
        "eval_metric": "ndcg@10",
    }
    with timer() as elapsed_train:
        booster = xgb.train(params, dtrain, num_boost_round=BENCH_HP["iterations"])
    train_seconds = elapsed_train()
    with timer() as elapsed_pred:
        y_score = booster.predict(dtest)
    predict_seconds = elapsed_pred()
    return y_score, train_seconds, predict_seconds


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

    print(f"[xgboost/{args.dataset}/seed={args.seed}] "
          f"train={X_tr.shape}  test={X_te.shape}  task={task}", file=sys.stderr)

    if task == "classification":
        y_tr_int = y_tr.astype(np.int64)
        y_te_int = y_te.astype(np.int64)
        y_proba, train_s, pred_s, n_classes = _train_classifier(
            X_tr, y_tr_int, X_te, y_te_int, cat_indices, args.seed,
        )
        metric_value = logloss(y_te_int, y_proba)
        metric_name = "logloss"
    elif task == "ranking":
        y_score, train_s, pred_s = _train_ranker(
            X_tr, y_tr, g_tr, X_te, y_te, g_te, args.seed,
        )
        metric_value = ndcg_at_k(y_te, y_score, g_te, k=10)
        metric_name = "ndcg@10"
    else:
        raise ValueError(f"Unsupported task: {task!r}")

    result = BenchResult(
        framework=FRAMEWORK,
        framework_version=xgb.__version__,
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
