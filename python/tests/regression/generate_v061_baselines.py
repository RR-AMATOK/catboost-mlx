"""
generate_v061_baselines.py — One-time baseline generator for v0.6.1 predict-output gate.

Run ONCE on master at v0.6.1 to produce v0.6.1_predict_baselines.pkl.
That pickle is checked in and never regenerated unless the v0.6.1 reference point
must be re-established (e.g. intentional bit-breaking optimization with a new
version tag).

Usage
-----
    cd <repo-root>
    python python/tests/regression/generate_v061_baselines.py

Output
------
    python/tests/regression/v0.6.1_predict_baselines.pkl

Design note: model stored, not just predictions
-------------------------------------------------
MLX Metal training is non-deterministic across separate Python process invocations
(atomic float operations in the histogram accumulation kernel do not guarantee
identical thread-block ordering across runs). Two separate calls to fit() with
identical seed, data, and hyperparameters will produce float32 model weights that
differ by ~1-3 ULP. This means we cannot compare re-trained predictions to a
pickled baseline.

The solution: store the trained model JSON (model._model_data) in the baseline
pickle alongside the reference predictions. The test then loads the stored model
and calls predict_proba on the same test set. This eliminates training non-
determinism entirely. The gate tests exactly what is claimed: that _predict_inprocess
produces byte-identical float32 output given a fixed model.

Regeneration policy
-------------------
This script MUST be run on master at commit d3bc0e1d02 (v0.6.1 release merge)
or any equivalent tagged v0.6.1 state, with the following exact configuration:

    Hyperparameters : BENCH_HP from benchmarks/upstream/scripts/_runner_common.py
                      iterations=200, depth=6, learning_rate=0.1, l2_reg=3.0,
                      random_strength=0.0, bootstrap_type='no'
    Seed            : 42
    Higgs scope     : Full 1,000,000-row train / 100,000-row test
    Epsilon scope   : 50,000-row train subset (rows 0..49999) /
                      10,000-row test subset (rows 0..9999)
                      -- see EPSILON_TRAIN_ROWS / EPSILON_TEST_ROWS constants below

The Epsilon subset is intentional: full Epsilon iter=200 takes ~470s on M3 Max MLX,
exceeding a 5-minute CI budget. A 50k-train / 10k-test subset at iter=200 takes
~60-80s and exercises the same predict codepath. Any bit divergence in
_predict_inprocess will flip this gate regardless of dataset scale.

To regenerate after a version bump that intentionally changes predict output:
    1. Bump VERSION and COMMIT below to the new tag.
    2. Run this script on the new master tip.
    3. Commit the updated pickle alongside the version-bump commit.
    4. Update the test docstring in test_branch_b_regression.py accordingly.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

# ── Paths + sys.path ─────────────────────────────────────────────────────────

_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parents[2]        # python/tests/regression -> repo root
BENCH_SCRIPTS = REPO_ROOT / "benchmarks" / "upstream" / "scripts"
OUT_PATH = _THIS_DIR / "v0.6.1_predict_baselines.pkl"

sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT))

# ── Constants ────────────────────────────────────────────────────────────────

VERSION = "0.6.1"
COMMIT = "d3bc0e1d02"   # master v0.6.1 release merge

SEED = 42

# Epsilon subset sizes -- keep CI wall-clock under 5 min total.
# Full Epsilon: 400k train / 100k test. Subset: 50k train / 10k test.
EPSILON_TRAIN_ROWS = 50_000
EPSILON_TEST_ROWS = 10_000


def _load_runner_common():
    """Load _runner_common, registering in sys.modules so @dataclass works."""
    import importlib.util
    _MOD_NAME = "benchmarks.upstream.scripts._runner_common"
    if _MOD_NAME in sys.modules:
        return sys.modules[_MOD_NAME]
    mod_path = str(BENCH_SCRIPTS / "_runner_common.py")
    spec = importlib.util.spec_from_file_location(_MOD_NAME, mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MOD_NAME] = mod        # register BEFORE exec_module
    spec.loader.exec_module(mod)
    return mod


def _check_catboost_mlx():
    try:
        from catboost_mlx import CatBoostMLXClassifier  # noqa: F401
        return True
    except ImportError:
        return False


def _train(X_train, y_train, seed, hp):
    """Train CatBoostMLXClassifier and return the fitted model."""
    from catboost_mlx import CatBoostMLXClassifier
    model = CatBoostMLXClassifier(
        iterations=hp["iterations"],
        depth=hp["depth"],
        learning_rate=hp["learning_rate"],
        l2_reg_lambda=hp["l2_reg"],
        loss="logloss",
        cat_features=[],
        random_seed=seed,
        random_strength=hp["random_strength"],
        bootstrap_type=hp["bootstrap_type"],
        verbose=True,
    )
    model.fit(X_train, y_train.astype(np.int64))
    return model


def main():
    if not _check_catboost_mlx():
        print("ERROR: catboost_mlx not importable. Install the wheel first.", file=sys.stderr)
        sys.exit(1)

    rc = _load_runner_common()
    hp = dict(rc.BENCH_HP)
    print(f"Hyperparameters: {hp}")
    print(f"Seed: {SEED}")
    print(f"Epsilon subset: {EPSILON_TRAIN_ROWS} train rows / {EPSILON_TEST_ROWS} test rows")

    # ── Higgs-1M ────────────────────────────────────────────────────────────
    print("\n=== Higgs-1M (full 1M train / 100k test) ===")
    train_csv, test_csv, meta = rc.load_csv_pair("higgs")
    X_tr, y_tr, _ = rc.load_xy(train_csv, meta)
    X_te, y_te, _ = rc.load_xy(test_csv, meta)
    X_tr = X_tr.astype(np.float32)
    X_te = X_te.astype(np.float32)
    print(f"  Train shape: {X_tr.shape}, Test shape: {X_te.shape}")

    higgs_model = _train(X_tr, y_tr, SEED, hp)
    higgs_pred = higgs_model.predict_proba(X_te)[:, 1].astype(np.float32)
    higgs_model_data = higgs_model._model_data   # JSON-serializable dict
    print(f"  Higgs predict_proba shape: {higgs_pred.shape}, dtype: {higgs_pred.dtype}")
    print(f"  Sample [0..4]: {higgs_pred[:5]}")

    # ── Epsilon subset ───────────────────────────────────────────────────────
    print(f"\n=== Epsilon subset ({EPSILON_TRAIN_ROWS} train / {EPSILON_TEST_ROWS} test) ===")
    eps_train_csv, eps_test_csv, eps_meta = rc.load_csv_pair("epsilon")
    X_eps_tr_full, y_eps_tr_full, _ = rc.load_xy(eps_train_csv, eps_meta)
    X_eps_te_full, y_eps_te_full, _ = rc.load_xy(eps_test_csv, eps_meta)

    eps_train_idx = list(range(EPSILON_TRAIN_ROWS))
    eps_test_idx = list(range(EPSILON_TEST_ROWS))

    X_eps_tr = X_eps_tr_full[eps_train_idx].astype(np.float32)
    y_eps_tr = y_eps_tr_full[eps_train_idx]
    X_eps_te = X_eps_te_full[eps_test_idx].astype(np.float32)
    print(f"  Train subset shape: {X_eps_tr.shape}, Test subset shape: {X_eps_te.shape}")

    eps_model = _train(X_eps_tr, y_eps_tr, SEED, hp)
    eps_pred = eps_model.predict_proba(X_eps_te)[:, 1].astype(np.float32)
    eps_model_data = eps_model._model_data
    print(f"  Epsilon subset predict_proba shape: {eps_pred.shape}, dtype: {eps_pred.dtype}")
    print(f"  Sample [0..4]: {eps_pred[:5]}")

    # ── Serialize ────────────────────────────────────────────────────────────
    # Store the model JSON alongside the reference predictions.
    # The test loads the stored model and calls predict_proba -- this avoids
    # training non-determinism (Metal GPU atomic ops are not order-guaranteed
    # across process invocations; re-training produces ~1-3 ULP weight drift).
    baseline = {
        "version": VERSION,
        "commit": COMMIT,
        # Reference predictions (float32, shape=(n_test,))
        "higgs_1m_iter200_seed42_predict_proba": higgs_pred,
        "epsilon_iter200_seed42_subset_predict_proba": eps_pred,
        # Stored model weights (JSON-serializable dicts) for the test to load
        "higgs_1m_model_data": higgs_model_data,
        "epsilon_subset_model_data": eps_model_data,
        # Index contract for epsilon subset
        "test_indices_used": {
            "epsilon_subset": eps_test_idx,
        },
        "hyperparameters": hp,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(baseline, f, protocol=4)

    print(f"\nBaseline written to: {OUT_PATH}")
    print(f"  higgs array  : shape={higgs_pred.shape}, dtype={higgs_pred.dtype}")
    print(f"  epsilon array: shape={eps_pred.shape},  dtype={eps_pred.dtype}")
    print(f"  pickle size  : {OUT_PATH.stat().st_size / 1024:.1f} KB")
    print("\nNext step: commit this pickle alongside any intentional version bump.")
    print("\nVerification: the test loads the stored model_data and calls predict_proba.")
    print("Training is not re-run during the test. Only _predict_inprocess is tested.")


if __name__ == "__main__":
    main()
