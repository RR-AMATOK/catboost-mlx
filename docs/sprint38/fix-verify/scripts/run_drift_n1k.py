#!/usr/bin/env python3
"""
Sprint-38 fix-verify: N=1k seed=42..46 drift sweep.

Runs MLX csv_train (production binary — no probe flags) vs CPU CatBoost
at the PROBE-G anchor parameters: ST/Cosine/RMSE, N=1000, 5 seeds, 50 iters.

Purpose: confirm whether the rebuild from current source reduces the
13.96% aggregate drift observed with the pre-rebuild binary.

Output: docs/sprint38/fix-verify/data/drift_n1k.csv
Schema: seed,mlx_rmse,cpu_rmse,drift_pct
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import catboost

REPO_ROOT = Path(__file__).resolve().parents[4]
FIX_VERIFY_DATA = REPO_ROOT / "docs" / "sprint38" / "fix-verify" / "data"
BINARY = REPO_ROOT / "csv_train"

# Match PROBE-G Phase 2 parameters exactly
N = 1_000
FEATS = 20
SEEDS = [42, 43, 44, 45, 46]
ITERS = 50
DEPTH = 6
BINS = 128
LR = 0.03
L2 = 3.0
SCORE_FN = "Cosine"
GROW = "SymmetricTree"
LOSS = "rmse"


def make_data(n: int, seed: int):
    """Canonical anchor: matches PROBE-G make_anchor."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, FEATS)).astype("float32")
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(n) * 0.1).astype("float32")
    return X, y


def run_mlx(csv_path: Path, seed: int) -> float:
    """Run MLX csv_train, return final RMSE."""
    cmd = [
        str(BINARY), str(csv_path),
        "--iterations", str(ITERS),
        "--depth", str(DEPTH),
        "--lr", str(LR),
        "--bins", str(BINS),
        "--l2", str(L2),
        "--loss", LOSS,
        "--score-function", SCORE_FN,
        "--seed", str(seed),
        "--verbose",
    ]
    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/mlx/lib"
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"MLX csv_train failed (seed={seed}):\n{result.stderr[:500]}")
    # Parse final loss from stdout — last "iter=N ... loss=X.XXXXXX" line
    final_rmse = None
    for line in result.stdout.splitlines():
        if line.startswith("iter=") and "loss=" in line:
            for tok in line.split():
                if tok.startswith("loss="):
                    final_rmse = float(tok.split("=", 1)[1])
    if final_rmse is None:
        raise RuntimeError(f"Could not parse final RMSE from MLX output (seed={seed})")
    return final_rmse


def run_cpu(X, y, seed: int) -> float:
    """Run CatBoost CPU, return final RMSE on training set."""
    model = catboost.CatBoostRegressor(
        iterations=ITERS,
        depth=DEPTH,
        learning_rate=LR,
        l2_leaf_reg=L2,
        border_count=BINS,
        grow_policy=GROW,
        score_function=SCORE_FN,
        loss_function="RMSE",
        random_seed=seed,
        random_strength=0.0,
        bootstrap_type="No",
        verbose=False,
        thread_count=1,
    )
    feature_names = [f"f{i}" for i in range(FEATS)]
    pool = catboost.Pool(X, label=y, feature_names=feature_names)
    model.fit(pool)
    preds = model.predict(X)
    rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
    return rmse


def main():
    if not BINARY.exists():
        print(f"ERROR: {BINARY} not found.", file=sys.stderr)
        sys.exit(1)

    FIX_VERIFY_DATA.mkdir(parents=True, exist_ok=True)
    out_path = FIX_VERIFY_DATA / "drift_n1k.csv"

    rows = []
    print(f"Binary: {BINARY.name}")
    print(f"Sweep: N={N}, seeds={SEEDS}, iters={ITERS}, depth={DEPTH}")
    print(f"Config: {GROW}/{SCORE_FN}/{LOSS.upper()}, bins={BINS}, lr={LR}, l2={L2}")
    print()

    for seed in SEEDS:
        X, y = make_data(N, seed)

        # Write anchor CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tf:
            tmp_path = Path(tf.name)
            w = csv.writer(tf)
            w.writerow([f"f{i}" for i in range(FEATS)] + ["target"])
            for i in range(len(y)):
                w.writerow(list(map(float, X[i])) + [float(y[i])])

        try:
            mlx_rmse = run_mlx(tmp_path, seed)
            cpu_rmse = run_cpu(X, y, seed)
            drift_pct = abs(mlx_rmse - cpu_rmse) / cpu_rmse * 100.0
            row = {
                "seed": seed,
                "mlx_rmse": mlx_rmse,
                "cpu_rmse": cpu_rmse,
                "drift_pct": drift_pct,
            }
            rows.append(row)
            print(f"  seed={seed}: mlx_rmse={mlx_rmse:.6f}  cpu_rmse={cpu_rmse:.6f}  drift={drift_pct:.2f}%")
        finally:
            tmp_path.unlink(missing_ok=True)

    # Write CSV
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "mlx_rmse", "cpu_rmse", "drift_pct"])
        w.writeheader()
        w.writerows(rows)

    mean_drift = float(np.mean([r["drift_pct"] for r in rows]))
    print()
    print(f"Aggregate drift (mean): {mean_drift:.2f}%")
    print(f"Drift threshold:        2.00%")
    verdict = "PASS" if mean_drift < 2.0 else "FAIL"
    print(f"Gate verdict:           {verdict}")
    print(f"Output: {out_path}")

    return mean_drift


if __name__ == "__main__":
    main()
