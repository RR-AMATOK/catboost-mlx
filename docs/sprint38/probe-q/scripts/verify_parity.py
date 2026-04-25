"""Multi-seed RandomStrength parity verification at N=1k.

Confirms DEC-045 (S38 RESOLVED — harness config asymmetry was the cause):
  - At RS=0 on both runtimes: bit-identical RMSE across all seeds.
  - At RS=1.0 on both runtimes: mean drift ~3% (RNG-implementation difference),
    bounded and not algorithmic.
  - The 13.93% N=1k drift S38 chased was the asymmetric case
    (MLX RS=1.0 vs CPU RS=0.0) — phantom.

Usage:
    python docs/sprint38/probe-q/scripts/verify_parity.py
"""
import os
import subprocess
import tempfile
from pathlib import Path

import catboost
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
BINARY = REPO / "csv_train"
ANCHOR = REPO / "docs/sprint38/f2/data/anchor_n1000_seed42.csv"

ENV = os.environ.copy()
ENV["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/mlx/lib"


def run_mlx(seed: int, rs: float) -> float:
    out = tempfile.mktemp(suffix=".json")
    cmd = [
        str(BINARY), str(ANCHOR),
        "--iterations", "50", "--depth", "6", "--lr", "0.03",
        "--bins", "128", "--l2", "3", "--loss", "rmse",
        "--score-function", "Cosine", "--seed", str(seed),
        "--random-strength", str(rs), "--output", out,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV)
    final = None
    for line in r.stdout.splitlines():
        if line.startswith("iter="):
            for p in line.split():
                if p.startswith("loss="):
                    final = float(p.split("=")[1])
    os.unlink(out)
    return final


def run_cpu(X, y, seed: int, rs: float) -> float:
    m = catboost.CatBoostRegressor(
        iterations=50, depth=6, learning_rate=0.03, l2_leaf_reg=3,
        border_count=128, grow_policy="SymmetricTree",
        score_function="Cosine", loss_function="RMSE",
        random_seed=seed, random_strength=rs,
        bootstrap_type="No", verbose=False,
    )
    m.fit(X, y)
    pred = m.predict(X)
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def main() -> None:
    df = pd.read_csv(ANCHOR)
    X = df.iloc[:, :-1].values.astype("float32")
    y = df.iloc[:, -1].values.astype("float32")

    print(f"{'seed':>5} | RS=0 (parity):              | RS=1 (parity):")
    print(f"{'':>5} | {'mlx':>10} {'cpu':>10} {'drift_%':>8} | {'mlx':>10} {'cpu':>10} {'drift_%':>8}")
    print("-" * 82)
    rs0, rs1 = [], []
    for seed in [42, 43, 44, 45, 46]:
        m0 = run_mlx(seed, 0.0); c0 = run_cpu(X, y, seed, 0.0)
        m1 = run_mlx(seed, 1.0); c1 = run_cpu(X, y, seed, 1.0)
        d0 = (m0 - c0) / c0 * 100
        d1 = (m1 - c1) / c1 * 100
        rs0.append(d0); rs1.append(d1)
        print(f"{seed:>5} | {m0:>10.6f} {c0:>10.6f} {d0:>7.3f}% | {m1:>10.6f} {c1:>10.6f} {d1:>7.3f}%")
    print("-" * 82)
    print(f"{'mean':>5} | {' ':>22} {np.mean(rs0):>7.3f}% | {' ':>22} {np.mean(rs1):>7.3f}%")
    print(f"{'std':>5} | {' ':>22} {np.std(rs0):>7.3f}% | {' ':>22} {np.std(rs1):>7.3f}%")


if __name__ == "__main__":
    main()
