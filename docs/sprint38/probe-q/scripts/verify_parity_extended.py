"""Extended RS=1.0 parity verification at N=1k, 10 seeds.

Extends verify_parity.py (5 seeds, S38) to 10 seeds [42..51] at RS=1.0 to
determine whether the mean drift -3.43% +/- 1.06% from the original 5-seed
run is a real bounded RNG-implementation bias or sample noise.

RS=0.0 verification is NOT extended (already proven 0.000% at 5 seeds; further
extension is not informative).

Verdict rule:
  - 95% CI excludes 0 -> real bounded RNG-implementation bias
  - 95% CI includes 0 -> sample noise in the 5-seed run

Output: docs/sprint38/probe-q/data/parity_verification_rs1_extended.csv
Schema: seed, mlx_rmse, cpu_rmse, drift_pct

Usage:
    python docs/sprint38/probe-q/scripts/verify_parity_extended.py
"""
import csv
import math
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
OUT_CSV = REPO / "docs/sprint38/probe-q/data/parity_verification_rs1_extended.csv"

ENV = os.environ.copy()
ENV["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/mlx/lib"

SEEDS = list(range(42, 52))  # 10 seeds: 42..51
RS = 1.0


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
    if os.path.exists(out):
        os.unlink(out)
    if final is None:
        raise RuntimeError(f"No loss in stdout for seed={seed}: {r.stdout[:500]}")
    return final


def run_cpu(X: np.ndarray, y: np.ndarray, seed: int, rs: float) -> float:
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

    print(f"Extended RS={RS} parity verification: {len(SEEDS)} seeds {SEEDS[0]}..{SEEDS[-1]}")
    print(f"{'seed':>5} | {'mlx_rmse':>10} {'cpu_rmse':>10} {'drift_pct':>10}")
    print("-" * 45)

    rows = []
    drifts = []
    for seed in SEEDS:
        mlx_rmse = run_mlx(seed, RS)
        cpu_rmse = run_cpu(X, y, seed, RS)
        drift = (mlx_rmse - cpu_rmse) / cpu_rmse * 100.0
        drifts.append(drift)
        rows.append({"seed": seed, "mlx_rmse": mlx_rmse, "cpu_rmse": cpu_rmse, "drift_pct": drift})
        print(f"{seed:>5} | {mlx_rmse:>10.6f} {cpu_rmse:>10.6f} {drift:>9.3f}%")

    print("-" * 45)

    n = len(drifts)
    mean_d = float(np.mean(drifts))
    std_d = float(np.std(drifts, ddof=1))
    se = std_d / math.sqrt(n)
    ci_lo = mean_d - 2 * se
    ci_hi = mean_d + 2 * se

    print(f"\nStatistics (N={n}):")
    print(f"  mean drift   = {mean_d:+.4f}%")
    print(f"  std          = {std_d:.4f}%")
    print(f"  SE (std/sqrt(N)) = {se:.4f}%")
    print(f"  95% CI       = [{ci_lo:+.4f}%, {ci_hi:+.4f}%]")
    print()

    if ci_lo > 0 or ci_hi < 0:
        verdict = "REAL BOUNDED BIAS: 95% CI excludes 0 -- RNG-implementation difference is real"
    else:
        verdict = "SAMPLE NOISE: 95% CI includes 0 -- 5-seed mean was sample noise"
    print(f"Verdict: {verdict}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "mlx_rmse", "cpu_rmse", "drift_pct"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
