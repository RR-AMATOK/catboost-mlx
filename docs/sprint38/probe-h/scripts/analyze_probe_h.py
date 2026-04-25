#!/usr/bin/env python3
"""
S38-PROBE-H: Per-side formula divergence analysis.

Applies both MLX's actual formula and CPU CatBoost's CalcScoreOnSide formula
to PROBE-H tuples to localise the argmax divergence at iter=1.

Formulas under test
-------------------
CPU (short_vector_ops.h UpdateScoreBinKernelPlain, generic path)
    For each partition p:
        if wL > 0: cosNum += sL^2 / (wL + lambda)
                   cosDen += sL^2 * wL / (wL + lambda)^2
        if wR > 0: cosNum += sR^2 / (wR + lambda)
                   cosDen += sR^2 * wR / (wR + lambda)^2
    gain = cosNum / sqrt(cosDen)
    Threshold: w > 0 (CalcAverage: count > 0 ? 1/(count+lambda) : 0).
    Note: at iter=1 with integer doc counts, w > 0  ≡  w > 1e-15.

MLX (csv_train.cpp ordinal Cosine, pre-DEC-042 / current probe binary)
    For each partition p:
        if wL < 1e-15f OR wR < 1e-15f: SKIP (contribute 0 to cosNum/cosDen)
        else: cosNum += sL^2/(wL+lambda) + sR^2/(wR+lambda)
              cosDen += sL^2*wL/(wL+lambda)^2 + sR^2*wR/(wR+lambda)^2
    gain = cosNum / sqrt(cosDen)
    Called the OLD JOINT-SKIP formula.

PROBE-H column semantics
------------------------
cosNumL, cosDenL  -- left-side contribution under CPU formula (per-side mask, w > 1e-15)
cosNumR, cosDenR  -- right-side contribution under CPU formula
cosNumTotal       = cosNumL + cosNumR  (equals cpu_termNum from PROBE-E)
cosDenTotal       = cosDenL + cosDenR
gain_mlx          -- Cosine gain from COSINE_RESIDUAL_INSTRUMENT (per-side mask path)
                     Verified: gain_mlx = cosNumTotal_sum / sqrt(cosDenTotal_sum + 1e-20)
picked_by_mlx     -- 1 for the (feat,bin) the training binary selected as argmax

PROBE-E column semantics (cos_leaf_seed42_depth{d}.csv)
---------------------------------------------------------
mlx_termNum, mlx_termDen -- old joint-skip per-partition contribution (0 when either side empty)
cpu_termNum, cpu_termDen -- per-side mask per-partition contribution (matches cosNumTotal)

Usage
-----
    python docs/sprint38/probe-h/scripts/analyze_probe_h.py

Outputs
-------
    docs/sprint38/probe-h/data/divergence_iter1.csv
    Prints per-depth tables and FINDING summary to stdout.

Reproducibility
---------------
No randomness; all operations are deterministic pandas/numpy aggregations.
Seeds: N/A (no stochastic operations).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
PROBE_H_DATA = REPO_ROOT / "docs" / "sprint38" / "probe-h" / "data"
F2_DATA = REPO_ROOT / "docs" / "sprint38" / "f2" / "data"

L2_LAMBDA = 3.0     # l2_leaf_reg used in the anchor run
EPSILON_GUARD = 1e-20   # cosDen_d guard in MLX (mirrors float guard)

# Sanity check tolerance for gain_cpu == gain_mlx_captured (both per-side mask)
SANITY_TOL = 1e-5

# CPU iter=1 picks from cpu_model.json (tree[0]).
# MLX iter=1 picks from run_probe_h.py / F2 mlx_model.json (trees[0]).
# These establish the ground-truth split choices for each depth.
CPU_ITER1_PICKS = {
    0: {"feat": 0, "border": 0.10254748165607452},
    1: {"feat": 1, "border": 0.43849730491638184},
    2: {"feat": 0, "border": -0.8107502460479736},
    3: {"feat": 0, "border": 1.0352909564971924},
    4: {"feat": 1, "border": -0.8001531362533569},
    5: {"feat": 0, "border": 1.7465853691101074},
}

# MLX iter=1 picks (confirmed via picked_by_mlx==1 in PROBE-H output)
MLX_ITER1_PICKS = {
    0: {"feat": 0, "bin": 69},
    1: {"feat": 1, "bin": 64},
    2: {"feat": 0, "bin": 29},
    3: {"feat": 15, "bin": 28},
    4: {"feat": 1, "bin": 23},
    5: {"feat": 0, "bin": 98},
}


def load_probe_h(depth: int) -> pd.DataFrame:
    path = PROBE_H_DATA / f"probe_h_iter1_depth{depth}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing PROBE-H file: {path}")
    return pd.read_csv(path)


def load_probe_e(depth: int) -> pd.DataFrame:
    path = PROBE_H_DATA / f"cos_leaf_seed42_depth{depth}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing PROBE-E file: {path}")
    return pd.read_csv(path)


def compute_gains_both_formulas(
    probe_h: pd.DataFrame, probe_e: pd.DataFrame
) -> pd.DataFrame:
    """
    For each (feat, bin) candidate, compute:
      gain_cpu        -- CPU per-side formula: sum(cosNumTotal) / sqrt(sum(cosDenTotal))
      gain_mlx_actual -- OLD joint-skip formula: sum(mlxTermNum) / sqrt(sum(mlxTermDen))
      gain_mlx_col    -- gain_mlx column from PROBE-H (per-side mask capture from instrument)

    Returns merged DataFrame indexed by (feat, bin).
    """
    # CPU formula: aggregate cosNumTotal, cosDenTotal per (feat, bin)
    cpu_agg = probe_h.groupby(["feat", "bin"], sort=False).agg(
        cosNum_cpu=("cosNumTotal", "sum"),
        cosDen_cpu=("cosDenTotal", "sum"),
        gain_mlx_col=("gain_mlx", "first"),
        picked=("picked_by_mlx", "max"),
    ).reset_index()
    cpu_agg["gain_cpu"] = cpu_agg["cosNum_cpu"] / np.sqrt(cpu_agg["cosDen_cpu"] + EPSILON_GUARD)

    # Old joint-skip formula: aggregate mlxTermNum, mlxTermDen per (feat, bin)
    mlx_agg = probe_e.groupby(["featIdx", "bin"], sort=False).agg(
        cosNum_mlx=("mlx_termNum", "sum"),
        cosDen_mlx=("mlx_termDen", "sum"),
    ).reset_index().rename(columns={"featIdx": "feat"})
    mlx_agg["gain_mlx_formula"] = mlx_agg["cosNum_mlx"] / np.sqrt(
        mlx_agg["cosDen_mlx"] + EPSILON_GUARD
    )

    merged = cpu_agg.merge(mlx_agg, on=["feat", "bin"], how="left")
    return merged


def top5_by_gain(df: pd.DataFrame, gain_col: str) -> pd.DataFrame:
    return df.nlargest(5, gain_col)[["feat", "bin", gain_col]].reset_index(drop=True)


def run_analysis() -> None:
    print("=" * 72)
    print("S38-PROBE-H: Formula divergence analysis — iter=1, N=1k seed=42")
    print("=" * 72)
    print()

    rows = []
    sanity_ok = True

    for d in range(6):
        probe_h = load_probe_h(d)
        probe_e = load_probe_e(d)
        df = compute_gains_both_formulas(probe_h, probe_e)

        # --- Sanity check: gain_cpu (per-side mask) must match gain_mlx_col to within SANITY_TOL
        residual = (df["gain_cpu"] - df["gain_mlx_col"]).abs().max()
        if residual > SANITY_TOL:
            print(
                f"[FAIL] SANITY CHECK depth={d}: "
                f"max |gain_cpu - gain_mlx_col| = {residual:.3e} > {SANITY_TOL:.3e}. STOP.",
                file=sys.stderr,
            )
            sanity_ok = False

        # --- CPU formula argmax
        cpu_idx = df["gain_cpu"].idxmax()
        cpu_w_feat = int(df.loc[cpu_idx, "feat"])
        cpu_w_bin = int(df.loc[cpu_idx, "bin"])
        cpu_w_gain = float(df.loc[cpu_idx, "gain_cpu"])

        # --- Old joint-skip argmax
        mlx_idx = df["gain_mlx_formula"].idxmax()
        mlx_w_feat = int(df.loc[mlx_idx, "feat"])
        mlx_w_bin = int(df.loc[mlx_idx, "bin"])
        mlx_w_gain = float(df.loc[mlx_idx, "gain_mlx_formula"])

        # --- picked_by_mlx winner
        picked_rows = df[df["picked"] == 1]
        p_feat = int(picked_rows["feat"].values[0]) if len(picked_rows) > 0 else -1
        p_bin = int(picked_rows["bin"].values[0]) if len(picked_rows) > 0 else -1

        # --- CPU winner: rank under MLX formula (old joint-skip)
        cpu_w_row = df[(df["feat"] == cpu_w_feat) & (df["bin"] == cpu_w_bin)]
        cpu_w_gain_in_mlx = float(cpu_w_row["gain_mlx_formula"].values[0]) if len(cpu_w_row) > 0 else float("nan")
        cpu_rank_in_mlx = int((df["gain_mlx_formula"] > cpu_w_gain_in_mlx).sum()) + 1

        # --- MLX winner: rank under CPU formula
        mlx_w_row = df[(df["feat"] == mlx_w_feat) & (df["bin"] == mlx_w_bin)]
        mlx_w_gain_in_cpu = float(mlx_w_row["gain_cpu"].values[0]) if len(mlx_w_row) > 0 else float("nan")
        mlx_rank_in_cpu = int((df["gain_cpu"] > mlx_w_gain_in_cpu).sum()) + 1

        # --- Gain delta: CPU winner gain (cpu formula) - MLX winner gain (mlx formula)
        gain_delta_cpu_minus_mlx = cpu_w_gain - mlx_w_gain

        # --- Gain delta: CPU winner gain under CPU formula vs picked under CPU formula
        picked_row = df[(df["feat"] == p_feat) & (df["bin"] == p_bin)]
        picked_gain_cpu = float(picked_row["gain_cpu"].values[0]) if len(picked_row) > 0 else float("nan")
        picked_gain_mlx = float(picked_row["gain_mlx_formula"].values[0]) if len(picked_row) > 0 else float("nan")

        # --- CPU reference split (from cpu_model.json)
        cpu_ref = CPU_ITER1_PICKS[d]
        mlx_ref = MLX_ITER1_PICKS[d]

        # --- Verify: does CPU formula argmax match CPU reference?
        cpu_formula_matches_ref = (cpu_w_feat == cpu_ref["feat"])

        # --- Top-5 under each formula
        top5_cpu = top5_by_gain(df, "gain_cpu")
        top5_mlx = top5_by_gain(df, "gain_mlx_formula")

        row = {
            "depth": d,
            "sanity_residual": residual,
            "cpu_winner_feat": cpu_w_feat,
            "cpu_winner_bin": cpu_w_bin,
            "cpu_winner_gain": cpu_w_gain,
            "mlx_winner_feat": mlx_w_feat,
            "mlx_winner_bin": mlx_w_bin,
            "mlx_winner_gain": mlx_w_gain,
            "picked_feat": p_feat,
            "picked_bin": p_bin,
            "picked_gain_under_cpu": picked_gain_cpu,
            "picked_gain_under_mlx": picked_gain_mlx,
            "cpu_winner_rank_in_mlx": cpu_rank_in_mlx,
            "mlx_winner_rank_in_cpu": mlx_rank_in_cpu,
            "gain_delta_cpu_argmax_minus_mlx_argmax": gain_delta_cpu_minus_mlx,
            "cpu_winner_gain_in_mlx_formula": cpu_w_gain_in_mlx,
            "mlx_winner_gain_in_cpu_formula": mlx_w_gain_in_cpu,
            "cpu_ref_feat": cpu_ref["feat"],
            "cpu_ref_border": cpu_ref["border"],
            "argmax_agree": (cpu_w_feat == mlx_w_feat and cpu_w_bin == mlx_w_bin),
        }
        rows.append(row)

        # --- Print per-depth summary
        print(f"--- Depth {d} {'(d=0: ULP-identical sanity check)' if d == 0 else ''}")
        print(f"  n_partition_rows: {len(probe_h)}, n_feat_bin_candidates: {len(df)}")
        print(f"  Sanity: max |gain_cpu - gain_mlx_col| = {residual:.3e} {'PASS' if residual <= SANITY_TOL else 'FAIL'}")
        print()
        print(f"  CPU formula argmax:    feat={cpu_w_feat:>2}, bin={cpu_w_bin:>3}, gain={cpu_w_gain:.8f}")
        print(f"  Old joint-skip argmax: feat={mlx_w_feat:>2}, bin={mlx_w_bin:>3}, gain={mlx_w_gain:.8f}")
        print(f"  picked_by_mlx:         feat={p_feat:>2}, bin={p_bin:>3}")
        print()
        print(f"  CPU ref (cpu_model.json d={d}): feat={cpu_ref['feat']}, border={cpu_ref['border']:.10f}")
        print(f"  CPU formula feat matches CPU ref: {cpu_formula_matches_ref}")
        print()
        if cpu_w_feat != mlx_w_feat or cpu_w_bin != mlx_w_bin:
            print(f"  DIVERGE: CPU winner rank under MLX formula: {cpu_rank_in_mlx}")
            print(f"  DIVERGE: MLX winner rank under CPU formula: {mlx_rank_in_cpu}")
            print(f"  Gain delta CPU_argmax - MLX_argmax = {gain_delta_cpu_minus_mlx:.8f}")
        else:
            print(f"  AGREE: both formulas pick feat={cpu_w_feat}, bin={cpu_w_bin}")
        print()
        print("  Top-5 (CPU formula):")
        for _, r in top5_cpu.iterrows():
            print(f"    feat={int(r.feat):>2}, bin={int(r.bin):>3}, gain_cpu={r['gain_cpu']:.6f}")
        print()
        print("  Top-5 (Old joint-skip formula):")
        for _, r in top5_mlx.iterrows():
            print(f"    feat={int(r.feat):>2}, bin={int(r.bin):>3}, gain_mlx={r['gain_mlx_formula']:.6f}")
        print()

    if not sanity_ok:
        print("[ERROR] Sanity check failed. Formula understanding is incorrect. Stopping.", file=sys.stderr)
        sys.exit(1)

    # --- Save divergence_iter1.csv
    out_df = pd.DataFrame(rows)
    out_path = PROBE_H_DATA / "divergence_iter1.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print()

    # --- Print summary table
    print("=" * 72)
    print("SUMMARY: per-depth divergence (iter=1, N=1k seed=42)")
    print("=" * 72)
    print(
        f"{'d':<3} {'cpu_feat/bin':>14} {'mlx_feat/bin':>14} {'picked_feat/bin':>16}"
        f" {'sanity':>10} {'agree':>6}"
    )
    for r in rows:
        agree_str = "YES" if r["argmax_agree"] else "NO"
        print(
            f"{r['depth']:<3} "
            f"({r['cpu_winner_feat']:>2},{r['cpu_winner_bin']:>3}){'':<5}"
            f"({r['mlx_winner_feat']:>2},{r['mlx_winner_bin']:>3}){'':<5}"
            f"({r['picked_feat']:>2},{r['picked_bin']:>3}){'':<7}"
            f"{r['sanity_residual']:>10.2e}"
            f"{agree_str:>7}"
        )
    print()

    # --- D=0 sanity check detail
    r0 = rows[0]
    print("D=0 iter=1 sanity check (CPU and MLX should agree — known ULP-identical split):")
    print(f"  CPU formula: feat={r0['cpu_winner_feat']}, bin={r0['cpu_winner_bin']}, gain={r0['cpu_winner_gain']:.8f}")
    print(f"  Old joint-skip: feat={r0['mlx_winner_feat']}, bin={r0['mlx_winner_bin']}, gain={r0['mlx_winner_gain']:.8f}")
    print(f"  picked_by_mlx: feat={r0['picked_feat']}, bin={r0['picked_bin']}")
    print(f"  Gain CPU vs MLX at d=0 winner: |{r0['cpu_winner_gain']:.8f} - {r0['mlx_winner_gain']:.8f}| = {abs(r0['cpu_winner_gain']-r0['mlx_winner_gain']):.3e}")
    print(f"  RESULT: {'PASS' if r0['argmax_agree'] and r0['picked_feat']==r0['cpu_winner_feat'] and r0['picked_bin']==r0['cpu_winner_bin'] else 'FAIL'}")
    print()

    # --- Divergence count
    diverge_count = sum(1 for r in rows if not r["argmax_agree"])
    print(f"Depths where CPU formula argmax != MLX formula argmax: {diverge_count}/6")
    print(f"Depths where picked_by_mlx matches CPU formula argmax: "
          f"{sum(1 for r in rows if r['picked_feat']==r['cpu_winner_feat'] and r['picked_bin']==r['cpu_winner_bin'])}/6")
    print()

    print("[DONE] analyze_probe_h.py complete.")


if __name__ == "__main__":
    run_analysis()
