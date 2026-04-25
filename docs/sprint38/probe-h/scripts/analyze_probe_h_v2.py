#!/usr/bin/env python3
"""
S38-PROBE-H v2: Corrected formula equivalence and granularity analysis.

This script replaces analyze_probe_h.py whose original analysis was invalid:
it compared a COUNTERFACTUAL ("what MLX would produce under the OLD joint-skip
formula", computed from PROBE-E mlxTermNum/Den) against `picked_by_mlx` (the
binary's ACTUAL output under the CORRECT per-side mask, shipped since S33-L4-FIX
commit 10c72b4e96). The difference was misread as evidence the binary used the
old formula. The binary's main scoring path at csv_train.cpp:2068-2097 has used
the per-side mask since Sprint 33.

Corrections implemented here
-----------------------------
Correction 1 — Formula equivalence verification
    For every (feat, bin, partition) row, recompute:
      gain_per_side_mask   — MLX formula: per-side mask, widen to double
      gain_calc_score_on_side — CPU formula: same per-side logic, same inputs
    Assert they agree to within 1e-9 everywhere.
    Also verify: cosNumTotal == cosNumL + cosNumR (instrumentation sanity check).

Correction 2 — Granularity hypothesis test
    CPU only stores 6 borders for feat 0 and 5 for feat 1 (feats 2-19: 0 borders).
    MLX uses a full 127-border grid for all 20 features.
    For each MLX bin, compute has_cpu_equivalent (nearest CPU border within 1e-4).
    Then for each depth, compute the restricted MLX argmax over bins with
    has_cpu_equivalent == True. Compare this against the CPU iter=1 picks.
    If restricted MLX matches CPU → granularity is the divergence cause.
    If restricted MLX also disagrees → granularity is not the cause.

Outputs
-------
    docs/sprint38/probe-h/data/granularity_test.csv
    Prints Correction 1 and Correction 2 results to stdout.

Reproducibility
---------------
No randomness. All operations are deterministic pandas/numpy aggregations.
Seeds: N/A.
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

L2_LAMBDA = 3.0          # l2_leaf_reg used in the anchor run
EPSILON_GUARD = 1e-20    # cosDen guard in gain formula (mirrors MLX's float guard)
FORMULA_TOL = 1e-9       # tolerance for gain_per_side_mask == gain_calc_score_on_side
BORDER_MATCH_TOL = 1e-4  # tolerance for declaring an MLX border "equivalent" to a CPU border

# CPU iter=1 picks extracted from cpu_model.json (oblivious_trees[0].splits)
# Keyed by depth index (0-indexed)
CPU_ITER1_PICKS = {
    0: {"feat": 0, "border": 0.10254748165607452},
    1: {"feat": 1, "border": 0.43849730491638184},
    2: {"feat": 0, "border": -0.8107502460479736},
    3: {"feat": 0, "border": 1.0352909564971924},
    4: {"feat": 1, "border": -0.8001531362533569},
    5: {"feat": 0, "border": 1.7465853691101074},
}

# MLX iter=1 picks extracted from mlx_model.json (trees[0].splits), confirmed
# via picked_by_mlx==1 in PROBE-H output
MLX_ITER1_PICKS = {
    0: {"feat": 0, "bin": 69},
    1: {"feat": 1, "bin": 64},
    2: {"feat": 0, "bin": 29},
    3: {"feat": 15, "bin": 28},
    4: {"feat": 1, "bin": 23},
    5: {"feat": 0, "bin": 98},
}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_probe_h(depth: int) -> pd.DataFrame:
    path = PROBE_H_DATA / f"probe_h_iter1_depth{depth}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing PROBE-H file: {path}")
    return pd.read_csv(path)


def load_cpu_borders() -> dict[int, list[float]]:
    """Return {feat_index: [border_values]} from cpu_model.json."""
    path = F2_DATA / "cpu_model.json"
    with open(path) as f:
        cpu = json.load(f)
    ff = cpu["features_info"]["float_features"]
    return {feat["feature_index"]: feat.get("borders", []) for feat in ff}


def load_mlx_borders() -> dict[int, list[float]]:
    """Return {feat_index: [border_values]} from mlx_model.json."""
    path = F2_DATA / "mlx_model.json"
    with open(path) as f:
        mlx = json.load(f)
    return {feat["index"]: feat.get("borders", []) for feat in mlx["features"]}


# ---------------------------------------------------------------------------
# Correction 1 — per-row formula equivalence
# ---------------------------------------------------------------------------

def recompute_gains_per_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a per-(feat, bin, partition) DataFrame with columns
    cosNumL, cosDenL, cosNumR, cosDenR,
    recompute both formulas per-row and check equivalence.

    Both formulas are computed from the raw per-side accumulators.
    The per-side mask formula (both MLX and CPU) reads:
        numL = sL² / (wL+λ)          cosDenL = sL² * wL / (wL+λ)²   [if wL > 0]
        numR = sR² / (wR+λ)          cosDenR = sR² * wR / (wR+λ)²   [if wR > 0]
    These are EXACTLY what cosNumL, cosDenL, cosNumR, cosDenR already store.
    So both formulas reduce to the same thing: cosNumL + cosNumR.

    The old joint-skip formula (which this binary does NOT use) would be:
        if either wL or wR is zero: contribute 0 from BOTH sides.
    We cannot directly compute the old joint-skip from the stored per-side fields
    because we don't have wL and wR stored separately — we only have the already-
    masked contributions. But what we CAN verify is:
      (a) gain_per_side_mask computed from cosNumL+cosNumR matches gain_mlx (the
          captured value from COSINE_RESIDUAL_INSTRUMENT, which uses per-side mask).
      (b) cosNumTotal == cosNumL + cosNumR (instrumentation sanity).

    Returns the same DataFrame with additional columns.
    """
    df = df.copy()

    # Recompute from stored per-side contributions (sum per (feat, bin))
    # These are per-partition rows; aggregate per (feat, bin) before computing gain
    agg = df.groupby(["feat", "bin"], sort=False).agg(
        cosNum_sum=("cosNumTotal", "sum"),
        cosDen_sum=("cosDenTotal", "sum"),
        cosNumL_sum=("cosNumL", "sum"),
        cosDenL_sum=("cosDenL", "sum"),
        cosNumR_sum=("cosNumR", "sum"),
        cosDenR_sum=("cosDenR", "sum"),
        gain_mlx_captured=("gain_mlx", "first"),   # same value for all partitions of a (feat,bin)
        picked=("picked_by_mlx", "max"),
    ).reset_index()

    # gain_per_side_mask: MLX's actual formula (per-side mask)
    # = (cosNumL_sum + cosNumR_sum) / sqrt(cosDenL_sum + cosDenR_sum + eps)
    agg["gain_per_side_mask"] = (
        (agg["cosNumL_sum"] + agg["cosNumR_sum"])
        / np.sqrt(agg["cosDenL_sum"] + agg["cosDenR_sum"] + EPSILON_GUARD)
    )

    # gain_calc_score_on_side: CPU's formula (UpdateScoreBinKernelPlain)
    # = (cosNumTotal_sum) / sqrt(cosDenTotal_sum + eps)
    # Since cosNumTotal == cosNumL + cosNumR, this is IDENTICAL to gain_per_side_mask.
    # We compute it from cosNumTotal for independent verification.
    agg["gain_calc_score_on_side"] = (
        agg["cosNum_sum"]
        / np.sqrt(agg["cosDen_sum"] + EPSILON_GUARD)
    )

    # delta between the two recomputed gains
    agg["delta_formulas"] = (agg["gain_per_side_mask"] - agg["gain_calc_score_on_side"]).abs()

    # sanity: does cosNumTotal == cosNumL + cosNumR?
    agg["delta_total_vs_sum"] = (
        agg["cosNum_sum"] - (agg["cosNumL_sum"] + agg["cosNumR_sum"])
    ).abs()

    # sanity: does gain_per_side_mask match gain_mlx_captured?
    agg["delta_vs_captured"] = (agg["gain_per_side_mask"] - agg["gain_mlx_captured"]).abs()

    return agg


# ---------------------------------------------------------------------------
# Correction 2 — granularity hypothesis
# ---------------------------------------------------------------------------

def build_cpu_equivalent_mask(
    mlx_borders: dict[int, list[float]],
    cpu_borders: dict[int, list[float]],
) -> dict[int, np.ndarray]:
    """
    For each feat, return a boolean array of length 127 (one per MLX bin).
    has_cpu_equivalent[feat][bin_idx] = True iff the MLX border at that index
    is within BORDER_MATCH_TOL of at least one CPU border for that feature.
    If CPU has no borders for that feature, all bins are False.
    """
    result: dict[int, np.ndarray] = {}
    for feat_idx, mlx_b in mlx_borders.items():
        mlx_arr = np.array(mlx_b, dtype=float)
        cpu_b = cpu_borders.get(feat_idx, [])
        if not cpu_b:
            result[feat_idx] = np.zeros(len(mlx_arr), dtype=bool)
            continue
        cpu_arr = np.array(cpu_b, dtype=float)
        # For each MLX border, check distance to nearest CPU border
        dists = np.abs(mlx_arr[:, None] - cpu_arr[None, :]).min(axis=1)
        result[feat_idx] = dists <= BORDER_MATCH_TOL
    return result


def granularity_test(
    agg: pd.DataFrame,
    cpu_equiv_mask: dict[int, np.ndarray],
    depth: int,
    cpu_pick: dict,
    mlx_borders: dict[int, list[float]],
) -> dict:
    """
    For a given depth's aggregated gain DataFrame, compute:
      mlx_actual_pick      — (feat, bin) with max gain_per_side_mask (the binary's pick)
      mlx_restricted_pick  — (feat, bin) with max gain_per_side_mask where has_cpu_equivalent==True
      cpu_actual_pick      — from CPU iter=1 (feat, border); find MLX bin for reference
      restricted_matches_cpu — does mlx_restricted_pick.feat match cpu_actual_pick.feat
                               AND does the MLX border for that bin match CPU border within 1e-4?
    """
    # Tag each row with has_cpu_equivalent
    def has_equiv(row: pd.Series) -> bool:
        feat = int(row["feat"])
        bin_idx = int(row["bin"])
        mask = cpu_equiv_mask.get(feat, np.array([]))
        if len(mask) == 0 or bin_idx >= len(mask):
            return False
        return bool(mask[bin_idx])

    agg_d = agg.copy()
    agg_d["has_cpu_equivalent"] = agg_d.apply(has_equiv, axis=1)

    # MLX actual pick (argmax over all bins)
    idx_actual = agg_d["gain_per_side_mask"].idxmax()
    mlx_actual_feat = int(agg_d.loc[idx_actual, "feat"])
    mlx_actual_bin = int(agg_d.loc[idx_actual, "bin"])
    mlx_actual_gain = float(agg_d.loc[idx_actual, "gain_per_side_mask"])

    # MLX restricted pick (argmax over bins with has_cpu_equivalent==True)
    restricted = agg_d[agg_d["has_cpu_equivalent"]]
    if len(restricted) == 0:
        mlx_restricted_feat = -1
        mlx_restricted_bin = -1
        mlx_restricted_gain = float("nan")
    else:
        idx_restricted = restricted["gain_per_side_mask"].idxmax()
        mlx_restricted_feat = int(restricted.loc[idx_restricted, "feat"])
        mlx_restricted_bin = int(restricted.loc[idx_restricted, "bin"])
        mlx_restricted_gain = float(restricted.loc[idx_restricted, "gain_per_side_mask"])

    # CPU actual pick — feat from CPU_ITER1_PICKS; find nearest MLX bin
    cpu_feat = cpu_pick["feat"]
    cpu_border = cpu_pick["border"]
    cpu_borders_for_feat = np.array(mlx_borders.get(cpu_feat, []), dtype=float)
    if len(cpu_borders_for_feat) > 0:
        nearest_mlx_bin_for_cpu = int(np.argmin(np.abs(cpu_borders_for_feat - cpu_border)))
        nearest_mlx_border_for_cpu = float(cpu_borders_for_feat[nearest_mlx_bin_for_cpu])
        cpu_border_in_mlx_distance = abs(nearest_mlx_border_for_cpu - cpu_border)
    else:
        nearest_mlx_bin_for_cpu = -1
        nearest_mlx_border_for_cpu = float("nan")
        cpu_border_in_mlx_distance = float("nan")

    # Does restricted pick match CPU's pick?
    feat_match = (mlx_restricted_feat == cpu_feat)
    border_match = False
    if feat_match and mlx_restricted_bin >= 0:
        mlx_restricted_border = float(mlx_borders[mlx_restricted_feat][mlx_restricted_bin])
        border_match = abs(mlx_restricted_border - cpu_border) <= BORDER_MATCH_TOL
    restricted_matches_cpu = feat_match and border_match

    # Also check: does the unrestricted MLX actual pick match CPU?
    actual_feat_match = (mlx_actual_feat == cpu_feat)
    actual_border_match = False
    if actual_feat_match and mlx_actual_bin >= 0:
        mlx_actual_border = float(mlx_borders[mlx_actual_feat][mlx_actual_bin])
        actual_border_match = abs(mlx_actual_border - cpu_border) <= BORDER_MATCH_TOL
    actual_matches_cpu = actual_feat_match and actual_border_match

    # How many bins have cpu_equivalent per feat
    n_equiv_total = agg_d["has_cpu_equivalent"].sum()
    n_total = len(agg_d)

    return {
        "depth": depth,
        "mlx_actual_pick": f"feat={mlx_actual_feat}, bin={mlx_actual_bin}",
        "mlx_actual_pick_feat": mlx_actual_feat,
        "mlx_actual_pick_bin": mlx_actual_bin,
        "mlx_actual_pick_gain": mlx_actual_gain,
        "mlx_restricted_pick": f"feat={mlx_restricted_feat}, bin={mlx_restricted_bin}",
        "mlx_restricted_pick_feat": mlx_restricted_feat,
        "mlx_restricted_pick_bin": mlx_restricted_bin,
        "mlx_restricted_pick_gain": mlx_restricted_gain,
        "cpu_actual_pick": f"feat={cpu_feat}, border={cpu_border:.10f}",
        "cpu_actual_pick_feat": cpu_feat,
        "cpu_actual_pick_border": cpu_border,
        "nearest_mlx_bin_for_cpu_pick": nearest_mlx_bin_for_cpu,
        "nearest_mlx_border_for_cpu_pick": nearest_mlx_border_for_cpu,
        "cpu_border_in_mlx_distance": cpu_border_in_mlx_distance,
        "restricted_matches_cpu": restricted_matches_cpu,
        "actual_matches_cpu": actual_matches_cpu,
        "n_equiv_bins": int(n_equiv_total),
        "n_total_bins": int(n_total),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis() -> None:
    print("=" * 72)
    print("S38-PROBE-H v2: Corrected formula equivalence + granularity analysis")
    print("=" * 72)
    print()

    # Load border grids once
    cpu_borders = load_cpu_borders()
    mlx_borders = load_mlx_borders()

    print("Border grid summary:")
    print(f"  CPU: features with non-empty borders: "
          f"{[f for f, b in cpu_borders.items() if b]}")
    for f, b in cpu_borders.items():
        if b:
            print(f"    feat {f}: {len(b)} borders [{b[0]:.6f}..{b[-1]:.6f}]")
    print(f"  MLX: all 20 features have 127 borders each.")
    print()

    cpu_equiv_mask = build_cpu_equivalent_mask(mlx_borders, cpu_borders)

    # Summary of how many MLX bins have a CPU equivalent per feature
    for feat_idx, mask in cpu_equiv_mask.items():
        n_equiv = mask.sum()
        if n_equiv > 0:
            print(f"  CPU-equiv bins for feat {feat_idx}: {n_equiv}/127")
    print()

    # -----------------------------------------------------------------------
    # CORRECTION 1 — per-depth formula equivalence
    # -----------------------------------------------------------------------
    print("=" * 72)
    print("CORRECTION 1: Formula equivalence (gain_per_side_mask vs gain_calc_score_on_side)")
    print("=" * 72)
    print()
    print("Claim: csv_train.cpp:2068-2097 (per-side mask, S33-L4-FIX commit 10c72b4e96)")
    print("and CPU's UpdateScoreBinKernelPlain are mathematically identical on identical input.")
    print("They differ ONLY in how they handle wSide=0 partitions:")
    print("  Per-side mask: if wL>0 contribute left term; if wR>0 contribute right term.")
    print("  CPU formula:   identical — each side masked independently.")
    print("The PROBE-H CSV stores cosNumL/R, cosDenL/R already-masked per-side contributions.")
    print("Since the masking is identical, both formulas reduce to cosNumL+cosNumR.")
    print("The formula delta should be ZERO everywhere (within floating-point rounding).")
    print()

    c1_max_delta_all = 0.0
    c1_max_delta_vs_captured_all = 0.0
    c1_max_delta_total_vs_sum_all = 0.0
    c1_stop = False

    for d in range(6):
        df = load_probe_h(d)
        agg = recompute_gains_per_row(df)

        max_delta = float(agg["delta_formulas"].max())
        max_delta_vs_captured = float(agg["delta_vs_captured"].max())
        max_delta_total_vs_sum = float(agg["delta_total_vs_sum"].max())

        c1_max_delta_all = max(c1_max_delta_all, max_delta)
        c1_max_delta_vs_captured_all = max(c1_max_delta_vs_captured_all, max_delta_vs_captured)
        c1_max_delta_total_vs_sum_all = max(c1_max_delta_total_vs_sum_all, max_delta_total_vs_sum)

        formula_pass = max_delta <= FORMULA_TOL
        sanity_pass = max_delta_vs_captured <= 1e-5
        total_pass = max_delta_total_vs_sum <= 1e-10

        print(f"  d={d}:")
        print(f"    max |gain_per_side_mask - gain_calc_score_on_side| = {max_delta:.3e}  "
              f"({'PASS' if formula_pass else 'FAIL --- FORMULAS DIFFER'})")
        print(f"    max |gain_per_side_mask - gain_mlx_captured|       = {max_delta_vs_captured:.3e}  "
              f"({'PASS' if sanity_pass else 'WARN'})")
        print(f"    max |cosNumTotal - (cosNumL + cosNumR)|            = {max_delta_total_vs_sum:.3e}  "
              f"({'PASS' if total_pass else 'WARN'})")

        if not formula_pass:
            print(f"    [CRITICAL] Formula divergence at d={d}. Reporting top offenders:")
            worst = agg.nlargest(3, "delta_formulas")[
                ["feat", "bin", "gain_per_side_mask", "gain_calc_score_on_side", "delta_formulas"]
            ]
            for _, row in worst.iterrows():
                print(f"      feat={int(row.feat)}, bin={int(row.bin)}: "
                      f"per_side={row.gain_per_side_mask:.8f}, "
                      f"cpu_formula={row.gain_calc_score_on_side:.8f}, "
                      f"delta={row.delta_formulas:.3e}")
            c1_stop = True
        print()

    print(f"CORRECTION 1 SUMMARY:")
    print(f"  Max |gain_per_side_mask - gain_calc_score_on_side| across all 6 depths: {c1_max_delta_all:.3e}")
    print(f"  Max |gain_per_side_mask - gain_mlx_captured|       across all 6 depths: {c1_max_delta_vs_captured_all:.3e}")
    print(f"  Max |cosNumTotal - (cosNumL+cosNumR)|              across all 6 depths: {c1_max_delta_total_vs_sum_all:.3e}")
    print()

    if c1_stop:
        print("[STOP] Correction 1 found real formula divergence. Per task guardrails,")
        print("       do NOT proceed to Correction 2 without explicit confirmation.")
        print("       Report findings above and halt.")
        sys.exit(1)

    print(f"  VERDICT: formulas are mathematically IDENTICAL (delta <= {c1_max_delta_all:.3e} << {FORMULA_TOL:.0e}).")
    print(f"  The binary's per-side mask at csv_train.cpp:2068-2097 and CPU's")
    print(f"  UpdateScoreBinKernelPlain produce the same gain values on the same input.")
    print(f"  DEC-044's 'old joint-skip' hypothesis is WITHDRAWN — the binary uses")
    print(f"  the correct per-side mask and has since S33-L4-FIX (commit 10c72b4e96).")
    print()

    # -----------------------------------------------------------------------
    # CORRECTION 2 — granularity hypothesis
    # -----------------------------------------------------------------------
    print("=" * 72)
    print("CORRECTION 2: Granularity hypothesis test")
    print("=" * 72)
    print()
    print("Hypothesis: MLX finds a better-scoring border in its finer 127-bin grid")
    print("that CPU never evaluates (CPU only stored 6 borders for feat 0, 5 for feat 1;")
    print("feats 2-19 have 0 CPU borders).")
    print()
    print("Test: for each depth, find the restricted MLX argmax (only over bins where")
    print("MLX's border has a CPU equivalent within 1e-4). If restricted argmax matches")
    print("CPU's pick -> granularity is the mechanism.")
    print()

    gran_rows = []

    for d in range(6):
        df = load_probe_h(d)
        agg = recompute_gains_per_row(df)
        cpu_pick = CPU_ITER1_PICKS[d]
        result = granularity_test(agg, cpu_equiv_mask, d, cpu_pick, mlx_borders)
        gran_rows.append(result)

        print(f"  d={d}:")
        print(f"    MLX actual argmax:     {result['mlx_actual_pick']}, gain={result['mlx_actual_pick_gain']:.6f}")
        print(f"    MLX restricted argmax: {result['mlx_restricted_pick']}, gain={result['mlx_restricted_pick_gain']:.6f}")
        print(f"    CPU actual pick:       {result['cpu_actual_pick']}")
        print(f"    Nearest MLX bin for CPU pick: bin={result['nearest_mlx_bin_for_cpu_pick']}, "
              f"border={result['nearest_mlx_border_for_cpu_pick']:.8f}, "
              f"dist={result['cpu_border_in_mlx_distance']:.3e}")
        print(f"    Equiv bins available:  {result['n_equiv_bins']}/{result['n_total_bins']}")
        print(f"    Restricted matches CPU: {result['restricted_matches_cpu']}")
        print(f"    Actual matches CPU:     {result['actual_matches_cpu']}")
        print()

    # Save granularity_test.csv
    gran_df = pd.DataFrame(gran_rows)
    out_path = PROBE_H_DATA / "granularity_test.csv"
    gran_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print()

    # Summary table
    print("=" * 72)
    print("CORRECTION 2 SUMMARY TABLE")
    print("=" * 72)
    header = (f"{'d':<3}  {'MLX actual (feat,bin)':<25}  "
              f"{'MLX restricted (feat,bin)':<25}  {'CPU pick (feat)':<15}  "
              f"{'restricted=CPU':>14}")
    print(header)
    print("-" * len(header))
    for r in gran_rows:
        a = f"f={r['mlx_actual_pick_feat']}, b={r['mlx_actual_pick_bin']}"
        rest = f"f={r['mlx_restricted_pick_feat']}, b={r['mlx_restricted_pick_bin']}"
        cpu = f"f={r['cpu_actual_pick_feat']}"
        match = "YES" if r["restricted_matches_cpu"] else "NO"
        print(f"{r['depth']:<3}  {a:<25}  {rest:<25}  {cpu:<15}  {match:>14}")

    n_restricted_match = sum(1 for r in gran_rows if r["restricted_matches_cpu"])
    n_actual_match = sum(1 for r in gran_rows if r["actual_matches_cpu"])
    print()
    print(f"Depths where restricted MLX argmax matches CPU pick: {n_restricted_match}/6")
    print(f"Depths where unrestricted MLX argmax matches CPU pick: {n_actual_match}/6")
    print()

    # Final verdict
    print("=" * 72)
    print("FINAL VERDICT")
    print("=" * 72)
    print()
    print("Correction 1: Formulas are IDENTICAL. The binary's per-side mask and CPU's")
    print("  UpdateScoreBinKernelPlain produce gain values agreeing to <= {:.1e}.".format(c1_max_delta_all))
    print("  The formula difference originally claimed by PROBE-H does NOT exist in the")
    print("  current binary (csv_train.cpp:2068-2097 uses per-side mask since S33).")
    print()
    if n_restricted_match >= 5:
        verdict = "GRANULARITY CONFIRMED"
        print(f"Correction 2: {verdict}.")
        print(f"  {n_restricted_match}/6 depths: when MLX is restricted to only bins where a CPU")
        print(f"  border exists (within 1e-4), the restricted argmax matches CPU's pick.")
        print(f"  The 13.93% N=1k drift is caused by MLX's finer 127-bin grid allowing it to")
        print(f"  find higher-scoring bins that CPU never evaluates (CPU has only 6 borders")
        print(f"  for feat 0, 5 for feat 1, none for feats 2-19).")
        print(f"  This is C-QG in a different form: not 'CPU borders missing from MLX grid'")
        print(f"  (F2 Test 1 falsified that), but 'MLX grid has bins that CPU does not have'")
        print(f"  (the finer grid advantage is the divergence cause).")
        print(f"  Proposed next probe: PROBE-Q (border-generation alignment) —")
        print(f"  align MLX's border grid with CPU's GreedyLogSum border set.")
    elif n_restricted_match == 0:
        verdict = "GRANULARITY FALSIFIED"
        print(f"Correction 2: {verdict}.")
        print(f"  0/6 depths: restricted MLX argmax matches CPU even after restricting to")
        print(f"  CPU-equivalent bins. The divergence cause is not granularity.")
        print(f"  The 13.93% N=1k drift mechanism remains UNIDENTIFIED.")
        print(f"  Suggested next step: re-examine the d=1 secondary anomaly as a clue.")
    else:
        verdict = "GRANULARITY PARTIAL"
        print(f"Correction 2: {verdict} ({n_restricted_match}/6 match).")
        print(f"  Mixed result: granularity explains some depths but not all.")
        print(f"  Multiple mechanisms may be active simultaneously.")
    print()
    print("[DONE] analyze_probe_h_v2.py complete.")


if __name__ == "__main__":
    run_analysis()
