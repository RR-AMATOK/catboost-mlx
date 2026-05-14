#!/usr/bin/env python3
"""
S48-T1 child-imbalance analysis script.

Reads per-split child-count CSVs produced by bench_boosting_s48_t1
and computes per-shape geomean of min(L,R)/(L+R) across all splits,
all trees, all seeds.

Usage:
    python3 analyze.py

Outputs:
    - Per-shape geomean + 3-seed variance table (stdout)
    - Per-depth geomean breakdown (stdout)
    - C6 verdict per shape (stdout)
    - Writes docs/sprint48/T1/child-imbalance/analysis.md
"""

import math
import os
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]  # catboost-mlx repo root
DATA_DIR = Path(__file__).parent / "data"
ANALYSIS_MD = Path(__file__).parent / "analysis.md"

SEEDS = [42, 43, 44]
SHAPES = {
    "higgs": "Higgs-1M-proxy (1M x 28, binary, depth 6, bins 128)",
    "epsilon": "Epsilon-proxy (400k x 2000, binary, depth 6, bins 128)",
}
MAX_DEPTH = 6

# DEC-052 T0c decision thresholds
THRESHOLD_KEEP   = 0.35   # geomean <= this => C6 projects >=2x => KEEP
THRESHOLD_RETIRE = 0.45   # geomean >= this => C6 bounded below 1.6x => likely RETIRE
# 0.35 < geomean < 0.45 => AMBIGUOUS


def load_csv(path: Path):
    """Load a child-count CSV into list of dicts."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "tree":        int(row["tree"]),
                "level":       int(row["level"]),
                "leaf":        int(row["leaf"]),
                "left_count":  int(row["left_count"]),
                "right_count": int(row["right_count"]),
            })
    return rows


def ratio(row):
    """Compute min(L,R)/(L+R) for a split row. Returns None if parent is empty."""
    total = row["left_count"] + row["right_count"]
    if total == 0:
        return None
    return min(row["left_count"], row["right_count"]) / total


def geomean(values):
    """Geometric mean of a list of positive floats."""
    if not values:
        return float("nan")
    log_sum = sum(math.log(v) for v in values if v > 0)
    return math.exp(log_sum / len(values))


def verdict(gm):
    if math.isnan(gm):
        return "UNKNOWN (no data)"
    if gm <= THRESHOLD_KEEP:
        return f"KEEP (geomean={gm:.4f} <= {THRESHOLD_KEEP}; C6 projects >=2x)"
    if gm >= THRESHOLD_RETIRE:
        return f"RETIRE (geomean={gm:.4f} >= {THRESHOLD_RETIRE}; C6 bounded < 1.6x iter)"
    return f"AMBIGUOUS (geomean={gm:.4f}; 0.35 < gm < 0.45; C6 T4 measurement load-bearing)"


def analyze_shape(shape_key):
    """
    Returns:
        per_seed_geomeans: {seed: geomean}
        overall_geomean: float
        per_depth_geomeans: {depth: geomean}
        total_splits: int
    """
    all_ratios = []
    per_depth = {d: [] for d in range(MAX_DEPTH)}
    per_seed_ratios = {}

    for seed in SEEDS:
        fname = DATA_DIR / f"{shape_key}_seed{seed}.csv"
        if not fname.exists():
            print(f"  WARNING: missing {fname}")
            continue
        rows = load_csv(fname)
        seed_ratios = []
        for row in rows:
            r = ratio(row)
            if r is None:
                continue
            all_ratios.append(r)
            seed_ratios.append(r)
            d = row["level"]
            if 0 <= d < MAX_DEPTH:
                per_depth[d].append(r)
        per_seed_ratios[seed] = geomean(seed_ratios)

    overall_gm = geomean(all_ratios)
    per_depth_gm = {d: geomean(per_depth[d]) for d in range(MAX_DEPTH)}
    return per_seed_ratios, overall_gm, per_depth_gm, len(all_ratios)


def main():
    print("=" * 68)
    print("S48-T1 child-imbalance analysis")
    print("=" * 68)

    results = {}
    for shape_key, shape_desc in SHAPES.items():
        print(f"\nShape: {shape_desc}")
        per_seed, gm, per_depth, n_splits = analyze_shape(shape_key)
        results[shape_key] = {
            "desc": shape_desc,
            "per_seed": per_seed,
            "geomean": gm,
            "per_depth": per_depth,
            "n_splits": n_splits,
            "verdict": verdict(gm),
        }

        print(f"  Total split records: {n_splits}")
        print(f"  Per-seed geomeans:")
        for seed, gm_s in sorted(per_seed.items()):
            print(f"    seed {seed}: {gm_s:.4f}")
        print(f"  Overall geomean (all seeds): {gm:.4f}")
        print(f"  Per-depth breakdown:")
        for d in range(MAX_DEPTH):
            gm_d = per_depth[d]
            if not math.isnan(gm_d):
                print(f"    depth {d}: {gm_d:.4f}")
            else:
                print(f"    depth {d}: (no data)")
        print(f"  C6 verdict: {results[shape_key]['verdict']}")

    # Combined verdict
    higgs_gm  = results.get("higgs",   {}).get("geomean", float("nan"))
    epsilon_gm = results.get("epsilon", {}).get("geomean", float("nan"))

    any_keep = any(
        not math.isnan(v["geomean"]) and v["geomean"] <= THRESHOLD_KEEP
        for v in results.values()
    )
    both_retire = all(
        not math.isnan(v["geomean"]) and v["geomean"] >= THRESHOLD_RETIRE
        for v in results.values()
        if not math.isnan(v["geomean"])
    ) and len(results) > 0

    if both_retire:
        combined = "C6 RETIRED — both shapes geomean >= 0.45 (near-balanced; subtraction savings bounded below 1.6x iter)"
        next_step = "C6 RETIRED — T2 inspects L6 + C4 only."
    elif any_keep:
        combined = "C6 KEEP — at least one shape geomean <= 0.35 (skewed splits; subtraction projects >=2x)"
        next_step = "READY FOR @SILICON-ARCHITECT T2 INSPECTION."
    else:
        combined = "AMBIGUOUS — geomean in [0.35, 0.45) on all measured shapes; C6 T4 measurement is load-bearing"
        next_step = "READY FOR @SILICON-ARCHITECT T2 INSPECTION (with AMBIGUOUS flag)."

    print(f"\n{'=' * 68}")
    print(f"Combined C6 verdict: {combined}")
    print(f"Next step: {next_step}")

    # Write analysis.md
    write_analysis_md(results, combined, next_step)
    print(f"\nAnalysis written to {ANALYSIS_MD}")


def write_analysis_md(results, combined_verdict, next_step):
    lines = []
    lines.append("# S48-T1 child-imbalance analysis")
    lines.append("")
    lines.append("**Date:** 2026-05-13")
    lines.append("**Branch:** `mlx/sprint-48-t0-brainstorm`")
    lines.append("**Authority:** DEC-052 T0c LOCK (2026-05-12)")
    lines.append("**Generated by:** `docs/sprint48/T1/child-imbalance/analyze.py`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Decision rule (locked at T0c, DEC-052)")
    lines.append("")
    lines.append("Metric: geomean of `min(|L|,|R|) / (|L|+|R|)` across all splits,")
    lines.append("all 100 trees, all 3 seeds (42/43/44) per shape.")
    lines.append("")
    lines.append("| Threshold | Interpretation | C6 outcome |")
    lines.append("|---|---|---|")
    lines.append("| geomean <= 0.35 | Skewed splits dominant | C6 projects >=2x -> KEEP |")
    lines.append("| 0.35 < geomean < 0.45 | Ambiguous zone | C6 T4 measurement load-bearing |")
    lines.append("| geomean >= 0.45 | Near-balanced splits dominant | C6 bounded < 1.6x iter -> likely RETIRE |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Sweep configuration")
    lines.append("")
    lines.append("| Shape | rows x features | depth | bins | iters | seeds |")
    lines.append("|---|---|---|---|---|---|")
    lines.append("| Higgs-1M-proxy | 1,000,000 x 28 | 6 | 128 | 100 | 42, 43, 44 |")
    lines.append("| Epsilon-proxy | 400,000 x 2,000 | 6 | 128 | 100 | 42, 43, 44 |")
    lines.append("")
    lines.append("Binary: `bench_boosting_s48_t1` compiled with `-DCATBOOST_MLX_LOG_CHILD_IMBALANCE`.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Per-shape results")
    lines.append("")

    for shape_key, r in results.items():
        lines.append(f"### {r['desc']}")
        lines.append("")
        lines.append(f"Total split records: {r['n_splits']:,}")
        lines.append("")
        lines.append("#### 3-seed geomeans")
        lines.append("")
        lines.append("| Seed | Geomean min(L,R)/(L+R) |")
        lines.append("|---|---|")
        for seed in SEEDS:
            gm_s = r["per_seed"].get(seed, float("nan"))
            gm_str = f"{gm_s:.4f}" if not math.isnan(gm_s) else "N/A (no data)"
            lines.append(f"| {seed} | {gm_str} |")
        lines.append("")
        gm_all = r["geomean"]
        gm_str = f"{gm_all:.4f}" if not math.isnan(gm_all) else "N/A"
        lines.append(f"**Overall geomean (all seeds, all trees):** {gm_str}")
        lines.append("")
        lines.append("#### Per-depth breakdown")
        lines.append("")
        lines.append("Deeper levels split smaller partitions; near-balanced splits at depth 0")
        lines.append("are less C6-favorable than at depth 5 (fewer docs/partition).")
        lines.append("")
        lines.append("| Depth | Geomean min(L,R)/(L+R) | Docs/partition (approx) |")
        lines.append("|---|---|---|")
        rows_map = {"higgs": 1_000_000, "epsilon": 400_000}
        total_rows = rows_map.get(shape_key, 100_000)
        for d in range(MAX_DEPTH):
            gm_d = r["per_depth"][d]
            gm_str = f"{gm_d:.4f}" if not math.isnan(gm_d) else "N/A"
            docs_per_part = total_rows // (1 << d) if d < 20 else "< 1"
            lines.append(f"| {d} | {gm_str} | ~{docs_per_part:,} |")
        lines.append("")
        lines.append(f"**C6 verdict for {shape_key}:** {r['verdict']}")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Combined C6 verdict")
    lines.append("")
    lines.append(combined_verdict)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Verification: bit-exactness check")
    lines.append("")
    lines.append("The instrumented binary (`bench_boosting_s48_t1`) must produce")
    lines.append("identical `BENCH_FINAL_LOSS` values to the uninstrumented baseline")
    lines.append("(`bench_boosting_baseline`) on the same shape/seed.")
    lines.append("")
    lines.append("Procedure:")
    lines.append("```bash")
    lines.append("# Run instrumented build (guard ON)")
    lines.append('T1="$BUILDDIR/bench_boosting_s48_t1"')
    lines.append('BASE="$BUILDDIR/bench_boosting_baseline"')
    lines.append("")
    lines.append('LOSS_T1=$("$T1"  --rows 1000000 --features 28 --classes 2 --depth 6 \\')
    lines.append('    --bins 128 --iters 10 --seed 42 2>/dev/null | grep BENCH_FINAL_LOSS | awk -F= \'{print $2}\')')
    lines.append('LOSS_BASE=$("$BASE" --rows 1000000 --features 28 --classes 2 --depth 6 \\')
    lines.append('    --bins 128 --iters 10 --seed 42 2>/dev/null | grep BENCH_FINAL_LOSS | awk -F= \'{print $2}\')')
    lines.append('[ "$LOSS_T1" = "$LOSS_BASE" ] && echo "PASS: bit-exact" || echo "FAIL: $LOSS_T1 vs $LOSS_BASE"')
    lines.append("```")
    lines.append("")
    lines.append("**Result:** [TO BE FILLED IN AFTER RUN]")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"## Next step")
    lines.append("")
    lines.append(next_step)
    lines.append("")

    ANALYSIS_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
