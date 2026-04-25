#!/usr/bin/env python3
"""
PROBE-Q Steps 3 & 4: Compare CPU vs MLX border sets per feature.

Inputs:
  data/cpu_borders_full.json  — 128 borders per feature (CPU CatBoost)
  data/mlx_borders_full.json  — 127 borders per feature (MLX csv_train)

Alignment strategy:
  CPU has 128 borders, MLX has 127 borders.  For each MLX border we find the
  nearest CPU border (minimum absolute distance).  This is a nearest-neighbour
  matching — NOT a 1:1 aligned pairing — because the two grids have different
  cardinalities.  We report:
    - max_delta   : worst-case |mlx_border - nearest_cpu_border|
    - mean_delta  : mean over all 127 MLX borders
    - median_delta: median over all 127 MLX borders
    - n_close     : n MLX borders with nearest-CPU delta < 1e-4
    - n_far       : n MLX borders with nearest-CPU delta > 1e-2

  A feature is "aligned" if max_delta < 1e-4 (all MLX borders have a CPU
  counterpart within 1e-4).  A feature is "diverged" if it has at least one
  border pair with delta > 1e-2.

Step 4 also cross-references the 6 CPU-stored borders for feat_0 from
docs/sprint38/f2/data/cpu_iter2_splits.json (the borders that ended up in the
trained tree), verifying those serialized borders exist in MLX's grid too.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
PROBE_Q_DIR = REPO_ROOT / "docs" / "sprint38" / "probe-q"
DATA_DIR = PROBE_Q_DIR / "data"
F2_DATA_DIR = REPO_ROOT / "docs" / "sprint38" / "f2" / "data"

ANCHOR_FEATS = 20
CLOSE_THRESH = 1e-4
FAR_THRESH = 1e-2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nearest_delta(mlx_val: float, cpu_sorted: np.ndarray) -> float:
    """Return |mlx_val - nearest_cpu_border|."""
    idx = np.searchsorted(cpu_sorted, mlx_val)
    candidates = []
    if idx > 0:
        candidates.append(abs(mlx_val - cpu_sorted[idx - 1]))
    if idx < len(cpu_sorted):
        candidates.append(abs(mlx_val - cpu_sorted[idx]))
    return min(candidates) if candidates else float("inf")


# ---------------------------------------------------------------------------
# Step 3: per-feature comparison
# ---------------------------------------------------------------------------

def compare_all_features(
    cpu_borders: dict[str, list[float]],
    mlx_borders: dict[str, list[float]],
) -> list[dict]:
    rows = []
    all_top5: list[tuple[float, int, float, float]] = []  # (delta, feat, mlx_b, cpu_b)

    print(f"\n{'feat':<6}{'n_cpu':<7}{'n_mlx':<7}{'max_delta':<14}{'mean_delta':<14}"
          f"{'median_delta':<14}{'n_close':<9}{'n_far':<7}{'verdict'}")
    print("-" * 100)

    for fi in range(ANCHOR_FEATS):
        key = f"feat_{fi}"
        cpu = np.array(sorted(cpu_borders.get(key, [])), dtype=np.float64)
        mlx = np.array(sorted(mlx_borders.get(key, [])), dtype=np.float64)

        n_cpu = len(cpu)
        n_mlx = len(mlx)

        if n_cpu == 0 or n_mlx == 0:
            print(f"{fi:<6}{n_cpu:<7}{n_mlx:<7}{'N/A':<14}{'N/A':<14}{'N/A':<14}{'N/A':<9}{'N/A':<7}SKIP")
            rows.append({
                "feat": fi, "n_cpu": n_cpu, "n_mlx": n_mlx,
                "max_delta": None, "mean_delta": None, "median_delta": None,
                "n_close": None, "n_far": None,
            })
            continue

        deltas = np.array([nearest_delta(b, cpu) for b in mlx])

        max_d = float(np.max(deltas))
        mean_d = float(np.mean(deltas))
        median_d = float(np.median(deltas))
        n_close = int(np.sum(deltas < CLOSE_THRESH))
        n_far = int(np.sum(deltas > FAR_THRESH))
        verdict = "ALIGNED" if max_d < CLOSE_THRESH else ("DIVERGED" if n_far > 0 else "BORDERLINE")

        print(f"{fi:<6}{n_cpu:<7}{n_mlx:<7}{max_d:<14.2e}{mean_d:<14.2e}"
              f"{median_d:<14.2e}{n_close:<9}{n_far:<7}{verdict}")

        rows.append({
            "feat": fi, "n_cpu": n_cpu, "n_mlx": n_mlx,
            "max_delta": max_d, "mean_delta": mean_d, "median_delta": median_d,
            "n_close": n_close, "n_far": n_far,
        })

        # Collect per-border top-5 candidates
        for j, (b_mlx, d) in enumerate(zip(mlx, deltas)):
            # Find the nearest CPU border value
            idx = np.searchsorted(cpu, b_mlx)
            if idx > 0 and (idx >= len(cpu) or abs(b_mlx - cpu[idx - 1]) <= abs(b_mlx - cpu[idx])):
                nearest_cpu = float(cpu[idx - 1])
            elif idx < len(cpu):
                nearest_cpu = float(cpu[idx])
            else:
                nearest_cpu = float("nan")
            all_top5.append((d, fi, float(b_mlx), nearest_cpu))

    return rows, all_top5


# ---------------------------------------------------------------------------
# Step 4: feat_0 used-border alignment
# ---------------------------------------------------------------------------

def feat0_used_borders(mlx_borders: dict[str, list[float]]) -> None:
    """For each CPU-stored border from F2 iter2 splits, find nearest MLX border."""
    splits_path = F2_DATA_DIR / "cpu_iter2_splits.json"
    if not splits_path.exists():
        print(f"\n[step4] {splits_path} not found; skipping.", file=sys.stderr)
        return

    with open(splits_path) as f:
        splits = json.load(f)

    # Collect all borders for feat=0 across all depths
    feat0_used: list[tuple[int, float]] = []
    for depth_key, info in splits.items():
        if isinstance(info, dict) and info.get("feat") == 0:
            b = info.get("border")
            if b is not None:
                d = int(depth_key.replace("depth_", ""))
                feat0_used.append((d, float(b)))

    mlx_feat0 = np.array(sorted(mlx_borders.get("feat_0", [])), dtype=np.float64)

    out_path = DATA_DIR / "feat0_used_borders_alignment.csv"
    print(f"\n[step4] Feat 0 used-border alignment ({len(feat0_used)} borders from F2 iter2):")
    print(f"  {'depth':<8}{'cpu_border':<22}{'nearest_mlx_border':<22}{'delta'}")
    rows = []
    for depth, cpu_b in sorted(feat0_used):
        d = nearest_delta(cpu_b, mlx_feat0)
        # Find nearest MLX border value
        idx = np.searchsorted(mlx_feat0, cpu_b)
        if idx > 0 and (idx >= len(mlx_feat0) or
                        abs(cpu_b - mlx_feat0[idx - 1]) <= abs(cpu_b - mlx_feat0[idx])):
            nearest_mlx = float(mlx_feat0[idx - 1])
        elif idx < len(mlx_feat0):
            nearest_mlx = float(mlx_feat0[idx])
        else:
            nearest_mlx = float("nan")
        print(f"  d={depth:<6} {cpu_b:<22.10f} {nearest_mlx:<22.10f} {d:.2e}")
        rows.append({"depth": depth, "cpu_border": cpu_b, "nearest_mlx_border": nearest_mlx, "delta": d})

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["depth", "cpu_border", "nearest_mlx_border", "delta"])
        w.writeheader()
        w.writerows(rows)
    print(f"[step4] Saved -> {out_path}")

    # Also check F2 MLX iter2 splits for comparison
    mlx_splits_path = F2_DATA_DIR / "mlx_iter2_splits.json"
    if mlx_splits_path.exists():
        with open(mlx_splits_path) as f:
            mlx_splits = json.load(f)
        print(f"\n[step4] F2 MLX iter2 splits for reference:")
        for dk, info in sorted(mlx_splits.items()):
            if isinstance(info, dict):
                print(f"  {dk}: feat={info.get('feat')}, bin_idx={info.get('bin_idx')}, border={info.get('border'):.10f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    cpu_path = DATA_DIR / "cpu_borders_full.json"
    mlx_path = DATA_DIR / "mlx_borders_full.json"

    for p in [cpu_path, mlx_path]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run extract_borders.py first.", file=sys.stderr)
            sys.exit(1)

    with open(cpu_path) as f:
        cpu_borders = json.load(f)
    with open(mlx_path) as f:
        mlx_borders = json.load(f)

    print(f"[compare_borders] CPU features loaded: {len(cpu_borders)}")
    print(f"[compare_borders] MLX features loaded: {len(mlx_borders)}")

    # Step 3
    rows, all_top5 = compare_all_features(cpu_borders, mlx_borders)

    # Headline aggregate
    n_aligned = sum(1 for r in rows if r["max_delta"] is not None and r["max_delta"] < CLOSE_THRESH)
    n_diverged = sum(1 for r in rows if r["n_far"] is not None and r["n_far"] > 0)
    print(f"\n[HEADLINE] {n_aligned}/20 features have ALL MLX borders aligned within {CLOSE_THRESH:.0e}")
    print(f"[HEADLINE] {n_diverged}/20 features have at least one border pair with delta > {FAR_THRESH:.0e}")

    # Top-5 largest deltas
    all_top5_sorted = sorted(all_top5, reverse=True)[:5]
    print(f"\n[TOP-5 LARGEST DELTAS]")
    print(f"  {'rank':<6}{'feat':<7}{'mlx_border':<22}{'nearest_cpu_border':<22}{'delta'}")
    for rank, (d, fi, mlx_b, cpu_b) in enumerate(all_top5_sorted, 1):
        print(f"  {rank:<6}{fi:<7}{mlx_b:<22.10f}{cpu_b:<22.10f}{d:.4e}")

    # Feat 0 and feat 1 detail
    for fi_name in ("feat_0", "feat_1"):
        r = next((r for r in rows if r["feat"] == int(fi_name.split("_")[1])), None)
        if r and r["max_delta"] is not None:
            verdict = "ALIGNED" if r["max_delta"] < CLOSE_THRESH else (
                "DIVERGED" if r["n_far"] > 0 else "BORDERLINE")
            print(f"\n[{fi_name}] max_delta={r['max_delta']:.2e}  n_close={r['n_close']}/127  "
                  f"n_far={r['n_far']}  verdict={verdict}")

    # Save comparison CSV
    out_csv = DATA_DIR / "border_comparison.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["feat", "n_cpu", "n_mlx", "max_delta", "mean_delta",
                                           "median_delta", "n_close", "n_far"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[compare_borders] Saved comparison table -> {out_csv}")

    # Step 4
    feat0_used_borders(mlx_borders)

    # Step 4b: overall verdict
    print("\n" + "=" * 80)
    max_delta_all = max((r["max_delta"] for r in rows if r["max_delta"] is not None), default=float("nan"))
    n_far_total = sum(r["n_far"] for r in rows if r["n_far"] is not None)
    print(f"OVERALL max_delta across all 20 features: {max_delta_all:.4e}")
    print(f"OVERALL n_far (delta > {FAR_THRESH:.0e}) across all 20 features: {n_far_total}")
    if max_delta_all < 1e-3:
        print("VERDICT: Q-ALIGNED — borders agree everywhere (max delta < 1e-3).")
        print("  Granularity is NOT the divergence mechanism. The d=4-5 drift comes from elsewhere.")
    else:
        print("VERDICT: Q-DIVERGED — borders genuinely differ (max delta > 1e-2 on multiple features).")
        print("  Granularity-via-different-quantization is a candidate mechanism.")
        print("  Recommend porting CatBoost's full border_count=128 to MLX as a proper fix.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
