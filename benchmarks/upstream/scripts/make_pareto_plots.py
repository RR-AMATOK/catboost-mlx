"""Generate per-dataset Pareto-frontier scatter plots from results JSONs.

Usage:
    python -m benchmarks.upstream.scripts.make_pareto_plots \\
        --results-dir benchmarks/upstream/results \\
        --out-dir docs/benchmarks/plots

For each dataset present in the results, produces a single PNG: x = train
wall-clock (log scale), y = test metric (linear). One point per framework
(error bars across seeds when present). Frameworks on the Pareto frontier
(non-dominated by any other framework's metric × wall-clock pair) are
highlighted; dominated points get a faded marker.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_grouped(results_dir: Path) -> Dict[str, Dict[str, List[dict]]]:
    """Return {dataset: {framework: [records...]}}."""
    out: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for p in sorted(results_dir.glob("*.json")):
        try:
            r = json.loads(p.read_text())
            out[r["dataset"]][r["framework"]].append(r)
        except Exception as exc:
            print(f"  skipping {p.name}: {exc}", file=sys.stderr)
    return out


def pareto_indices(times: np.ndarray, metrics: np.ndarray, *, lower_is_better: bool) -> np.ndarray:
    """Boolean mask: which (time, metric) pairs are Pareto-optimal?
    A point is dominated if some other point has both lower time AND
    metric-better-or-equal AND any of those strict.
    """
    n = times.size
    on_frontier = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            t_better = times[j] <= times[i]
            if lower_is_better:
                m_better = metrics[j] <= metrics[i]
                strict = (times[j] < times[i]) or (metrics[j] < metrics[i])
            else:
                m_better = metrics[j] >= metrics[i]
                strict = (times[j] < times[i]) or (metrics[j] > metrics[i])
            if t_better and m_better and strict:
                on_frontier[i] = False
                break
    return on_frontier


# Stable per-framework styling
COLORS = {
    "lightgbm":            "#4D9DE0",
    "xgboost":             "#E15554",
    "catboost_cpu":        "#7768AE",
    "catboost_mlx":        "#3BB273",
    "catboost_mlx_no_cat": "#A6CFA1",
}
MARKERS = {
    "lightgbm":            "s",
    "xgboost":             "^",
    "catboost_cpu":        "D",
    "catboost_mlx":        "o",
    "catboost_mlx_no_cat": "v",
}


def plot_dataset(dataset: str, frameworks: Dict[str, List[dict]], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5))

    points = []
    for fw, recs in frameworks.items():
        if not recs:
            continue
        times = [r["train_seconds"] for r in recs]
        metrics = [r["metric_value"] for r in recs]
        points.append((fw, np.mean(times), np.std(times) if len(times) > 1 else 0,
                       np.mean(metrics), np.std(metrics) if len(metrics) > 1 else 0,
                       recs[0].get("metric_name", ""), recs[0].get("notes", "")))

    if not points:
        return

    metric_name = points[0][5]
    lower_better = metric_name.lower() in {"logloss", "rmse", "mae"}
    times_arr = np.array([p[1] for p in points])
    metrics_arr = np.array([p[3] for p in points])
    on_pareto = pareto_indices(times_arr, metrics_arr, lower_is_better=lower_better)

    for i, (fw, t, t_err, m, m_err, _, _) in enumerate(points):
        color = COLORS.get(fw, "#888888")
        marker = MARKERS.get(fw, "o")
        alpha = 1.0 if on_pareto[i] else 0.45
        edgecolor = "black" if on_pareto[i] else "none"
        ax.errorbar(
            t, m, xerr=t_err, yerr=m_err,
            fmt=marker, markersize=12, markeredgewidth=1.2,
            color=color, ecolor=color, capsize=3,
            markerfacecolor=color, markeredgecolor=edgecolor, alpha=alpha,
            label=f"{fw}{'  (Pareto)' if on_pareto[i] else ''}",
            zorder=3 if on_pareto[i] else 2,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Train wall-clock (s)  ←  faster")
    metric_arrow = "lower is better" if lower_better else "higher is better"
    ax.set_ylabel(f"Test {metric_name}  ({metric_arrow})")
    ax.set_title(f"{dataset} — Pareto frontier on Apple Silicon")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    hw = points[0][6] if False else (
        list(frameworks.values())[0][0].get("hardware", "")
    )
    fig.text(0.99, 0.01, hw, ha="right", va="bottom", fontsize=7,
             color="#666", style="italic")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path.name}  ({len(points)} frameworks; "
          f"{int(on_pareto.sum())} on Pareto frontier)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default=str(Path(__file__).resolve().parents[1] / "results"))
    ap.add_argument("--out-dir", default="docs/benchmarks/plots")
    args = ap.parse_args()

    grouped = load_grouped(Path(args.results_dir))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset, frameworks in sorted(grouped.items()):
        plot_dataset(dataset, frameworks, out_dir / f"{dataset}_pareto.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
