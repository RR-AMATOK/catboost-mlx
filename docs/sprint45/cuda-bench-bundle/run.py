"""Run the CatBoost-CUDA cross-class sweep.

Mirrors the M3 Max iter-grid methodology used in S44-T2:
  Adult, Higgs-1M, Epsilon, Amazon at iter ∈ {200, 500, 1000, 2000} × seeds {42, 43, 44}
  Higgs-11M at iter=200 only (huge dataset; ~hours per run)

Total: 48 + 3 = 51 runs. Estimated wall-time depends on GPU; for an A100
expect ~1-2 hours total; for a 1080 Ti expect 4-8 hours.

Output: JSONs in benchmarks/cuda/results/ following the same schema as the
M3 Max sweep, so they can be aggregated against M3 Max numbers directly.

Run: python run.py
     python run.py --skip higgs_11m epsilon  # skip slow workloads
     python run.py --only higgs_1m           # one dataset only (smoke test)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


# Mirrors S44-T2 methodology — see docs/sprint44/sprint-plan.md and
# docs/benchmarks/v0.6.0-pareto.md §2.
DEFAULT_DATASETS = ["adult", "higgs_1m", "epsilon", "amazon"]
DEFAULT_ITERS = [200, 500, 1000, 2000]
HIGGS_11M_ITERS = [200]  # Higgs-11M only at iter=200 (huge wall-clock at 1000+)
DEFAULT_SEEDS = [42, 43, 44]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_ROOT = Path(__file__).resolve().parent / "cache"


def cell_filename(dataset: str, iter_n: int, seed: int) -> str:
    """Match the M3 Max naming convention so cross-class aggregation works."""
    if iter_n == 200:
        return f"{dataset}_catboost_cuda_{seed}.json"
    return f"{dataset}_iter{iter_n}_catboost_cuda_{seed}.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip", nargs="*", default=[],
                    help="Datasets to skip (e.g. --skip higgs_11m).")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Only run these datasets (overrides --skip).")
    ap.add_argument("--iters", nargs="*", type=int, default=None,
                    help="Override iter grid (default: 200 500 1000 2000).")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="Override seeds (default: 42 43 44).")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--cache-root", default=str(CACHE_ROOT),
                    help=f"Cache root directory (default: {CACHE_ROOT}).")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.only:
        datasets = [d for d in args.only if d in DEFAULT_DATASETS + ["higgs_11m"]]
    else:
        datasets = [d for d in DEFAULT_DATASETS if d not in args.skip]
        if "higgs_11m" not in args.skip and (args.only is None or "higgs_11m" in args.only):
            include_11m = True
        else:
            include_11m = False
    iters = args.iters if args.iters else DEFAULT_ITERS
    seeds = args.seeds if args.seeds else DEFAULT_SEEDS
    include_11m = (args.only is None and "higgs_11m" not in args.skip) or \
                  (args.only is not None and "higgs_11m" in args.only)

    n_main = len(datasets) * len(iters) * len(seeds)
    n_11m = (len(HIGGS_11M_ITERS) * len(seeds)) if include_11m else 0
    total = n_main + n_11m

    print("=== CatBoost-CUDA cross-class sweep ===")
    print(f"  Datasets:    {datasets}")
    print(f"  Iters:       {iters}")
    print(f"  Seeds:       {seeds}")
    print(f"  Higgs-11M:   {'YES (iter=200 only)' if include_11m else 'NO'}")
    print(f"  Cache root:  {args.cache_root}")
    print(f"  Results dir: {results_dir}")
    print(f"  Total runs:  {total}")
    print()

    done = 0
    cached = 0
    failed = 0
    t_start = time.time()

    def run_cell(dataset: str, iter_n: int, seed: int) -> None:
        nonlocal done, cached, failed
        done += 1
        out = results_dir / cell_filename(dataset, iter_n, seed)
        if out.exists():
            print(f"[{done}/{total}] {dataset}/iter={iter_n}/seed={seed}  CACHED")
            cached += 1
            return
        print(f"[{done}/{total}] {dataset}/iter={iter_n}/seed={seed}  RUNNING")
        t0 = time.time()
        try:
            subprocess.run([
                sys.executable, "-m", "scripts.run_catboost_cuda",
                "--dataset", dataset,
                "--seed", str(seed),
                "--iterations", str(iter_n),
                "--results-dir", str(results_dir),
                "--cache-root", args.cache_root,
            ], check=True)
            print(f"           ok in {time.time()-t0:.1f}s")
        except subprocess.CalledProcessError as exc:
            print(f"           FAILED: {exc}", file=sys.stderr)
            failed += 1

    # Main 4 datasets at full iter grid
    for iter_n in iters:
        for dataset in datasets:
            for seed in seeds:
                run_cell(dataset, iter_n, seed)

    # Higgs-11M only at iter=200
    if include_11m:
        for iter_n in HIGGS_11M_ITERS:
            for seed in seeds:
                run_cell("higgs_11m", iter_n, seed)

    elapsed = time.time() - t_start
    print()
    print(f"=== Done in {elapsed/60:.1f} min ===")
    print(f"  Total: {total}  ok: {done - cached - failed}  cached: {cached}  failed: {failed}")
    print(f"  Results: {results_dir}/*.json")
    print(f"  Bring back the entire {results_dir} folder + hardware.txt.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
