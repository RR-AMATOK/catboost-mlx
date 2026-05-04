"""Prepare all datasets for the CUDA cross-class benchmark.

Adult, Higgs-1M, Higgs-11M are auto-downloading. Epsilon and Amazon need
manual data acquisition (the adapter prints the recipe). Run this before
run.py.

By default, prepared datasets are written to ./cache/ inside the bundle
(the same place bundle_cache.py copies pre-prepared datasets to). Use
--cache-root to override.

Run: python prepare.py
     python prepare.py --skip higgs_11m epsilon amazon  # to skip slow ones
     python prepare.py --only higgs_11m                  # one missing dataset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python prepare.py` from the bundle root by ensuring the bundle
# directory is on sys.path (so `adapters.X` imports resolve).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters._common import cache_dir
from adapters import adult, higgs, epsilon, amazon

# Default cache root is the bundle-local ./cache/ directory, NOT the
# global ~/.cache/catboost-mlx-benchmarks/. This keeps everything
# self-contained and disposable.
DEFAULT_CACHE_ROOT = Path(__file__).resolve().parent / "cache"


DATASETS = {
    "adult":     "Auto-downloads ~3.8 MB from UCI.",
    "higgs_1m":  "Auto-downloads ~2.7 GB compressed; 1M-row subset of HIGGS.csv.",
    "higgs_11m": "Auto-downloads ~2.7 GB compressed; full 10.5M/500k split. ~7 GB on disk after decompress.",
    "epsilon":   "AUTO-DOWNLOADS ~15 GB compressed from libsvm (slow, 20-60 min). Decompresses to ~20 GB, writes ~14 GB CSVs.",
    "amazon":    "MANUAL: requires Kaggle CLI auth + competition acceptance. Adapter prints recipe.",
}


def prep_adult(cache_root: Path):
    out_dir = cache_dir("adult", root=cache_root)
    return adult.prepare(out_dir)


def prep_higgs_1m(cache_root: Path):
    """1M train + 100k test subset of HIGGS for cross-class anchor."""
    out_dir = cache_dir("higgs_1m", root=cache_root)
    return higgs.prepare(out_dir, materialize=True,
                         n_train=1_000_000, n_test=100_000,
                         name="higgs_1m")


def prep_higgs_11m(cache_root: Path):
    """Full 10.5M / 500k upstream split."""
    out_dir = cache_dir("higgs_11m", root=cache_root)
    return higgs.prepare(out_dir, materialize=True, name="higgs_11m")


def prep_epsilon(cache_root: Path):
    out_dir = cache_dir("epsilon", root=cache_root)
    # Auto-downloads ~15 GB compressed from libsvm, decompresses to ~20 GB,
    # then writes train.csv + test.csv (~14 GB total). This is slow.
    return epsilon.prepare(out_dir, materialize=True)


def prep_amazon(cache_root: Path):
    out_dir = cache_dir("amazon", root=cache_root)
    return amazon.prepare(out_dir)


PREP_FNS = {
    "adult":     prep_adult,
    "higgs_1m":  prep_higgs_1m,
    "higgs_11m": prep_higgs_11m,
    "epsilon":   prep_epsilon,
    "amazon":    prep_amazon,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip", nargs="*", default=[],
                    help="Dataset names to skip (e.g. --skip epsilon amazon).")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Only prep these datasets (overrides --skip).")
    ap.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT),
                    help=f"Cache root directory (default: {DEFAULT_CACHE_ROOT}).")
    args = ap.parse_args()

    if args.only:
        targets = [d for d in args.only if d in DATASETS]
    else:
        targets = [d for d in DATASETS if d not in args.skip]

    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    print(f"=== Dataset prep ({len(targets)} datasets) ===")
    print(f"  Cache root: {cache_root}")
    for name in targets:
        print(f"  {name}: {DATASETS[name]}")
    print()

    failures = []
    for name in targets:
        # Skip prep if dataset already exists in cache (bundled or previously prepped)
        if (cache_root / name / "meta.json").exists():
            print(f"\n=== {name} already in cache; skipping ===")
            continue
        print(f"\n=== preparing {name} ===")
        try:
            PREP_FNS[name](cache_root)
            print(f"  {name}: OK")
        except Exception as exc:
            print(f"  {name}: FAILED — {exc}", file=sys.stderr)
            failures.append((name, str(exc)))

    print()
    print(f"=== Summary: {len(targets) - len(failures)}/{len(targets)} OK ===")
    for name, msg in failures:
        print(f"  FAILED: {name}: {msg}", file=sys.stderr)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
