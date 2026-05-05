"""Bundle the user's prepared datasets into ./cache/ for USB transfer.

Run this on the Mac BEFORE copying the bundle to USB. Reads from the local
~/.cache/catboost-mlx-benchmarks/ and writes a trimmed copy to
cuda-bench-bundle/cache/. Strips raw download files (HIGGS.csv, *.libsvm)
since the runner only needs the materialized {train,test}.csv + meta.json.

Subdirs are renamed to match the v0.6.0 dataset naming convention:
  ~/.cache/.../higgs/   (1M subset)  →  cache/higgs_1m/
  ~/.cache/.../epsilon/              →  cache/epsilon/
  ~/.cache/.../adult/                →  cache/adult/
  ~/.cache/.../amazon/               →  cache/amazon/

Higgs-11M is NOT bundled — it requires re-prep (no current cache subdir).
The Windows side will auto-download via prepare.py if requested.

Run: python bundle_cache.py                  # bundle everything available
     python bundle_cache.py --skip epsilon   # leave out the 20 GB Epsilon data
     python bundle_cache.py --only adult amazon higgs_1m  # minimal subset
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


# Mapping from local cache subdir name -> bundle/cache/ subdir name
DEFAULT_MAPPING = {
    "adult":   "adult",
    "amazon":  "amazon",
    "higgs":   "higgs_1m",   # local cache has the 1M subset under "higgs/"
    "epsilon": "epsilon",
}

# Files we WANT to copy from each dataset cache subdir.
WANTED = {"meta.json", "train.csv", "test.csv"}

# Files we explicitly STRIP (raw downloads — runner doesn't need them).
STRIP = {"HIGGS.csv", "HIGGS.csv.gz", "epsilon_normalized", "epsilon_normalized.t",
         "split.json"}


def humansize(bytes_: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if bytes_ < 1024:
            return f"{bytes_:.1f} {unit}"
        bytes_ /= 1024
    return f"{bytes_:.1f} PB"


def bundle_one(src_dir: Path, dst_dir: Path) -> tuple[int, int]:
    """Copy WANTED files from src to dst. Returns (n_files, total_bytes)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    n_files = 0
    total_bytes = 0
    for item in sorted(src_dir.iterdir()):
        if item.name in STRIP or item.name.startswith("."):
            continue
        if item.name not in WANTED:
            print(f"    skipping {item.name} (not in WANTED set)")
            continue
        if item.is_file():
            sz = item.stat().st_size
            print(f"    {item.name} ({humansize(sz)})...", end="", flush=True)
            shutil.copy2(item, dst_dir / item.name)
            n_files += 1
            total_bytes += sz
            print(" OK")
    return n_files, total_bytes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache-source",
                    default=str(Path.home() / ".cache" / "catboost-mlx-benchmarks"),
                    help="Local prepared-cache root (default: ~/.cache/catboost-mlx-benchmarks).")
    ap.add_argument("--bundle-cache",
                    default=str(Path(__file__).resolve().parent / "cache"),
                    help="Output cache directory inside the bundle (default: ./cache).")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="Bundle-side dataset names to skip (e.g. --skip epsilon).")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Bundle-side dataset names to include (overrides --skip).")
    args = ap.parse_args()

    src_root = Path(args.cache_source)
    dst_root = Path(args.bundle_cache)

    if not src_root.exists():
        print(f"FAIL: source cache {src_root} does not exist. "
              f"Prepare datasets first via the upstream adapters.", file=sys.stderr)
        return 1

    # Determine which datasets to bundle.
    targets = {}  # bundle_name -> source_subdir
    for src_name, bundle_name in DEFAULT_MAPPING.items():
        if args.only and bundle_name not in args.only:
            continue
        if not args.only and bundle_name in args.skip:
            continue
        src_dir = src_root / src_name
        if not src_dir.exists():
            print(f"  WARN: {src_dir} does not exist; skipping bundle of {bundle_name}")
            continue
        targets[bundle_name] = src_dir

    if not targets:
        print("Nothing to bundle (no matching cache subdirs found).", file=sys.stderr)
        return 1

    print(f"=== Bundling {len(targets)} dataset(s) into {dst_root} ===\n")

    grand_total_bytes = 0
    for bundle_name, src_dir in targets.items():
        dst_dir = dst_root / bundle_name
        print(f"-> {bundle_name}  (from {src_dir} → {dst_dir})")
        n, b = bundle_one(src_dir, dst_dir)
        print(f"   {n} file(s), {humansize(b)} total\n")
        grand_total_bytes += b

    print(f"=== Done: {len(targets)} dataset(s), {humansize(grand_total_bytes)} total ===")
    print(f"  Bundle cache root: {dst_root}")
    print(f"  Next: copy the entire bundle directory to USB.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
