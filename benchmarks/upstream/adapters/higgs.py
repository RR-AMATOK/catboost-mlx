"""Adapter for the UCI Higgs Boson dataset.

Source: https://archive.ics.uci.edu/ml/datasets/HIGGS
File:   https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
        (~2.7 GB compressed, ~7.5 GB uncompressed; downloaded once and cached)

Task: binary classification. Used by upstream catboost/benchmarks
`training_speed/` as the canonical large-but-runnable GBDT benchmark.

11,000,000 rows × 28 numeric features. Standard split (matching upstream
practice and most published benchmarks): first 10,500,000 rows train, last
500,000 rows test.

NO categorical features. cat_indices = [].

This adapter is download-on-first-run. Subsequent invocations reuse the
cached HIGGS.csv. **The download is ~2.7 GB and the decompressed CSV is
~7.5 GB on disk — confirm you have headroom before running.** UCI rate
limits can also throttle large downloads; if the download stalls, retry.

Output: <cache>/{train,test}.csv plus meta.json.
The output train.csv duplicates the source (just a slice), so for storage
we instead write a small "split.json" that records (train_start, train_end,
test_start, test_end) into the cache and let runner scripts memory-map
HIGGS.csv with those slices. Set --materialize to write actual CSVs.

Run: python -m benchmarks.upstream.adapters.higgs
     python -m benchmarks.upstream.adapters.higgs --materialize  # ~7 GB on disk
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._common import (
    DatasetMeta,
    cache_dir,
    download_if_missing,
    gunzip_if_missing,
)

NAME = "higgs"
URL_GZ = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"

N_TOTAL = 11_000_000
N_TRAIN = 10_500_000
N_TEST = 500_000
N_FEATURES = 28


def _materialize_split(higgs_csv: Path, out_dir: Path) -> tuple[Path, Path]:
    """Write train.csv and test.csv as actual files (~7 GB on disk).
    Most users should NOT do this — runner scripts can stream from HIGGS.csv
    directly using split.json offsets. This is here for completeness.
    """
    train_csv = out_dir / "train.csv"
    test_csv = out_dir / "test.csv"

    print(f"  Writing train.csv (first {N_TRAIN:,} rows)...", file=sys.stderr)
    with open(higgs_csv) as src, open(train_csv, "w") as f:
        # HIGGS.csv has no header; csv_train --target-col 0 handles raw layout.
        # Add a header so the runner scripts can introspect it consistently.
        feature_names = [f"f{i}" for i in range(N_FEATURES)]
        f.write("target," + ",".join(feature_names) + "\n")
        for i, line in enumerate(src):
            if i >= N_TRAIN:
                break
            f.write(line)

    print(f"  Writing test.csv (next {N_TEST:,} rows)...", file=sys.stderr)
    with open(higgs_csv) as src, open(test_csv, "w") as f:
        feature_names = [f"f{i}" for i in range(N_FEATURES)]
        f.write("target," + ",".join(feature_names) + "\n")
        for i, line in enumerate(src):
            if i < N_TRAIN:
                continue
            if i >= N_TRAIN + N_TEST:
                break
            f.write(line)

    return train_csv, test_csv


def prepare(out_dir: Path, *, materialize: bool = False) -> DatasetMeta:
    higgs_gz = out_dir / "HIGGS.csv.gz"
    higgs_csv = out_dir / "HIGGS.csv"

    download_if_missing(URL_GZ, higgs_gz, label="HIGGS.csv.gz (~2.7 GB)")
    gunzip_if_missing(higgs_gz, higgs_csv)

    # Verify the decompressed file has the expected row count
    print(f"  Verifying row count of {higgs_csv.name}...", file=sys.stderr)
    n_rows = 0
    with open(higgs_csv) as f:
        for _ in f:
            n_rows += 1
            if n_rows > N_TOTAL:
                break
    if n_rows < N_TOTAL:
        raise RuntimeError(
            f"HIGGS.csv has only {n_rows:,} rows; expected {N_TOTAL:,}. "
            f"Re-download the gz."
        )

    split_meta = {
        "source_csv": str(higgs_csv),
        "header_present": False,
        "target_col": 0,
        "feature_cols": list(range(1, N_FEATURES + 1)),
        "train_start": 0,
        "train_end": N_TRAIN,
        "test_start": N_TRAIN,
        "test_end": N_TRAIN + N_TEST,
    }
    (out_dir / "split.json").write_text(json.dumps(split_meta, indent=2))

    if materialize:
        _materialize_split(higgs_csv, out_dir)

    meta = DatasetMeta(
        name=NAME,
        task="classification",
        metric="logloss",
        n_train=N_TRAIN,
        n_test=N_TEST,
        n_features=N_FEATURES,
        cat_indices=[],
        target_col=0,
        notes=(
            "UCI Higgs Boson, binary classification, all numeric. Standard split: "
            f"first {N_TRAIN:,} rows train, next {N_TEST:,} rows test. Source CSV is "
            f"~7.5 GB uncompressed; runner scripts should stream rather than fully "
            f"materialize unless --materialize was passed."
        ),
        source_url=URL_GZ,
    )
    meta.write(out_dir)

    print(f"Higgs prepared:")
    print(f"  out: {out_dir}")
    print(f"  train: {meta.n_train:,} rows  test: {meta.n_test:,} rows  features: {meta.n_features}")
    if not materialize:
        print(f"  Streaming mode (split.json offsets); pass --materialize to write actual {N_FEATURES+1}-col CSVs.")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--materialize", action="store_true",
                        help="Write actual train.csv/test.csv files (~7 GB on disk).")
    args = parser.parse_args()

    out_dir = cache_dir(NAME, root=Path(args.cache_root) if args.cache_root else None)
    try:
        prepare(out_dir, materialize=args.materialize)
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
