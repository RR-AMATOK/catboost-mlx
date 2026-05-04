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
from typing import Optional

from ._common import (
    DatasetMeta,
    cache_dir,
    download_if_missing,
    gunzip_if_missing,
)

NAME = "higgs"  # default; overridable via --name (e.g. higgs_1m, higgs_11m)
URL_GZ = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"

N_TOTAL = 11_000_000
N_TRAIN = 10_500_000
N_TEST = 500_000
N_FEATURES = 28


def _materialize_subset(
    higgs_csv: Path, out_dir: Path, n_train: int, n_test: int,
) -> tuple[Path, Path]:
    """Write train.csv (first n_train rows) + test.csv (next n_test rows)
    with a header row. Used by both full-scale and subset materialization.
    """
    train_csv = out_dir / "train.csv"
    test_csv = out_dir / "test.csv"
    feature_names = [f"f{i}" for i in range(N_FEATURES)]
    header = "target," + ",".join(feature_names) + "\n"

    print(f"  Writing train.csv (first {n_train:,} rows)...", file=sys.stderr)
    with open(higgs_csv) as src, open(train_csv, "w") as f:
        f.write(header)
        for i, line in enumerate(src):
            if i >= n_train:
                break
            f.write(line)

    print(f"  Writing test.csv (next {n_test:,} rows)...", file=sys.stderr)
    with open(higgs_csv) as src, open(test_csv, "w") as f:
        f.write(header)
        for i, line in enumerate(src):
            if i < n_train:
                continue
            if i >= n_train + n_test:
                break
            f.write(line)

    return train_csv, test_csv


def prepare(
    out_dir: Path, *,
    materialize: bool = False,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    name: str = NAME,
) -> DatasetMeta:
    """Prepare Higgs at full scale (default) or a subset (--n-train / --n-test).

    Subset mode is useful for smoke-testing the runner pipeline on a tractable
    fraction of the data; for the canonical training_speed/ comparison numbers
    use the default (materialize the full 10.5M / 500k split).
    """
    higgs_gz = out_dir / "HIGGS.csv.gz"
    higgs_csv = out_dir / "HIGGS.csv"

    download_if_missing(URL_GZ, higgs_gz, label="HIGGS.csv.gz (~2.7 GB)")
    gunzip_if_missing(higgs_gz, higgs_csv)

    actual_n_train = n_train if n_train is not None else N_TRAIN
    actual_n_test = n_test if n_test is not None else N_TEST
    is_subset = (actual_n_train, actual_n_test) != (N_TRAIN, N_TEST)

    if actual_n_train + actual_n_test > N_TOTAL:
        raise ValueError(
            f"n_train + n_test = {actual_n_train + actual_n_test:,} exceeds the "
            f"Higgs total of {N_TOTAL:,} rows."
        )

    split_meta = {
        "source_csv": str(higgs_csv),
        "header_present": False,
        "target_col": 0,
        "feature_cols": list(range(1, N_FEATURES + 1)),
        "train_start": 0,
        "train_end": actual_n_train,
        "test_start": actual_n_train,
        "test_end": actual_n_train + actual_n_test,
        "is_subset": is_subset,
    }
    (out_dir / "split.json").write_text(json.dumps(split_meta, indent=2))

    if materialize:
        _materialize_subset(higgs_csv, out_dir, actual_n_train, actual_n_test)

    suffix = " (SUBSET)" if is_subset else ""
    meta = DatasetMeta(
        name=name,
        task="classification",
        metric="logloss",
        n_train=actual_n_train,
        n_test=actual_n_test,
        n_features=N_FEATURES,
        cat_indices=[],
        target_col=0,
        notes=(
            f"UCI Higgs Boson{suffix}, binary classification, all numeric. "
            f"Split: first {actual_n_train:,} rows train, next {actual_n_test:,} rows test. "
            f"Canonical full-scale upstream split is {N_TRAIN:,} train / {N_TEST:,} test; "
            f"this prep is{'' if not is_subset else ' a subset of'} that."
        ),
        source_url=URL_GZ,
    )
    meta.write(out_dir)

    print(f"Higgs prepared{suffix}:")
    print(f"  out: {out_dir}")
    print(f"  train: {meta.n_train:,} rows  test: {meta.n_test:,} rows  features: {meta.n_features}")
    if not materialize:
        print(f"  Streaming mode (split.json offsets); pass --materialize to write CSVs.")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--materialize", action="store_true",
                        help="Write actual train.csv/test.csv files (~7 GB on disk for full-scale).")
    parser.add_argument("--n-train", type=int, default=None,
                        help=f"Train rows (default: {N_TRAIN:,}). Use a smaller value for "
                             "a tractable subset; e.g. --n-train 1000000 --n-test 100000.")
    parser.add_argument("--n-test", type=int, default=None,
                        help=f"Test rows (default: {N_TEST:,}).")
    parser.add_argument("--name", default=NAME,
                        help=f"Override cache subdir name (default: {NAME}). "
                             "Use 'higgs_1m' for the 1M subset or 'higgs_11m' for full scale.")
    args = parser.parse_args()

    out_dir = cache_dir(args.name, root=Path(args.cache_root) if args.cache_root else None)
    try:
        prepare(out_dir, materialize=args.materialize,
                n_train=args.n_train, n_test=args.n_test, name=args.name)
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
