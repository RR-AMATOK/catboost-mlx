"""Adapter for the Epsilon dataset (Pascal Large Scale Learning Challenge).

Source: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
        Listed as 'epsilon' under "Pascal Challenge"
Files:  epsilon_normalized.bz2 (~12 GB compressed; train, ~400k × 2000)
        epsilon_normalized.t.bz2 (~3 GB compressed; test, ~100k × 2000)

Task: binary classification, all numeric. Used by upstream
catboost/benchmarks `training_speed/` as the canonical large-feature-count
GBDT benchmark.

Standard split (per upstream practice): the libsvm files come pre-split
into train (epsilon_normalized) and test (epsilon_normalized.t).

400,000 train × 2,000 features
100,000 test  × 2,000 features
NO categorical features. cat_indices = [].

This adapter does NOT auto-download because the files are large and the
LIBSVM mirror does not serve well via urllib (slow, prone to truncation).
Place the bz2 files in the cache dir before running:

    # Manual download recipe
    DATA_DIR=~/.cache/catboost-mlx-benchmarks/epsilon
    mkdir -p "$DATA_DIR"
    cd "$DATA_DIR"
    curl -O https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2
    curl -O https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.bz2
    bunzip2 epsilon_normalized.bz2
    bunzip2 epsilon_normalized.t.bz2

After bunzip, this adapter parses the libsvm format and writes train.csv +
test.csv. **The output CSVs will be 6-8 GB each — make sure you have disk
headroom.** Same caveat as Higgs about runner scripts streaming the source
files rather than the materialized CSVs.

Run: python -m benchmarks.upstream.adapters.epsilon  # validates only
     python -m benchmarks.upstream.adapters.epsilon --materialize  # ~14 GB out
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ._common import (
    DatasetMeta,
    cache_dir,
    parse_libsvm,
    write_dense_csv,
)

NAME = "epsilon"
SOURCE_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html"

N_TRAIN = 400_000
N_TEST = 100_000
N_FEATURES = 2_000


def _check_raw(out_dir: Path) -> tuple[Path, Path]:
    train_path = out_dir / "epsilon_normalized"
    test_path = out_dir / "epsilon_normalized.t"
    missing = [p for p in (train_path, test_path) if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Epsilon raw files missing in {out_dir}:\n"
            + "\n".join(f"  {p.name}" for p in missing)
            + "\n\nDownload + decompress before running:\n"
            f"  cd '{out_dir}'\n"
            f"  curl -O https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2\n"
            f"  curl -O https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.bz2\n"
            f"  bunzip2 epsilon_normalized.bz2\n"
            f"  bunzip2 epsilon_normalized.t.bz2"
        )
    return train_path, test_path


def prepare(out_dir: Path, *, materialize: bool = False) -> DatasetMeta:
    train_raw, test_raw = _check_raw(out_dir)

    split_meta = {
        "source_train_libsvm": str(train_raw),
        "source_test_libsvm": str(test_raw),
        "target_col": 0,
        "n_features": N_FEATURES,
    }
    (out_dir / "split.json").write_text(json.dumps(split_meta, indent=2))

    if materialize:
        print(f"  Parsing {train_raw.name} ({N_TRAIN:,} rows × {N_FEATURES})...", file=sys.stderr)
        X_tr, y_tr = parse_libsvm(train_raw, N_FEATURES)
        # Convert -1/+1 labels to 0/1 (libsvm convention -> our convention)
        y_tr = ((y_tr + 1) // 2).astype(np.int64)
        train_csv = out_dir / "train.csv"
        write_dense_csv(train_csv, X_tr, y_tr)
        del X_tr, y_tr

        print(f"  Parsing {test_raw.name} ({N_TEST:,} rows × {N_FEATURES})...", file=sys.stderr)
        X_te, y_te = parse_libsvm(test_raw, N_FEATURES)
        y_te = ((y_te + 1) // 2).astype(np.int64)
        test_csv = out_dir / "test.csv"
        write_dense_csv(test_csv, X_te, y_te)
        del X_te, y_te

    meta = DatasetMeta(
        name=NAME,
        task="classification",
        metric="logloss",
        n_train=N_TRAIN,
        n_test=N_TEST,
        n_features=N_FEATURES,
        cat_indices=[],
        target_col=N_FEATURES,
        notes=(
            "Epsilon (Pascal LSC), libsvm format, all numeric. "
            "Source labels are -1/+1; this adapter remaps to 0/1 for "
            "compatibility with csv_train's logloss target convention. "
            "Materialized CSVs are ~7 GB each; runner scripts should prefer "
            "to stream from the libsvm files when possible."
        ),
        source_url=SOURCE_URL,
    )
    meta.write(out_dir)

    print(f"Epsilon prepared:")
    print(f"  out: {out_dir}")
    print(f"  train: {meta.n_train:,} rows  test: {meta.n_test:,} rows  features: {meta.n_features}")
    if not materialize:
        print(f"  Validation-only mode; pass --materialize to write CSVs (~14 GB on disk).")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--materialize", action="store_true",
                        help="Parse libsvm and write train.csv + test.csv (~14 GB).")
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
