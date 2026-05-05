"""Adapter for the Amazon Employee Access Challenge (Kaggle).

Source: https://www.kaggle.com/c/amazon-employee-access-challenge/data
Task: binary classification (predict whether an employee should be granted
resource access). Used in upstream catboost/benchmarks `quality_benchmarks/`
as "Amazon".

This dataset is the **canonical demonstration of the DEC-046 CTR rare-class
characterized gap**: 32k rows, 9 features, ALL categorical, with a heavily
imbalanced binary target (~94% positive). MLX's CTR encoding will produce
slightly different rare-class behavior than CatBoost-CPU here — this is the
"include with footnote" dataset per the S40 advisory-board synthesis.

Kaggle requires authentication, so the adapter does NOT auto-download.
Place the raw files in the cache dir before running:

    pip install kaggle  # one-time
    # Configure ~/.kaggle/kaggle.json with your API token
    kaggle competitions download -c amazon-employee-access-challenge \\
        -p ~/.cache/catboost-mlx-benchmarks/amazon
    cd ~/.cache/catboost-mlx-benchmarks/amazon
    unzip amazon-employee-access-challenge.zip

After unzip, the directory should contain `train.csv` (32769 rows × 10 cols).
The Kaggle test.csv has no labels (it's a competition test), so this adapter
splits the labelled train.csv into our own train/test (80/20, deterministic
seed=42) for benchmark evaluation.

Output: <cache>/{train,test}.csv with 9 cat features + target,
plus meta.json with cat_indices=[0..8] and target_col=9.

Run: python -m benchmarks.upstream.adapters.amazon
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ._common import (
    DatasetMeta,
    cache_dir,
    deterministic_split,
)

NAME = "amazon"
SOURCE_URL = "https://www.kaggle.com/c/amazon-employee-access-challenge/data"

# Original Kaggle column order: ACTION + 9 categorical role-related codes.
COLUMNS = [
    "ACTION",          # target (1 = granted, 0 = denied)
    "RESOURCE",        # 0
    "MGR_ID",          # 1
    "ROLE_ROLLUP_1",   # 2
    "ROLE_ROLLUP_2",   # 3
    "ROLE_DEPTNAME",   # 4
    "ROLE_TITLE",      # 5
    "ROLE_FAMILY_DESC",  # 6
    "ROLE_FAMILY",     # 7
    "ROLE_CODE",       # 8
]


def _check_raw(out_dir: Path) -> Path:
    """Locate the unzipped Kaggle train.csv in `out_dir`. Raise with clear
    instructions if missing."""
    candidate = out_dir / "train.csv"
    # Don't confuse with our own output — Kaggle train.csv has the ACTION column
    # in position 0; ours has it last. Distinguish by reading the header.
    if candidate.exists():
        with open(candidate) as f:
            header = f.readline().strip().split(",")
        if header and header[0] == "ACTION":
            return candidate

    # Look for the zip in the cache dir, even if not yet extracted.
    zip_candidates = list(out_dir.glob("amazon-employee-access-challenge*.zip"))
    if zip_candidates:
        raise RuntimeError(
            f"Found Kaggle zip at {zip_candidates[0]} but it is not extracted yet. "
            f"Run: cd '{out_dir}' && unzip {zip_candidates[0].name}"
        )

    raise RuntimeError(
        f"Amazon Kaggle data not found in {out_dir}. To download:\n"
        f"  pip install kaggle\n"
        f"  # Configure ~/.kaggle/kaggle.json with your Kaggle API token\n"
        f"  kaggle competitions download -c amazon-employee-access-challenge -p '{out_dir}'\n"
        f"  cd '{out_dir}' && unzip amazon-employee-access-challenge.zip\n"
        f"\nThen re-run this adapter."
    )


def _load_kaggle_train(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the Kaggle train.csv into (X, y) with ACTION as y."""
    rows = path.read_text().splitlines()[1:]  # skip header
    n = len(rows)
    X = np.empty((n, 9), dtype=np.int64)
    y = np.empty(n, dtype=np.int64)
    for i, line in enumerate(rows):
        cells = line.strip().split(",")
        if len(cells) != 10:
            raise RuntimeError(f"Unexpected row at line {i+2}: {len(cells)} cells")
        y[i] = int(cells[0])
        for j in range(9):
            X[i, j] = int(cells[j + 1])
    return X, y


def _write_csv(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    """Amazon CSV writer: 9 int-coded cats + binary target. No NaNs in this dataset."""
    with open(path, "w") as f:
        f.write(",".join(COLUMNS[1:] + ["target"]) + "\n")
        for i in range(X.shape[0]):
            cells = [str(int(X[i, j])) for j in range(9)]
            cells.append(str(int(y[i])))
            f.write(",".join(cells) + "\n")


def prepare(out_dir: Path, *, seed: int = 42) -> DatasetMeta:
    """Validate cached data, split 80/20, write {train,test}.csv + meta.json."""
    raw_csv = _check_raw(out_dir)

    X, y = _load_kaggle_train(raw_csv)

    # Move the Kaggle file to a non-conflicting name BEFORE writing our train.csv,
    # otherwise we'd overwrite the source. The check above ensures train.csv
    # is the Kaggle ACTION-first file, so renaming is safe.
    archived = out_dir / "kaggle_train_raw.csv"
    if not archived.exists():
        raw_csv.rename(archived)

    train_idx, test_idx = deterministic_split(len(y), test_fraction=0.2, seed=seed)
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    train_csv = out_dir / "train.csv"
    test_csv = out_dir / "test.csv"
    _write_csv(train_csv, X_tr, y_tr)
    _write_csv(test_csv, X_te, y_te)

    pos_rate = float(y.mean())

    meta = DatasetMeta(
        name=NAME,
        task="classification",
        metric="logloss",
        n_train=int(X_tr.shape[0]),
        n_test=int(X_te.shape[0]),
        n_features=9,
        cat_indices=list(range(9)),
        target_col=9,
        notes=(
            f"Amazon Employee Access (Kaggle), 9 categorical features, no numeric. "
            f"Imbalanced binary target ({pos_rate:.1%} positive). 80/20 deterministic "
            f"split (seed={seed}) of the Kaggle labelled train set since the Kaggle "
            f"test.csv has no labels. THIS DATASET TRIGGERS THE DEC-046 CTR rare-class "
            f"characterized gap — include in benchmark writeup with the documented "
            f"footnote, do not cherry-pick around."
        ),
        source_url=SOURCE_URL,
    )
    meta.write(out_dir)

    print(f"Amazon prepared:")
    print(f"  out: {out_dir}")
    print(f"  train: {meta.n_train} rows  test: {meta.n_test} rows  features: {meta.n_features}")
    print(f"  cat_indices: {meta.cat_indices}  positive rate: {pos_rate:.4f}")
    print(f"  NOTE: DEC-046 characterized gap visible on this dataset.")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = cache_dir(NAME, root=Path(args.cache_root) if args.cache_root else None)
    try:
        prepare(out_dir, seed=args.seed)
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
