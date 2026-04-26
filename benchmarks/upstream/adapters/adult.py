"""Adapter for the UCI Adult Census income dataset.

Source: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/
Task: binary classification (income >50K vs ≤50K). Used in the upstream
catboost/benchmarks `quality_benchmarks/` suite under "Adult". Train and
test files are pre-split by the UCI source (~32k train / ~16k test).

14 features:
- 6 numeric:    age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week
- 8 categorical: workclass, education, marital_status, occupation, relationship,
                 race, sex, native_country

Quirks of the upstream files:
- 'NA' / '?' tokens used for missing categorical values
- adult.test has a leading "|1x3 Cross validator" comment line we must skip
- adult.test target labels are suffixed with '.' (e.g. '<=50K.' vs '<=50K')

Output: <cache>/{train,test}.csv with the 14 features + target column,
plus meta.json with cat_indices and target_col fields.

Run: python -m benchmarks.upstream.adapters.adult
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ._common import (
    DatasetMeta,
    cache_dir,
    download_if_missing,
)

NAME = "adult"
URL_TRAIN = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
URL_TEST = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "target",
]
NUMERIC_COLS = {"age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"}
CAT_COLS = [c for c in COLUMNS[:-1] if c not in NUMERIC_COLS]


def _read_adult_file(path: Path, *, is_test: bool) -> list[list[str]]:
    """Read an adult.{data,test} file, skipping the test-file comment line and
    stripping the trailing '.' from test-file target labels.
    """
    lines = path.read_text().splitlines()
    if is_test and lines and lines[0].startswith("|"):
        lines = lines[1:]

    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        cells = [c.strip() for c in line.split(",")]
        if len(cells) != len(COLUMNS):
            continue
        if is_test:
            cells[-1] = cells[-1].rstrip(".")
        rows.append(cells)
    return rows


def _encode_cat_columns(
    train_rows: list[list[str]], test_rows: list[list[str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Build dense (X, y) arrays for train and test with consistent categorical
    integer codes across both splits. Missing values ('?', 'NA', '') are
    encoded as a dedicated integer (-1 is unsafe with most cat encoders, so
    we reserve code 0 for unknown).
    """
    n_features = len(COLUMNS) - 1

    # Build per-column code maps using the union of train + test categories
    cat_indices = [i for i, c in enumerate(COLUMNS[:-1]) if c in CAT_COLS]
    cat_code_maps: dict[int, dict[str, int]] = {j: {"": 0} for j in cat_indices}

    def assign_codes(rows: list[list[str]]) -> None:
        for row in rows:
            for j in cat_indices:
                tok = row[j]
                if tok in ("?", "NA"):
                    tok = ""
                if tok not in cat_code_maps[j]:
                    cat_code_maps[j][tok] = len(cat_code_maps[j])
            for j_num in range(n_features):
                if j_num in cat_indices:
                    continue
                # Numeric columns: '?' / 'NA' / '' will become NaN; nothing to do here.

    assign_codes(train_rows)
    assign_codes(test_rows)

    def materialize(rows: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
        n = len(rows)
        X = np.empty((n, n_features), dtype=np.float64)
        y = np.empty(n, dtype=np.int64)
        for i, row in enumerate(rows):
            for j in range(n_features):
                tok = row[j]
                if j in cat_indices:
                    if tok in ("?", "NA"):
                        tok = ""
                    X[i, j] = float(cat_code_maps[j][tok])
                else:
                    if tok in ("?", "NA", ""):
                        X[i, j] = float("nan")
                    else:
                        X[i, j] = float(tok)
            label = row[-1]
            y[i] = 1 if label == ">50K" else 0
        return X, y

    X_tr, y_tr = materialize(train_rows)
    X_te, y_te = materialize(test_rows)
    return X_tr, y_tr, X_te, y_te, cat_indices


def _write_csv(path: Path, X: np.ndarray, y: np.ndarray, cat_indices: list[int]) -> None:
    """Adult-specific CSV writer: cats as int codes, numerics as floats, NaN as empty."""
    n, p = X.shape
    cat_set = set(cat_indices)
    feature_names = COLUMNS[:-1]
    with open(path, "w") as f:
        f.write(",".join(feature_names + ["target"]) + "\n")
        for i in range(n):
            cells = []
            for j in range(p):
                v = X[i, j]
                if np.isnan(v):
                    cells.append("")
                elif j in cat_set:
                    cells.append(str(int(v)))
                else:
                    cells.append(repr(float(v)))
            cells.append(str(int(y[i])))
            f.write(",".join(cells) + "\n")


def prepare(out_dir: Path) -> DatasetMeta:
    """Download Adult if missing, write {train,test}.csv + meta.json under out_dir."""
    train_raw = out_dir / "adult.data"
    test_raw = out_dir / "adult.test"

    download_if_missing(URL_TRAIN, train_raw, label="adult.data")
    download_if_missing(URL_TEST, test_raw, label="adult.test")

    train_rows = _read_adult_file(train_raw, is_test=False)
    test_rows = _read_adult_file(test_raw, is_test=True)

    X_tr, y_tr, X_te, y_te, cat_indices = _encode_cat_columns(train_rows, test_rows)

    train_csv = out_dir / "train.csv"
    test_csv = out_dir / "test.csv"
    _write_csv(train_csv, X_tr, y_tr, cat_indices)
    _write_csv(test_csv, X_te, y_te, cat_indices)

    meta = DatasetMeta(
        name=NAME,
        task="classification",
        metric="logloss",
        n_train=int(X_tr.shape[0]),
        n_test=int(X_te.shape[0]),
        n_features=int(X_tr.shape[1]),
        cat_indices=cat_indices,
        target_col=int(X_tr.shape[1]),  # column 14 (0-based: features 0..13, target at 14)
        notes=(
            "UCI Adult Census income; binary classification. Categorical columns "
            "encoded as integer codes; missing tokens ('?', 'NA') -> code 0. "
            "Used by upstream catboost/benchmarks quality_benchmarks/ as 'Adult'."
        ),
        source_url=URL_TRAIN,
    )
    meta.write(out_dir)

    print(f"Adult prepared:")
    print(f"  out: {out_dir}")
    print(f"  train: {meta.n_train} rows  test: {meta.n_test} rows  features: {meta.n_features}")
    print(f"  cat_indices: {meta.cat_indices}")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--cache-root", default=None,
        help="Override the cache root directory (default: ~/.cache/catboost-mlx-benchmarks)",
    )
    args = parser.parse_args()

    out_dir = cache_dir(NAME, root=Path(args.cache_root) if args.cache_root else None)
    try:
        prepare(out_dir)
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
