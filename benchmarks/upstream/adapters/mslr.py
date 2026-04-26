"""Adapter for the MSLR-WEB10K Microsoft Learning to Rank dataset.

Source: https://www.microsoft.com/en-us/research/project/mslr/
File:   MSLR-WEB10K.zip (~3.7 GB; downloaded after registration form)

Task: ranking. Target = relevance label 0–4 (5-level graded relevance).
Used by upstream catboost/benchmarks `ranking/` as the canonical learning-
to-rank benchmark. We measure NDCG@10 on Fold1 test (matching upstream).

10,000 queries, ~1.2M query-doc pairs total.
136 numeric features per query-doc.
NO categorical features. cat_indices = [].

Microsoft requires a registration form to obtain the download URL, so this
adapter cannot auto-download. Place the unzipped Fold1 files in the cache
dir before running:

    DATA_DIR=~/.cache/catboost-mlx-benchmarks/mslr
    mkdir -p "$DATA_DIR"
    # Download MSLR-WEB10K.zip after registering at:
    #   https://www.microsoft.com/en-us/research/project/mslr/
    # Place it in $DATA_DIR, then:
    cd "$DATA_DIR"
    unzip MSLR-WEB10K.zip       # produces Fold1/, Fold2/, ..., Fold5/
    mv Fold1/*.txt .             # we use Fold1 only (canonical split)

After unzip you should have train.txt, vali.txt, test.txt in $DATA_DIR.
This adapter parses the Microsoft format (relevance qid:N feat:val ...)
and writes train.csv + test.csv with columns: target, qid, f1, f2, ..., f136.

Run: python -m benchmarks.upstream.adapters.mslr
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
)

NAME = "mslr"
SOURCE_URL = "https://www.microsoft.com/en-us/research/project/mslr/"

N_FEATURES = 136


def _check_raw(out_dir: Path) -> tuple[Path, Path]:
    train_path = out_dir / "train.txt"
    test_path = out_dir / "test.txt"
    missing = [p for p in (train_path, test_path) if not p.exists()]
    if missing:
        raise RuntimeError(
            f"MSLR-WEB10K raw files missing in {out_dir}:\n"
            + "\n".join(f"  {p.name}" for p in missing)
            + "\n\nMicrosoft requires a registration form to obtain the download URL.\n"
              f"Register at: {SOURCE_URL}\n"
              f"Then place MSLR-WEB10K.zip in {out_dir} and:\n"
              f"  cd '{out_dir}' && unzip MSLR-WEB10K.zip && mv Fold1/*.txt ./"
        )
    return train_path, test_path


def _parse_mslr_split(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a Microsoft LETOR text file: one query-doc per line, format:
        <relevance> qid:<N> 1:<v1> 2:<v2> ... 136:<v136> #docid = ...
    Returns (X, y, qid) where qid is the per-row group id (int64).
    """
    rows = path.read_text().splitlines()
    n = len(rows)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)
    y = np.zeros(n, dtype=np.int8)
    qid = np.zeros(n, dtype=np.int64)
    for i, line in enumerate(rows):
        # Strip any trailing comment (after '#')
        comment_idx = line.find("#")
        if comment_idx >= 0:
            line = line[:comment_idx]
        parts = line.split()
        if not parts:
            continue
        y[i] = int(parts[0])
        qid_token = parts[1]  # 'qid:<N>'
        if not qid_token.startswith("qid:"):
            raise RuntimeError(f"Line {i+1} missing qid: {line!r}")
        qid[i] = int(qid_token[4:])
        for token in parts[2:]:
            idx_s, val_s = token.split(":", 1)
            X[i, int(idx_s) - 1] = float(val_s)
    return X, y, qid


def _write_csv(path: Path, X: np.ndarray, y: np.ndarray, qid: np.ndarray) -> None:
    """MSLR CSV writer: target,qid,f1..f136. csv_train uses --group-col 1."""
    n, p = X.shape
    feature_names = [f"f{i+1}" for i in range(p)]
    with open(path, "w") as f:
        f.write("target,qid," + ",".join(feature_names) + "\n")
        for i in range(n):
            cells = [str(int(y[i])), str(int(qid[i]))]
            cells.extend(repr(float(X[i, j])) for j in range(p))
            f.write(",".join(cells) + "\n")


def prepare(out_dir: Path) -> DatasetMeta:
    train_raw, test_raw = _check_raw(out_dir)

    print(f"  Parsing {train_raw.name}...", file=sys.stderr)
    X_tr, y_tr, qid_tr = _parse_mslr_split(train_raw)
    print(f"  Parsing {test_raw.name}...", file=sys.stderr)
    X_te, y_te, qid_te = _parse_mslr_split(test_raw)

    train_csv = out_dir / "train.csv"
    test_csv = out_dir / "test.csv"
    _write_csv(train_csv, X_tr, y_tr, qid_tr)
    _write_csv(test_csv, X_te, y_te, qid_te)

    meta = DatasetMeta(
        name=NAME,
        task="ranking",
        metric="ndcg@10",
        n_train=int(X_tr.shape[0]),
        n_test=int(X_te.shape[0]),
        n_features=N_FEATURES,
        cat_indices=[],
        target_col=0,
        group_col=1,
        notes=(
            "Microsoft Learning to Rank (MSLR-WEB10K), Fold1 train/test. "
            "Five-level graded relevance (target 0..4). NDCG@10 metric. "
            "csv_train invocation: --loss yetirank --target-col 0 --group-col 1 "
            f"--cat-features '' (numeric only); 136 features in columns 2..{N_FEATURES+1}."
        ),
        source_url=SOURCE_URL,
    )
    meta.write(out_dir)

    print(f"MSLR-WEB10K prepared:")
    print(f"  out: {out_dir}")
    print(f"  train: {meta.n_train:,} rows  test: {meta.n_test:,} rows  features: {meta.n_features}")
    print(f"  unique queries train/test: {len(np.unique(qid_tr)):,} / {len(np.unique(qid_te)):,}")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache-root", default=None)
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
