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

This adapter AUTO-DOWNLOADS the .bz2 files from the LIBSVM mirror and
decompresses them. The mirror is slow (~5-15 MB/s typical from the
Taiwan academic server) — expect 20-60 minutes for the train file alone.
Resume is supported by the underlying urllib (interrupted .part files are
discarded; re-run to retry from scratch). If the auto-download fails,
fall back to the manual recipe below.

    # Manual download recipe (only if auto-download fails)
    DATA_DIR=~/.cache/catboost-mlx-benchmarks/epsilon
    mkdir -p "$DATA_DIR"
    cd "$DATA_DIR"
    curl -O https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2
    curl -O https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.bz2
    bunzip2 epsilon_normalized.bz2
    bunzip2 epsilon_normalized.t.bz2

After bunzip, this adapter parses the libsvm format and writes train.csv +
test.csv. **The output CSVs will be 6-8 GB each — make sure you have disk
headroom (~20 GB total: 15 GB compressed sources + 14 GB CSVs, half of
which can be deleted after the CSVs are written).**

Run: python -m benchmarks.upstream.adapters.epsilon  # auto-download + parse
     python -m benchmarks.upstream.adapters.epsilon --materialize  # also write CSVs
"""
from __future__ import annotations

import argparse
import bz2
import json
import shutil
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from ._common import (
    DatasetMeta,
    cache_dir,
    download_if_missing,
    parse_libsvm,
    write_dense_csv,
)


def _install_unverified_ssl_opener() -> None:
    """Install a global urllib opener that bypasses SSL cert verification.

    The LIBSVM Taiwan academic mirror occasionally serves with cert chains
    that some Python builds (e.g. the python.org installer on macOS without
    `Install Certificates.command` run, or a corporate-MITM proxy on
    Windows) don't trust. The dataset itself is not security-sensitive —
    falling back to unverified is acceptable for benchmark data download.
    Subsequent urllib.request calls in the same process will use this
    opener.
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    https_handler = urllib.request.HTTPSHandler(context=ctx)
    opener = urllib.request.build_opener(https_handler)
    urllib.request.install_opener(opener)

NAME = "epsilon"
SOURCE_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html"

URL_TRAIN_BZ2 = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2"
URL_TEST_BZ2 = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.bz2"

N_TRAIN = 400_000
N_TEST = 100_000
N_FEATURES = 2_000


def _bunzip2_if_missing(src_bz2: Path, dst: Path) -> Path:
    """Decompress src_bz2 -> dst if dst is missing. Streams in 1 MB chunks
    to keep memory low (the test file is ~3 GB compressed -> ~7 GB raw)."""
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    print(f"  Decompressing {src_bz2.name} -> {dst.name} (this is slow)...", file=sys.stderr)
    with bz2.open(src_bz2, "rb") as f_in, open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=1 << 20)
    return dst


def _download_with_ssl_fallback(url: str, dst: Path, label: str) -> Path:
    """Try a normal download first; on SSL cert failure, retry unverified."""
    try:
        return download_if_missing(url, dst, label=label)
    except RuntimeError as exc:
        if "CERTIFICATE_VERIFY_FAILED" not in str(exc) and "SSL" not in str(exc):
            raise
        print(f"  WARNING: SSL cert verification failed for {url}", file=sys.stderr)
        print(f"           retrying with unverified SSL (benchmark data; not security-sensitive)",
              file=sys.stderr)
        _install_unverified_ssl_opener()
        return download_if_missing(url, dst, label=label)


def _ensure_raw(out_dir: Path) -> tuple[Path, Path]:
    """Download + decompress the libsvm files if they aren't already present."""
    train_bz2 = out_dir / "epsilon_normalized.bz2"
    test_bz2 = out_dir / "epsilon_normalized.t.bz2"
    train_path = out_dir / "epsilon_normalized"
    test_path = out_dir / "epsilon_normalized.t"

    # Already-decompressed raw files? Use them as-is, don't re-download.
    if train_path.exists() and test_path.exists():
        return train_path, test_path

    # Otherwise download .bz2 + decompress.
    if not train_path.exists():
        _download_with_ssl_fallback(URL_TRAIN_BZ2, train_bz2,
                                    label="epsilon_normalized.bz2 (~12 GB compressed)")
        _bunzip2_if_missing(train_bz2, train_path)
    if not test_path.exists():
        _download_with_ssl_fallback(URL_TEST_BZ2, test_bz2,
                                    label="epsilon_normalized.t.bz2 (~3 GB compressed)")
        _bunzip2_if_missing(test_bz2, test_path)

    return train_path, test_path


def prepare(out_dir: Path, *, materialize: bool = False) -> DatasetMeta:
    train_raw, test_raw = _ensure_raw(out_dir)

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
