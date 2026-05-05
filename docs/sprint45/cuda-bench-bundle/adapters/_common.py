"""Shared utilities for upstream-benchmark dataset adapters.

Each adapter under benchmarks/upstream/adapters/ converts a dataset from its
canonical source format (UCI .data, libsvm, Kaggle CSV, MSLR text) into the
CSV form that catboost-mlx's csv_train expects. Common concerns (cache dir
location, download with progress, libsvm parsing, output writer with explicit
NaN sentinels) live here.

S42-T1 scaffold. See docs/sprint42/sprint-plan.md.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np


# ── Cache layout ─────────────────────────────────────────────────────────────

DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "catboost-mlx-benchmarks"


def cache_dir(dataset_name: str, root: Optional[Path] = None) -> Path:
    """Return the canonical cache directory for a dataset, creating it if missing."""
    base = root or DEFAULT_CACHE_ROOT
    out = base / dataset_name
    out.mkdir(parents=True, exist_ok=True)
    return out


# ── Dataset metadata record ─────────────────────────────────────────────────

@dataclass
class DatasetMeta:
    """Machine-readable summary of a prepared dataset.

    Adapters dump one of these to <cache>/meta.json after preparing the CSV
    so downstream runner scripts can configure each framework consistently.
    """
    name: str
    task: str                      # "classification" | "regression" | "ranking"
    metric: str                    # "logloss" | "rmse" | "ndcg"
    n_train: int
    n_test: int
    n_features: int
    cat_indices: List[int] = field(default_factory=list)
    target_col: int = -1           # 0-based index in the produced CSV
    group_col: Optional[int] = None  # 0-based index for ranking datasets
    notes: str = ""
    source_url: str = ""

    def write(self, out_dir: Path) -> Path:
        path = out_dir / "meta.json"
        path.write_text(json.dumps(self.__dict__, indent=2))
        return path


# ── Download helpers ────────────────────────────────────────────────────────

def download_if_missing(url: str, dst: Path, *, label: Optional[str] = None) -> Path:
    """Download `url` to `dst` if `dst` is missing. Show byte-progress on stderr.

    Returns dst. Raises RuntimeError on failure (the caller should `print` the
    manual-download instructions to the user).
    """
    if dst.exists() and dst.stat().st_size > 0:
        return dst

    label = label or dst.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")

    print(f"  Downloading {label} from {url}", file=sys.stderr)
    t0 = time.time()
    last_print = 0.0

    def _report(blocknum: int, blocksize: int, totalsize: int) -> None:
        nonlocal last_print
        bytes_read = blocknum * blocksize
        now = time.time()
        if now - last_print < 0.5:
            return
        last_print = now
        if totalsize > 0:
            pct = min(100.0, bytes_read * 100.0 / totalsize)
            print(
                f"    {bytes_read/1e6:8.1f} MB / {totalsize/1e6:8.1f} MB "
                f"({pct:5.1f}%, {bytes_read/(now-t0)/1e6:6.2f} MB/s)",
                end="\r",
                file=sys.stderr,
            )
        else:
            print(
                f"    {bytes_read/1e6:8.1f} MB ({bytes_read/(now-t0)/1e6:6.2f} MB/s)",
                end="\r",
                file=sys.stderr,
            )

    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_report)
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    print(file=sys.stderr)
    tmp.rename(dst)
    return dst


def gunzip_if_missing(src_gz: Path, dst: Path) -> Path:
    """Decompress src_gz to dst if dst is missing."""
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    print(f"  Decompressing {src_gz.name} -> {dst.name}", file=sys.stderr)
    with gzip.open(src_gz, "rb") as f_in, open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=1 << 20)
    return dst


# ── CSV writers ─────────────────────────────────────────────────────────────

def write_dense_csv(
    path: Path,
    X: np.ndarray,
    y: Optional[np.ndarray],
    *,
    feature_names: Optional[Sequence[str]] = None,
    target_name: str = "target",
    cat_indices: Sequence[int] = (),
) -> None:
    """Write a dense numeric matrix + target as CSV in the layout csv_train expects.

    - Header row: <feature_names...>,<target_name>
    - Categorical columns are written as integer codes (no string escaping).
    - NaN values are written as empty strings (csv_train treats empty as NaN).
    """
    n, p = X.shape
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(p)]
    elif len(feature_names) != p:
        raise ValueError(f"feature_names length {len(feature_names)} != X cols {p}")

    cat_set = set(cat_indices)

    with open(path, "w") as f:
        if y is None:
            f.write(",".join(feature_names) + "\n")
        else:
            f.write(",".join(list(feature_names) + [target_name]) + "\n")

        for i in range(n):
            row = []
            for j in range(p):
                v = X[i, j]
                if j in cat_set:
                    row.append("" if (isinstance(v, float) and np.isnan(v)) else str(int(v)))
                else:
                    if isinstance(v, float) and np.isnan(v):
                        row.append("")
                    else:
                        row.append(repr(float(v)))
            if y is not None:
                yv = y[i]
                if isinstance(yv, float) and np.isnan(yv):
                    row.append("")
                elif isinstance(yv, (np.integer, int)):
                    row.append(str(int(yv)))
                else:
                    row.append(repr(float(yv)))
            f.write(",".join(row) + "\n")


def parse_libsvm(path: Path, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """Parse a libsvm-format text file into a dense (X, y) pair.

    Used by the Epsilon adapter (Pascal LSC challenge format). Memory-heavy
    for large datasets; caller is responsible for chunking if n_rows × n_features
    exceeds available RAM.
    """
    rows = path.read_text().splitlines()
    n = len(rows)
    X = np.zeros((n, n_features), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    for i, line in enumerate(rows):
        parts = line.split()
        if not parts:
            continue
        y[i] = float(parts[0])
        for token in parts[1:]:
            idx_s, val_s = token.split(":", 1)
            X[i, int(idx_s) - 1] = float(val_s)
    return X, y


# ── Reproducibility helpers ─────────────────────────────────────────────────

def deterministic_split(
    n: int,
    *,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) for a deterministic random split."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(round(n * test_fraction))
    return perm[n_test:], perm[:n_test]
