#!/usr/bin/env bash
# Build csv_train_probe_g for S38-PROBE-G (small-N residual capture).
#
# Mirrors S33-PROBE-E build flags exactly. Same compile-time gates that the
# instrumentation in csv_train.cpp expects:
#
#   -DCOSINE_RESIDUAL_INSTRUMENT  enables fp32-vs-fp64 double-shadow (PROBE-D infra)
#                                 and per-bin cos_accum dump.
#   -DPROBE_E_INSTRUMENT          enables per-(feat, bin, partition) leaf record dump
#                                 with both MLX-actual contribution AND CPU-counterfactual
#                                 (per-side mask, CalcAverage) for joint cosNum/cosDen sums.
#   -DPROBE_D_ARM_AT_ITER=1       arm at iter=1 (1-indexed iter=2), all 6 depths.
#
# The instrumentation lives in FindBestSplit (csv_train.cpp:1899-1950 etc.).
# After DEC-042, the actual MLX behavior matches CPU's per-side-mask formula
# (cpu_termNum/Den columns); the mlx_termNum/Den columns describe what the
# pre-DEC-042 joint-skip would have produced. Their diff thus characterises
# what DEC-042 closed at the captured anchor.
#
# Output: per-depth `cos_leaf_seed42_depth{0..5}.csv` and per-bin
#         `cos_accum_seed42_depth{0..5}.csv` files into the directory pointed
#         to by COSINE_RESIDUAL_OUTDIR (must lie under repo `docs/`).
#
# Usage: ./scripts/build_probe_g.sh [output_path]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
OUT="${1:-${REPO_ROOT}/csv_train_probe_g}"

MLX_INC="/opt/homebrew/opt/mlx/include"
MLX_LIB="/opt/homebrew/opt/mlx/lib"
SRC="${REPO_ROOT}/catboost/mlx/tests/csv_train.cpp"

KERNEL_SOURCES="${REPO_ROOT}/catboost/mlx/kernels/kernel_sources.h"
EXPECTED_MD5="9edaef45b99b9db3e2717da93800e76f"
ACTUAL_MD5="$(md5 -q "${KERNEL_SOURCES}" 2>/dev/null || md5sum "${KERNEL_SOURCES}" | cut -d' ' -f1)"
if [[ "${ACTUAL_MD5}" != "${EXPECTED_MD5}" ]]; then
    echo "ERROR: kernel_sources.h md5 mismatch (got ${ACTUAL_MD5}, expected ${EXPECTED_MD5})" >&2
    exit 2
fi
echo "[build] kernel_sources.h md5 OK: ${ACTUAL_MD5}"

clang++ -std=c++17 -O2 \
    -DCOSINE_RESIDUAL_INSTRUMENT \
    -DPROBE_E_INSTRUMENT \
    -DPROBE_D_ARM_AT_ITER=1 \
    -I"${REPO_ROOT}" \
    -I"${MLX_INC}" \
    -L"${MLX_LIB}" -lmlx \
    -framework Metal -framework Foundation \
    -Wno-c++20-extensions \
    "${SRC}" \
    -o "${OUT}"

echo "[build] done: ${OUT}"
file "${OUT}"
