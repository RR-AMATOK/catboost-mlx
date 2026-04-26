#!/usr/bin/env bash
# HISTORICAL: PROBE_H_INSTRUMENT block removed in S39 (DEC-044 withdrawn). This script preserved as historical record. Re-running requires restoring the source block.
# Build csv_train_probe_h for S38-PROBE-H (per-side cosNum/cosDen capture at iter=1).
#
# Compile-time gates:
#
#   -DCOSINE_RESIDUAL_INSTRUMENT  enables fp32-vs-fp64 double-shadow (PROBE-D infra)
#                                 and per-bin binRecords used for gain_mlx lookup.
#   -DPROBE_E_INSTRUMENT          enables the TLeafRecord capture infrastructure
#                                 (per-(feat, bin, partition) partition stats snapshot).
#   -DPROBE_H_INSTRUMENT          extends TLeafRecord with per-side fields (cosNumL/R,
#                                 cosDenL/R) and writes probe_h_iter1_depth{0..5}.csv.
#   -DPROBE_D_ARM_AT_ITER=0       arm at iter=0 (0-indexed), i.e. the first tree
#                                 (iter=1 in user-facing 1-indexed terminology). This is
#                                 the constant-basePred tree — cleanest signal for
#                                 formula-level divergence (C-PSF), no prior tree state.
#
# The instrumentation lives in csv_train.cpp:
#   - TLeafRecord extension (PROBE_H_INSTRUMENT fields): near line 161
#   - WriteProbeHCSV():                                  near line 231
#   - Per-side field population (capture block):         near line 1993
#   - Flush block (post-FindBestSplit):                  near line 5050
#
# Anchor: np.random.default_rng(42), N=1000, 20 features, ST/Cosine/RMSE,
#         depth=6, bins=128, l2=3, lr=0.03. Matches PROBE-G/F2 anchor exactly.
#
# Output: probe_h_iter1_depth{0..5}.csv into COSINE_RESIDUAL_OUTDIR.
#         Binary at repo root as csv_train_probe_h.
#
# Kernel md5 guard: kernel_sources.h must be 9edaef45b99b9db3e2717da93800e76f.
#   PROBE-H is host-side instrumentation only; kernel sources are not modified.
#
# Usage: ./scripts/build_probe_h.sh [output_path]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
OUT="${1:-${REPO_ROOT}/csv_train_probe_h}"

MLX_INC="/opt/homebrew/opt/mlx/include"
MLX_LIB="/opt/homebrew/opt/mlx/lib"
SRC="${REPO_ROOT}/catboost/mlx/tests/csv_train.cpp"

KERNEL_SOURCES="${REPO_ROOT}/catboost/mlx/kernels/kernel_sources.h"
EXPECTED_MD5="9edaef45b99b9db3e2717da93800e76f"
ACTUAL_MD5="$(md5 -q "${KERNEL_SOURCES}" 2>/dev/null || md5sum "${KERNEL_SOURCES}" | cut -d' ' -f1)"
if [[ "${ACTUAL_MD5}" != "${EXPECTED_MD5}" ]]; then
    echo "ERROR: kernel_sources.h md5 mismatch (got ${ACTUAL_MD5}, expected ${EXPECTED_MD5})" >&2
    echo "PROBE-H is host-side only — kernel changes are unexpected. Stop." >&2
    exit 2
fi
echo "[build] kernel_sources.h md5 OK: ${ACTUAL_MD5}"

clang++ -std=c++17 -O2 \
    -DCOSINE_RESIDUAL_INSTRUMENT \
    -DPROBE_E_INSTRUMENT \
    -DPROBE_H_INSTRUMENT \
    -DPROBE_D_ARM_AT_ITER=0 \
    -I"${REPO_ROOT}" \
    -I"${MLX_INC}" \
    -L"${MLX_LIB}" -lmlx \
    -framework Metal -framework Foundation \
    -Wno-c++20-extensions \
    "${SRC}" \
    -o "${OUT}"

echo "[build] done: ${OUT}"
file "${OUT}"
