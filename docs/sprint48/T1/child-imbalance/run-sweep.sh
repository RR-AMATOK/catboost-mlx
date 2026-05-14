#!/usr/bin/env bash
# S48-T1 child-imbalance sweep operator script.
# Run from the catboost-mlx repo root.
#
# Prerequisites:
#   - bench_boosting_s48_t1 built (see BUILD STEP below)
#   - docs/sprint48/T1/child-imbalance/data/ directory exists
#
# Usage:
#   bash docs/sprint48/T1/child-imbalance/run-sweep.sh

set -euo pipefail

REPO="/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/catboost-mlx"
BUILDDIR="$REPO/python/build/temp.macosx-11.1-arm64-cpython-313/catboost_mlx._core"
T1_BIN="$BUILDDIR/bench_boosting_s48_t1"
BASE_BIN="$BUILDDIR/bench_boosting_baseline"
OUTDIR="$REPO/docs/sprint48/T1/child-imbalance/data"

echo "================================================================"
echo "S48-T1 child-imbalance sweep"
echo "================================================================"
echo ""

# ── BUILD STEP ────────────────────────────────────────────────────────────────
# Run this section once before measurements if the binary is not yet built.

build_binary() {
    echo "[BUILD] Configuring CMake with BUILD_S48_T1=ON ..."
    cmake -S "$REPO/python/catboost_mlx/_core" \
          -B "$BUILDDIR" \
          -DBUILD_S48_T1=ON \
          2>&1 | tail -5

    echo "[BUILD] Compiling bench_boosting_s48_t1 ..."
    cmake --build "$BUILDDIR" --target bench_boosting_s48_t1 2>&1 | tail -10

    echo "[BUILD] Done."
    ls -lh "$T1_BIN"
}

# Uncomment to build:
# build_binary

# ── BIT-EXACTNESS VERIFICATION ────────────────────────────────────────────────
# Verify instrumented build produces identical final loss to baseline.
# Run BEFORE the full sweep to confirm zero behavior change.

verify_bit_exactness() {
    echo "[VERIFY] Bit-exactness check (Higgs-1M proxy, 10 iters, seed 42) ..."
    LOSS_T1=$("$T1_BIN" \
        --rows 1000000 --features 28 --classes 2 --depth 6 \
        --bins 128 --iters 10 --seed 42 \
        2>/dev/null | grep BENCH_FINAL_LOSS | awk -F= '{print $2}')
    LOSS_BASE=$("$BASE_BIN" \
        --rows 1000000 --features 28 --classes 2 --depth 6 \
        --bins 128 --iters 10 --seed 42 \
        2>/dev/null | grep BENCH_FINAL_LOSS | awk -F= '{print $2}')

    if [ "$LOSS_T1" = "$LOSS_BASE" ]; then
        echo "[VERIFY] PASS: bit-exact (loss=$LOSS_T1)"
    else
        echo "[VERIFY] FAIL: T1=$LOSS_T1 vs BASE=$LOSS_BASE"
        exit 1
    fi
}

# ── SWEEP ─────────────────────────────────────────────────────────────────────
# 3 seeds x 2 shapes x 100 trees = 6 runs.
# Each run logs per-split child counts to a CSV.
# Wall-clock estimate: ~5-15 min/run at 100 iters on M3 Max.
# Total: ~30-90 min.

mkdir -p "$OUTDIR"

run_sweep() {
    local SEEDS=(42 43 44)

    for seed in "${SEEDS[@]}"; do
        echo ""
        echo "--- Higgs-1M proxy  seed=$seed ---"
        "$T1_BIN" \
            --rows 1000000 --features 28 --classes 2 --depth 6 \
            --bins 128 --iters 100 --seed "$seed" \
            --child-log "$OUTDIR/higgs_seed${seed}.csv" \
            2>&1 | tee "$OUTDIR/higgs_seed${seed}.stdout.txt"

        echo ""
        echo "--- Epsilon proxy  seed=$seed ---"
        "$T1_BIN" \
            --rows 400000 --features 2000 --classes 2 --depth 6 \
            --bins 128 --iters 100 --seed "$seed" \
            --child-log "$OUTDIR/epsilon_seed${seed}.csv" \
            2>&1 | tee "$OUTDIR/epsilon_seed${seed}.stdout.txt"
    done

    echo ""
    echo "================================================================"
    echo "Sweep complete. CSV files:"
    ls -lh "$OUTDIR"/*.csv 2>/dev/null || echo "(no CSVs found — did the binary run?)"
}

# ── MAIN ──────────────────────────────────────────────────────────────────────

echo "Step 1: verify bit-exactness"
verify_bit_exactness

echo ""
echo "Step 2: run 6-config sweep (3 seeds x 2 shapes)"
run_sweep

echo ""
echo "Step 3: run analysis script"
python3 "$REPO/docs/sprint48/T1/child-imbalance/analyze.py"

echo ""
echo "Done. analysis.md written to:"
echo "  $REPO/docs/sprint48/T1/child-imbalance/analysis.md"
