#!/usr/bin/env bash
# Axis C — argmin_iter parity at iso-accuracy on Epsilon.
#
# @mathematician's experiment: does catboost-mlx reach the same logloss
# floor in fewer iterations than catboost-cpu?  Predicted ratio
# T*_mlx / T*_cpu ∈ [0.55, 0.85] from variance-reduction theory; falsified
# if ratio ∈ [0.95, 1.05].
#
# Sweep:  Epsilon × {catboost_cpu, catboost_mlx} × seeds {0..9} × iters
#         {200, 500, 1000, 2000, 4000} = 100 runs.
# CPU runs use --metric-period 1 to capture per-iter trajectory (free —
# CatBoost computes it internally).  MLX runs use the standard 5-point
# discrete trajectory (one final-iter logloss per iter level).
#
# Estimated wall-time on M3 Max: ~5 hours.  Cache-aware (skips runs whose
# output JSON already exists).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

RESULTS_DIR="benchmarks/axisC/results"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
ITERS="${ITERS:-200 500 1000 2000 4000}"

mkdir -p "$RESULTS_DIR"

echo "=== Axis C — argmin_iter parity sweep on Epsilon ===" >&2
echo "  ROOT:        $ROOT" >&2
echo "  RESULTS_DIR: $RESULTS_DIR" >&2
echo "  SEEDS:       $SEEDS" >&2
echo "  ITERS:       $ITERS" >&2
echo "  Estimated:   ~5 hours on M3 Max (10 seeds × 5 iters × 2 backends)" >&2
echo >&2

OK=0; SKIP=0; FAIL=0; TOTAL=0

# CatBoost-CPU sweep — per-iter trajectory via metric_period=1.
for ITER in $ITERS; do
  for SEED in $SEEDS; do
    TOTAL=$((TOTAL + 1))
    if [ "$ITER" = "200" ]; then
      OUT_JSON="$RESULTS_DIR/epsilon_axisC_catboost_cpu_${SEED}.json"
    else
      OUT_JSON="$RESULTS_DIR/epsilon_axisC_iter${ITER}_catboost_cpu_${SEED}.json"
    fi
    if [ -f "$OUT_JSON" ]; then
      echo "[axisC/cpu/iter=$ITER/seed=$SEED] cached" >&2
      OK=$((OK + 1))
      continue
    fi
    echo "[axisC/cpu/iter=$ITER/seed=$SEED] running..." >&2
    if python3 -m benchmarks.upstream.scripts.run_catboost_cpu \
        --dataset epsilon --seed "$SEED" --iterations "$ITER" \
        --metric-period 1 --results-dir "$RESULTS_DIR"; then
      OK=$((OK + 1))
    else
      echo "[axisC/cpu/iter=$ITER/seed=$SEED] FAILED" >&2
      FAIL=$((FAIL + 1))
    fi
  done
done

# CatBoost-MLX sweep — final-iter logloss; the 5-point discrete trajectory
# is reconstructed in the analysis script from the iter-level JSONs.
for ITER in $ITERS; do
  for SEED in $SEEDS; do
    TOTAL=$((TOTAL + 1))
    if [ "$ITER" = "200" ]; then
      OUT_JSON="$RESULTS_DIR/epsilon_catboost_mlx_${SEED}.json"
    else
      OUT_JSON="$RESULTS_DIR/epsilon_iter${ITER}_catboost_mlx_${SEED}.json"
    fi
    if [ -f "$OUT_JSON" ]; then
      echo "[axisC/mlx/iter=$ITER/seed=$SEED] cached" >&2
      OK=$((OK + 1))
      continue
    fi
    echo "[axisC/mlx/iter=$ITER/seed=$SEED] running..." >&2
    if python3 -m benchmarks.upstream.scripts.run_catboost_mlx \
        --dataset epsilon --seed "$SEED" --iterations "$ITER" \
        --results-dir "$RESULTS_DIR"; then
      OK=$((OK + 1))
    else
      echo "[axisC/mlx/iter=$ITER/seed=$SEED] FAILED" >&2
      FAIL=$((FAIL + 1))
    fi
  done
done

echo >&2
echo "=== Axis C summary ===" >&2
echo "  total: $TOTAL  ok: $OK  fail: $FAIL" >&2
echo "  Results: $RESULTS_DIR/*.json" >&2
echo "  Next: paired-t-test on argmin_t L(t) per seed across backends." >&2
