#!/usr/bin/env bash
# One-shot driver for the S42 5-dataset × 4-framework × 3-seed benchmark sweep.
#
# Datasets must be prepared first via the adapters under
# benchmarks/upstream/adapters/. Adult downloads automatically; Higgs
# downloads automatically (large, ~2.7 GB); Amazon, Epsilon, MSLR require
# manual data acquisition (each adapter prints the recipe).
#
# Usage:
#   bash benchmarks/upstream/scripts/run_subset.sh           # all datasets, 3 seeds
#   bash benchmarks/upstream/scripts/run_subset.sh adult     # one dataset, 3 seeds
#   DATASETS="adult amazon" SEEDS="42" bash benchmarks/upstream/scripts/run_subset.sh
#
# Environment:
#   DATASETS  Space-separated dataset names (default: "adult amazon higgs epsilon mslr")
#   SEEDS     Space-separated seeds       (default: "42 43 44")
#   FRAMEWORKS Space-separated framework runners
#             (default: "lightgbm xgboost catboost_cpu catboost_mlx")
#   RESULTS_DIR  Output directory (default: benchmarks/upstream/results)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

DATASETS="${DATASETS:-${1:-adult amazon higgs epsilon mslr}}"
SEEDS="${SEEDS:-42 43 44}"
FRAMEWORKS="${FRAMEWORKS:-lightgbm xgboost catboost_cpu catboost_mlx}"
RESULTS_DIR="${RESULTS_DIR:-benchmarks/upstream/results}"

mkdir -p "$RESULTS_DIR"

echo "=== S42 benchmark sweep ===" >&2
echo "  ROOT:        $ROOT" >&2
echo "  DATASETS:    $DATASETS" >&2
echo "  SEEDS:       $SEEDS" >&2
echo "  FRAMEWORKS:  $FRAMEWORKS" >&2
echo "  RESULTS_DIR: $RESULTS_DIR" >&2
echo >&2

OK=0
FAIL=0
SKIP=0
TOTAL=0

for DATASET in $DATASETS; do
  META="$HOME/.cache/catboost-mlx-benchmarks/$DATASET/meta.json"
  if [ ! -f "$META" ]; then
    echo "[$DATASET] meta.json missing — running adapter..." >&2
    if ! python3 -m "benchmarks.upstream.adapters.$DATASET" >&2; then
      echo "[$DATASET] adapter failed; skipping all framework runs" >&2
      for FW in $FRAMEWORKS; do for SEED in $SEEDS; do
        SKIP=$((SKIP + 1)); TOTAL=$((TOTAL + 1))
      done; done
      continue
    fi
  fi

  for FW in $FRAMEWORKS; do
    for SEED in $SEEDS; do
      TOTAL=$((TOTAL + 1))
      OUT_JSON="$RESULTS_DIR/${DATASET}_${FW}_${SEED}.json"
      if [ -f "$OUT_JSON" ]; then
        echo "[$DATASET/$FW/$SEED] cached → $OUT_JSON" >&2
        OK=$((OK + 1))
        continue
      fi
      echo "[$DATASET/$FW/$SEED] running..." >&2
      if python3 -m "benchmarks.upstream.scripts.run_$FW" \
            --dataset "$DATASET" --seed "$SEED" \
            --results-dir "$RESULTS_DIR"; then
        OK=$((OK + 1))
      else
        echo "[$DATASET/$FW/$SEED] FAILED" >&2
        FAIL=$((FAIL + 1))
      fi
    done
  done
done

echo >&2
echo "=== Summary ===" >&2
echo "  total: $TOTAL  ok: $OK  fail: $FAIL  skip: $SKIP" >&2
echo >&2
echo "Results: $RESULTS_DIR/*.json" >&2
echo "Next: T3 — write docs/benchmarks/v0.5.x-pareto.md from these JSONs." >&2

if [ "$FAIL" -gt 0 ]; then
  exit 2
fi
