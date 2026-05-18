#!/usr/bin/env bash
# revert_option_b_baseline.sh — S49-T3: revert Option B patch, restore C6 expression
#
# Run from repo root AFTER the baseline pass completes:
#   bash docs/sprint49/T3/envelope-sweep/revert_option_b_baseline.sh
#
# Then rebuild and run the C6 sweep pass.

set -euo pipefail

TRAIN_CPP="catboost/mlx/train_lib/train.cpp"

if [ ! -f "$TRAIN_CPP" ]; then
    echo "ERROR: $TRAIN_CPP not found. Run from repo root." >&2
    exit 1
fi

# Verify the patch marker is present
if ! grep -q "S49-T3 Option B baseline patch" "$TRAIN_CPP"; then
    echo "ERROR: Option B patch marker not found in $TRAIN_CPP." >&2
    echo "       File may not be patched. Use git diff to inspect." >&2
    exit 1
fi

python3 - <<'PYEOF'
import sys

path = "catboost/mlx/train_lib/train.cpp"
with open(path) as f:
    content = f.read()

old = (
    "        const bool useHistogramSubtraction = false;  // S49-T3 Option B baseline patch"
)
new = (
    "        const bool useHistogramSubtraction =\n"
    "            (lossFunction == ELossFunction::Logloss)\n"
    "            || (lossFunction == ELossFunction::CrossEntropy)\n"
    "            || (lossFunction == ELossFunction::MultiClass);"
)

if old not in content:
    print("ERROR: baseline patch marker not found verbatim — aborting.", file=sys.stderr)
    sys.exit(1)

patched = content.replace(old, new)
with open(path, "w") as f:
    f.write(patched)

print(f"Reverted {path}: C6 loss-conditional dispatch restored.")
PYEOF

echo "Revert done. Now rebuild:"
echo "  cd python && pip install -e . --no-build-isolation -q && cd .."
echo "Then run:"
echo "  SWEEP_PASS=c6 python docs/sprint49/T3/envelope-sweep/run_t3_sweep.py"
