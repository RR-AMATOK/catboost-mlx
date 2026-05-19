#!/usr/bin/env bash
# apply_option_b_baseline.sh — S49-T3 Option B: force UseHistogramSubtraction=false
#
# Run from repo root BEFORE building the baseline pass:
#   bash docs/sprint49/T3/envelope-sweep/apply_option_b_baseline.sh
#
# This replaces ONLY the boolean initialization expression (lines 177-180 in
# catboost/mlx/train_lib/train.cpp) with a literal `false`, so the baseline
# pass always takes the production src-broadcast path regardless of loss type.
#
# Revert with:
#   bash docs/sprint49/T3/envelope-sweep/revert_option_b_baseline.sh
#
# Or: git checkout catboost/mlx/train_lib/train.cpp

set -euo pipefail

TRAIN_CPP="catboost/mlx/train_lib/train.cpp"

if [ ! -f "$TRAIN_CPP" ]; then
    echo "ERROR: $TRAIN_CPP not found. Run from repo root." >&2
    exit 1
fi

# Verify the target lines are present (guard against double-apply or wrong branch)
if ! grep -q "lossFunction == ELossFunction::Logloss" "$TRAIN_CPP"; then
    echo "ERROR: Expected C6 expression not found in $TRAIN_CPP." >&2
    echo "       File may already be patched (check git diff), or wrong branch." >&2
    exit 1
fi

# The patch: replace lines 177-180 (the 3-way OR bool) with a single `false`.
# We use Python for the replacement to avoid sed portability issues on macOS.
python3 - <<'PYEOF'
import re, sys

path = "catboost/mlx/train_lib/train.cpp"
with open(path) as f:
    content = f.read()

# Match the exact 4-line block that sets useHistogramSubtraction
old = (
    "        const bool useHistogramSubtraction =\n"
    "            (lossFunction == ELossFunction::Logloss)\n"
    "            || (lossFunction == ELossFunction::CrossEntropy)\n"
    "            || (lossFunction == ELossFunction::MultiClass);"
)
new = (
    "        const bool useHistogramSubtraction = false;  // S49-T3 Option B baseline patch"
)

if old not in content:
    print("ERROR: target block not found verbatim in train.cpp — aborting.", file=sys.stderr)
    sys.exit(1)

count = content.count(old)
if count != 1:
    print(f"ERROR: expected 1 occurrence, found {count} — aborting.", file=sys.stderr)
    sys.exit(1)

patched = content.replace(old, new)
with open(path, "w") as f:
    f.write(patched)

print(f"Patched {path}: UseHistogramSubtraction forced to false for baseline pass.")
PYEOF

echo "Patch applied. Now rebuild:"
echo "  cd python && pip install -e . --no-build-isolation -q && cd .."
echo "Then run:"
echo "  SWEEP_PASS=baseline python docs/sprint49/T3/envelope-sweep/run_t3_sweep.py"
