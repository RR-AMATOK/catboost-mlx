# S46-T4 Phase 3 — Measurement Commands

**Prerequisite:** Phase 1 build completed. Binaries at:
- `$BUILDDIR/bench_boosting_baseline`
- `$BUILDDIR/bench_boosting_probe_b`
- `$BUILDDIR/bench_boosting_probe_d`

Where `BUILDDIR="/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/catboost-mlx/python/build/temp.macosx-11.1-arm64-cpython-313/catboost_mlx._core"`

---

## Step 0: Regression guard

```bash
REPO="/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/catboost-mlx"
cd "$REPO"
python -m pytest python/tests/regression/test_branch_b_regression.py -v
# Expected: all PASS. If any fail: stop and investigate before benchmarking.
```

---

## Step 1: Baseline measurements (3 shapes × 3 seeds)

```bash
REPO="/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/catboost-mlx"
BUILDDIR="$REPO/python/build/temp.macosx-11.1-arm64-cpython-313/catboost_mlx._core"
BASE="$BUILDDIR/bench_boosting_baseline"
OUTDIR="$REPO/docs/sprint46/T4/probe-d/data"
mkdir -p "$OUTDIR"

for seed in 42 43 44; do
  # Gate-config
  "$BASE" --rows 50000 --features 100 --classes 2 --depth 6 --bins 128 \
    --iters 12 --seed $seed --per-kernel-profile \
    > "$OUTDIR/baseline_gate_seed${seed}.txt" 2>&1

  # Higgs-1M proxy
  "$BASE" --rows 1000000 --features 28 --classes 2 --depth 6 --bins 128 \
    --iters 12 --seed $seed --per-kernel-profile \
    > "$OUTDIR/baseline_higgs_seed${seed}.txt" 2>&1

  # Epsilon proxy (primary decision anchor)
  "$BASE" --rows 400000 --features 2000 --classes 2 --depth 6 --bins 128 \
    --iters 12 --seed $seed --per-kernel-profile \
    > "$OUTDIR/baseline_epsilon_seed${seed}.txt" 2>&1
done
echo "Baseline measurements complete"
```

---

## Step 2: Probe B measurements

```bash
PROBE_B="$BUILDDIR/bench_boosting_probe_b"

for seed in 42 43 44; do
  "$PROBE_B" --rows 50000 --features 100 --classes 2 --depth 6 --bins 128 \
    --iters 12 --seed $seed --per-kernel-profile \
    > "$OUTDIR/probe_b_gate_seed${seed}.txt" 2>&1

  "$PROBE_B" --rows 1000000 --features 28 --classes 2 --depth 6 --bins 128 \
    --iters 12 --seed $seed --per-kernel-profile \
    > "$OUTDIR/probe_b_higgs_seed${seed}.txt" 2>&1

  "$PROBE_B" --rows 400000 --features 2000 --classes 2 --depth 6 --bins 128 \
    --iters 12 --seed $seed --per-kernel-profile \
    > "$OUTDIR/probe_b_epsilon_seed${seed}.txt" 2>&1
done
echo "Probe B measurements complete"
```

---

## Step 3: Probe D measurements (D2 with corrected dispatcher)

```bash
PROBE_D="$BUILDDIR/bench_boosting_probe_d"

for seed in 42 43 44; do
  "$PROBE_D" --rows 50000 --features 100 --classes 2 --depth 6 --bins 128 \
    --iters 12 --seed $seed --per-kernel-profile \
    > "$OUTDIR/probe_d_gate_seed${seed}.txt" 2>&1

  "$PROBE_D" --rows 1000000 --features 28 --classes 2 --depth 6 --bins 128 \
    --iters 12 --seed $seed --per-kernel-profile \
    > "$OUTDIR/probe_d_higgs_seed${seed}.txt" 2>&1

  "$PROBE_D" --rows 400000 --features 2000 --classes 2 --depth 6 --bins 128 \
    --iters 12 --seed $seed --per-kernel-profile \
    > "$OUTDIR/probe_d_epsilon_seed${seed}.txt" 2>&1
done
echo "Probe D measurements complete"
```

---

## Step 4: Parse and aggregate results

```python
#!/usr/bin/env python3
"""Parse bench_boosting --per-kernel-profile output and compute speedup ratios."""
import re
import glob
import json
from pathlib import Path
import statistics

OUTDIR = Path("/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/catboost-mlx/docs/sprint46/T4/probe-d/data")

def parse_timings(path):
    """Extract histogram_mean and iter_total_mean from bench output."""
    text = Path(path).read_text()
    hm = re.search(r'histogram\s+mean=\s*([\d.]+)', text)
    im = re.search(r'iter_total\s+mean=\s*([\d.]+)', text)
    if hm and im:
        return float(hm.group(1)), float(im.group(1))
    return None, None

shapes = ['gate', 'higgs', 'epsilon']
probes = ['baseline', 'probe_b', 'probe_d']
seeds = [42, 43, 44]

results = {}
for probe in probes:
    results[probe] = {}
    for shape in shapes:
        hist_vals, iter_vals = [], []
        for seed in seeds:
            path = OUTDIR / f"{probe}_{shape}_seed{seed}.txt"
            if path.exists():
                h, it = parse_timings(str(path))
                if h is not None:
                    hist_vals.append(h)
                    iter_vals.append(it)
        if hist_vals:
            results[probe][shape] = {
                'hist_mean_ms': statistics.mean(hist_vals),
                'hist_std_ms': statistics.stdev(hist_vals) if len(hist_vals) > 1 else 0,
                'iter_mean_ms': statistics.mean(iter_vals),
                'iter_std_ms': statistics.stdev(iter_vals) if len(iter_vals) > 1 else 0,
            }

# Compute speedup ratios vs baseline
print("\n=== Speedup ratios vs baseline ===")
print(f"{'Probe':<15} {'Shape':<10} {'hist speedup':>14} {'iter speedup':>14}")
for probe in ['probe_b', 'probe_d']:
    for shape in shapes:
        if shape in results[probe] and shape in results['baseline']:
            base_hist = results['baseline'][shape]['hist_mean_ms']
            base_iter = results['baseline'][shape]['iter_mean_ms']
            probe_hist = results[probe][shape]['hist_mean_ms']
            probe_iter = results[probe][shape]['iter_mean_ms']
            hist_speedup = base_hist / probe_hist
            iter_speedup = base_iter / probe_iter
            print(f"{probe:<15} {shape:<10} {hist_speedup:>14.3f}x {iter_speedup:>14.3f}x")

# Save results JSON
out_path = OUTDIR.parent / 'results.json'
json.dump(results, open(str(out_path), 'w'), indent=2)
print(f"\nResults saved to {out_path}")
```

Save as `$REPO/docs/sprint46/T4/probe-d/parse_results.py` and run:
```bash
python3 "$REPO/docs/sprint46/T4/probe-d/parse_results.py"
```

---

## Step 5: Parity check (ULP comparison)

```bash
# Compare final loss values across 50 iters (regression mode: 1 class)
for probe in baseline probe_b probe_d; do
  for seed in 42 43 44; do
    "$BUILDDIR/bench_boosting_${probe}" \
      --rows 50000 --features 100 --classes 1 --depth 6 --bins 128 \
      --iters 50 --seed $seed \
      2>&1 | grep -E "^(RMSE|loss|final)" | tee "$OUTDIR/parity_${probe}_seed${seed}.txt"
  done
done

# Manual ULP check: compare floating-point outputs between baseline and probes.
# Max ULP <= 4 for RMSE (DEC-008); <= 8 for MultiClass.
python3 - << 'EOF'
import re, glob
from pathlib import Path

OUTDIR = Path("/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/catboost-mlx/docs/sprint46/T4/probe-d/data")

def extract_loss(path):
    text = Path(path).read_text()
    # Look for final loss line like "loss=0.123456" or similar
    m = re.search(r'(?:loss|RMSE)\s*=?\s*([\d.e+-]+)', text)
    return float(m.group(1)) if m else None

import struct
def ulp_diff(a, b):
    ia = struct.unpack('I', struct.pack('f', a))[0]
    ib = struct.unpack('I', struct.pack('f', b))[0]
    return abs(int(ia) - int(ib))

for probe in ['probe_b', 'probe_d']:
    print(f"\n=== {probe} parity vs baseline ===")
    for seed in [42, 43, 44]:
        base_path = OUTDIR / f"parity_baseline_seed{seed}.txt"
        probe_path = OUTDIR / f"parity_{probe}_seed{seed}.txt"
        if base_path.exists() and probe_path.exists():
            base_loss = extract_loss(str(base_path))
            probe_loss = extract_loss(str(probe_path))
            if base_loss and probe_loss:
                ulp = ulp_diff(base_loss, probe_loss)
                status = "PASS" if ulp <= 4 else "FAIL"
                print(f"  seed {seed}: baseline={base_loss:.8f} probe={probe_loss:.8f} ULP={ulp} [{status}]")
EOF
```

---

## Decision table (post-measurement)

Fill in from parsed results:

| Candidate | Shape | hist speedup | iter speedup | ULP | Outcome |
|---|---|---|---|---|---|
| B | Epsilon | ___ | ___ | ___ | A/B/C |
| B | Higgs | ___ | ___ | ___ | — |
| B | Gate | ___ | ___ | ___ | — |
| D2 | Epsilon | ___ | ___ | ___ | A/B/C |
| D2 | Higgs | ___ | ___ | ___ | — |
| D2 | Gate | ___ | ___ | ___ | — |

**Outcome mapping:**
- iter speedup ≥ 3.0× → Outcome A (COMMIT to S47)
- 1.5× ≤ iter speedup < 3.0× → Outcome B (user call)
- iter speedup < 1.5× → Outcome C (HALT)
- Any ULP > 4 (RMSE) or > 8 (MultiClass) → RETIRE (parity failure)
