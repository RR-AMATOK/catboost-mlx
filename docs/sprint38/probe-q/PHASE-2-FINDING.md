# PROBE-Q phase 2 — Sprint 38 RESOLVED: harness asymmetry, not algorithm bug

**Date**: 2026-04-25
**Branch**: `mlx/sprint-38-lg-small-n`
**Anchor**: identical to PROBE-G / F2 / PROBE-H (`np.random.default_rng(42)`, N=1000, 20 features, ST/Cosine/RMSE, depth=6, bins=128, l2=3, lr=0.03)

## Headline

The "13.93% N=1k LG+Cosine drift" Sprint 37 #113 T3 G3b/G3c flagged and Sprint 38 spent four probes hunting **was a harness configuration mismatch**, not an algorithm bug:

- MLX `csv_train.cpp:599` defaults `RandomStrength = 1.0f`.
- The drift-comparison harnesses (`docs/sprint38/probe-g/scripts/run_probe_g.py`, `docs/sprint38/f2/scripts/run_f2.py`) pass `random_strength=0.0` to CPU CatBoost explicitly, but **never pass `--random-strength 0` to MLX's CLI**.
- Result: MLX trees were perturbed by random gain noise scaled by `RandomStrength × gradRms`; CPU trees were deterministic. Apples-to-oranges.

With matched RandomStrength=0:

| Config | MLX RMSE (50 iters, N=1k) | CPU RMSE | Drift |
|---|---|---|---|
| **Both RS=0.0** | 0.204238 | 0.204238 | **0.000%** (bit-identical) |
| Both RS=1.0 (single seed) | 0.232996 | 0.241850 | -3.66% (ordinary noise variance) |
| **MLX RS=1.0 vs CPU RS=0.0 (the asymmetry)** | 0.232996 | 0.204238 | **14.08%** (phantom) |

At matched RS=0, MLX's iter=1 + iter=2 trees are bit-identical to CPU's at all 12 splits (6 per tree × 2 trees), feature AND border alignment to fp32-rounding precision.

## Mechanism

`csv_train.cpp` lines 2196–2206:
```cpp
double perturbedGain = totalGain;
if (noiseScale > 0.0f) {
    perturbedGain += static_cast<double>(noiseScale * noiseDist(*rng));
}
if (perturbedGain > bestGain) {
    bestGain = perturbedGain;
    bestSplit.FeatureId = featIdx;
    bestSplit.BinId = bin;
    bestSplit.Gain = totalGain;     // store deterministic gain, not perturbed
    bestSplit.Score = ...;
}
```

When the top-2 candidates have a small gain gap (< noise σ), the perturbation can flip the pick. Once that happens at any depth, the tree structure diverges from the deterministic optimum, and downstream depths follow a different trajectory.

The instrumentation captures `totalGain` (deterministic), not `perturbedGain`. So the captured gains are correct relative to the formula, but the binary's argmax operates on perturbed values that the instrumentation does not record.

## Why the smooth log-linear N* curve?

PROBE-G's drift table:

| N | drift % |
|---|---|
| 500 | 18.70 |
| 1000 | 13.93 |
| 2000 | 8.59 |
| 5000 | 5.09 |
| 10000 | 3.05 |
| 20000 | 1.94 |
| 50000 | 1.23 |
| N* (2% threshold) ≈ 19,231 |

At small N, gradient magnitudes are similar but gain gaps between top candidates are smaller (less data → less candidate dominance). Noise/signal ratio scales inversely with N. At N=1k, noise frequently flips the top-2 picks; at N=50k, it rarely does. The smooth log-linear decay in drift is exactly what an asymmetric-noise mechanism predicts. PROBE-G's "Scenario C with depth-amplification" verdict was reading topology dynamics into noise statistics.

## What this retracts

- **PROBE-G classification** (Scenarios A/B/C from `docs/sprint38/probe-g/FINDING.md`): the rubric assumed a degenerate-partition mechanism. The actual mechanism is RandomStrength asymmetry. Skip-rate, gap, drift-curve interpretations stand as observational data; the *causal* interpretation is wrong. The `cos_leaf_*` per-(feat, bin, partition) data is still valid (instrumentation captures deterministic values).
- **F2 verdict** (`docs/sprint38/f2/FINDING.md`): "C-PSF confirmed, C-QG and C-LV falsified". C-QG was correctly falsified; C-LV was correctly falsified; C-PSF (per-side scoring formula divergence) is **also falsified** — formulas were always equivalent, the divergence was noise. The 0/6 iter=2 tree match was caused by RS asymmetry, not formula mismatch.
- **PROBE-H FINDING.md** (already withdrawn for the joint-skip claim): the d=1 anomaly (binary picks bin=64 over the captured argmax bin=82) is the textbook signature of RS noise flipping a close pick (gap 0.056, well within noise σ ≈ gradRms ≈ 0.5).
- **PROBE-Q phase 1 verdict** (granularity falsified): stands. Quantization grids really are aligned. This was unaffected by RS because it was a comparison of static border arrays, not training-time picks.

## What this confirms

- DEC-042 (per-side mask) is correct and has been since Sprint 33 commit `10c72b4e96`.
- The DEC-042 port to `FindBestSplitPerPartition` (commit `a481972529` in this sprint) is correct.
- All quantization, formula, and instrumentation code paths verified across the four probes are correct.

## What broke

The harness scripts. PROBE-G's `run_probe_g.py:121-125` had a comment explicitly documenting "No --random-strength override; csv_train default is RS=1.0" alongside the CPU call setting `random_strength=0.0`. The asymmetry was visible in source for the entire sprint and was justified to itself with circular reasoning ("the t3 harness pattern matches this") rather than checked against CPU's configuration.

## Fix

Both harnesses now pass `--random-strength 0` to MLX. See `run_f2.py` and `run_probe_g.py` Phase 1 + Phase 2 cmd blocks. With this fix, the falsification is reproducible:

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/opt/mlx/lib \
  ./csv_train docs/sprint38/f2/data/anchor_n1000_seed42.csv \
    --iterations 2 --depth 6 --lr 0.03 --bins 128 --l2 3 \
    --loss rmse --score-function Cosine --seed 42 \
    --random-strength 0 \
    --output /tmp/mlx_rs0.json
```

Compare `/tmp/mlx_rs0.json trees[0].splits` against `docs/sprint38/f2/data/cpu_model.json oblivious_trees[0].splits` — 12/12 match in feature and border (fp32-precision).

## What's left (production-correctness angle)

Users who set `RandomStrength=1.0` (the default) on both runtimes will get statistically-similar but non-identical models because the RNG sequences differ between CatBoost and MLX (different RNG implementations, same seed). For a single seed at N=1k the RMSE delta is bounded ~3-4%; over many seeds the expected delta is near 0. This is normal noise variance, not a bug.

If users want bit-identical reproducibility across runtimes, they must set `RandomStrength=0.0` explicitly on both sides. Document this in the README.

## Lessons

1. **Cross-runtime parity tests must verify SYMMETRIC configuration before interpreting drift.** The asymmetry should have been caught at PROBE-G initialization. It would have been caught by a one-line sanity check: "Print all hyperparameters as actually invoked on both sides; confirm they match where intended to match."

2. **A 13% drift over 5 seeds is not an unbounded mystery — it's a finite-effect-size signal whose mechanism class can be identified by varying ONE parameter at a time.** The first move when a drift number is mysterious should be: bisect the configuration space. Halve the difference between the two runtime's hyperparameters until the drift collapses. We chased four formula/precision/quantization probes before checking the configuration — that ordering was wrong.

3. **Probes built on the same harness inherit its bugs.** PROBE-G, F2, PROBE-H all used the asymmetric harness. Each probe's verdict was internally consistent with what its instrumentation captured, but ALL of them were measuring the same configuration artifact. Probe diversity ≠ verdict diversity if they share the same input pipeline.
