#!/usr/bin/env python3
"""Parse bench_boosting --per-kernel-profile output and compute speedup ratios.

Usage:
    python3 docs/sprint46/T4/probe-d/parse_results.py

Outputs:
    - Console table of speedup ratios
    - docs/sprint46/T4/probe-d/results.json
    - docs/sprint46/T4/probe-d/analysis.md (T5 decision table)
"""
import re
import json
import struct
import statistics
from pathlib import Path

REPO = Path("/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/catboost-mlx")
OUTDIR = REPO / "docs/sprint46/T4/probe-d/data"
RESULTS_JSON = REPO / "docs/sprint46/T4/probe-d/results.json"
ANALYSIS_MD = REPO / "docs/sprint46/T4/probe-d/analysis.md"

SHAPES = ['gate', 'higgs', 'epsilon']
PROBES = ['baseline', 'probe_b', 'probe_d']
SEEDS = [42, 43, 44]
SHAPE_LABELS = {
    'gate': 'Gate-config (50k×100)',
    'higgs': 'Higgs-1M (1M×28)',
    'epsilon': 'Epsilon-proxy (400k×2000)',
}


def parse_timings(path):
    """Extract histogram mean and iter_total mean from bench_boosting stdout."""
    text = path.read_text()
    hm = re.search(r'histogram\s+mean=\s*([\d.]+)', text)
    im = re.search(r'iter_total\s+mean=\s*([\d.]+)', text)
    if hm and im:
        return float(hm.group(1)), float(im.group(1))
    return None, None


def parse_loss(path):
    """Extract final loss value from bench output."""
    text = path.read_text()
    # bench_boosting outputs "final loss = X.XXXXXX" or "RMSE = X" at end
    m = re.search(r'(?:final\s+loss|RMSE)\s*[=:]\s*([\d.e+\-]+)', text, re.IGNORECASE)
    return float(m.group(1)) if m else None


def ulp_diff_f32(a, b):
    """Compute ULP distance between two float32 values."""
    ia = struct.unpack('I', struct.pack('f', float(a)))[0]
    ib = struct.unpack('I', struct.pack('f', float(b)))[0]
    return abs(int(ia) - int(ib))


def compute_ci95(values):
    """Compute 95% CI as mean ± 1.96 * stderr."""
    if len(values) < 2:
        return 0.0
    std = statistics.stdev(values)
    stderr = std / (len(values) ** 0.5)
    return 1.96 * stderr


# ── Collect timing results ────────────────────────────────────────────────────
results = {}
for probe in PROBES:
    results[probe] = {}
    for shape in SHAPES:
        hist_vals, iter_vals = [], []
        for seed in SEEDS:
            path = OUTDIR / f"{probe}_{shape}_seed{seed}.txt"
            if path.exists():
                h, it = parse_timings(path)
                if h is not None:
                    hist_vals.append(h)
                    iter_vals.append(it)
        if hist_vals:
            results[probe][shape] = {
                'hist_mean_ms': statistics.mean(hist_vals),
                'hist_std_ms': statistics.stdev(hist_vals) if len(hist_vals) > 1 else 0.0,
                'iter_mean_ms': statistics.mean(iter_vals),
                'iter_std_ms': statistics.stdev(iter_vals) if len(iter_vals) > 1 else 0.0,
                'n_seeds': len(hist_vals),
            }
        else:
            results[probe][shape] = None

# ── Compute speedups ──────────────────────────────────────────────────────────
speedups = {}
for probe in ['probe_b', 'probe_d']:
    speedups[probe] = {}
    for shape in SHAPES:
        base = results.get('baseline', {}).get(shape)
        probe_r = results.get(probe, {}).get(shape)
        if base and probe_r:
            hist_sp = base['hist_mean_ms'] / probe_r['hist_mean_ms']
            iter_sp = base['iter_mean_ms'] / probe_r['iter_mean_ms']
            speedups[probe][shape] = {
                'hist_speedup_mean': hist_sp,
                'iter_speedup_mean': iter_sp,
                'iter_speedup_ci95': compute_ci95(
                    [results['baseline'][shape]['iter_mean_ms'] / r
                     for r in [probe_r['iter_mean_ms']]]
                ),
            }

# ── Parity check ─────────────────────────────────────────────────────────────
parity = {}
for probe in ['probe_b', 'probe_d']:
    parity[probe] = {}
    for shape in SHAPES:
        ulps = []
        for seed in SEEDS:
            base_path = OUTDIR / f"parity_baseline_{shape}_seed{seed}.txt"
            probe_path = OUTDIR / f"parity_{probe}_{shape}_seed{seed}.txt"
            if base_path.exists() and probe_path.exists():
                bl = parse_loss(base_path)
                pl = parse_loss(probe_path)
                if bl is not None and pl is not None:
                    ulps.append(ulp_diff_f32(bl, pl))
        parity[probe][shape] = {
            'max_ulp': max(ulps) if ulps else None,
            'mean_ulp': statistics.mean(ulps) if ulps else None,
            'n': len(ulps),
        }

# ── Outcome classification ────────────────────────────────────────────────────
def classify_outcome(iter_speedup, max_ulp, loss_type='rmse'):
    ulp_ceiling = 4 if loss_type == 'rmse' else 8
    if max_ulp is not None and max_ulp > ulp_ceiling:
        return 'RETIRE (parity failure)'
    if iter_speedup is None:
        return 'NOT MEASURED'
    if iter_speedup >= 3.0:
        return 'Outcome A (COMMIT)'
    if iter_speedup >= 1.5:
        return 'Outcome B (user call)'
    return 'Outcome C (HALT)'

# ── Print console report ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("S46-T4 Option 2 — Phase 3 Measurement Results")
print("=" * 70)
print(f"\n{'Candidate':<12} {'Shape':<22} {'hist speedup':>14} {'iter speedup':>14} {'max ULP':>10}")
print("-" * 75)
for probe, label in [('probe_b', 'B (hierarchical)'), ('probe_d', 'D2 (split-K)')]:
    for shape in SHAPES:
        sp = speedups.get(probe, {}).get(shape, {})
        par = parity.get(probe, {}).get(shape, {})
        hist_sp = f"{sp.get('hist_speedup_mean', 0):.3f}x" if sp else "N/A"
        iter_sp = f"{sp.get('iter_speedup_mean', 0):.3f}x" if sp else "N/A"
        max_ulp = par.get('max_ulp')
        ulp_str = str(max_ulp) if max_ulp is not None else "N/A"
        shape_short = {'gate': 'Gate', 'higgs': 'Higgs-1M', 'epsilon': 'Epsilon'}[shape]
        print(f"{label:<12} {shape_short:<22} {hist_sp:>14} {iter_sp:>14} {ulp_str:>10}")

print("\n" + "=" * 70)
print("EPSILON OUTCOMES (primary decision anchor for T5)")
print("=" * 70)
for probe, label in [('probe_b', 'B'), ('probe_d', 'D2')]:
    sp = speedups.get(probe, {}).get('epsilon', {})
    par = parity.get(probe, {}).get('epsilon', {})
    iter_sp = sp.get('iter_speedup_mean') if sp else None
    max_ulp = par.get('max_ulp')
    outcome = classify_outcome(iter_sp, max_ulp)
    iter_str = f"{iter_sp:.3f}x" if iter_sp else "N/A"
    print(f"  {label}: iter speedup = {iter_str}, max ULP = {max_ulp} → {outcome}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
json_output = {
    'B': {},
    'D2': {},
}
for probe, key in [('probe_b', 'B'), ('probe_d', 'D2')]:
    for shape in SHAPES:
        sp = speedups.get(probe, {}).get(shape, {})
        par = parity.get(probe, {}).get(shape, {})
        if sp:
            json_output[key][shape] = {
                'iter_speedup_mean': sp.get('iter_speedup_mean'),
                'iter_speedup_ci95': [
                    sp.get('iter_speedup_mean', 0) - sp.get('iter_speedup_ci95', 0),
                    sp.get('iter_speedup_mean', 0) + sp.get('iter_speedup_ci95', 0),
                ],
                'histogram_speedup_mean': sp.get('hist_speedup_mean'),
                'parity_ulp': par.get('max_ulp'),
            }

RESULTS_JSON.write_text(json.dumps(json_output, indent=2))
print(f"\nResults JSON written to {RESULTS_JSON}")

print("\nRun complete. Fill analysis.md with outcomes for T5 decision gate.")
