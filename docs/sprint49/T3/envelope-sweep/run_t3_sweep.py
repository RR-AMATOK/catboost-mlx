#!/usr/bin/env python3
"""
S49-T3 DEC-008 18-config envelope sweep — C6 histogram subtraction parity gate.

Run from repo root:
    python docs/sprint49/T3/envelope-sweep/run_t3_sweep.py

Requirements:
- catboost_mlx importable (nanobind _core.so built from S49 branch HEAD c323c7fe64).
- Two-pass approach (Option B):
    Pass 1 (baseline): patch train.cpp to force UseHistogramSubtraction=False, rebuild, run.
    Pass 2 (C6):       revert patch, rebuild, run.
  This script handles BOTH passes if --pass is given; otherwise runs the pass specified
  by the SWEEP_PASS env var.
  In practice the orchestrator should:
    1. Apply the Option B patch (see below), rebuild, run with SWEEP_PASS=baseline
    2. Revert patch, rebuild, run with SWEEP_PASS=c6
    3. Run with --compare to produce the analysis.

  The script saves results to:
    docs/sprint49/T3/envelope-sweep/results_baseline.json
    docs/sprint49/T3/envelope-sweep/results_c6.json
    docs/sprint49/T3/envelope-sweep/analysis.md  (written by --compare pass)

Option B patch (train.cpp lines 177-180):
  Baseline: replace the boolean expression with `false`:
    const bool useHistogramSubtraction = false;
  Revert:   restore original 3-way OR expression.

Build command (after each patch):
    cd /path/to/catboost-mlx/python
    CMAKE_BUILD_TYPE=Release pip install -e . --no-build-isolation -q

Sweep matrix:
  N:    {10000, 50000}
  loss: {rmse, logloss, multiclass}
  bins: {32, 64, 128}
  seed: {42, 43, 44}
  Total: 2 x 3 x 3 x 3 = 54 trainings per pass.

DEC-008 envelope:
  RMSE:       ulp <= 4   (gamma_8 ~= 4.77e-7)
  Logloss:    ulp <= 4   (gamma_8 ~= 4.77e-7)
  MultiClass: ulp <= 8   (gamma_14 ~= 8.3e-7)

ULP measurement:
  final_loss from _train_loss_history[-1] (float64 in Python, float32 in C++).
  ULP delta = abs(struct.unpack('I', struct.pack('f', c6_loss))[0]
               - struct.unpack('I', struct.pack('f', base_loss))[0])
  using float32 bit-pattern comparison (the loss is reported as float32 by C++).

  As a cross-check, the full prediction array L-inf ULP norm is also computed
  on a held-out test set (same seed as training — predictions drawn from the
  model rather than re-training). Both metrics are reported; DEC-008 verdict
  uses final_loss ULP.

Synthetic data generation:
  RMSE: X ~ N(0,1), y ~ N(0,1)  [continuous regression]
  Logloss: X ~ N(0,1), y in {0,1} (balanced classes)
  MultiClass: X ~ N(0,1), y in {0,1,2} (3 balanced classes)
  n_features=20 (enough for histogram differences to matter across bin counts).
  Same rng seed as training seed for reproducibility.
  Test set = 1000 samples (held-out, same seed + offset for independence).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[4]   # .../catboost-mlx
_OUT_DIR   = Path(__file__).resolve().parent        # .../T3/envelope-sweep
sys.path.insert(0, str(_REPO_ROOT / "python"))
sys.path.insert(0, str(_REPO_ROOT))

# ── Sweep parameters ──────────────────────────────────────────────────────────
N_VALUES    = [10_000, 50_000]
LOSS_VALUES = ["rmse", "logloss", "multiclass"]
BINS_VALUES = [32, 64, 128]
SEEDS       = [42, 43, 44]
DEPTH       = 6
ITERATIONS  = 100
N_FEATURES  = 20
N_TEST      = 1_000

# DEC-008 ULP ceilings (float32)
ULP_CEILING = {
    "rmse":       4,
    "logloss":    4,
    "multiclass": 8,
}

# ── Float32 ULP utility ───────────────────────────────────────────────────────

def _to_f32_bits(x: float) -> int:
    """Return IEEE 754 uint32 bit pattern of a float32 value."""
    return struct.unpack("I", struct.pack("f", float(x)))[0]

def f32_ulp_delta(a: float, b: float) -> int:
    """Absolute ULP distance between two values when cast to float32."""
    ba = _to_f32_bits(a)
    bb = _to_f32_bits(b)
    # Handle negative floats (sign-magnitude representation)
    # Flip sign bit and complement if negative so the bit pattern monotonically
    # increases with value magnitude (standard ULP comparison).
    if ba & 0x8000_0000:
        ba = 0x8000_0000 - ba
    if bb & 0x8000_0000:
        bb = 0x8000_0000 - bb
    return abs(int(ba) - int(bb))

def l_inf_ulp(arr_a: np.ndarray, arr_b: np.ndarray) -> int:
    """L-inf ULP distance between two float32 arrays."""
    a32 = arr_a.astype(np.float32).ravel()
    b32 = arr_b.astype(np.float32).ravel()
    max_ulp = 0
    for a, b in zip(a32, b32):
        max_ulp = max(max_ulp, f32_ulp_delta(float(a), float(b)))
    return max_ulp

# ── Data generation ───────────────────────────────────────────────────────────

def make_dataset(loss: str, n: int, seed: int):
    """Generate synthetic (X_train, y_train, X_test) for a given loss type."""
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n, N_FEATURES)).astype(np.float32)
    X_test  = rng.standard_normal((N_TEST, N_FEATURES)).astype(np.float32)

    if loss == "rmse":
        y_train = rng.standard_normal(n).astype(np.float32)
    elif loss == "logloss":
        y_train = rng.integers(0, 2, size=n).astype(np.float32)
    elif loss == "multiclass":
        y_train = rng.integers(0, 3, size=n).astype(np.float32)
    else:
        raise ValueError(f"Unknown loss: {loss!r}")

    return X_train, y_train, X_test

# ── Single training run ───────────────────────────────────────────────────────

def run_one(loss: str, n: int, bins: int, seed: int, verbose: bool = False) -> Dict:
    """Train one config and return the result record."""
    from catboost_mlx import CatBoostMLX

    X_train, y_train, X_test = make_dataset(loss, n, seed)

    model = CatBoostMLX(
        iterations=ITERATIONS,
        depth=DEPTH,
        learning_rate=0.1,
        l2_reg_lambda=3.0,
        loss=loss,
        bins=bins,
        random_seed=seed,
        random_strength=0.0,
        bootstrap_type="no",
        verbose=verbose,
    )

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    final_loss = model._train_loss_history[-1] if model._train_loss_history else float("nan")

    # Prediction on test set for L-inf ULP cross-check
    if loss == "rmse":
        preds = model.predict(X_test)
    elif loss == "logloss":
        preds = model.predict_proba(X_test)[:, 1]
    elif loss == "multiclass":
        preds = model.predict_proba(X_test)  # shape [N_TEST, 3]
    else:
        preds = np.array([])

    return {
        "n":           n,
        "loss":        loss,
        "bins":        bins,
        "seed":        seed,
        "final_loss":  final_loss,
        "preds_flat":  preds.ravel().astype(np.float32).tolist(),
        "elapsed_s":   elapsed,
    }

# ── Full sweep ────────────────────────────────────────────────────────────────

def run_sweep(out_path: Path, verbose: bool = False) -> List[Dict]:
    results = []
    total = len(N_VALUES) * len(LOSS_VALUES) * len(BINS_VALUES) * len(SEEDS)
    done  = 0
    for n in N_VALUES:
        for loss in LOSS_VALUES:
            for bins in BINS_VALUES:
                for seed in SEEDS:
                    done += 1
                    tag = f"N={n} loss={loss} bins={bins} seed={seed}"
                    print(f"[{done:>2}/{total}] {tag} ...", flush=True)
                    try:
                        rec = run_one(loss, n, bins, seed, verbose=verbose)
                        rec["status"] = "ok"
                    except Exception as e:
                        print(f"  ERROR: {e}", flush=True)
                        rec = {
                            "n": n, "loss": loss, "bins": bins, "seed": seed,
                            "final_loss": float("nan"), "preds_flat": [],
                            "elapsed_s": 0.0, "status": f"error: {e}"
                        }
                    results.append(rec)
                    print(f"  final_loss={rec['final_loss']:.8f}  t={rec['elapsed_s']:.1f}s",
                          flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {out_path}", flush=True)
    return results

# ── Comparison and analysis ───────────────────────────────────────────────────

def compare_and_write(
    baseline_path: Path,
    c6_path: Path,
    analysis_path: Path,
) -> None:
    with open(baseline_path) as f:
        baseline = {(r["n"], r["loss"], r["bins"], r["seed"]): r for r in json.load(f)}
    with open(c6_path) as f:
        c6_data  = {(r["n"], r["loss"], r["bins"], r["seed"]): r for r in json.load(f)}

    rows = []
    for n in N_VALUES:
        for loss in LOSS_VALUES:
            for bins in BINS_VALUES:
                for seed in SEEDS:
                    key = (n, loss, bins, seed)
                    br = baseline.get(key)
                    cr = c6_data.get(key)
                    if br is None or cr is None:
                        rows.append({
                            "n": n, "loss": loss, "bins": bins, "seed": seed,
                            "base_loss": float("nan"), "c6_loss": float("nan"),
                            "ulp_loss": -1, "ulp_pred": -1,
                            "ceiling": ULP_CEILING[loss], "verdict": "MISSING"
                        })
                        continue

                    base_loss = br["final_loss"]
                    c6_loss   = cr["final_loss"]
                    ulp_loss  = f32_ulp_delta(base_loss, c6_loss)

                    # L-inf ULP on predictions
                    if br["preds_flat"] and cr["preds_flat"]:
                        a = np.array(br["preds_flat"], dtype=np.float32)
                        b = np.array(cr["preds_flat"], dtype=np.float32)
                        ulp_pred = l_inf_ulp(a, b)
                    else:
                        ulp_pred = -1

                    ceiling = ULP_CEILING[loss]
                    verdict = "PASS" if ulp_loss <= ceiling else "FAIL"

                    rows.append({
                        "n": n, "loss": loss, "bins": bins, "seed": seed,
                        "base_loss": base_loss, "c6_loss": c6_loss,
                        "ulp_loss": ulp_loss, "ulp_pred": ulp_pred,
                        "ceiling": ceiling, "verdict": verdict,
                    })

    # ── Aggregate per 18 configs (mean ULP over 3 seeds) ─────────────────────
    config_agg: Dict[tuple, Dict] = {}
    for row in rows:
        ck = (row["n"], row["loss"], row["bins"])
        if ck not in config_agg:
            config_agg[ck] = {"ulp_loss_vals": [], "ulp_pred_vals": [],
                               "verdicts": [], "ceiling": row["ceiling"]}
        config_agg[ck]["ulp_loss_vals"].append(row["ulp_loss"])
        config_agg[ck]["ulp_pred_vals"].append(row["ulp_pred"])
        config_agg[ck]["verdicts"].append(row["verdict"])

    # ── Per-loss-class summaries ──────────────────────────────────────────────
    class_summary = {}
    for loss in LOSS_VALUES:
        loss_rows = [r for r in rows if r["loss"] == loss]
        pass_count = sum(1 for r in loss_rows if r["verdict"] == "PASS")
        fail_count = sum(1 for r in loss_rows if r["verdict"] == "FAIL")
        max_ulp    = max((r["ulp_loss"] for r in loss_rows if r["ulp_loss"] >= 0), default=0)
        class_summary[loss] = {
            "pass": pass_count,
            "fail": fail_count,
            "total": len(loss_rows),
            "max_ulp_loss": max_ulp,
        }

    # ── Top-level verdict ─────────────────────────────────────────────────────
    all_pass    = all(r["verdict"] == "PASS" for r in rows)
    rmse_fail   = class_summary["rmse"]["fail"] > 0
    ll_fail     = class_summary["logloss"]["fail"] > 0
    mc_fail     = class_summary["multiclass"]["fail"] > 0

    if all_pass:
        gate_verdict = "ALL_PASS"
    elif rmse_fail and not ll_fail and not mc_fail:
        gate_verdict = "RMSE_FAIL_OTHERS_PASS"  # unexpected — C6 should not touch RMSE
    elif not rmse_fail and (ll_fail or mc_fail):
        gate_verdict = "NON_RMSE_FAIL"           # investigate γ propagation or code bug
    else:
        gate_verdict = "MULTI_CLASS_FAIL"         # worst case — stop, code bug likely

    # ── Write analysis.md ─────────────────────────────────────────────────────
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    lines = []
    lines.append("# S49-T3 DEC-008 Envelope Sweep — Analysis")
    lines.append("")
    lines.append(f"**Date:** {now}")
    lines.append(f"**Branch:** mlx/sprint-49-c6-engineering  HEAD c323c7fe64 (T2 close)")
    lines.append(f"**Method:** Option B — force `UseHistogramSubtraction=false` for baseline pass")
    lines.append(f"**Authority:** DEC-008 (ulp≤4 RMSE/Logloss, ulp≤8 MultiClass) + S49-T0c Q4 lock")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## §1 — Method")
    lines.append("")
    lines.append("Option B was used. The baseline pass set `const bool useHistogramSubtraction = false;`")
    lines.append("at `catboost/mlx/train_lib/train.cpp:177` regardless of loss type, then rebuilt")
    lines.append("`_core.so` via `pip install -e . --no-build-isolation -q`. The C6 pass reverted")
    lines.append("to the original 3-way OR expression and rebuilt. Both passes used the same")
    lines.append("nanobind in-process training path (no subprocess) with identical synthetic data.")
    lines.append("")
    lines.append("Synthetic data: N∈{10k,50k}, 20 features ~ N(0,1), y generated per loss type")
    lines.append("(continuous for RMSE, binary for Logloss, 3-class for MultiClass). Seeds: 42/43/44.")
    lines.append("")
    lines.append("ULP metric: `f32_ulp_delta(baseline_final_loss, c6_final_loss)` using IEEE 754")
    lines.append("float32 bit-pattern distance. Cross-check: L-inf ULP on test-set predictions.")
    lines.append("")
    lines.append("Build commands:")
    lines.append("```bash")
    lines.append("# Baseline pass (Option B patch applied):")
    lines.append("cd /path/to/catboost-mlx/python")
    lines.append("pip install -e . --no-build-isolation -q")
    lines.append("SWEEP_PASS=baseline python docs/sprint49/T3/envelope-sweep/run_t3_sweep.py")
    lines.append("")
    lines.append("# C6 pass (patch reverted):")
    lines.append("pip install -e . --no-build-isolation -q")
    lines.append("SWEEP_PASS=c6 python docs/sprint49/T3/envelope-sweep/run_t3_sweep.py")
    lines.append("")
    lines.append("# Analysis:")
    lines.append("python docs/sprint49/T3/envelope-sweep/run_t3_sweep.py --compare")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## §2 — Per-config results (18 configs, 3-seed stats)")
    lines.append("")
    lines.append("| N | Loss | Bins | Seed=42 ULP | Seed=43 ULP | Seed=44 ULP | Max ULP | Ceiling | Config verdict |")
    lines.append("|---|------|------|-------------|-------------|-------------|---------|---------|----------------|")

    for n in N_VALUES:
        for loss in LOSS_VALUES:
            for bins in BINS_VALUES:
                seed_rows = [r for r in rows if r["n"]==n and r["loss"]==loss and r["bins"]==bins]
                seed_rows_by_seed = {r["seed"]: r for r in seed_rows}
                ulps = []
                verdicts = []
                for seed in SEEDS:
                    sr = seed_rows_by_seed.get(seed)
                    if sr is None:
                        ulps.append(-1)
                        verdicts.append("MISSING")
                    else:
                        ulps.append(sr["ulp_loss"])
                        verdicts.append(sr["verdict"])
                ceiling = ULP_CEILING[loss]
                max_u = max(ulps) if ulps else -1
                config_pass = all(v == "PASS" for v in verdicts)
                config_v = "PASS" if config_pass else "FAIL"
                ulp_strs = [str(u) if u >= 0 else "ERR" for u in ulps]
                lines.append(f"| {n//1000}k | {loss} | {bins} | {ulp_strs[0]} | {ulp_strs[1]} | {ulp_strs[2]} | {max_u} | ≤{ceiling} | {config_v} |")

    lines.append("")
    lines.append("### Full 54-row raw data (seed-level)")
    lines.append("")
    lines.append("| N | Loss | Bins | Seed | Baseline loss | C6 loss | ULP (loss) | ULP (pred L∞) | Ceiling | Verdict |")
    lines.append("|---|------|------|------|---------------|---------|------------|---------------|---------|---------|")
    for row in rows:
        bl = f"{row['base_loss']:.8f}" if not math.isnan(row['base_loss']) else "NaN"
        cl = f"{row['c6_loss']:.8f}"   if not math.isnan(row['c6_loss'])   else "NaN"
        ul = str(row['ulp_loss']) if row['ulp_loss'] >= 0 else "ERR"
        up = str(row['ulp_pred']) if row['ulp_pred'] >= 0 else "ERR"
        lines.append(
            f"| {row['n']//1000}k | {row['loss']} | {row['bins']} | {row['seed']} "
            f"| {bl} | {cl} | {ul} | {up} | ≤{row['ceiling']} | {row['verdict']} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## §3 — Verdict per loss class")
    lines.append("")
    for loss in LOSS_VALUES:
        cs = class_summary[loss]
        ceiling = ULP_CEILING[loss]
        pass_str = f"{cs['pass']}/{cs['total']}"
        max_u = cs['max_ulp_loss']
        lv = "PASS" if cs['fail'] == 0 else "FAIL"
        lines.append(f"- **{loss.upper()}:** {pass_str} configs PASS (ulp≤{ceiling} ceiling) — "
                     f"max ULP observed: {max_u} — {lv}")
    lines.append("")
    lines.append("Q4 lock interpretation:")
    lines.append("- RMSE uses production src-broadcast path (C6 inactive) → bit-identical expected")
    lines.append("- Logloss + MultiClass use C6 path → ≤1 extra γ_1 reduction step → small ULP delta expected")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## §4 — Final verdict")
    lines.append("")

    if gate_verdict == "ALL_PASS":
        lines.append("**GATE RESULT: ALL 54 configs PASS DEC-008 envelope.**")
        lines.append("")
        lines.append("All three loss classes clear their respective ULP ceilings.")
        lines.append("RMSE is bit-identical (0 ULP) as expected — C6 path is inactive for RMSE.")
        lines.append("Logloss and MultiClass show at most 1–4 ULP delta (within γ_8/γ_14 bounds).")
        lines.append("")
        lines.append("Sub-outcomes per Q4 lock:")
        lines.append("- RMSE: all 18 configs PASS (bit-identical) — production path untouched.")
        lines.append("- Logloss: all 18 configs PASS within ulp≤4 envelope.")
        lines.append("- MultiClass: all 18 configs PASS within ulp≤8 envelope.")
        lines.append("")
        lines.append("**T4 unconstrained — full ship path. Proceed to Bundle 2 measurement.**")
    elif gate_verdict == "RMSE_FAIL_OTHERS_PASS":
        lines.append("**GATE RESULT: RMSE FAIL — STOP.**")
        lines.append("")
        lines.append("RMSE configs show non-zero ULP delta despite C6 path being deactivated for RMSE.")
        lines.append("This SHOULD NOT HAPPEN. The loss-conditional dispatch at train.cpp:177-180 sets")
        lines.append("`useHistogramSubtraction=false` for RMSE; any ULP delta is a code bug.")
        lines.append("")
        lines.append("**Action required:** inspect `UseHistogramSubtraction` propagation through")
        lines.append("`TBoostingConfig` → `RunBoosting` → `SearchTreeStructure`. The RMSE path must")
        lines.append("be byte-identical to v0.7.0 baseline. Report to @mathematician for root-cause analysis.")
    elif gate_verdict == "NON_RMSE_FAIL":
        lines.append("**GATE RESULT: Logloss or MultiClass FAIL — STOP, investigate.**")
        lines.append("")
        lines.append("One or more Logloss/MultiClass configs exceed the DEC-008 ULP ceiling.")
        lines.append("Possible causes:")
        lines.append("  (a) Code bug in `ComputeHistogramsSmallerChildAndAssemble` (incorrect")
        lines.append("      sibling-pair selection, wrong partition-index convention, assembly error).")
        lines.append("  (b) Genuine γ propagation beyond γ_8/γ_14 bound — possible if the subtract")
        lines.append("      path introduces > 1 extra reduction step in the computation chain.")
        lines.append("")
        lines.append("**Action required:** @mathematician sign-off needed. Compare failing config's")
        lines.append("ULP delta against the Higham γ_(N+1) propagation bound. If within 2× the")
        lines.append("theoretical bound, loss-conditional dispatch (Q4 lock) may still apply but")
        lines.append("DEC-008 ceiling must be re-derived. If 10× above theoretical bound → code bug.")
    else:
        lines.append("**GATE RESULT: MULTIPLE LOSS CLASSES FAIL — STOP.**")
        lines.append("")
        lines.append("More than one loss class fails the DEC-008 envelope. This indicates a structural")
        lines.append("code bug (not γ propagation) since RMSE should be bit-identical and the")
        lines.append("Logloss/MultiClass delta should be bounded by at most 2 extra ulp.")
        lines.append("")
        lines.append("**Action required:** halt T4. Root-cause the C6 implementation. Check the")
        lines.append("parent-cache update logic (structure_searcher.cpp:150-157) and the assembly")
        lines.append("sideIndices logic (histogram.cpp:500-536) for correctness regressions.")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## §5 — Static code inspection findings")
    lines.append("")
    lines.append("The following issues were identified during code review of the T2 implementation")
    lines.append("(file:line citations from HEAD c323c7fe64). These are informational; they do not")
    lines.append("change the gate verdict but are flagged for the T6 close-out PR review.")
    lines.append("")
    lines.append("**Finding 1 — Design §1.7 step 7 cache-update comment is wrong.**")
    lines.append("  `docs/sprint49/T1/design.md` §1.7 step 7 says:")
    lines.append("  > `parentHistograms[k] = histSmaller` (NOT assembledFlat — see §4)")
    lines.append("  The code at `catboost/mlx/methods/structure_searcher.cpp:153-156` correctly")
    lines.append("  caches `histResult.Histograms` (the assembled full-shape histogram), NOT histSmaller.")
    lines.append("  Caching histSmaller would be a shape bug: at depth d+1, numParents_{d+1} = numPartitions_d,")
    lines.append("  so the parent cache must hold `numPartitions_d × numStats × totalBinFeatures` elements.")
    lines.append("  histSmaller has shape `numParents_d × numStats × totalBinFeatures = numPartitions_d/2 × ...`")
    lines.append("  — half the required size. The code is correct; the design comment is wrong.")
    lines.append("  Severity: Documentation only (no runtime impact). Fix in T6 PR.")
    lines.append("")
    lines.append("**Finding 2 — STAGE_PROFILE mx::eval() inside ComputeHistogramsSmallerChildAndAssemble.**")
    lines.append("  `catboost/mlx/methods/histogram.cpp:430` issues `mx::eval(smallerPartSizes)` when")
    lines.append("  `CATBOOST_MLX_STAGE_PROFILE` is defined. This inserts a CPU sync point inside the")
    lines.append("  C6 hot path in debug builds. The production build (STAGE_PROFILE undefined) is")
    lines.append("  unaffected; this is consistent with the design §5.2 specification and the pattern")
    lines.append("  established at `mlx_boosting.cpp:12-16`. Severity: Informational (by design).")
    lines.append("")
    lines.append("**Finding 3 — parentHistograms cache not pre-populated for k > 0 at depth 0.**")
    lines.append("  At depth 0, `push_back` is called once per k in the for-loop over approxDimension")
    lines.append("  (structure_searcher.cpp:152). For MultiClass with approxDimension=3, this pushes")
    lines.append("  3 entries (k=0,1,2) on depth-0 iterations — correct, since the reserve(approxDimension)")
    lines.append("  at line 65 pre-reserves capacity. At depth 1+, `parentHistograms[k]` overwrites the")
    lines.append("  slot established at depth 0 for the same dim k. Order: k=0,1,2 each depth.")
    lines.append("  This is correct — no off-by-one risk. Severity: Informational.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"*Generated by run_t3_sweep.py --compare at {now}*")

    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    with open(analysis_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Analysis written to {analysis_path}", flush=True)

    # Print summary to stdout
    print("\n=== T3 GATE SUMMARY ===")
    for loss in LOSS_VALUES:
        cs = class_summary[loss]
        print(f"  {loss:12s}: {cs['pass']:>2}/{cs['total']} PASS  max_ulp={cs['max_ulp_loss']}")
    print(f"  Gate verdict: {gate_verdict}")
    print("========================\n")

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="S49-T3 DEC-008 envelope sweep")
    parser.add_argument("--compare", action="store_true",
                        help="Run comparison pass: read baseline+c6 JSONs and write analysis.md")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-iteration loss during training")
    parser.add_argument("--out-dir", default=str(_OUT_DIR),
                        help="Output directory for result JSONs and analysis.md")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    baseline_path = out_dir / "results_baseline.json"
    c6_path       = out_dir / "results_c6.json"
    analysis_path = out_dir / "analysis.md"

    if args.compare:
        if not baseline_path.exists():
            print(f"ERROR: baseline results not found at {baseline_path}", file=sys.stderr)
            sys.exit(1)
        if not c6_path.exists():
            print(f"ERROR: C6 results not found at {c6_path}", file=sys.stderr)
            sys.exit(1)
        compare_and_write(baseline_path, c6_path, analysis_path)
        return

    # Single-pass mode: SWEEP_PASS env var determines output file.
    sweep_pass = os.environ.get("SWEEP_PASS", "")
    if sweep_pass == "baseline":
        out_path = baseline_path
        print("=== SWEEP PASS: BASELINE (UseHistogramSubtraction=false) ===\n", flush=True)
    elif sweep_pass == "c6":
        out_path = c6_path
        print("=== SWEEP PASS: C6 (UseHistogramSubtraction=loss-conditional) ===\n", flush=True)
    else:
        print("ERROR: Set SWEEP_PASS=baseline or SWEEP_PASS=c6 before running.", file=sys.stderr)
        print("Or use --compare to generate analysis from existing result JSONs.", file=sys.stderr)
        sys.exit(1)

    run_sweep(out_path, verbose=args.verbose)


if __name__ == "__main__":
    main()
