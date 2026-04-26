# PROBE-G Finding — small-N residual mechanism capture

**Date**: 2026-04-25
**Branch**: `mlx/sprint-38-lg-small-n`
**Anchor**: `np.random.default_rng(42)`, **N=1000**, 20 features,
y = 0.5·X[0] + 0.3·X[1] + 0.1·noise, ST/Cosine/RMSE, depth=6, bins=128, l2=3, lr=0.03
**Build**: `csv_train_probe_g` with `-DCOSINE_RESIDUAL_INSTRUMENT
-DPROBE_E_INSTRUMENT -DPROBE_D_ARM_AT_ITER=1`
**Kernel md5**: `9edaef45b99b9db3e2717da93800e76f` (host-side instrumentation only — kernels untouched)

> **Status**: COMPLETE — empirical capture validated. **Classification AMENDED (2026-04-25,
> @devils-advocate stress-test): Scenario C confirmed at d≤2 only. d≥3 is a regime change —
> a different mechanism class (plausibly continuous-precision-class at small leaves), NOT
> depth-amplification of the topology mechanism. PROBE-G structurally cannot localize the d≥3
> residual. F2 (CPU-tree split comparison at the N=1k seed=42 anchor) is the cheaper next step
> before PROBE-H.** Data landed 2026-04-25 under `docs/sprint38/probe-g/data/`. All four
> phases executed cleanly.

---

## Why this probe

S38 T0a (math, `docs/sprint38/lg-small-n/math-derivation.md`) and T0b (code,
`docs/sprint38/lg-small-n/code-reading.md`) hypothesised **H3** as the
small-N drift mechanism — that `FindBestSplitPerPartition` carries an
unfixed joint-skip `continue` (csv_train.cpp:2304 one-hot, :2388 ordinal)
that DEC-042 left in place when it landed the per-side mask in
`FindBestSplit` only. H3 predicted ST+Cosine at N=1k, depth=6 should drift
**< 2%** because ST uses the FIXED `FindBestSplit` ordinal path.

The cheap discriminator (ST+Cosine N=1k, 3 seeds) was run separately and
**refuted H3 as the dominant mechanism**: aggregate drift = **13.96%**
(per-seed: 13.38%, 14.08%, 14.41%). Since ST uses the post-DEC-042 fixed
path, a **separate small-N mechanism (H1 family or unknown) is real and
load-bearing**. The math agent's pre-emptive falsification of H1 ("the
per-side mask injects signal, not noise, at small leaves") is empirically
contradicted at this regime.

PROBE-G is the empirical capture — mirrors **PROBE-E from S33**
(`docs/sprint33/probe-e/FINDING.md`) but at the small-N regime — to surface
what mechanism is actually firing.

## Method

The PROBE_E_INSTRUMENT instrumentation (lives in
`catboost/mlx/tests/csv_train.cpp:1899-1950` inside `FindBestSplit`'s inner
per-(p, k) loop) captures, for every (feat, bin, partition) candidate at
iter=2 d=0..5:

- `mlx_skipped` — whether the pre-DEC-042 joint-skip would have fired
- `mlx_termNum`, `mlx_termDen` — pre-DEC-042 (joint-skip) contribution
- `cpu_termNum`, `cpu_termDen` — per-side mask formula (= post-DEC-042 MLX-actual)

**Critical semantic note**: in the post-DEC-042 codebase, MLX's actual
behavior matches `cpu_termNum/Den` (the per-side mask). The diff
`cpu - mlx` measures what DEC-042 closed at the captured anchor. **It does
NOT directly measure MLX-vs-CPU-CatBoost divergence**, because both columns
assume MLX's path; CPU CatBoost's runtime trace would require a separate
instrumentation hook in the CPU build. Thus PROBE-G can:

1. Confirm whether DEC-042 had any structural effect at N=1k (skip rate,
   gap between post-fix and pre-fix per-bin gains).
2. Surface mechanisms that are **proxies** for MLX-vs-CPU divergence:
   per-bin contribution magnitude, per-bin precision noise, MLX argmax
   stability across runs.
3. Identify what classes are NOT captured (quantization borders,
   basePred init, leaf-value computation, fp32 reduction noise) — these
   would require additional probes if PROBE-G's diagnostics come back
   inconclusive.

## Phases (run by `scripts/run_probe_g.py`)

1. **Phase 1 — anchor capture**: build `csv_train_probe_g`; run on N=1k
   seed=42 anchor with iters=2 (instrumentation arms at iter=2). Drops
   `data/cos_leaf_seed42_depth{0..5}.csv` and
   `data/cos_accum_seed42_depth{0..5}.csv`.
2. **Phase 2 — scaling sweep**: ST+Cosine at
   N ∈ {500, 1000, 2000, 5000, 10000, 20000, 50000}, 5 seeds each,
   iters=50, mirrors the canonical `t3-measure` harness exactly. Writes
   `data/scaling_sweep.csv`.
3. **Phase 3 — diagnostics**: per-depth skip rate, post-fix vs pre-fix
   argmax per (feat, bin), per-bin contribution magnitudes. Compares
   to PROBE-E's N=50k reference. Writes `data/diagnostics.json` and
   `data/diagnostics_summary.txt`.
4. **Phase 4 — boundary estimation**: linear interpolation in (log N,
   drift) to find N* where aggregate drift crosses 2%.

## Reproducibility

```bash
# 1. Build the probe binary (host-side only; kernels untouched)
bash docs/sprint38/probe-g/scripts/build_probe_g.sh

# 2. Run all phases (anchor capture + scaling sweep + diagnostics + boundary)
python docs/sprint38/probe-g/scripts/run_probe_g.py

# 3. Or run individual phases
python docs/sprint38/probe-g/scripts/run_probe_g.py --phase 1
python docs/sprint38/probe-g/scripts/run_probe_g.py --phase 2
python docs/sprint38/probe-g/scripts/run_probe_g.py --phase 3
python docs/sprint38/probe-g/scripts/run_probe_g.py --phase 4
```

Outputs land in `docs/sprint38/probe-g/data/`. The
`COSINE_RESIDUAL_OUTDIR` env-var policy (S37 SA-L3-S30) requires the path
to be under `docs/`; the harness handles this.

## What we expect to find (predictions, NOT yet validated)

The empirical context: 14% aggregate drift at N=1k with the post-DEC-042
fixed `FindBestSplit` path. PROBE-E showed that at N=50k, the joint-skip
fired on 5% of (feat, bin, partition) cells at d=2 and the d=2 gain gap
between MLX's pick and CPU's pick was 26 gain units. After DEC-042, the
N=50k drift collapsed to 1.27%.

At N=1k, we predict three plausible scenarios:

| Scenario | Skip rate at d=2 | Gap (post-fix - pre-fix) | Implication |
|---|---|---|---|
| (A) DEC-042 inert at small N | low | small | Residual drift is from a separate mechanism not captured here. Need follow-up probe (quantization borders / basePred / leaf values). |
| (B) DEC-042 active but insufficient | ≫ 5% | large | Per-side mask is correct but per-bin signal is noise-dominated at small leaf size — fp32 precision class is the residual. K4 fix may need extension. |
| (C) Same mechanism, different magnitude | similar to PROBE-E | proportional | Skip frequency scales with depth × bins/leaf — small N just has more skips. |

The math derivation predicted scenario (C) at the LG path; the ST refutation
suggests neither (B) nor (C) cleanly, since ST's path is fixed. Scenario (A)
is the most consistent with the refutation — but PROBE-G's instrumentation
can only confirm (A) negatively (by showing low impact of DEC-042 at N=1k);
naming the actual mechanism requires a follow-up probe.

## Result

### Per-depth diagnostics (N=1000, seed=42, ST+Cosine, iter=2)

| d | rows | skip% | PE_skip% | postfix_pick | prefix_pick | gap |
|---|------|--------|----------|--------------|-------------|-----|
| 0 | 2540 | 0.00 | 0.00 | (0,69):12.4648 | (0,69):12.4648 | 0.000 |
| 1 | 5080 | 2.54 | 2.50 | (1,82):14.6068 | (1,82):14.6068 | 0.000 |
| 2 | 10160 | 5.36 | 5.00 | (0,21):15.7691 | (14,112):14.5916 | 1.177 |
| 3 | 20320 | 29.97 | 7.60 | (0,109):16.1764 | (7,83):15.5545 | 0.622 |
| 4 | 40640 | 33.06 | 10.60 | (1,29):16.1741 | (7,92):15.5857 | 0.588 |
| 5 | 81280 | 39.69 | 14.60 | (1,29):16.1529 | (11,57):15.5709 | 0.582 |

Columns: `skip%` = fraction of (feat, bin, partition) cells where the pre-DEC-042 joint-skip
fires at N=1k. `PE_skip%` = the PROBE-E N=50k reference skip rate. `postfix_pick` = argmax
under post-DEC-042 per-side mask (actual MLX behavior). `prefix_pick` = argmax under the
pre-DEC-042 joint-skip counterfactual. `gap` = postfix_gain − prefix_gain.

### Aggregate scaling sweep (mean drift across 5 seeds, ST+Cosine, iter=50)

| N | drift % |
|---|---------|
| 500 | 18.703 |
| 1000 | 13.929 |
| 2000 | 8.593 |
| 5000 | 5.093 |
| 10000 | 3.052 |
| 20000 | 1.937 |
| 50000 | 1.233 |

**N* (2% threshold, log-N linear interpolation between N=10k and N=20k)**: **19,231**.

The math derivation predicted N* ≈ 5k–10k (based on the `B/n_leaf > 1` threshold crossing at
N = 64 × 128 = 8,192). The empirical crossing at ~19k is approximately 2× higher than predicted
— the derivation underestimated the N* because the per-leaf skip rate does not rise as sharply as
the `bins/n_leaf` proxy suggests (skip rate depends on how many leaves encounter a genuinely
degenerate split, which is a structural function of the data, not just leaf size).

---

## Classification — AMENDED: Scenario C at d≤2 only; d≥3 is a regime change

**Amended verdict (2026-04-25, @devils-advocate stress-test):** Scenario C is confirmed at
d=0, d=1, and d=2 — the degenerate-partition handling mechanism from PROBE-E is active at these
depths at small N. At d≥3, the data is inconsistent with depth-amplification of the same
mechanism. Instead, d≥3 represents a regime change to a different class — plausibly
continuous-precision-class at small leaves (N/64 ≈ 16 docs/leaf at d=6). **PROBE-G
structurally cannot localize the d≥3 mechanism** (see §Critical caveat below): both columns
assume MLX's path and neither observes CPU CatBoost's actual runtime behavior.

The original "Scenario C dominant with depth-amplification" verdict is retracted. Three
falsifications forced the amendment:

**(1) Skip-ratio does NOT rise monotonically with depth — it peaks at d=3 then falls.**

The ratio (MLX-skip% / PROBE-E-skip%) by depth:

| d | MLX skip% | PROBE-E skip% | Ratio |
|---|-----------|---------------|-------|
| 2 | 5.36 | 5.00 | 1.07× |
| 3 | 29.97 | 7.60 | 3.94× |
| 4 | 33.06 | 10.60 | 3.12× |
| 5 | 39.69 | 14.60 | 2.72× |

A genuine "depth-amplification of the same mechanism" would require the ratio to rise
monotonically with d (more depth → more skips relative to the large-N baseline). Instead
the ratio peaks at d=3 then falls at d=4 and d=5. By contrast, PROBE-E (N=50k) showed a
monotonic skip-rate increase with depth. The fall-off at d=4/d=5 is consistent with leaf-size
saturation: N=1k / 64 ≈ 16 docs/leaf at d=6, so degenerate skips happen by exhaustion of doc
count, not by growing partition topology.

**(2) Gap collapses while skip rate explodes — inconsistent with "same mechanism, more
frequent."**

| d | gap (postfix − prefix) | skip% | mag_median_termN_nonzero |
|---|------------------------|-------|--------------------------|
| 2 | 1.177 | 5.36% | 60.16 |
| 3 | 0.622 | 29.97% | 19.19 |
| 4 | 0.588 | 33.06% | 7.04 |
| 5 | 0.582 | 39.69% | 2.44 |

If d≥3 were the same mechanism at higher frequency, accumulated gap should rise with skip rate
(more cells corrected → larger total effect on the argmax). Instead the gap decays ~2× while
`mag_median_termN_nonzero` decays ~25× (60.16 → 2.44). At d≥3 each skipped partition
contributes a numerically negligible term; the per-side mask is correcting noise-scale cells.
This is incompatible with the topology-amplification hypothesis.

**(3) The drift curve is smooth power-law — no knee near N=8k where the `B/n_leaf > 1`
threshold would predict a regime change.**

| N | drift % |
|---|---------|
| 500 | 18.703 |
| 1000 | 13.929 |
| 2000 | 8.593 |
| 5000 | 5.093 |
| 10000 | 3.052 |
| 20000 | 1.937 |
| 50000 | 1.233 |

Drift halves roughly per doubling of N across the full range. A topology-threshold mechanism
(`B/n_leaf > 1` ↔ `N < 8192`) would show a knee near N=8k. There is no knee — the curve is
smooth power-law decay. This is the signature of a **continuous precision/noise class**
mechanism, not a discrete topology one. It also implies the math derivation in
`docs/sprint38/lg-small-n/math-derivation.md` was structurally inadequate (not just off by 2×
in its N* estimate): it assumed topology-class and derived a threshold, but the data
contradicts the threshold model.

**Residual implication:** DEC-036's "precision class exhausted" closure was established at
large N only. At small N the precision-class mechanism may not have been exhausted — it may
be the primary driver of the 14% N=1k residual. This requires F2 to discriminate (see
§Recommended next step).

---

## Critical caveat (re-stated)

The following is quoted verbatim from the Method section above and is essential for interpreting
all PROBE-G numbers:

> **Critical semantic note**: in the post-DEC-042 codebase, MLX's actual
> behavior matches `cpu_termNum/Den` (the per-side mask). The diff
> `cpu - mlx` measures what DEC-042 closed at the captured anchor. **It does
> NOT directly measure MLX-vs-CPU-CatBoost divergence**, because both columns
> assume MLX's path; CPU CatBoost's runtime trace would require a separate
> instrumentation hook in the CPU build. Thus PROBE-G can:
>
> 1. Confirm whether DEC-042 had any structural effect at N=1k (skip rate,
>    gap between post-fix and pre-fix per-bin gains).
> 2. Surface mechanisms that are **proxies** for MLX-vs-CPU divergence:
>    per-bin contribution magnitude, per-bin precision noise, MLX argmax
>    stability across runs.
> 3. Identify what classes are NOT captured (quantization borders,
>    basePred init, leaf-value computation, fp32 reduction noise) — these
>    would require additional probes if PROBE-G's diagnostics come back
>    inconclusive.

PROBE-G compared post-DEC-042 MLX against a pre-DEC-042 counterfactual **within MLX**. Both
the "postfix" and "prefix" columns represent MLX execution paths — neither column is CPU
CatBoost's actual runtime behavior. The classification above (Scenario C) describes the
relative structural change that DEC-042 makes at N=1k, not the gap between MLX and CPU.

---

## What this implies for DEC-042

**DEC-042 is structurally correct at d=0, 1, 2.** The per-side mask fires at d=2 (gap = 1.177,
selecting signal feat=0 bin=21 over noise feat=14 bin=112) and operates on the same
partition-state mechanism class that PROBE-E identified at N=50k. At these depths, DEC-042
behaves identically to the large-N regime.

**At d≥3, DEC-042 fires on numerically negligible cells.** The `mag_median_termN_nonzero` decays
from 60.16 at d=2 to 2.44 at d=5. The per-side mask is firing — the skip rates are real (29.97%,
33.06%, 39.69%) — but each individual correction is tiny. DEC-042 neither helps nor hurts
meaningfully at d≥3; it is neither the cause of the d≥3 residual nor its remedy.

**The 13.93% aggregate drift at N=1k is from a different class than d≤2's topology mechanism.**
This was already known from the cheap discriminator run (ST+Cosine, 3 seeds: 13.96% drift with
the post-DEC-042 fixed path). PROBE-G now confirms: the d=2 topology fix accounts for the
1.177-gap correction, but the 13.93% aggregate drift is accumulating from a mechanism that
PROBE-G cannot observe — because PROBE-G is MLX-internal and never measures CPU CatBoost's
actual runtime values.

The 13.93% residual's locus candidates remain:

- (i) per-side formula divergence vs CPU's `UpdateScoreBinKernelPlain` at small leaf sizes;
- (ii) quantization border deficit (DEC-039: 127-vs-128 borders) amplifying at small N;
- (iii) basePred init effects compounding at small-sample leaves;
- (iv) leaf-value precision — Newton leaf computation near the L2 regularization floor.

F2 (see §Recommended next step) provides the cheapest discrimination between (i) and (ii)–(iv).

---

## Predicted N* boundary

The `B/n_leaf` threshold model (`N < 64 × 128 = 8192`) predicted N* ≈ 5k–10k. The empirical
crossing at N* ≈ 19,231 is ~2× higher. As the amended classification above argues, the
smooth power-law curve with no knee near N=8k indicates the threshold model is structurally
wrong — the mechanism is continuous, not threshold-based. N* = 19,231 is the empirical fact
and the right number to cite for production guidance. The `B/n_leaf` derivation should not
be used to predict further scaling behavior.

## Recommended next step — F2 first, then PROBE-H or PROBE-I

**AMENDED (2026-04-25):** PROBE-H (CPU-side instrumentation) remains the eventual probe, but
a cheaper falsification step (F2) can run first with ~2 hours of effort and no new
instrumentation.

### F2 — CPU-tree split comparison at the N=1k seed=42 anchor (~2h)

Run the standard `catboost` Python package (CPU CatBoost) on the same N=1k seed=42 anchor
used by PROBE-G, with the same hyperparameters (depth=6, bins=128, l2=3, lr=0.03, ST,
Cosine, RMSE, iters=2). Dump the iter=2 tree splits using CatBoost's model-introspection API
(e.g., `model.get_tree_splits()` or equivalent). Compare to MLX's iter=2 tree splits from
PROBE-G's postfix column at the same anchor.

**No new instrumentation required.** Uses CatBoost's existing model API.

**Discriminator:**

| F2 result | Interpretation | Next step |
|-----------|---------------|-----------|
| CPU picks match MLX postfix at d=2 (both pick feat=0, bin≈21) | DEC-042 is equivalent to CPU's `CalcScoreOnSide` at d=2. The residual is at d≥3 — small-leaf precision noise, not topology. | Open **PROBE-I** targeting precision/leaf-value at d≥3. Do NOT open PROBE-H. |
| CPU picks differ from MLX postfix at d=2 | DEC-042's per-side formula is not equivalent to CPU's `CalcScoreOnSide`. The formula divergence IS the residual. | Open **PROBE-H** to dump CPU per-bin per-side gain and find the formula divergence. |

### PROBE-H (if F2 shows d=2 divergence) — CPU-side instrumentation (~1 sprint)

**PROBE-H goal**: at the same N=1k seed=42 anchor, capture CPU CatBoost's runtime
per-side per-bin gain and compare it against MLX's post-DEC-042 picks (the "postfix" column
from PROBE-G). This is a cross-runtime comparison, not an MLX-internal counterfactual.

**New instrumentation needed**: a CPU-side hook in CatBoost's score calculator — likely in
`catboost/private/libs/algo/score_calcers.cpp` inside `TCosineScoreCalcer::CalcMetric` or
the per-partition reducer `UpdateScoreBinKernelPlain` in `short_vector_ops.h`. The hook should
emit the same `(feat, bin, partition, gain)` tuples that MLX emits via `PROBE_E_INSTRUMENT`
at iter=2, armed at the same depth range (d=0..5) and the same anchor configuration
(N=1k, seed=42, ST+Cosine, RMSE, depth=6, bins=128, l2=3, lr=0.03).

**Estimate**: ~1 sprint — instrumentation (~2 days), anchor capture (~0.5 day),
classification (~0.5 day).

**Directory**: `docs/sprint38/probe-h/` or `docs/sprint39/probe-h/` — Ramos decides.

### PROBE-I (if F2 shows d=2 match) — precision/leaf-value at d≥3

Target one of: quantization borders (does the 127-vs-128 border count shift leaf membership
enough to flip gain rankings at small N?), basePred initialization, or leaf-value Newton
computation near the L2 regularization floor at small leaf sizes.

## Artifacts

All files verified present on disk under `docs/sprint38/probe-g/`.

- `data/anchor_n1000_seed42.csv` — N=1k anchor data
- `data/cos_leaf_seed42_depth{0..5}.csv` — per-(feat, bin, partition) capture (6 files)
- `data/cos_accum_seed42_depth{0..5}.csv` — per-bin fp32 vs fp64 shadow (6 files)
- `data/scaling_sweep.csv` — per-(N, seed) drift table, HISTORICAL: generated under asymmetric RS (MLX RS=1.0 vs CPU RS=0.0); mean drift ranged 1.2%–18.7% across N values
- `data/scaling_sweep_parity.csv` — canonical RS=0 parity rerun (S39); 35 rows, mean drift 0.023%, max 0.086% — confirms near-zero MLX-vs-CPU drift once RS is matched
- `data/diagnostics.json` — Phase 3 structured output (per-depth skip rates, picks, gaps)
- `data/diagnostics_summary.txt` — Phase 3 human-readable summary
- `data/boundary.json` — Phase 4 N* estimate (n_cross=19231, interpolated log-N)
- `data/probe_run_stdout.txt`, `data/probe_run_stderr.txt` — capture run logs
- `data/probe_g_phase{1..4}.log` — per-phase run logs
- `data/probe_g_run.log` — combined orchestrator log
- `data/leaf_sum_seed42.csv`, `data/approx_update_seed42.csv` — auxiliary per-iter dumps
- `scripts/build_probe_g.sh` — build script (mirrors PROBE-E)
- `scripts/run_probe_g.py` — phased orchestrator and analysis harness

## Build invariants

- `kernel_sources.h` md5 = `9edaef45b99b9db3e2717da93800e76f` (unchanged)
- All instrumentation gated under `#ifdef PROBE_E_INSTRUMENT`. Production
  builds (no flag) compile to bit-identical machine code as before.
- Capture fires only when `g_cosInstr.dumpCosDen && Cosine && K==1` —
  same arming as PROBE-D/E.
