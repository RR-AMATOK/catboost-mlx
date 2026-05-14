# S48-T0b-DELTA — Devils-Advocate Stress-Test of C6/C7/C8

**Date:** 2026-05-08
**Companion:** `docs/sprint48/T0/dac-stress-test.md` (T0b on v2 5-candidate roster)
**Authority:** DEC-049 OUTCOME, DEC-051, DEC-052 OPEN; LESSONS-LEARNED MANDATORY-CODE-INSPECTION + SIMD routing invariant.

---

## §1 — Seven-DEC rejection lookup

### C6 — Histogram subtraction (parent-minus-sibling)

| DEC | Echo? |
|---|---|
| DEC-013 (writeback-plurality / batched-atomic) | NO. Subtraction is dense BW-bound elementwise; no atomics. |
| DEC-014 (per-doc VGPR tiling) | NO. No per-doc state at all in the subtract kernel. |
| DEC-015 (col-major gather) | NO. Subtract operates on already-built dense histograms. |
| DEC-017 (per-TG amortization regime flip) | NO — but the SMALLER-CHILD pass still runs the production src-broadcast kernel at a smaller doc count. Per-TG amortization shrinks. **Q: does it cross into DEC-017's failure regime when the smaller child is very small (e.g., 200 docs/partition)?** Flag for T2. |
| DEC-019 (stats pre-permute) | NO. |
| DEC-048 (dispatch fusion) | NO. Adds a dispatch (subtract kernel) — opposite direction. |
| DEC-049 (simd_shuffle removal) | NO. The smaller-child histogram still uses production src-broadcast; routing is unchanged. |

**Verdict: KEEP (FLAGGED-FOR-T2).** C6 is genuinely outside the 7-falsification chain. It does NOT change kernel internals; it changes WORKLOAD COUNT — orthogonal to all 7 retired DECs.

### C7 — Cross-iteration histogram delta

**Verdict: RETIRE-AT-PREMISE.** Collapses to histogram-of-deltas under visionary's own gate (see §2).

### C8 — AMX matmul reformulation (sparse boolean SpMV)

| DEC | Echo? |
|---|---|
| DEC-013 | **YES (indirect).** `scatter_add` (only available primitive — verified `mlx/mlx/ops.h:1206`) IS atomic accumulation. Calling it `SpMV` doesn't change the underlying op. |
| DEC-049 | YES — same routing question; SpMV must route stat[d] to the correct (f,b) cell, same impossibility. |

**Verdict: RETIRE-AT-PREMISE.**

---

## §2 — Hidden-rebrand check

**C6 — "is this just LightGBM in disguise?"** Yes — and that's a feature, not a flaw. LightGBM has independent industrial validation (≥500M production users, peer-reviewed Ke et al. 2017). **CRITICAL: LightGBM applies subtraction in leaf-wise growth.** CatBoost-MLX uses oblivious (depthwise). Visionary asserts "compatible with oblivious" but cites no implementation. **MANDATORY at T2:** silicon-architect must confirm subtraction works under oblivious topology where ALL siblings at depth d are computed for ALL leaves before any depth-d+1 split — i.e., parent histograms must persist across the full level. Memory: at depth 6 with Epsilon (2000 features × 128 bins × 64 partitions × 8B = 130 MB per level for parent cache). Plausible but not free.

**C7 — does the "leaf-constant stat-delta" save anything?** No. Visionary's own pre-flight gate (correlation `hist[T]` vs `hist[T+1]` > 0.9) is a proxy for the wrong thing. Even if hist[T+1] ≈ hist[T] in value, the *delta* is `O(N · F)` in changed entries because every doc's gradient changes every iteration. The "delta" IS itself a histogram of gradient-changes — same kernel cost, +overhead of base-state read.

**C8 — `mx::scatter_reduce` exists?** **NO.** Verified `mlx/mlx/ops.h`: only `scatter_add` (line 1206) and `segmented_mm` (line 1579). `scatter_reduce` does not exist. Sparse SpMV path requires either (a) `scatter_add` (atomic — DEC-013 territory at production, falsified) or (b) custom Metal kernel (collapses to current src-broadcast, DEC-049 territory). Identical to L3's missing-primitive collapse. **C8 = L3 with linear-algebra branding.**

---

## §3 — LESSONS-LEARNED check (C6 only — sole KEEP)

- **Routing-honest (S46-T6):** PASS. Smaller-child histogram inherits production routing. Subtract kernel has no routing question (dense elementwise).
- **Toy-to-production transfer:** MEDIUM RISK. LightGBM ships subtraction in *CUDA leaf-wise*; we apply in *Metal oblivious*. Different topology, different memory hierarchy. Needs T4 production-shape measurement, not toy.
- **MANDATORY-CODE-INSPECTION at T2:**
  1. **Parent-cache memory:** at Epsilon depth 6, 130 MB per level. Cite Metal buffer allocation path. File:line for the dispatch graph change at `histogram.cpp` that schedules parent-build-before-children-build.
  2. **Smaller-child-selection mechanism:** doc-count-per-child is computed CPU-side from partition assignment. Cite the code path where this branch decision lives. If selection requires GPU→CPU sync per split, the sync cost may exceed the savings.
  3. **DEC-017 amortization at small-child shapes:** at very-skewed splits (the high-payoff case), smaller child may have <500 docs/partition — putting the smaller-child kernel into DEC-017's failure regime. Silicon-architect must compute per-TG docs/thread for the Epsilon depth-6 high-skew case and confirm it stays ≥30.

---

## §4 — Updated final shortlist

```
PRIOR SHORTLIST (T0b):
1. L6 — Hybrid CPU+GPU concurrent
2. C4 — Persistent-kernel pipelining
3. C5 — Leaf-wise (CONDITIONAL on user admit)

DELTA (this stress-test):
- C6: KEEP (FLAGGED-FOR-T2) — promotes to TOP of shortlist
- C7: RETIRE-AT-PREMISE (collapses to histogram-of-deltas)
- C8: RETIRE-AT-PREMISE (scatter_reduce doesn't exist; = L3 with linear-algebra branding)

UPDATED FULL SHORTLIST (priority order):
1. C6 — Histogram subtraction (workload reduction; cross-domain validated; lowest eng cost)
2. L6 — Hybrid CPU+GPU concurrent (UMA-axis novel)
3. C4 — Persistent-kernel pipelining (sync-topology axis novel)
4. C5 — Leaf-wise (CONDITIONAL on user admit at T0c)

UPDATED RETIRED:
- L4 (T0b: DEC-014 + DEC-017 + DEC-049 echoes)
- C1 (T0b: missing primitive + L2 territory + DEC-019)
- C7 (DELTA: collapses to histogram-of-deltas)
- C8 (DELTA: scatter_reduce missing; = L3)
```

C6 promotes to **#1** because it has the highest visionary P(survive) (0.7), lowest engineering cost (1-2 sprints), lowest mechanism risk (dense BW-bound subtract is GPU-trivial), and *independent industrial proof of concept* — qualitatively stronger evidence than any prior sprint candidate.

---

## §5 — Verdict on visionary's rubric clause

Visionary proposes: *"If a candidate has (a) low engineering cost ≤2 sprints, (b) cross-domain industrial validation, AND (c) measured ≥1.5× iter speedup on Higgs-1M, it is eligible for Outcome A regardless of whether it projects to ≥2×."*

**Recommendation: AMEND** with two tightening conditions:

1. Qualifiers (a) and (b) must be **certified at T0c (pre-measurement)**, not invoked at T5 to rescue marginal results.
2. The (c) threshold should be **≥1.7×, not ≥1.5×.** Reasoning: f_hist Higgs-1M ≈ 0.90, so 1.5× iter implies only ~1.6× histogram — within S46 measurement noise (±5% iter-time, see S46-T4). 1.7× iter implies ~1.9× histogram — clear of noise.

Borderline NO on goalpost-moving: DEC-051's hard gate is **≤5× MLX/CUDA on Higgs-1M iter=1000** — that does NOT change. The proposed clause amends only the **Outcome A trigger**.

---

## §6 — Updated meta-verdict

**Prior (post-T0b):** YELLOW.
**Now (post-DELTA):** **YELLOW-leaning-GREEN.**

C6 materially shifts the picture. Three reasons:
1. **Workload-reduction axis is genuinely orthogonal** to all 7 retired DECs and all 5 v2-roster candidates I previously stressed.
2. **Independent industrial validation** (LightGBM) is qualitatively stronger evidence than any prior sprint candidate.
3. **Engineering cost low** (subtraction kernel is BW-bound elementwise — easiest GPU primitive). Failure mode is dispatch-graph wiring, not novel algorithm.

**Most likely outcome shift:**
- P(C — retire+pivot): 0.55 → **0.40**
- P(B — user-call zone): 0.25 → **0.35**
- P(A — measured ≥2× / amended ≥1.7×): 0.20 → **0.25**

C6's MEAN child-imbalance at Higgs-1M and Epsilon is the load-bearing unknown — visionary's pre-flight gate is the single cheapest falsification test in the entire arc. **This should be T1's first task**, ahead of f_hist remeasurement.

**YELLOW (not GREEN)** because two C6 risks remain unresolved until T2 code-inspection: (i) parent-cache memory at Epsilon depth 6 (130 MB/level), and (ii) GPU→CPU sync per smaller-child selection.

---

## Cheapest falsification test (single most efficient experiment)

Instrument v0.7.0 baseline to log per-split child-size ratio for 100 trees × 3 seeds on Higgs-1M and Epsilon. Compute geomean `min(|L|,|R|)/|P|` per shape.
- If ≥0.45 (near-balanced splits dominant): C6's expected speedup is bounded below 1.6× iter — trap zone, rubric-clause debate becomes load-bearing.
- If ≤0.35 (skewed splits dominant): C6 projects ≥2× and Outcome A plausible without rubric amendment.

Estimated effort: 4 hours instrumentation + 1 hour run on existing benchmark harness. Falsifies/confirms C6's premise before any kernel work.

**READY FOR T0c USER-CALL — final shortlist of 3 candidates (C6, L6, C4) plus 1 conditional (C5).**
