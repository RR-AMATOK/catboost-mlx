# Sprint 49 Plan — C6 Engineering (Histogram Subtraction / Parent-minus-Sibling)

**Sprint:** 49
**Status:** ACTIVE — T0c locks approved by user 2026-05-14 (3-agent panel review + 5-question batch decision)
**Branch:** `mlx/sprint-49-c6-engineering`
**Cut from:** master `5d1ae685fc` (S48 close, DEC-052 OUTCOME A locked 2026-05-13)
**Theme:** Engineering implementation of C6 — sole surviving candidate from S48 v0.8.0 throughput arc. Cache parent histograms at depth d−1; dispatch unchanged `kHistOneByteSource` on SMALLER child; derive larger via `hist[R] = hist[P] - hist[L]`. CatBoost-canonical mechanism (CPU `FixUpStats` precedent).
**Mode:** ENGINEERING. Production code changes in scope; loss-conditional dispatch (no feature flag — see Q4 below).
**Duration:** ~6.5 days agent work. **Hard 7-day timebox.** S49+S50 ≤ 10-day v0.8.0 budget per DEC-052.

---

## §1 — Strategic context and authority

DEC-052 OUTCOME A authorized C6 for S49 engineering after T0+T1+T2+T3 cleared at S48. The probe-spec at `docs/sprint48/T3/probe-spec-c6.md` is LOAD-BEARING input.

**Authority chain:** DEC-049 OUTCOME (RETIRED) → DEC-050 (Option α) → DEC-051 (PyPI gated on CUDA-class throughput) → DEC-052 OUTCOME A → **S49-T0c LOCKS (2026-05-14)** below.

**Why this is admissible after 7 falsifications:** C6 is workload reduction, not kernel optimization. The kernel body is byte-identical to v0.7.0; only dispatch frequency and shape change. **F5 finding:** CatBoost-CPU ships this trick already (`catboost/private/libs/algo/scoring.cpp:315-332` `FixUpStats`). Algorithmic novelty risk: zero.

---

## §2 — S49-T0c LOCKS (5 batch decisions approved 2026-05-14)

After 3-agent panel review (@ml-product-owner sprint plan + @devils-advocate stress-test + @strategist synthesis), user approved all 5 recommendations as batch lock.

### Q1 — Bundle 2 hard gate AMENDED (Amazon carve-out)

Devils-advocate's load-bearing new finding: Amazon MLX/CUDA at v0.7.0 baseline is **already 0.91× (MLX faster)** — but MLX runs a degenerate workload due to DEC-046 uint8 aliasing folding RESOURCE feature (cardinality 799) into 255 bins. Logloss diverges (MLX 0.2195 vs CPU 0.1332). **Amazon Bundle 2 gate currently tests a fictional measurement.**

This wasn't known at S48-T0c on 2026-05-12. **NOT goalpost-moving** — correcting a measurement-validity error before T1 fires.

**LOCKED:** Bundle 2 hard gate becomes:
- ≤5× MLX/CUDA on Higgs-1M iter=1000 (primary)
- ≤5× MLX/CUDA on Epsilon iter=2000 (cross-shape robustness)
- Amazon iter=1000: **CARVED OUT** of S49 Bundle 2 hard gate. Re-enters Bundle 2 in a future sprint conditional on DEC-046 fix.
- v0.8.x release notes will document Amazon as a known limitation pending DEC-046.

DEC-052 OUTCOME A clauses on Higgs-1M ≥1.7× iter + parity intact + ≤5× on the two retained datasets remain LOCKED.

### Q2 — Amazon T0 child-imbalance kill rubric

Even with Amazon carved out of Bundle 2, S49-T0 still measures Amazon child-imbalance — it's the cheapest signal we have on C6's behavior under categorical-heavy workloads, AND it pre-flights the v0.8.x DEC-046 dependency.

**LOCKED:**
- Amazon geomean child-imbalance ≤ 0.35 → PASS (matches Higgs/Epsilon framework)
- 0.35–0.45 → user-call (one-shot escalation, NOT auto-retire — Amazon is informational at S49)
- > 0.45 → C6 ships but flagged as "Amazon-sub-optimal" in release notes; doesn't retire arc

This is softer than the Higgs/Epsilon kill rule because Amazon is no longer Bundle 2 hard gate.

### Q3 — Outcome B(1.5–1.7×) trap-zone discipline

Devils-advocate: "Outcome B(1.5-1.7× Higgs) = auto-retire, NO user-call, NO feature flag. Feature flags = dead-code rot ratchet."

**LOCKED:** If Higgs-1M measured 1.5–1.7× iter speedup → arc auto-retires under sunk-cost rail. No user re-deliberation. Pivot to ordered boosting per DEC-052 T0c Q3.

Single exception: if Higgs ≥1.7× AND Epsilon also clears Bundle 2 ≤5× AND Higgs-only (not Bundle 2) measures 1.5-1.7× → Outcome A (rubric clause trigger is Higgs ≥1.7×, the rest is icing).

### Q4 — RMSE envelope strategy (if DEC-008 Gate B fails)

γ_13 ≈ 7.7e-7 vs RMSE γ_8 ≈ 4.77e-7 ceiling. C6 may exceed RMSE envelope at depth 6.

**LOCKED:** Loss-conditional dispatch — NO feature flag. If T3 18-config sweep shows RMSE FAIL + Logloss/MultiClass PASS:
- C6 path applies to Logloss + MultiClass loss configurations only
- RMSE configurations use the production src-broadcast path (unchanged)
- Dispatch decision is loss-type-based, made at training start
- No `use_histogram_subtraction` runtime flag; no per-iter branching cost
- Honest scope-narrowing, not Outcome-B-trap-zone hedge

If T3 shows RMSE+Logloss+MultiClass ALL fail → STOP, investigate (likely code bug). If RMSE-fail but Logloss/Multi pass → loss-conditional dispatch ships.

### Q5 — Sprint plan shape (Modified β)

T0 and T1 are KILL-ABLE gates. T2 commits engineering investment; once T2 starts, sprint runs through T6 with classify-only outcomes at T3/T4/T5.

---

## §3 — Task structure

| # | Task | Owner | Days | Kill-able? |
|---|------|-------|------|-----------|
| **T0** | Amazon child-imbalance pre-flight measurement (Q2 rubric) | @performance-engineer | 0.5 | YES (>0.45 → "Amazon-suboptimal" flag) |
| **T1** | Dispatch graph design + smaller-child selection — DESIGN ONLY | @research-scientist + @silicon-architect | 0.5 | YES (no viable MLX primitive within 0.5 sprint → retire arc) |
| **T2** | Engineering implementation — parent cache + smaller-child dispatch + subtract kernel + loss-conditional dispatch | @ml-engineer | 2.0 | NO (commit point; classify-only T3-T5) |
| **T3** | DEC-008 18-config envelope sweep (Q4 rubric) | @qa-engineer + @mathematician | 1.0 | Classify (full ship vs Logloss/Multi only) |
| **T4** | Bundle 2 multi-dataset measurement (Higgs-1M + Epsilon ONLY per Q1; Amazon informational) | @performance-engineer | 1.5 | Classify (Outcome A/B/C tier) |
| **T5** | Decision gate per Q3 rubric | @strategist + user | 0.5 | Classify |
| **T6** | Close-out PR | @ml-product-owner + @technical-writer | 0.5 | — |

Total: 6.5 days; 7-day hard timebox.

### T0 — Amazon child-imbalance pre-flight (0.5 day)

Rebuild `bench_boosting_s48_t1` on S49 branch (instrumentation already on master post-PR-#50; CMakeLists already gated by `BUILD_S48_T1=ON`). Run on Amazon iter=100 × 3 seeds × depth 6. Compute geomean per Q2 rubric.

Output: `docs/sprint49/T0/amazon-child-imbalance/analysis.md`.

### T1 — Design (0.5 day)

Design (NOT implement) the dispatch graph change. Specify exact MLX primitive sequence:
- Smaller-child mask: `mx::less(layout.PartSizes[L], layout.PartSizes[R])`
- Selection: `mx::where(mask, L_indices, R_indices)`
- Derived larger: `mx::subtract(hist_parent, hist_smaller)`
- Output assembly: `mx::take_along_axis(...)` — **NOT `mx::concatenate`** (Risk 2: would force `mx::eval()` boundary)

Verify lazy-graph fusion via `CATBOOST_MLX_STAGE_PROFILE` sync-point count parity vs v0.7.0.

Output: `docs/sprint49/T1/design.md`.

### T2 — Engineering (2 days)

Implement on `mlx/sprint-49-c6-engineering`:
- `catboost/mlx/methods/histogram.cpp`: new `ComputeHistogramsSmallerChild(...)` overload; dispatch grid math change at d ≥ 1
- `catboost/mlx/methods/structure_searcher.cpp:60-108`: parent cache lifetime; smaller-child selection insertion; output assembly
- `catboost/mlx/methods/structure_searcher.h:22`: `TPartitionLayout` extension if needed
- **Loss-conditional dispatch** (Q4 lock): C6 path activates for Logloss + MultiClass only; RMSE uses production path
- NO runtime feature flag; loss-type checked at training start
- Branch-B regression GREEN — predict path untouched (probe-spec §7.1)

Pre-condition before T2 starts: confirm pending `_predict_utils.py` modification in current `git status` is unrelated to C6 (else revert).

### T3 — DEC-008 envelope sweep (1 day)

Standard 18-config parity sweep: {10k, 50k} × {RMSE, Logloss, MultiClass} × {32, 64, 128} bins × oblivious. Compare ULP delta vs v0.7.0 baseline.

Sub-outcomes per Q4 lock:
- All 18 PASS DEC-008 envelope → full ship
- RMSE-only FAIL but Logloss + MultiClass PASS → loss-conditional dispatch (RMSE uses production, others use C6)
- Multiple-loss-class FAIL → STOP, investigate (likely code bug, not γ propagation)

### T4 — Bundle 2 measurement (1.5 days)

Per Q1 lock: **2 datasets** (Higgs-1M iter=1000 + Epsilon iter=2000) are Bundle 2 hard gates. Amazon iter=1000 measured for information only — does NOT gate publish.

Protocol: 3-seed median (CV<8%), warm-runs-only, M3 Max anchor, `median(MLX)/median(CUDA)`.

Per-kernel profile capture per probe-spec §5.3 (verify `f_hist` decreased; subtract kernel time ~1 ms at Epsilon depth 6).

### T5 — Decision (0.5 day)

Per Q3 lock:
- **Outcome A:** Higgs-1M ≥1.7× iter AND Bundle 2 datasets (Higgs+Epsilon) both ≤5× AND DEC-008 envelope intact (or loss-conditional Q4 path)
  → S50 cutover greenlight; PyPI publish prep
- **Outcome B(1.5–1.7×):** AUTO-RETIRE under sunk-cost rail; pivot to ordered boosting; NO user re-deliberation
- **Outcome C:** <1.5× Higgs OR Epsilon >5× → DEC-052 OUTCOME revised to RETIRED-EMPIRICALLY; pivot to ordered boosting
- **Outcome D:** >8× Higgs after S49 → stop-loss floor, retire v0.8.0 throughput arc

### T6 — Close-out (0.5 day)

PR `mlx/sprint-49-c6-engineering` → master. Update HANDOFF/TODOS/CHANGELOG-DEV/DECISIONS.md (DEC-052 OUTCOME update). LESSONS-LEARNED entry per outcome. S50 plan stub queued.

---

## §4 — Sequential gates (decision tree)

```
T0 — Amazon child-imbalance pre-flight (informational)
  │
  ├─ geomean > 0.45 → flag Amazon-suboptimal in release notes; PROCEED
  ├─ 0.35–0.45    → user-call one-shot; PROCEED with note
  └─ ≤ 0.35       → PROCEED full speed
        │
        ▼
     T1 — Design (no commit yet)
        │
        ├─ no viable MLX primitive within 0.5 day → retire arc (8b unlikely)
        └─ design signed off → PROCEED
              │
              ▼
           T2 — Implementation (engineering commit point)
              │
              ├─ Branch-B FAIL with flag OFF → STOP, root-cause (likely predict edit)
              └─ Branch-B GREEN → PROCEED
                    │
                    ▼
                 T3 — DEC-008 envelope sweep (classify-only)
                    │
                    ├─ All 18 PASS → unconstrained build to T4
                    ├─ RMSE FAIL + Logloss/Multi PASS → loss-conditional build to T4
                    └─ Multiple-class FAIL → STOP, code-bug investigation
                          │
                          ▼
                       T4 — Bundle 2 measurement (Higgs + Epsilon hard; Amazon informational)
                          │
                          ▼
                       T5 — Outcome classification
                          │
                          ├─ Higgs ≥1.7× AND Bundle 2 ≤5× AND parity intact  → OUTCOME A → S50 cutover
                          ├─ Higgs 1.5–1.7×                                  → OUTCOME B → AUTO-RETIRE
                          ├─ Higgs <1.5× OR Epsilon >5×                      → OUTCOME C → RETIRE, pivot
                          └─ Higgs >8×                                       → OUTCOME D → stop-loss
                                │
                                ▼
                             T6 — Close-out PR
```

---

## §5 — Risk register (per probe-spec §7 + S49 specifics)

| # | Risk | Likelihood | Impact | Mitigation | Owner |
|---|------|-----------|--------|-----------|-------|
| R1 | Amazon child-imbalance > 0.45 (categorical features may produce different skew) | Med | LOW (Amazon carved out per Q1) | T0 pre-flight; release-notes flag if sub-optimal | @performance-engineer |
| R2 | MLX lazy-graph fusion broken by wrong primitive | Med | HIGH | T1 mandates `mx::where` + `take_along_axis`; T4 verifies via `STAGE_PROFILE` sync count | @silicon-architect (T1), @performance-engineer (T4) |
| R3 | DEC-008 RMSE envelope violation at depth 6 (γ_13 vs γ_8) | Med | MED | T3 18-config sweep; Q4 LOCK: loss-conditional dispatch (no flag) | @qa-engineer, @mathematician |
| R4 | DEC-017 cliff at small smaller-children (T1 data shows ≥22 docs/thread; Amazon may differ) | Low-Med | MED | T2 instruments docs/thread distribution; if >5% dispatches <10 docs/thread, add per-partition fallback | @performance-engineer |
| R5 | Memory pressure from per-iter parent-cache allocation | Low | LOW | T2 hoists allocation out of per-iter scope; reuse buffer | @ml-engineer |
| R6 | Predict-path bit-equivalence regression (Branch-B FAIL) | Low | HIGH | T2 hard rule: NO predict-side edits. Pre-condition check on pending `_predict_utils.py` mod | @ml-engineer, @qa-engineer |
| R7 | Sprint scope blow past 7 days | Low-Med | MED | Hard timebox at T6 day 7; single-failure fallbacks pre-defined | @ml-product-owner |

---

## §6 — Cross-cutting hard rules

1. **Branch-B regression GREEN on master throughout.** Predict-only test; training not exercised. C6 must not touch predict path.
2. **MANDATORY-CODE-INSPECTION at every mechanism claim** (LESSONS-LEARNED standing rule from S46). File:line citations are NON-NEGOTIABLE.
3. **NO `mx::eval()` boundary in per-depth loop** (Risk 2 mitigation). Only fusable primitives (`mx::less`/`where`/`take_along_axis`/`subtract`/`reshape`).
4. **DEC-008 envelope honored** OR loss-conditional dispatch (Q4 lock).
5. **Loss-conditional dispatch, NO runtime feature flag** (Q4 lock). No `use_histogram_subtraction` runtime branching.
6. **NO push to upstream `catboost/catboost`** (DEC-004). All commits + PRs target RR-AMATOK/catboost-mlx.
7. **NO Co-Authored-By Claude trailer.**
8. **3-seed median, warm-runs-only, M3 Max anchor** for all T4 measurements (DEC-052 T0c protocol).
9. **Outcome B(1.5–1.7×) = auto-retire** (Q3 lock). NO user re-deliberation; pivot to ordered boosting per DEC-052 T0c Q3.
10. **Amazon carved out of Bundle 2 hard gate** (Q1 lock). Amazon measurement informational; v0.8.x dependency on DEC-046 fix documented.

---

## §7 — Definition of Done

1. T0 Amazon child-imbalance verdict recorded.
2. T1 design doc with file:line citations for every MLX primitive choice.
3. T2 implementation merged to S49 branch; Branch-B GREEN on master with code merged (RMSE path unchanged); loss-conditional dispatch active for Logloss/MultiClass.
4. T3 DEC-008 18-config envelope sweep complete; verdict recorded.
5. T4 Bundle 2 measurement on Higgs-1M + Epsilon (hard gate); Amazon (informational); per-kernel profile captured; CV<8%.
6. T5 DEC-052 OUTCOME entry updated in DECISIONS.md with empirical justification.
7. T6 close-out: state files updated; LESSONS-LEARNED entry; single squash-merge PR.
8. If Outcome A: S50 plan stub queued (cutover, depthwise extension, PyPI prep). If C/D: S50 plan = ordered-boosting kickoff.

S49 is "successful" under all 4 outcomes if gates are honored and decision is empirically grounded.

---

## §8 — Cross-references

- `docs/sprint48/T3/probe-spec-c6.md` — PRIMARY input (probe-spec)
- `docs/sprint48/T2/feasibility.md` — silicon-architect MANDATORY-CODE-INSPECTION
- `docs/sprint48/T1/child-imbalance/analysis.md` — T1 empirical data (Higgs 0.3064, Epsilon 0.2830)
- `.claude/state/DECISIONS.md` DEC-052 OUTCOME A + S49 T0c amendments
- `docs/benchmarks/cross-class-cuda-comparison.md` — Amazon DEC-046 degenerate workload context (Q1 rationale)
- `catboost/private/libs/algo/scoring.cpp:315-332` — `FixUpStats` CatBoost-CPU precedent (F5)
- S46 + S47 sprint plans — template style

---

## §9 — Probability-weighted expected outcome (post-panel)

- P(A — full ship, ≥1.7× Higgs, Bundle 2 clean): **0.42**
- P(B — auto-retire at 1.5–1.7×): **0.30**
- P(C — retire empirically, Higgs <1.5× or Epsilon >5×): **0.25**
- P(D — stop-loss >8× Higgs): **0.03**

Bundle 2 is now Higgs + Epsilon only (Q1 amendment). Amazon DEC-046 carve-out simplifies the Bundle 2 gate but doesn't change C6's core probability profile.

---

**S49 ACTIVE. T0 ready to fire.**
