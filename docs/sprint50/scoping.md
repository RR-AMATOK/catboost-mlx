# Sprint 50 — Categorical Handling Closure — SCOPING

**Sprint:** 50 (re-sequenced from ordered boosting per DEC-053; categorical-handling closure FIRST)
**Date:** 2026-05-18
**Branch:** `mlx/sprint-50-categorical-closure`
**Cut-from:** S49 close tip (`42e1089cba` C6 retirement); will rebase onto master after PR #51 lands
**Authority:** DEC-053 (re-sequence decision); DEC-046 (S40 CTR RNG ordering context); S48-T0c Q1 Amendment (Amazon Bundle 2 carve-out evidence)
**Mode:** ENGINEERING (bounded defect repair); 1-2 sprint scope.
**Theme:** Fix the Amazon bin-aliasing defect that makes Amazon Bundle 2 measurement fictional. Re-enable Amazon into Bundle 2 hard gate. Unblock future throughput investigation infrastructure.

---

## §1 — Strategic context

### The defect

Amazon dataset training in catboost-mlx produces a **degenerate model**: logloss MLX 0.2195 vs CPU CatBoost 0.1332 on the same data/seed. The root cause (per S48-T0c Q1 Amendment) is **bin-quantization aliasing**: RESOURCE feature (cardinality 799) is folded into 255 bins via uint8 encoding, losing ~70% of the feature's information.

Verification needed (T1): exact code path producing this defect. Candidates:
- `catboost/libs/data/objects.h` or related quantization headers
- `catboost/mlx/gpu_data/` quantization adapter
- `catboost/private/libs/data_types/` feature encoding

### Why this is S50

After C6 retired (DEC-052 OUTCOME REVISED 2026-05-18), the T0c Q3 pivot to ordered boosting fired automatically per pre-commit. 3-agent panel review surfaced that ordered-first sequencing would ship a feature against a known-broken Amazon substrate, repeating the fictional-measurement pattern C6 retirement just exposed. Per DEC-053, S50 re-sequences: categorical closure first, ordered boosting deferred to S52+ on honest substrate.

### Why this is bounded

- **Known defect** (not lever-search): we know what's wrong (aliasing) and roughly where (categorical quantization path). Engineering scope, not research.
- **No throughput claim**: S50 is correctness, not speedup. No sunk-cost-rail-class failure modes.
- **Bounded measurement gate**: Amazon logloss within DEC-008 envelope of CPU CatBoost = success; otherwise fix is incomplete.

### What this DOES NOT solve

- **Does NOT unblock PyPI publish** per DEC-051 (still gated on throughput).
- **Does NOT ship ordered boosting** (that's S52+; @ml-product-owner sprint plan + @research-scientist deep-dive preserved at `docs/sprint52/preliminary/` for direct use when S52 fires).
- **Does NOT address DEC-046 CTR RNG ordering** (separate categorical concern; can be revisited in S50 if scope permits, otherwise carries on as documented characterized limitation per Lane B).

---

## §2 — Task structure (T0–T6, bounded engineering)

| # | Task | Owner | Days | Notes |
|---|------|-------|------|-------|
| **T0** | Scaffold + advisory board kickoff | @ml-product-owner | 0.5 | Visionary brainstorm on alternative fix approaches; devils-advocate stress-test fix scope. Mirror S46/S48 T0 pattern. |
| **T1** | Current-state characterization (root cause + code path) | @ml-engineer + @data-scientist | 1.0 | Reproduce Amazon defect (logloss MLX 0.2195 vs CPU 0.1332); trace uint8 aliasing to specific file:line in quantization path; document the exact mechanism (where cardinality > 255 features get folded). |
| **T2** | Fix design | @architect + @ml-engineer | 0.5 | Three candidate approaches: (a) uint8 → uint16 binning, (b) dynamic hashing for high-cardinality features, (c) feature-engineering preprocessing. Pick one based on T1 root cause. |
| **T3** | Implementation | @ml-engineer | 1.0 | Atomic commits per logical unit. Branch-B regression GREEN throughout. |
| **T4** | Amazon validity verification | @qa-engineer | 0.5 | Re-train Amazon with fix; logloss MLX within ±0.005 of CPU CatBoost. Plus regression on Higgs-1M + Epsilon (must not degrade). |
| **T5** | Decision: ship + re-enable Bundle 2 OR scope-adjust | @strategist + user | 0.25 | If T4 passes: ship + amend DEC-051 to re-include Amazon in Bundle 2 hard gate. If T4 fails: root-cause and S51 continuation. |
| **T6** | Close-out | @ml-product-owner + @technical-writer | 0.25 | DEC-053 OUTCOME; HANDOFF/TODOS/CHANGELOG-DEV/LESSONS-LEARNED update; PR to master. |

**Total: ~4 days agent work; hard 7-day timebox.** Engineering depth varies on T1 root cause complexity.

---

## §3 — Success criteria (LOCK at T0c, anti-goalpost-moving)

- **G1 — Amazon parity:** logloss MLX within ±0.005 of CPU CatBoost on Amazon iter=1000 (3 seeds, median). DEC-008 envelope extension acceptable; bit-exact unrealistic given encoding may differ.
- **G2 — No regression:** Higgs-1M + Epsilon logloss within bounded envelope of v0.7.0 baseline (Branch-B regression GREEN).
- **G3 — Bundle 2 re-eligibility:** post-fix, Amazon MLX/CUDA wall-clock ratio is meaningful (not the current 0.91× fictional value); document the new ratio for future Bundle 2 hard gate inclusion.
- **G4 — Performance floor:** post-fix Amazon training time within 2× pre-fix (avoid pathological slowdown from new encoding).

---

## §4 — Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|-----------|
| R1 | Root cause is deeper than uint8 aliasing (e.g., upstream CatBoost quantization shared with MLX path) | Med | HIGH | T1 specifies "could not verify" as legitimate outcome → scope-adjust or escalate |
| R2 | Fix breaks Higgs/Epsilon (regression) | Low | HIGH | T4 Branch-B + G2 regression check; atomic commits with revertability |
| R3 | Fix is pathologically slow (high-cardinality features ~10× slowdown) | Med | MED | G4 hard floor; if exceeded, hybrid approach (uint8 for ≤255, uint16 for >255) |
| R4 | Sprint scope creep beyond 7 days | Low-Med | MED | Hard timebox; T5 cuts at end of day 5 even if T4 incomplete |
| R5 | Fix conflicts with CTR encoding (DEC-046 RNG ordering) | Low | LOW | Document interaction; CTR ordering separate issue per Lane B characterization |

---

## §5 — Cross-cutting hard rules

1. **Branch-B regression GREEN on master throughout** — predict-path bit-equivalence to v0.6.1 baselines.
2. **MANDATORY-CODE-INSPECTION** at every mechanism claim (file:line citation).
3. **Atomic commits per logical unit** — easy revertability if T4 surfaces issue.
4. **No production-default behavior change without G2 passing** — fix opt-in via internal config until verified.
5. **NO Co-Authored-By Claude trailer.**
6. **No push to upstream catboost/catboost** — RR-AMATOK fork only.

---

## §6 — Cross-references

- DEC-046 — CTR RNG ordering (S40 characterized limitation, separate but related)
- DEC-051 — PyPI publish gate; S50 does NOT change this gate
- DEC-052 OUTCOME REVISED — C6 retirement that triggered T0c Q3 pivot fire
- DEC-053 — Re-sequence decision (this scoping doc is its companion)
- S48-T0c Q1 Amendment — Amazon Bundle 2 carve-out evidence (load-bearing for S50)
- `docs/sprint52/preliminary/` — preserved ordered-boosting plan + algorithmic deep-dive (for S52+ when categorical closure lands)

---

## §7 — Definition of Done

- [ ] T0–T6 documented under `docs/sprint50/`
- [ ] DEC-053 OUTCOME finalized in DECISIONS.md
- [ ] G1–G4 verified empirically (Amazon parity + no regression + Bundle 2 re-eligibility + performance floor)
- [ ] DEC-051 amendment: Amazon re-included in Bundle 2 hard gate (or documented why not)
- [ ] HANDOFF/TODOS/CHANGELOG-DEV/LESSONS-LEARNED updated
- [ ] PR `mlx/sprint-50-categorical-closure` → master
- [ ] S52 ordered-boosting kickoff sprint plan inherits from `docs/sprint52/preliminary/` (NEXT)

**S50 is "successful" under all outcomes** if defect is properly diagnosed and either fixed OR documented with concrete next-step plan.

---

**S50 READY. T0 ready to fire when user greenlights.**
