# Sprint 45 Plan — Performance Spike: H-Dispatch Probe + Cross-Class Lock

**Sprint:** 45  |  **Status:** APPROVED — execution starting 2026-05-04  |  **Branch:** `mlx/sprint-45-perf-spike-and-decide`
**Cut from:** master `d3bc0e1d02` (post v0.6.1 release)
**Theme:** Resolve the throughput question with a single focused probe (H-Dispatch) informed by fresh CUDA cross-class data, lock the three-platform parity claim, and gate any v0.7.0 release on measured perf delta.

## Strategic context

v0.6.1 shipped the reproducibility-grade frame on Apple Silicon (`docs/benchmarks/v0.6.0-pareto.md` §1). The CUDA cross-class sweep landed on 2026-05-04 (RTX 5070 Ti, Blackwell, 16 GB VRAM, CatBoost 1.2.10) — confirming three-platform bit-equivalence at fair convergence (≤0.0002 logloss) and revealing the cross-class wall-clock gap structure:

| Workload | M3 MLX | M3 CPU | Win CUDA | MLX/CUDA |
|---|---|---|---|---|
| Higgs-1M iter=1000 | 128.79s | 24.51s | 5.55s | ~23× |
| Epsilon iter=2000 | 4480s | 282s | 50.6s | ~88× |

The cross-class ratio scales sharply with feature dimensionality (Higgs 28 features ≈ 23×; Epsilon 2000 features ≈ 88×). This is the single most diagnostic finding in the data — and a 6-agent advisory panel (silicon-architect, mathematician, performance-engineer, devils-advocate, strategist, visionary) independently identified its mechanism: **dispatch overhead, not kernel arithmetic**. Epsilon issues ~3,000 Metal kernel dispatches per iteration (2000 features ÷ 4 features/group × 6 depth levels) vs Higgs's 42. This was NOT in the original throughput-pivot framing (W1/W2/W3) and was NOT in the original 3-hypothesis spike (H-Sibling, H-Bandwidth, H-Sync).

**v0.7.0 release condition:** ≥3× MLX/CPU iter speedup on Epsilon (or ≥2× on Higgs-1M iter=1000), measured at production dispatch shape, 3 seeds × 3 runs warm. PyPI publish is gated on this delta — without performance, no PyPI release.

## What changed vs the original (2026-05-02) plan

The original S45 plan called for a 3–5 day spike testing three hypotheses (H-Sibling, H-Bandwidth, H-Sync) across multiple datasets. The CUDA data published the day before sprint start changed three priors:

1. **H-Bandwidth is no longer diagnostic.** Mathematician's roofline calc shows MLX achieves ~0.13% of M3 Max bandwidth on Epsilon and ~0.007% on Higgs. The kernel is not bandwidth-bound; it is dispatch/orchestration-bound. The bandwidth probe would confirm a number we already have.
2. **H-Sibling is unlikely to dominate at this scale.** Sibling subtraction halves histogram-build time at depth ≥ 2, but when histogram is no longer the plurality of iter wall-clock (post-S22 sort-by-bin, DEC-020), this is a 10–15% lever, not a 3× lever.
3. **H-Dispatch is the missing hypothesis.** Three agents independently arrived at this finding from different angles. Probe-D for H-Dispatch is testable in 1–2 days at production dispatch shape with no kernel engineering.

This sprint replaces the broad 3-hypothesis spike with a single focused H-Dispatch probe, plus two parallel artifacts (cross-class writeup, parity oracle CLI) that ship regardless of T3 outcome.

## Items table

| # | Description | Status | Owner | Reviewers |
|---|---|---|---|---|
| **T0** | Sprint scaffold (branch, plan revision, S45 task tracking) | **IN PROGRESS** | @ml-product-owner | — |
| **T1** | Branch-B regression test (CI gate locking v0.6.1 predict-output bit-equivalence on Higgs-1M + Epsilon) | TODO | @performance-engineer | @qa-engineer, @mathematician |
| **T2** | **H-Dispatch probe** — instrument dispatch count + per-dispatch latency on Higgs-1M (iter=200) and Epsilon (iter=200, iter=2000); simulate single-dispatch upper bound. **NO engineering commits.** | TODO | @performance-engineer | @silicon-architect |
| **T3** | Decision gate — A (≥3×) / B (1.5–3×, user-call) / C (<1.5×, HALT). DEC-048 filed regardless. | TODO | @strategist | @devils-advocate |
| **T4** | Cross-class CUDA writeup at `docs/benchmarks/cross-class-cuda-comparison.md` — three-platform bit-equivalence; per-feature-dim wall-clock structure; honest cross-class framing | TODO (parallel with T1/T2) | @data-scientist | @technical-writer |
| **T5** | `catboost-tripoint` CLI sketch — 150–200 LoC predict + diff harness verifying CPU/MLX/CUDA agreement against the derived fp32 floor (visionary's parity-oracle artifact) | TODO (parallel with T1/T2) | @ml-engineer | @mathematician |
| **T6** | Sprint close-out — DEC-048 entry, HANDOFF/TODOS/CHANGELOG-DEV update, PR. v0.7.0 release ceremony deferred to S46 conditional on T3 = COMMIT + S46 engineering landing the perf gate. | TODO | @ml-product-owner + @technical-writer | @strategist |

T2's outcome determines whether S46 runs (engineer the H-Dispatch lever) and consequently whether v0.7.0 ships in this development cycle. T4 + T5 ship in S45 regardless of T3, because the cross-class parity story is itself a load-bearing v0.7.0 artifact — not contingent on perf engineering.

## What the prior verdict said (and why this plan does not repeat it)

The original 2026-04-26 advisory round produced these verdicts, all of which are preserved and informed today's hypothesis revision:

- **@silicon-architect (April):** 32 KB threadgroup-memory ceiling caps histogram tiling; sibling subtraction is the actual missing lever; realistic landing Higgs 5.4× → ~2.5× over 5 sprints. **(May update: H-Dispatch is the more likely lever on high-feature workloads; sibling subtraction is real but marginal at production dispatch shape.)**
- **@performance-engineer (April):** Probe-D protocol mandatory; toy-bench projections do not transfer to production. **(May update: confirmed; H-Dispatch is testable via dispatch-count instrumentation + single-dispatch ablation, no toy bench needed.)**
- **@mathematician (April):** Both Higgs and Epsilon at fp32 floors; bandwidth roofline gap. **(May update: bandwidth gap confirmed at 0.007–0.13% of roof — kernel is dispatch-bound, not bandwidth-bound.)**
- **@devils-advocate (April):** Demanded a 3–5 day spike before committing 4 sprints. **(May update: spike narrowed to 1–2 days on a single hypothesis; threshold raised from 1.5× to 3× MLX/CPU iter speedup for COMMIT.)**
- **@strategist (April–May):** Throughput epic deferred to v0.7.x with kill criteria; reproducibility-grade frame validated. **(May update: throughput epic narrowed to H-Dispatch only; v0.7.0 PyPI publish gated on perf delta landing.)**

This plan does NOT repeat the W1/W2/W3 framing. The plan does not assume a multi-lever throughput pivot. It commits to one focused experiment with binary outcome gating.

## T1 — Branch-B regression test (owner: @performance-engineer)

CI gate that protects v0.6.1's bit-equivalence claim from any future perf optimization. Deliverables in `python/tests/regression/`:

1. **`test_branch_b_regression.py`** — pytest that:
   - Loads a fixed seed (42), trains catboost-mlx on Higgs-1M iter=200 and Epsilon iter=200
   - Compares `predict_proba()` output on test set against pickled reference (`v0.6.1_predict_baselines.pkl`)
   - Asserts byte-identical match (allclose with `atol=0, rtol=0`)
2. **`v0.6.1_predict_baselines.pkl`** — checked-in reference outputs from current master (v0.6.1)
3. **CI integration** — added to existing `.github/workflows/python-tests.yml` Python-tests job

Exit criteria: regression test green on master (current state); test fails reproducibly when `_predict_inprocess` returns altered output; CI runs the test on every PR.

This MUST be green at all times during T2/S46. Any optimization that flips a bit fails this gate.

## T2 — H-Dispatch probe (owner: @performance-engineer, advised by: @silicon-architect)

The single load-bearing experiment for v0.7.0. Output: a verdict document, not engineering code.

**Hypothesis (H-Dispatch):** Epsilon iter=2000 issues ~3,000 Metal kernel dispatches per iteration (one per feature group per depth level). The MLX graph-construction overhead per dispatch (~30–50 µs per `mx::fast::metal_kernel` call based on DEC-014) compounds with 2000-feature workloads in a way it does not with 28-feature workloads. Closing this requires fusing the per-feature-group dispatches into a single multi-group dispatch per depth level, which the kernel signature already supports via the `numGroups` parameter — the lever is in the dispatcher, not the kernel.

**Probe-D experiment design:**

1. **Measure** per-dispatch latency × dispatch count per iter on Epsilon (iter=200, iter=2000) and Higgs-1M (iter=200) using `CATBOOST_MLX_STAGE_PROFILE` instrumentation. Confirm Epsilon issues ~3000 dispatches/iter; confirm dispatch overhead is X% of iter wall-clock.
2. **Ablate** by replacing the per-group histogram dispatch with a `mx::zeros()` of the correct output shape — keeps dispatch-call count and graph construction cost; eliminates kernel work. Measure resulting `histogram_ms`. Difference between this and production = pure kernel work cost; remaining = dispatch + graph overhead.
3. **Single-dispatch upper bound** — modify dispatcher to issue ONE multi-group dispatch per depth level (not per feature group), keeping kernel work intact via the existing `numGroups` parameter. This is a 1-day implementation in `histogram.cpp:DispatchHistogramBatched` (the kernel already accepts numGroups). Measure iter wall-clock improvement on Epsilon iter=2000.

Exit criteria: a verdict document with one of three outcomes:
- **Outcome A — Single-dispatch upper bound shows ≥3× iter speedup on Epsilon iter=2000:** T3 = COMMIT to S46 dispatch-fusion engineering.
- **Outcome B — Upper bound shows 1.5–3× speedup OR ≥3× plausible only with multi-sprint engineering:** T3 = user-decision call.
- **Outcome C — Upper bound shows <1.5× speedup, OR dispatch overhead is <20% of iter wall-clock and not the load-bearing cost:** T3 = HALT throughput epic. DEC-048 = KILL with empirical justification.

**Branch-B constraint:** the single-dispatch ablation in step 3 may break bit-equivalence by altering accumulation order. The probe is allowed to break the regression test temporarily in the probe branch. T3=COMMIT requires planning for bit-equivalence preservation in S46 engineering.

## T3 — Decision gate (owner: @strategist, reviewed by: @devils-advocate)

Strategist synthesizes T2 verdict + cross-class CUDA writeup state. Devils-advocate stress-tests against LESSONS-LEARNED and the meta-criterion ("if T2 = Outcome C, throughput epic is retired permanently — gap is hardware-class-bound, further pursuit is sunk-cost theater").

Output: explicit COMMIT or HALT decision recorded as `DEC-048` in `.claude/state/DECISIONS.md` regardless of direction.

**T3 happens BEFORE any S46 engineering is scheduled.** If T3 = HALT, S46 is repurposed (likely toward MSLR or csv_train uint8 cat-bin fix per DEC-047 follow-up); if T3 = COMMIT, S46 scope is "engineer dispatch fusion + Branch-B regression green + ≥3× exit gate."

## T4 — Cross-class CUDA writeup (owner: @data-scientist, reviewed by: @technical-writer)

`docs/benchmarks/cross-class-cuda-comparison.md`. Ships in S45 regardless of T3. Sections:

1. **TL;DR** — three-platform bit-equivalence; cross-class wall-clock structure; per-feature-dim ratio scaling; honest cross-class methodology disclaimer (different hardware classes, not same-machine)
2. **Hardware** — M3 Max (M3 silicon, ~5–7 TFLOPS, ~400 GB/s) vs RTX 5070 Ti (Blackwell, 30 TFLOPS, 896 GB/s)
3. **Methodology** — same hyperparameters, same datasets, same iter grid, same CatBoost version (1.2.10) across both. Window opened by user's Windows GPU box on 2026-05-04.
4. **Per-dataset results** — full 5-dataset matrix with paired CPU/MLX/CUDA wall-clock + logloss
5. **Accuracy alignment** — three-platform bit-equivalence at fair convergence (≤0.0002 logloss); MLX architectural floor structurally characterized
6. **Wall-clock structure** — per-feature-dim ratio scaling (Higgs 23× → Epsilon 88×); attribution to dispatch overhead
7. **Honest limitations** — cross-class is informational, NOT a same-machine perf claim; closing 88× is partly hardware physics

Sources: `docs/sprint45/cuda-bench-bundle/results/*.json` (51 files, RTX 5070 Ti) + `benchmarks/upstream/results/*.json` (M3 Max).

## T5 — `catboost-tripoint` CLI sketch (owner: @ml-engineer, reviewed by: @mathematician)

Visionary's parity-oracle artifact. Single CLI tool: `catboost-tripoint verify --model X.cbm --data Y.parquet`. Runs the same model on (a) catboost-cpu, (b) catboost-mlx, (c) catboost-cuda if available; emits a numerical-agreement report:
- Per-tree leaf-value max-abs-diff
- Per-row prediction divergence (max + mean)
- Theoretical fp32 floor for `(T, depth)` model: `ε_mach × T × √L` (mathematician's formula)
- PASS/FAIL against derived bound

Output: signed JSON with the verification result. Audience: regulated-ML teams (fintech, healthcare) needing cross-platform model-validation receipts; MLOps platforms certifying GPU-portability.

Scope: 150–200 LoC sketch. NOT production-grade in S45 — just enough to demonstrate the artifact and validate the framing on Higgs-1M + Epsilon. v0.7.x evolution: signed reports, multi-model verification, CI integration.

## T6 — Sprint close-out

- DEC-048 entry in `.claude/state/DECISIONS.md` (KILL or COMMIT direction, with empirical justification from T2)
- HANDOFF.md update with T3 outcome
- TODOS.md: T1–T6 status; S46 scope conditional on T3
- CHANGELOG-DEV.md: 2026-05-XX session entry
- LESSONS-LEARNED.md: at least one entry — either "narrow probe with strong prior beats broad spike" (if T3=COMMIT confirms H-Dispatch) or "throughput epic is hardware-class-bound; reproducibility-grade is the durable v0.x story" (if T3=HALT)
- Single PR (sprint-45 → master) once all six tasks land

## Branch + PR plan

- Single branch `mlx/sprint-45-perf-spike-and-decide`
- T0/T1/T2/T4/T5 commit to branch (T2 commits are profiling artifacts and verdict doc — engineering is deferred to S46)
- T3 commits the DEC-048 entry
- T6 commits the sprint close-out
- Single PR after T6 lands, regardless of T3 direction
- A sprint that closes with T3=HALT and no engineering work IS a successful sprint (v0.7.0 = "reproducibility-grade extension only" path) — see DEC-048 KILL outcome

## Files in scope vs explicitly NOT in scope

**In scope (S45):**
- `docs/sprint45/sprint-plan.md`, `docs/sprint45/sprint-close.md`, `docs/sprint45/T2/` (probe artifacts)
- `docs/benchmarks/cross-class-cuda-comparison.md` (T4)
- `docs/sprint45/cuda-bench-bundle/` (already in working tree; commit excluding cache/)
- `docs/sprint45/cuda-bench-bundle/results/*.json` (51 cells from RTX 5070 Ti) + `hardware.txt`
- `python/tests/regression/test_branch_b_regression.py` (T1)
- `tests/regression/v0.6.1_predict_baselines.pkl` (T1 reference data)
- `tools/catboost_tripoint/` or similar (T5 — exact path TBD by @ml-engineer)
- `.claude/state/{HANDOFF,TODOS,CHANGELOG-DEV,DECISIONS}.md` (T3 + T6)
- `Frameworks/LESSONS-LEARNED.md` (T6)

**Explicitly NOT in scope (deferred to S46 conditional on T3=COMMIT):**
- `catboost/mlx/methods/histogram.cpp` engineering changes (T2 probe may include a measurement-only modification on a probe branch but does not commit; S46 commits the production fix)
- `catboost/mlx/kernels/**.metal` source changes
- `python/catboost_mlx/**` API changes
- Anything that would alter `predict()` numeric output (would fail T1's regression test)

**Explicitly NOT in S45 under any outcome:**
- PyPI publish — gated on v0.7.0 perf delta landing (S46 outcome dependent)
- Sibling subtraction implementation (deprioritized in light of H-Dispatch finding)
- W1/W2/W3 throughput-pivot framing — replaced by single-hypothesis probe
- New launch artifacts beyond cross-class writeup + parity oracle CLI sketch
- Multi-sprint engineering commitments (T3=COMMIT scope must be 1-sprint)

## Risk register

| Risk | Mitigation | Owner |
|---|---|---|
| **H-Dispatch falsified** | T3 = HALT is a pre-defined valid outcome. Sprint closes with T4 (writeup) + T5 (oracle CLI) + DEC-048 KILL. v0.7.0 path becomes "reproducibility extension only" with PyPI publish deferred indefinitely OR shipped on current perf. | @strategist |
| **H-Dispatch upper bound only 1.5–3×** | Outcome B path: explicit user-decision call. Plan does not commit engineering to a marginal win without confirmation. | @ml-product-owner |
| **Branch-B regression breaks during T2 ablation** | Allowed temporarily on probe branch. T3=COMMIT requires bit-equivalence preservation in S46 engineering plan. | @qa-engineer |
| **T1 regression test infrastructure cost** | T1 budget = 1 day. If pickle round-trip is more than that, switch to JSON-serialized reference outputs (smaller, version-controllable). | @performance-engineer |
| **T4 writeup gates on additional CUDA runs** | T4 ships with current 51-cell coverage. Higgs-11M iter=1000 not in this dataset; document as future work. | @data-scientist |
| **T5 catboost-tripoint over-scopes to production tool** | T5 = sketch only. ≤200 LoC. Production hardening deferred to v0.7.x. | @ml-engineer |
| **Multi-week creep from T3=COMMIT scope** | T3 decision MUST include a 1-sprint scope. Multi-sprint engineering = T3=HALT or Outcome B user-call, not commit. | @strategist |

## Definition of Done

**S45 closes successfully if:**
- T1 regression test exists and is green on master
- T2 verdict document exists and identifies one of {A, B, C}
- T3 DEC-048 entry exists with empirical justification
- T4 cross-class writeup exists at `docs/benchmarks/cross-class-cuda-comparison.md`
- T5 `catboost-tripoint` CLI sketch exists and runs on Higgs-1M + Epsilon
- T6 close-out artifacts updated; PR opened

**v0.7.0 ships only if (separately, post-S45):**
- T3 = COMMIT path
- S46 engineering lands ≥3× iter speedup on Epsilon iter=2000 measured at production shape, 3 seeds × 3 runs warm
- Branch-B regression green throughout
- PyPI publish ceremony executes cleanly

If S46 fails to land the perf gate, v0.7.0 does not ship; the project remains at v0.6.1 indefinitely until either a research-grade lever appears or the audience proposition shifts.

## Agent panel

**Load-bearing trio:** @performance-engineer (T1, T2 owner), @silicon-architect (T2 advisor), @mathematician (correctness preservation + T5 fp32-floor formula)

**Decision authorities:** @strategist (T3 synthesis, T6 retro), @devils-advocate (T3 challenge)

**Implementation:** @ml-engineer (T5 sketch, S46 conditional), @qa-engineer (T1 test infrastructure), @code-reviewer (every commit)

**Data + writing:** @data-scientist (T4 writeup), @technical-writer (T4 review + T6 close-out)

**NOT consulted in S45:** @research-scientist, @hardware-researcher, @security-auditor, @mlops-engineer, @visionary (visionary's input was the T5 framing; that work is now @ml-engineer's)
