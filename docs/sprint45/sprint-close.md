# Sprint 45 Close — Performance Spike: H-Dispatch Probe + Cross-Class Lock

**Sprint:** 45  |  **Status:** CLOSED  |  **Branch:** `mlx/sprint-45-perf-spike-and-decide`
**Cut from:** master `d3bc0e1d02` (post v0.6.1 release)
**Authoritative records:** this file + `docs/sprint45/T2/probe-verdict.md` + `docs/sprint45/T3/decision-synthesis.md`
**Theme:** Run a single focused probe against the H-Dispatch throughput hypothesis, produce a cross-class CUDA writeup and a parity-oracle CLI sketch, and lock the decision gate on v0.7.0's release path.

---

## TL;DR

S45 probed the H-Dispatch hypothesis — that Epsilon's 88× cross-class slowdown vs CUDA was driven by ~3,000 Metal dispatches per iteration — and falsified it by code inspection alone: `DispatchHistogramBatched` already fuses all feature groups into a single Metal dispatch per depth level, producing 6 dispatches/iter regardless of feature count. Dispatch overhead is 0.008% of Epsilon's 2241ms/iter. DEC-048 permanently kills the dispatch-fusion lever. Two parallel artifacts ship regardless of the probe outcome: a 5,300-word cross-class CUDA writeup (`docs/benchmarks/cross-class-cuda-comparison.md`) and a `catboost-tripoint` parity-oracle CLI sketch (`tools/catboost_tripoint/`). v0.7.0 is now gated on a separate, fresh investigation into the `simd_shuffle_xor` serial chain — the actual attributed cost driver — not on the dispatch route.

---

## Items Completed

| Item | Description | Outcome | Commit(s) |
|---|---|---|---|
| **T0** | Sprint scaffold — branch, sprint-plan revision, S45 task tracking | Sprint plan with single H-Dispatch probe; risk register; three-outcome gate | `bd4e65c29e` |
| **T1** | Branch-B regression test — bit-equivalence CI gate on Higgs-1M + Epsilon at v0.6.1 baseline | GREEN. `test_branch_b_regression.py` wired to CI; byte-identical match against pickled reference | `04fe8ef894` |
| **T2** | H-Dispatch probe — dispatch count + dispatch overhead measurement on Higgs-1M and Epsilon | **Outcome C — HALT.** Both C-triggers fired: dispatch overhead 0.008% (threshold: <20%) + Step 3 engineering already production code | T6 commit (no eng commits permitted pre-T3) |
| **T3** | Decision gate — DEC-048 KILL with "narrowed not killed" scope; devils-advocate YELLOW review | DEC-048 committed. KILL on dispatch-fusion; simd_shuffle lever explicitly NOT closed | `253f6ce3d5` |
| **T4** | Cross-class CUDA writeup — three-platform bit-equivalence; per-feature-dim wall-clock structure | `docs/benchmarks/cross-class-cuda-comparison.md` (5,300 words). Ships regardless of T3. | T6 commit |
| **T5** | `catboost-tripoint` parity-oracle CLI sketch — CPU/MLX/CUDA agreement report against fp32 floor | `tools/catboost_tripoint/` (8 files, ~180 LoC). Ships regardless of T3. | T6 commit |
| **T6** | Sprint close-out — this file, HANDOFF/TODOS/CHANGELOG-DEV, LESSONS-LEARNED (×2), PR | This commit | T6 commit |

---

## What Landed

### T1 — Branch-B Regression Test

`python/tests/regression/test_branch_b_regression.py` is the production bit-equivalence gate for v0.6.1. It trains catboost-mlx on Higgs-1M (iter=200) and Epsilon (iter=200) at seed 42, loads pickled reference outputs (`v0.6.1_predict_baselines.pkl`), and asserts byte-identical match (`atol=0, rtol=0`). Added to `.github/workflows/python-tests.yml`. No future optimization that flips a prediction bit will pass this gate without an explicit reference-update commit.

### T4 — Cross-Class CUDA Writeup

`docs/benchmarks/cross-class-cuda-comparison.md` covers three-platform bit-equivalence (M3 Max MLX, M3 Max CPU, RTX 5070 Ti / Blackwell CUDA) across the full 5-dataset suite. The headline finding is that the cross-class wall-clock ratio scales sharply with feature dimensionality:

| Workload | M3 MLX | M3 CPU | CUDA (RTX 5070 Ti) | MLX/CUDA |
|---|---|---|---|---|
| Higgs-1M iter=1000 (28 features) | 128.79s | 24.51s | 5.55s | ~23× |
| Epsilon iter=2000 (2000 features) | 4480s | 282s | 50.6s | ~88× |

The writeup explicitly frames cross-class comparisons as informational, not same-machine claims, and attributes the per-feature-dim scaling to kernel work volume (batch-TG-ops) rather than dispatch count. All 51 CUDA result JSONs from the RTX 5070 Ti sweep are in `docs/sprint45/cuda-bench-bundle/results/`.

### T5 — `catboost-tripoint` CLI Sketch

`tools/catboost_tripoint/` is a ~180-LoC single-entrypoint CLI (`catboost-tripoint verify --model X.cbm --data Y.parquet`) that runs the same model on CatBoost-CPU, CatBoost-MLX, and CatBoost-CUDA (when available), computes per-tree leaf-value max-abs-diff, per-row prediction divergence (max + mean), and the mathematician's theoretical fp32 floor (`ε_mach × T × √L`), and emits a signed JSON verification report with PASS/FAIL against the derived bound. Demonstrated on Higgs-1M and Epsilon. This is the v0.6.x reproducibility-grade parity oracle — not production-hardened in S45, but the framing and artifact are live.

---

## What Did Not Land (and Why)

### T2 — H-Dispatch Probe: Outcome C

The probe hypothesis was that Epsilon issues ~3,000 Metal kernel dispatches per iteration and that fusing these into one multi-group dispatch per depth level would deliver the ≥3× speedup needed to gate v0.7.0. The probe falsified this at Step 1 — no instrumented build was needed.

**The production implementation already is the proposed fix.** `DispatchHistogramBatched` (`catboost/mlx/methods/histogram.cpp:31`) encodes all feature groups into the X dimension of a 3D Metal dispatch grid (`256 * maxBlocksPerPart * numGroups`). The dispatch count is 6/iter on both Epsilon (2000 features, numGroups=500) and Higgs-1M (28 features, numGroups=7) — identical. The "2000 features ÷ 4 per group × 6 depth = 3,000 dispatches" arithmetic that led six advisory agents to the same wrong conclusion describes independent per-group dispatches; the production dispatcher fused them via the `numGroups` parameter before S45 began.

Dispatch overhead: 6 × ~30 µs = 0.18 ms/iter = 0.008% of Epsilon's 2241 ms/iter. The Outcome C threshold was <20%; the actual measurement missed it by 2,500×.

**The actual cost driver**, per S19-01c (DEC-020), is the `simd_shuffle_xor` serial chain in the histogram accumulation kernel: 86% of accumulation time, scaling linearly with batch-TG-ops (~200M for Epsilon vs ~438k for Higgs — a 456× ratio that closely tracks the observed cross-class wall-clock gap).

---

## DEC-048

**KILL — H-Dispatch throughput lever permanently retired** (filed in `.claude/state/DECISIONS.md`, Sprint 45, 2026-05-04).

`DispatchHistogramBatched` already batches all feature groups into a single Metal dispatch per depth level. Dispatch count is 6/iter for both Epsilon and Higgs-1M; dispatch overhead is 0.008% of Epsilon iter wall-clock. The proposed S46 engineering is already production code. There is no speedup available via the dispatch-fusion route; this hypothesis is closed and forbidden from re-opening absent evidence of a regression to per-group dispatching.

**Scope of the KILL is the dispatch-fusion lever specifically.** The `simd_shuffle_xor` serial chain was not in the S45 hypothesis set and is not closed by this DEC. DEC-048 narrows the throughput problem; it does not retire it permanently. The throughput epic is one lever from permanently dead — not zero levers.

Companion documents: `docs/sprint45/T2/probe-verdict.md` (Step 1 code inspection + analytical ablation + Step 3 determination), `docs/sprint45/T3/decision-synthesis.md` (strategist's verdict + Path A/C framing), `docs/sprint45/T3/devils-advocate-review.md` (YELLOW on "narrowed not killed"; §4 MANDATORY-CODE-INSPECTION gate).

---

## What's Next: S46 as a Fresh Research Arc

The user's choice is Path C — pursue the `simd_shuffle_xor` serial chain, the lever S19-01c identified as 86% of accumulation cost.

**Critical framing constraint:** S46 must open as a fresh research investigation, not a "throughput epic continuation." The devils-advocate's YELLOW review on DEC-048 correctly identifies that three consecutive sprints (S43, S44, S45) approached the throughput question, each ending with "this specific lever doesn't work but the broader question remains open." Continuing that pattern under the same W1/W2/W3 throughput-pivot framing is sunk-cost protection. DEC-048 retires the throughput-pivot framing entirely. If simd_shuffle work begins, it opens a NEW DEC and a NEW sprint with its own gates and kill criteria — not a continuation of the W1/W2/W3 arc.

**S46 scope — the spike-then-commit pattern from S45:**

S46 is a scoping and research sprint, not an engineering sprint. The pattern that made S45 productive (define a single hypothesis with a binary outcome gate, produce a verdict document before committing any production engineering) must be re-applied.

- **S46 deliverables:** feasibility report on `simd_shuffle_xor` redesign + design proposal + engineering plan with kill criteria. NOT production kernel commits.
- **S46 design questions:** what warp-shuffle reduction strategy replaces the serial `simd_shuffle_xor` chain? What is the realistic speedup bound on M3 Max? What is the Branch-B regression-test constraint on accumulation-order change? S19-01c's own conclusion was "requires algorithmic restructuring" — S20 was supposed to address it and didn't ship. S46 must answer why S20 failed and whether the proposed path avoids that failure mode.
- **Multi-sprint engineering only if S46 produces a viable plan.** The engineering window (S47+) opens only when S46 delivers a design with stated kill criteria and a one-sprint falsification gate.
- **New DEC required.** The simd_shuffle scope decision must be filed as a new DEC (DEC-049 or as-numbered) before any S47 engineering begins.

---

## v0.7.0 Status

**Indefinite hold.** The v0.7.0 release condition (≥3× MLX/CPU iter speedup on Epsilon iter=2000, or ≥2× on Higgs-1M iter=1000, measured at production dispatch shape, 3 seeds × 3 runs warm) is not met by any currently-shipped or in-flight change. The dispatch route is permanently closed by DEC-048. The simd_shuffle route is unproven and requires S46 feasibility work before any engineering commitment.

The project remains at v0.6.1 indefinitely until S46+ delivers a measurable perf delta. PyPI publish is gated on v0.7.0, which is gated on S46+ outcome. If S46 produces Outcome C, the "reproducibility-grade extension only" path (v0.7.0 = T4 + T5 artifacts without perf gate) is on the table as an explicit user decision — per devils-advocate's Option 1 (retire v0.7.0 throughput framing; shift proposition from "fast GBDT" to "auditable cross-platform GBDT").

Two v0.6.x artifacts that ship in any case: `docs/benchmarks/cross-class-cuda-comparison.md` (T4) and `tools/catboost_tripoint/` (T5). Both are load-bearing for the reproducibility-grade proposition regardless of v0.7.0 outcome.

---

## CI Status

| Workflow | Status |
|---|---|
| Compile csv_train (Apple Silicon) | green (no source changes in S45) |
| MLX Python Test Suite | green; `test_branch_b_regression.py` PASS |
| `mlx-perf-regression.yaml` | green (speedup-ratio gate; no kernel changes) |
| Branch-B regression gate (new T1) | GREEN — v0.6.1 bit-equivalence protected |

---

## Files in This Sprint

```
docs/sprint45/
  sprint-plan.md                            (T0)
  sprint-close.md                           (T6, this file)
  T2/probe-verdict.md                       (T2 — Outcome C verdict)
  T3/decision-synthesis.md                  (T3 — strategist)
  T3/devils-advocate-review.md              (T3 — YELLOW review)
  cuda-bench-bundle/results/*.json          (51 cells — RTX 5070 Ti)
  cuda-bench-bundle/hardware.txt

docs/benchmarks/
  cross-class-cuda-comparison.md            (T4 — 5,300 words)

tools/catboost_tripoint/                    (T5 — 8 files, ~180 LoC)

python/tests/regression/
  test_branch_b_regression.py               (T1)
  v0.6.1_predict_baselines.pkl              (T1 reference)

.claude/state/
  DECISIONS.md                              (DEC-048 appended, T3)
  HANDOFF.md                                (T6 — this close-out block)
  TODOS.md                                  (T6 — S45 DONE, S46 pending)
  CHANGELOG-DEV.md                          (T6 — 2026-05-04 session entry)

Frameworks/LESSONS-LEARNED.md               (T6 — two new entries appended)
```

No `catboost/mlx/**.{cpp,h,metal}` source changes. No kernel changes. Production kernel v5 (`784f82a891`) byte-identical from S30 → S45.

---

## Branch Lifecycle

`mlx/sprint-45-perf-spike-and-decide`: T0/T1 commits `bd4e65c29e` + `04fe8ef894`; T3 commit `253f6ce3d5`; T6 close-out commit (this). Single PR to master after T6 lands.

S45 with T3=HALT and no engineering commits is a successful sprint. DEC-048 + T4 + T5 are permanent assets. The probe cost was one sprint cycle to confirm the dispatch-fusion engineering is already done; the Probe-D protocol recovered the cost by preventing an engineering sprint from writing and committing code that would have produced 0.008% speedup.
