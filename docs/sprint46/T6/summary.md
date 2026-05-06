# S46 Sprint Summary — simd_shuffle Redesign Research Arc

**Date:** 2026-05-05
**Branch:** `mlx/sprint-46-simd-shuffle-research`
**Tip commit:** `94a260468b`
**Duration:** 1 day actual (1-week budget)
**Outcome:** DEC-049 RETIRED — Outcome C (HALT)

---

## TL;DR

S46 executed a 4-candidate bounded research arc to determine whether the histogram accumulation `simd_shuffle` src-broadcast chain could be restructured for ≥3× speedup on the Epsilon 400k×2000 production shape (v0.7.0 gate). After a build-env fix, a dispatcher rewrite, and a 27-run sweep, all four candidates were falsified. No production code changed. The sprint closes as designed — negative research results with quantitative evidence are a legitimate deliverable. v0.7.0 path is now a user call.

---

## T0–T6 Chronology

### T0 — Scaffold

Branch `mlx/sprint-46-simd-shuffle-research` cut from master at S45 PR #46 merge. `docs/sprint46/sprint-plan.md` committed. DEC-049 OPEN filed in `.claude/state/DECISIONS.md`. TODOS.md S46 row added. Terminology correction documented: S45 and DEC-048 refer to "simd_shuffle_xor" but the production kernel at `kernel_sources.h:209–224` uses `simd_shuffle` (broadcast), not `simd_shuffle_xor` (butterfly). The xor butterfly was REMOVED in S18 DEC-012. Sprint plan uses "src-broadcast chain" to avoid the legacy misnomer.

### T1 — Current-state characterization

`docs/sprint46/T1/current-state.md` documents the `kHistOneByteSource` accumulation loop at `kernel_sources.h:209–224` with file:line citations for every claim. Key findings: f_hist at Epsilon-proxy = 0.9772 (histogram phase is 97.7% of Epsilon iteration wall-clock); the src-broadcast chain runs 32 iterations per batch window, each dispatching 2 `simd_shuffle` calls; the ownership predicate `(bin & 31) == lane` gates writes so each lane accumulates only the bins it owns. Threadgroup memory at the 32 KB ceiling (DEC-011): `simdHist[8][1024] × 4 B = 32 KB`. No headroom for additive threadgroup state.

### T2 — Feasibility analysis

`docs/sprint46/T2/feasibility.md` evaluated all four bounded candidates against DEC-017/023/025 precedents. Candidate A (atomic-add accumulation) was RETIRED at T2 per DEC-025 re-entry policy — no new evidence vs DEC-017 (T3b +42.3% production regression) and DEC-023 (atomic-float race at config #8). Three candidates survived to T3: B (per-lane register accumulation), C (sort-by-bin extension), D (split-K merge kernel). f_hist = 0.9772 at Epsilon means any candidate with ≥3× hist speedup maps to ≥2.93× iter speedup — all three were analytically viable for Outcome A.

### T3 — Probe-D spec

`docs/sprint46/T3/probe-d-spec.md` specified the measurement protocol: 3 seeds × 12 iters (1 cold + 11 warm, 10%-trimmed mean) × 3 shapes (Gate 50k×100, Higgs-1M 1M×28, Epsilon 400k×2000). Kill threshold: Epsilon iter speedup < 1.5× → RETIRE. Probe binaries compiled under `#ifdef SIMD_SHUFFLE_PROBE_{B,C,D}` guards (S33 PROBE-E pattern), never built into production. T3 erratum: Probe D1 (intra-kernel K-split) requires `partialHist[4][8][1024] × 4B = 128 KB`, 4× the 32 KB threadgroup ceiling — retired structurally. D2 (separate dispatch + merge kernel) is the only viable D path.

### T4 — Probe-D execution

**Build blocker surfaced and resolved.** MLX 0.31.2 headers (Homebrew) were incompatible with Darwin 25.3 SDK (Apple Clang 21.0.0, macOS 15.4). This blocked standalone clang++ probe builds. Resolution (Option 2 / Route A, `docs/sprint46/T4/build-env/status.md`): probe binary targets integrated into `python/catboost_mlx/_core/CMakeLists.txt` via `BUILD_S46_PROBES=ON`, sharing the same `find_package(MLX)` resolution used by `_core.so`. The Python-installed MLX headers (conda/pip) are compatible; Homebrew headers are not.

Probe C was retired structurally (DEC-023 race — sort + per-lane accumulation cannot deterministically cover features 1-3). Probe D2 dispatcher was rewritten: the initial `mx::add` sequential chain serialized K dispatches in the MLX compute graph; replaced with K independent `kSlices` arrays + `mx::eval(kSlices)` + `mx::concatenate`.

**27-run sweep** (3 shapes × 3 seeds × 3 probes = 27 runs; raw data at `docs/sprint46/T4/probe-d/results.json`):

| Probe | Epsilon iter speedup | Parity |
|---|---|---|
| B (per-lane register) | 9.79× | FAIL — Δloss 0.005–0.008, iter-0 |
| D2 (split-K merge) | 1.006× | FAIL — Δloss 0.03+, Epsilon |

### T5 — Decision gate

`docs/sprint46/T5/decision.md` records the full verdict. Probe B's 9.79× speedup is diagnosed as artifact: `kernel_sources.h:1374-1407` confuses processor-lane (doc index mod 32) with owner-lane (bin value mod 32). P(match) = 1/32. ~96.9% of contributions silently dropped. Probe D2 is empirically flat (1.006×); dispatch amortization + `mx::concatenate` copy cost absorb the K-fold src-loop reduction. Structural conclusion: the premise "eliminate src-broadcast without restoring routing" is logically impossible — bin-owner mapping is intrinsic to bin values; any routing-free accumulation scheme silently drops contributions in proportion to 1/SIMD_SIZE. **Outcome C — RETIRED. DEC-049 = KILL.**

### T6 — Sprint close-out (this document)

DEC-049 OUTCOME appended to `.claude/state/DECISIONS.md`. LESSONS-LEARNED entries filed (project-local + Frameworks-wide). HANDOFF.md updated with BLOCKING DECISION FOR S47. TODOS.md and CHANGELOG-DEV.md updated. Memory index updated. PR `mlx/sprint-46-simd-shuffle-research` → master prepared for user review.

---

## What Shipped

**Probe variants (all ifdef-gated, never production):**
- `catboost/mlx/kernels/kernel_sources.h` — Probe B block (`#ifdef SIMD_SHUFFLE_PROBE_B`), Probe D blocks (`#ifdef SIMD_SHUFFLE_PROBE_D`). Production path bit-identical to v0.6.1 baseline throughout.
- `catboost/mlx/tests/bench_boosting.cpp` — `DispatchHistogramProbeDKBench` dispatcher under `SIMD_SHUFFLE_PROBE_D` guard, with corrected K-slot pre-allocation + `mx::concatenate` + merge kernel pattern.
- `python/catboost_mlx/_core/CMakeLists.txt` — `BUILD_S46_PROBES=ON` option with three executable targets (`bench_boosting_baseline`, `bench_boosting_probe_b`, `bench_boosting_probe_d`); build-env fix via conda-path `find_package(MLX)`.

**Documentation (all under `docs/sprint46/`):**
- `T1/current-state.md` — `kHistOneByteSource` characterization with file:line citations
- `T2/feasibility.md` — 4-candidate feasibility analysis; Candidate A retired at T2
- `T3/probe-d-spec.md` — measurement protocol + D1 erratum
- `T4/f_hist/analysis.md` — f_hist measurement; Epsilon 0.9772 confirmed
- `T4/B/probe-verdict.md`, `T4/C/probe-verdict.md`, `T4/D/probe-verdict.md` — per-candidate verdicts
- `T4/build-env/status.md` — build blocker root cause and resolution
- `T4/probe-d/results.json` — 27-run sweep raw data
- `T4/probe-d/parse_results.py` — parsing and speedup computation
- `T5/decision.md` — full T5 decision gate

**State files updated in T6:**
- `.claude/state/DECISIONS.md` — DEC-049 OUTCOME appended
- `.claude/state/LESSONS-LEARNED.md` — "SIMD histogram routing invariant" entry
- `.claude/state/HANDOFF.md` — S46 CLOSED + BLOCKING DECISION FOR S47
- `.claude/state/TODOS.md` — S46-T5/T6 COMPLETED; S47 BLOCKED placeholder
- `.claude/state/CHANGELOG-DEV.md` — S46 entry
- `Frameworks/LESSONS-LEARNED.md` — Frameworks-wide routing invariant entry

---

## What Did NOT Ship

No production code changed. `catboost/mlx/kernels/kernel_sources.h` is byte-identical to v0.6.1 on the production code path (all probe blocks ifdef-gated out). No changes to `histogram.cpp`, `structure_searcher.cpp`, or any Python API. Branch-B regression gate (`python/tests/regression/test_branch_b_regression.py`) GREEN on master throughout. No v0.7.0 release — perf gate not cleared.

---

## Process Wins

**Build-env unblocked for v0.7.x.** The `BUILD_S46_PROBES=ON` CMake integration establishes a reusable pattern for future probe binary targets without re-diagnosing the MLX 0.31.2 / Darwin 25.3 SDK incompatibility.

**LESSONS-LEARNED MANDATORY-CODE-INSPECTION rule fired correctly.** The requirement to cite file:line for every mechanism claim led T4 to inspect `kernel_sources.h:1374-1407` after observing the 9.79× speedup — which is what surfaced the processor/owner-lane confusion. Without the inspection requirement, Probe B would have routed to T5 as a candidate for Outcome A and on to S47 engineering with a broken kernel.

**First throughput falsification caught before engineering.** This is the 7th throughput-hypothesis falsification in this codebase (DEC-013, 014, 015, 017, 019, 048, 049), but the first caught entirely within the research sprint before any production-bound engineering began.

---

## Process Gaps

**Code inspection happened post-speedup, not pre-sweep.** The Probe B code inspection was triggered by the anomalous measurement result, not by a mandatory pre-sweep sign-off gate. The rule was applied to mechanism claims in T1/T2/T3 but not enforced as a gate on probe kernel accumulation invariants before measurement.

**Standing rule added (S46-T6):** future probe specs MUST require a code-inspection sign-off on the probe kernel's accumulation invariant BEFORE the sweep runs. The sign-off must answer "how does each (bin, stat) pair reach the lane that owns that bin?" Any probe that cannot answer this question is blocked from the sweep.

---

## Cost

| Item | Budget | Actual |
|---|---|---|
| Duration | 1 week | 1 day |
| T5 outcome | Unknown (Commit/user-call/Halt) | Halt (Outcome C) |
| S47 | Engineering if T5=Commit | BLOCKED — user call |

---

## Outputs

- **Branch:** `mlx/sprint-46-simd-shuffle-research`
- **PR:** `mlx/sprint-46-simd-shuffle-research` → master (pending user review)
- **DEC-049:** RETIRED (2026-05-05)
- **v0.7.0:** BLOCKED — user call required (see HANDOFF.md §"BLOCKING DECISION FOR S47")
