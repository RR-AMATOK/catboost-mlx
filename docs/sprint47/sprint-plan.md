# Sprint 47 Plan — v0.7.0 Release Engineering

**Sprint:** 47  |  **Status:** ACTIVE — T0 complete, T1 next  |  **Branch:** `mlx/sprint-47-release-0.7.0`
**Cut from:** master at commit `7a97db638f` (post S46 PR #47 merge)
**Theme:** Release engineering. Ship v0.7.0 to PyPI as a reproducibility-grade release.
**Mode:** RELEASE ENGINEERING. No kernel changes, no probe code, no new benchmarks, no throughput research.
**Duration target:** 1 week (2–3 calendar days of agent work).
**Scope locked:** Per DEC-050. Any kernel or probe change in this sprint is out of scope.

---

## Strategic context

S43 locked Branch B: CatBoost-MLX is a deterministic, bit-equivalent Apple Silicon-native CatBoost-Plain port. S44 (DEC-047) confirmed that at fair convergence CatBoost-MLX is statistically indistinguishable from CatBoost-CPU on numeric workloads — establishing the "reproducibility-grade" frame. S45 shipped the three platform-parity artifacts that make that claim defensible: the Branch-B regression CI gate (T1), the cross-class CUDA bit-equivalence writeup (T4), and the `catboost-tripoint` parity oracle (T5). S46 ran the final bounded throughput research arc: seven falsifications (DEC-013, 014, 015, 017, 019, 048, 049) have now exhausted the histogram-kernel-restructure design space under the existing topology and the DEC-008 parity envelope.

DEC-050 (decided 2026-05-05 by user) resolves the v0.7.0 blocking question. The decision is Option α: v0.7.0 ships as a **reproducibility-grade release**. No throughput delta vs v0.6.x is required. The key argument for α over β (try another lever) is that β would be the eighth throughput attempt on the same kernel topology without a structurally new premise. The key argument for α over γ ("relax the gate") is that α and γ ship the same artifact — α names it honestly. "Reproducibility-grade by design" is a positive product claim grounded in four shipped artifacts: Branch-B CI gate, cross-class CUDA writeup, `catboost-tripoint` oracle, and the full DEC-049 falsification record. γ presents the same artifact as a concession.

Throughput defers to v0.8.0, conditioned on a structurally NEW lever class (bin-distributed dispatch, sort-then-scan with race-free atomics, or a fundamentally different kernel topology). Any v0.8.0 arc must publish a fresh f_hist analysis at v0.7.0's measured baseline before any probe is greenlit.

S47 executes the release. No scope ambiguity is permitted.

---

## Tasks

| # | Task | Owner | Inputs | Outputs | Acceptance | Est. |
|---|------|-------|--------|---------|-----------|------|
| **T0** | Scaffold | @technical-writer | DEC-050, sprint brief | `docs/sprint47/sprint-plan.md`; HANDOFF.md S47 block updated | Plan committed; no advisory board required | 30 min |
| **T1** | Version bump | @ml-engineer | `python/catboost_mlx/__init__.py`, `python/setup.py`, any other `__version__` anchors | All version anchors → `0.7.0`; `pip install -e .` smoke import passes | `python -c "import catboost_mlx; print(catboost_mlx.__version__)"` prints `0.7.0` | 45 min |
| **T2** | User CHANGELOG | @technical-writer | `CHANGELOG.md`, `CHANGELOG-DEV.md` (S43–S46 entries), DEC-047/050 | `## [0.7.0]` section in `CHANGELOG.md` | Cites Branch-B regression test + tripoint + cross-class CUDA bit-equivalence; honest deferred-throughput statement; no hedging | 2 hr |
| **T3** | README posture | @technical-writer | `README.md` current Status block | Updated Status section + cross-links to DEC-047/050 + NOT-upstream-catboost disambiguation banner | "Reproducibility-grade by design" framing is explicit and grounded in evidence; disambiguation banner prevents catboost-package confusion | 1 hr |
| **T4** | Release validation | @qa-engineer | Branch-B regression test suite, `catboost-tripoint`, locally-built wheel | `docs/sprint47/T4/release-validation.md` | Branch-B GREEN; `catboost-tripoint verify` PASS on Higgs-1M + Epsilon; clean-env `pip install` + round-trip predict works | 2 hr |
| **T5** | PyPI publish | @mlops-engineer | T4 wheel, `~/.pypirc`, TestPyPI credentials | TestPyPI smoke → PyPI prod → GitHub Release `v0.7.0` | `pip install catboost-mlx==0.7.0` works in a clean conda env; GitHub Release published with reproducibility-grade release notes | 2 hr |
| **T6** | Close-out | @ml-product-owner + @technical-writer | All above | DEC-050 → IMPLEMENTED; HANDOFF/TODOS/DECISIONS/CHANGELOG-DEV updated; PR `mlx/sprint-47-release-0.7.0` → master squash-merged; `v0.7.0` git tag pushed | PR green, tag on master, PyPI live | 1 hr |

**Total estimated agent work:** ~9.5 hours over 2–3 calendar days. 1-week budget.

---

## Sequential gates

```
T0 (scaffold — DONE)
  └─▶  T1 (version bump)
         └─▶  T2 (CHANGELOG)  ─┐
              T3 (README)      ├─▶  T5 (PyPI publish)
              T4 (validation)  ─┘         └─▶  T6 (close-out)
```

T2, T3, and T4 are independent of each other and can run in parallel after T1. T5 requires all three. T6 requires T5.

---

## Risk pre-mortem

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| **PyPI namespace collision** — `catboost-mlx` already registered by an unrelated project | Low–Med | High | Search PyPI before wheel build: `pip index versions catboost-mlx`. If taken, fall back to `catboost-mlx-apple` (register in T1); update README + CHANGELOG accordingly. |
| **Confusion with upstream `catboost` package** — users `pip install catboost` expecting this | High | Med | T3 README adds an explicit disambiguation banner: "This is NOT the upstream catboost package. Install upstream CatBoost via `pip install catboost`." Cross-link to `catboost/catboost` upstream. |
| **TestPyPI → PyPI version trap** — 0.7.0 burned on TestPyPI, PyPI rejects it | Med | Med | T5 tests on TestPyPI with a `.devN` suffix (e.g., `0.7.0.dev0`) before the final PyPI push. If the real `0.7.0` is somehow pre-consumed, fall back to `0.7.0.post1` on PyPI and update version anchors. |
| **Wheel build fails on `macos-arm64`** — packaging toolchain issue (sdist vs wheel, Metal headers missing) | Med | Med | T4 builds the wheel locally first using `python -m build --wheel`. TestPyPI upload catches platform-tag surprises before the prod push. |
| **Branch-B regression baseline drift** — `v0.6.1_predict_baselines.pkl` was generated on a different machine / MLX version | Low | High | T4 runs the Branch-B regression test (`python/tests/regression/test_branch_b_regression.py`) first, before any other validation. If it drifts, block T5 until the cause is isolated. Do not re-baseline silently. |
| **`__version__` anchor missed in submodule or secondary path** — import returns `0.6.x` after version bump | Med | Low | T1 includes a grep audit before declaring done: `grep -r "__version__\|version=" python/catboost_mlx/ python/setup.py`. Every hit reviewed manually. Smoke test `catboost_mlx.__version__ == "0.7.0"` is a hard acceptance criterion. |

---

## Out of scope (explicit)

The following are **not** v0.7.0 blockers and are explicitly excluded from S47:

- Kernel changes of any kind (`kernel_sources.h`, `histogram.cpp`, `structure_searcher.cpp`)
- Probe code or ifdef-gated experimental variants
- New benchmark datasets or benchmark runs beyond release validation (Higgs-1M + Epsilon smoke)
- β-lever exploration (structurally new throughput levers) — per DEC-050, this is v0.8.0+ scope requiring a fresh f_hist analysis
- MSLR-WEB10K sweep — deferred from S44, remains deferred
- csv_train.cpp uint8 cat-bin aliasing fix (DEC-047 scope note) — v0.6.x, not v0.7.0
- Ordered Boosting implementation — v0.7.x follow-up, not a release blocker
- Upstream RFC posting (`docs/upstream_issue_draft.md`) — STAGED, not posted; timing is a separate user decision
- Multi-platform wheel builds (Linux x86_64, Windows) — this release is macOS arm64 only

---

## Two follow-up items deferred to S48

These are real gaps but not v0.7.0 blockers. Document them here so they do not get lost and are not reopened as v0.7.0 scope creep.

**S48-FU-1: Perf-gate robustness.** The CI perf-gate (`.github/workflows/mlx-perf-regression.yaml`) was redesigned in S42-T4 to compare CPU/MLX speedup ratios rather than absolute wall-clock. The histogram-stage gate retains `continue-on-error: true` pending a redesign (noted in S42 close-out). Post-v0.7.0, this should be addressed before any throughput work (v0.8.0 arc) could accidentally regress v0.7.0's ratio baseline. Cross-ref: S46-T4 CI investigation context.

**S48-FU-2: `stage_profile` API drift.** The `CATBOOST_MLX_STAGE_PROFILE` instrumentation used in S45-T2 and S46-T4 has drifted from the documented interface (observed during S46 build-env debugging). Not a user-facing regression, but it will block any future probe-D measurement if not fixed. Cross-ref: `docs/sprint46/T4/build-env/status.md`.

---

## Cross-references

| Pointer | Why it matters for S47 |
|---------|----------------------|
| DEC-046 (`docs/decisions.md`, Sprint 40) | Lane B lock — "characterized-difference port" framing that S47-T3 supersedes with "reproducibility-grade" |
| DEC-047 (Sprint 44) | v0.6.0 reproducibility-grade framing; Axis C verdict. S47-T2 and T3 build on this. |
| DEC-050 (Sprint 46) | The authority for S47's scope. α vs β vs γ decision. T6 marks it IMPLEMENTED. |
| S45-T1 (`python/tests/regression/test_branch_b_regression.py`) | Branch-B CI gate — must be GREEN throughout S47; T4 runs it explicitly |
| S45-T4 (`docs/benchmarks/cross-class-cuda-comparison.md`) | Cross-class CUDA bit-equivalence writeup — cited in T2 CHANGELOG |
| S45-T5 (`tools/catboost_tripoint/`) | `catboost-tripoint` parity oracle — smoke-tested in T4; cited in T2 CHANGELOG |
| v0.5.3 release pattern (commit `7d7d034ff1`) | The most recent PyPI release; T5 mirrors this workflow |
| `docs/sprint46/T6/summary.md` | S46 close-out narrative and what S47 inherits |

---

## Definition of Done

S47 closes successfully when ALL of the following are true:

1. `pip install catboost-mlx==0.7.0` succeeds in a clean conda environment.
2. `catboost-tripoint verify --model X.cbm --data Y.parquet` passes on Higgs-1M + Epsilon with v0.7.0 installed.
3. `python/tests/regression/test_branch_b_regression.py` GREEN (byte-identical predict output vs v0.6.1 baselines).
4. `CHANGELOG.md` has a `## [0.7.0]` entry with reproducibility-grade framing, cites Branch-B + tripoint + cross-class CUDA, and documents deferred throughput honestly.
5. `README.md` Status section says "reproducibility-grade by design" with cross-links and a disambiguation banner.
6. GitHub Release `v0.7.0` exists on `RR-AMATOK/catboost-mlx` with release notes.
7. `v0.7.0` git tag on master.
8. DEC-050 status updated to IMPLEMENTED in `.claude/state/DECISIONS.md`.
9. `HANDOFF.md` + `TODOS.md` + `CHANGELOG-DEV.md` reflect v0.7.0 IMPLEMENTED state.
10. Single PR `mlx/sprint-47-release-0.7.0` → master squash-merged.

---

## Files in scope vs explicitly NOT in scope

**In scope (S47):**
- `docs/sprint47/sprint-plan.md` (this file)
- `docs/sprint47/T4/release-validation.md`
- `python/catboost_mlx/__init__.py` (version bump only)
- `python/setup.py` (version bump only)
- `CHANGELOG.md` (new `## [0.7.0]` section)
- `README.md` (Status section update + disambiguation banner)
- `.claude/state/{HANDOFF,TODOS,CHANGELOG-DEV,DECISIONS}.md` (T6)

**Explicitly NOT in scope:**
- `catboost/mlx/kernels/kernel_sources.h`
- `catboost/mlx/methods/histogram.cpp`
- `catboost/mlx/methods/structure_searcher.cpp`
- Any file under `catboost/mlx/` except documentation
- `benchmarks/` (no new benchmark runs)
- `docs/sprint47/` beyond the plan and T4 validation doc
- `Frameworks/LESSONS-LEARNED.md` (unless T5 surfaces a surprising failure worth recording)

---

## Branch + PR plan

- Single branch `mlx/sprint-47-release-0.7.0` (already cut from master at `7a97db638f`).
- T0–T6 commit atomically per task per the project commit convention: `[mlx] sprint-47: T<N> — <description>`.
- Single PR after T6 lands, squash-merged to master.
- Push target: `origin` (`RR-AMATOK/catboost-mlx`) only. **Never push upstream** (`catboost/catboost`). Per DEC-004.
- `v0.7.0` git tag pushed to `origin` after master merge, by @mlops-engineer in T5 or @ml-product-owner in T6.

---

## Agent panel

**Writing:** @technical-writer (T0, T2, T3, T6 docs)
**Implementation:** @ml-engineer (T1 version bump)
**Quality:** @qa-engineer (T4 release validation)
**Infra:** @mlops-engineer (T5 PyPI publish + wheel build)
**Coordination:** @ml-product-owner (T6 close-out)

**NOT needed in S47:** @performance-engineer, @silicon-architect, @mathematician, @research-scientist, @strategist, @devils-advocate, @hardware-researcher (no kernel work, no throughput decisions, no new research).
