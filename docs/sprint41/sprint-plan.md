# Sprint 41 Plan — Polish-to-Trust (E1)

**Sprint:** 41  |  **Status:** IN PROGRESS  |  **Branch:** `mlx/sprint-41-polish`
**Cut from:** master `aac00046a1` (post v0.5.0)
**Theme:** Close every paper-cut a v0.5.0 user could hit on the happy path. Make the
README the single document a new user needs. Stage Track B (upstream RFC) without posting.

## Strategic context

Following the Sprint 40 advisory-board synthesis (DEC-046 lane lock; visionary's "RS=0
deterministic moat" reframing), the project's **distribution path is PyPI + community
evangelism, not upstream PR**. Upstream submission is opportunistic, gated on multiple
trigger conditions (Ordered Boosting, ya.make availability, maintainer engagement, etc.)
that are 6-12+ sprints away.

The committee's E1 epic (the polish-to-trust phase) is what unblocks Track A. The next
user must be able to install, run a hello-world, and trust the contract written in the
README. That is what S41 ships.

Out of S41 scope (deliberate):
- **Ordered Boosting** (E2, 5 sprints — major capability work, hero feature for v0.6.0)
- **CTR RNG ordering closure** (deferred per "characterized > unimplemented" trade-off)
- **HN/Twitter/MLX-Slack launch** (E3 — wait for v0.6.0 with Ordered Boosting)
- **Posting** the upstream RFC (T5 stages it; posting is a separate decision)

## Items

| # | Description | Effort | Status |
|---|---|---|---|
| **T1** | `bootstrap_type` validator case-insensitivity (matches CatBoost-CPU) | 5 min | DONE |
| **T2** | README install + test path (canonical Installation & Quick Start, 30-second smoke test) | 1 sprint | DONE |
| **T3** | `predict()` subprocess profiling — locate the 41× slowdown source; document or fix | 30 min – 1 sprint | TBD |
| **T4** | PyPI publish-readiness audit — wheel build, install verification, no secrets | 1 sprint | TBD |
| **T5** | Refresh `docs/upstream_issue_draft.md` for post-S30 reality (DON'T post) | 1 sprint | TBD |
| **T6** | Sprint close-out + (optional) v0.5.1 patch tag | 0.5 sprint | TBD |

## Verification gates

- T1: existing `test_invalid_bootstrap_type` still passes; new `test_bootstrap_type_case_insensitive` covers `'No'`, `'NO'`, `'no'`, `'Bayesian'`, `'BAYESIAN'`, `'Bernoulli'`, `'MVS'`. Existing CLI passthrough at `core.py:802` still receives lowercase value.
- T2: README's 30-second smoke test must run end-to-end on a fresh install and print `OK`. Verified by author before commit.
- T3: profile output captured; decision documented either way (fix vs. doc).
- T4: `pip wheel python/` produces a clean wheel; install of that wheel into a fresh venv produces a working package; the v0.5.0 release tarball install path also verified.
- T5: refreshed draft references DEC-036/042/045/046 and v0.5.0; explicitly marked NOT-POSTED with the trigger conditions for posting.

## Files in scope

- `python/catboost_mlx/core.py` — T1 validator
- `python/tests/test_basic.py` — T1 test
- `catboost/mlx/README.md` — T2 install/test path
- `docs/sprint41/*` — sprint docs
- `docs/upstream_issue_draft.md` — T5 refresh
- `.claude/state/{HANDOFF,TODOS,LESSONS-LEARNED,DECISIONS,CHANGELOG-DEV}.md` — close-out

## Files explicitly NOT in scope

- `catboost/mlx/**.cpp,.metal,.h` — no kernel changes; no source-of-truth changes
- `catboost/cuda/`, `catboost/private/`, upstream `library/`, `util/`, `contrib/` — fork stays as-is per session decision (no unfork, no strip)

## Branch + PR plan

Single branch `mlx/sprint-41-polish`. Commits per item (T1, T2, T3+T4 if combined, T5, close-out). Single PR. Optional v0.5.1 tag after merge if T1 + T2 + at least one of T3/T4 lands cleanly.
