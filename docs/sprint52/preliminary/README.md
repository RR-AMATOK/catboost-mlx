# S52+ Preliminary Artifacts — Ordered Boosting

**Created:** 2026-05-18 (S50 kickoff)
**Status:** PRESERVED FOR FUTURE USE — S52+ ordered-boosting kickoff will inherit these

## Why these are here

Per DEC-053, the S50 pivot was re-sequenced: **categorical handling closure first** (S50), **ordered boosting deferred to S52+**. The 3-agent panel work that produced detailed ordered-boosting plans on 2026-05-18 is preserved here so the future S52 sprint can pick up directly without re-running the panel.

## Contents

- `ml-product-owner-sprint-plan.md` — ~330-line S50/S52 scoping plan from @ml-product-owner: T0-T6 structure, parity contract options (G1: bit-exact / bounded / algorithmic), feature-flag strategy (opt-in `boosting_type='Ordered'`), risk register, multi-sprint roadmap (S52-S54)
- `research-scientist-deep-dive.md` — Algorithmic deep-dive from @research-scientist: Prokhorenkova 2018 mechanism; CPU CatBoost BodyTail batching (NOT N supplementary models — ~13 doubling batches at 1M scale; `fold.cpp:156-198`); MLX integration points (`mlx_boosting.h:41-60` TBoostingConfig needs BoostingType+PermutationCount; per-permutation cursors; tail-only derivatives at `tensor_search_helpers.cpp:451-455`); memory cost ~400-500 MB overhead; parity recommendation: bounded envelope ±0.005 logloss
- `devils-advocate-stress-test.md` — Pivot stress-test from @devils-advocate: failure-mode probabilities (P=0.35 quality failure on small-N regime we don't benchmark; P=0.45 scope blow-out); R1-R5 pre-commit rails (scope, quality, throughput, time, DEC-046 dependency); strategic coherence concerns

## When to use

When S50 completes (categorical handling closure verified; Amazon valid in Bundle 2 or documented why not):
- Spawn S52 sprint as `mlx/sprint-52-ordered-boosting-kickoff`
- Use `ml-product-owner-sprint-plan.md` as the base for `docs/sprint52/scoping.md`
- Use `research-scientist-deep-dive.md` as the algorithm reference for T0/T1/T3
- Use `devils-advocate-stress-test.md` to lock R1-R5 pre-commit rails at T0c
- Reconsider G1 parity envelope choice based on S50 lessons about MLX/CPU encoding consistency

## Important caveats from devils-advocate

1. **Ordered boosting does NOT unblock PyPI publish per DEC-051** (throughput-gated; ordered is intrinsically 2-4× slower than plain).
2. **Showcase dataset mismatch:** ordered shines on N<10k; existing benchmark suite is N=1M.
3. **Scope realism:** "2-3 sprints" estimate may stretch to 3-5 per S22-S24 over-budget precedent.
4. **R5: DEC-046 dependency** — categoricals require BOTH bin-aliasing fix (S50) AND CTR RNG ordering work; numerical-only at first S52 lock recommended.

## Files

All three preserved verbatim from 2026-05-18 panel review. Don't edit; treat as historical inputs.
