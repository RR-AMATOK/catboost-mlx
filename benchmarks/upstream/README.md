# Upstream benchmark suite

Adapters and runners that reproduce the [`catboost/benchmarks`](https://github.com/catboost/benchmarks) suite against catboost-mlx (and the three reference GBDT libraries: LightGBM, XGBoost, CatBoost-CPU) on Apple Silicon.

## Goal

Produce a Pareto frontier (test metric × wall-clock × peak RSS) on the upstream-canonical datasets — the same datasets the GBDT community evaluates CatBoost on — to enable head-to-head comparison without comparing against published numbers from different hardware.

## Honest-publishing constraints (Sprint 42 plan)

1. **Same M-series machine for all 4 frameworks.** No comparison vs upstream's published x86/CUDA numbers.
2. **Include Amazon (where the DEC-046 CTR rare-class characterized gap is visible) with a footnote**, do not cherry-pick.
3. **Show our depth-6 against their depth-6**, never against their tuned depth-8/10.
4. **Re-frame `gpu_vs_cpu_training_speed/`** as "MLX-GPU vs CatBoost-CPU on M-series" — different question than upstream's CUDA-vs-CPU.

## Layout

```
adapters/    Per-dataset preprocessing scripts (raw -> CSV + cat_features config)
scripts/     Per-framework runners + driver script
results/     Per-run JSON output (machine-readable)
```

## Status

Scaffolded in Sprint 42 (2026-04-26). See `docs/sprint42/sprint-plan.md` for the
detailed plan and target dataset subset.
