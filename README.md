<img src=http://storage.mds.yandex.net/get-devtools-opensource/250854/catboost-logo.png width=300/>

> **This is NOT the upstream `catboost` package.** This is `catboost-mlx`, an
> independent port of CatBoost's gradient boosting decision trees to Apple Silicon
> GPU via Apple's MLX/Metal framework. For the upstream Python `catboost` package
> (CPU + CUDA, multi-platform), see [catboost.ai](https://catboost.ai) or
> `pip install catboost`. This package: Apple Silicon only, MLX/Metal backend,
> focused on bit-exact parity with upstream CatBoost on supported losses (RMSE,
> Logloss, MultiClass).

---

## CatBoost-MLX: GPU Acceleration on Apple Silicon

This fork adds a native Apple Silicon GPU backend using [MLX](https://github.com/ml-explore/mlx) and Metal compute shaders. Train gradient boosted decision trees directly on your Mac's GPU — no CUDA required.

- **Full Python API**: `CatBoostMLXRegressor`, `CatBoostMLXClassifier` with scikit-learn compatibility
- **4 Metal compute kernels**: histogram, score splits, leaf estimation, tree application
- **10 loss functions**: RMSE, Logloss, MultiClass, MAE, Quantile, Huber, Poisson, Tweedie, MAPE, PairLogit
- **684 tests passing** across Python 3.9–3.13

**Requirements**: Apple Silicon Mac (M1/M2/M3/M4), macOS 14+, MLX (`brew install mlx`)

> [Python quickstart and API reference](python/README.md) | [C++ architecture and CLI](catboost/mlx/README.md) | [Benchmarks](python/benchmarks/README.md)

---

## Status

**v0.7.0 — Reproducibility-grade release. Source-install only; no PyPI publish yet.**

PyPI publish is intentionally gated on closing the MLX-vs-CUDA throughput gap (see
[Throughput posture](#throughput-posture) below and DEC-051 in
[`.claude/state/DECISIONS.md`](.claude/state/DECISIONS.md)). Install from source:

```bash
git clone https://github.com/RR-AMATOK/catboost-mlx.git
cd catboost-mlx
pip install -e python/
```

### What "reproducibility-grade" means

- **Bit-equivalence with v0.6.1 predict output is enforced by CI.** The Branch-B
  regression test (`python/tests/regression/test_branch_b_regression.py`) compares
  predict output against committed baselines byte-for-byte on every push.
- **Three-platform bit-equivalence at fair convergence.** On all-numeric workloads,
  Mac CPU (CatBoost 1.2.10), Mac MLX (this package), and Windows CUDA (RTX 5070 Ti,
  CatBoost 1.2.10) produce logloss values within ≤0.0003 of each other. Full results
  in [`docs/benchmarks/cross-class-cuda-comparison.md`](docs/benchmarks/cross-class-cuda-comparison.md).
- **Parity oracle.** The `catboost-tripoint` CLI
  (`tools/catboost_tripoint/`) compares CPU/MLX/CUDA predict outputs against the
  derived fp32 floor and emits a signed JSON report. Exit code 0 = PASS; exit code
  1 = at least one pairwise diff exceeds the floor.
- **5-dataset Pareto benchmark suite.** Adult, Higgs-1M, Higgs-11M, Epsilon, and
  Amazon results (with full reproducibility receipts) are in
  [`docs/benchmarks/v0.6.0-pareto.md`](docs/benchmarks/v0.6.0-pareto.md).

### Throughput posture

The MLX path matches CatBoost-CPU on correctness but does not yet exceed it on
wall-clock. Training is 5–16× slower than CatBoost-CPU on the same M3 Max chip
across the five measured datasets; the gap is structural compute-throughput, not
launch overhead (falsified at 11M-row scale). Seven throughput-lever research arcs
were executed and falsified across Sprints 13–46 (DEC-013/014/015/017/019/048/049 in
[`.claude/state/DECISIONS.md`](.claude/state/DECISIONS.md)). Throughput improvement
is deferred to v0.8.0, conditioned on a structurally new lever class and a fresh
kernel-efficiency analysis. The engineering rationale is documented in DEC-049
OUTCOME and DEC-050.

---

## Install

```bash
git clone https://github.com/RR-AMATOK/catboost-mlx.git
cd catboost-mlx
pip install -e python/
```

PyPI publish is deferred (see [Status](#status) and DEC-051). When the MLX path
reaches CUDA-class throughput, the package will publish as `catboost-mlx` on PyPI
(name reservation verified 2026-05-06).

---

CatBoost is a machine learning method based on [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting) over decision trees.

Main advantages of CatBoost:
--------------
  - Superior quality [compared](https://github.com/catboost/benchmarks/blob/master/README.md) with other GBDT libraries on many datasets.
  - Best-in-class [prediction](https://catboost.ai/docs/concepts/c-plus-plus-api.html) speed.
  - Support for both [numerical and categorical](https://catboost.ai/docs/concepts/algorithm-main-stages.html) features.
  - Fast GPU and multi-GPU support for out-of-the box training.
  - Built-in [visualization tools](https://catboost.ai/docs/features/visualization.html).
  - Fast and reproducible distributed training with [Apache Spark](https://catboost.ai/en/docs/concepts/spark-overview) and [CLI](https://catboost.ai/en/docs/concepts/cli-distributed-learning).

Get Started and Documentation
--------------
All CatBoost documentation is available [here](https://catboost.ai/docs/).

For upstream CatBoost installation guides (CPU + CUDA, multi-platform):
 * [Python package](https://catboost.ai/en/docs/concepts/python-installation)
 * [R-package](https://catboost.ai/en/docs/concepts/r-installation)
 * [Command line](https://catboost.ai/en/docs/concepts/cli-installation)
 * [Package for Apache Spark](https://catboost.ai/en/docs/concepts/spark-installation)

Next you may want to explore:
* [Tutorials](https://github.com/catboost/tutorials/#readme)
* [Training modes and metrics](https://catboost.ai/docs/concepts/loss-functions.html)
* [Cross-validation](https://catboost.ai/docs/features/cross-validation.html#cross-validation)
* [Parameters tuning](https://catboost.ai/docs/concepts/parameter-tuning.html)
* [Feature importance calculation](https://catboost.ai/docs/features/feature-importances-calculation.html)
* [Regular](https://catboost.ai/docs/features/prediction.html#prediction) and [staged](https://catboost.ai/docs/features/staged-prediction.html#staged-prediction) predictions
* CatBoost for Apache Spark videos: [Introduction](https://youtu.be/47-mAVms-b8) and [Architecture](https://youtu.be/nrGt5VKZpzc)

If you cannot open documentation in your browser try adding yastatic.net and yastat.net to the list of allowed domains in your Privacy Badger.

CatBoost models in production
--------------
If you want to evaluate CatBoost model in your application read [model api documentation](https://github.com/catboost/catboost/tree/master/catboost/CatboostModelAPI.md).

Questions and bug reports
--------------
* For reporting bugs please use the [catboost/bugreport](https://github.com/catboost/catboost/issues) page.
* Ask a question on [CatBoost GitHub Discussions Q&A forum](https://github.com/catboost/catboost/discussions/categories/q-a).
* Ask a question on [Stack Overflow](https://stackoverflow.com/questions/tagged/catboost) with the catboost tag, we monitor this for new questions.
* Seek prompt advice at [Telegram group](https://t.me/catboost_en) or Russian-speaking [Telegram chat](https://t.me/catboost_ru)

Help to Make CatBoost Better
----------------------------
* Check out [open problems](https://github.com/catboost/catboost/blob/master/open_problems/open_problems.md) and [help wanted issues](https://github.com/catboost/catboost/labels/help%20wanted) to see what can be improved, or open an issue if you want something.
* Add your stories and experience to [Awesome CatBoost](AWESOME.md).
* [Instructions for contributors](https://github.com/catboost/catboost/blob/master/CONTRIBUTING.md).

News
--------------
Latest news are published on [twitter](https://twitter.com/catboostml).

Reference Paper
-------
Anna Veronika Dorogush, Andrey Gulin, Gleb Gusev, Nikita Kazeev, Liudmila Ostroumova Prokhorenkova, Aleksandr Vorobev ["Fighting biases with dynamic boosting"](https://arxiv.org/abs/1706.09516). arXiv:1706.09516, 2017.

Anna Veronika Dorogush, Vasily Ershov, Andrey Gulin ["CatBoost: gradient boosting with categorical features support"](http://learningsys.org/nips17/assets/papers/paper_11.pdf). Workshop on ML Systems
at NIPS 2017.

License
-------
© YANDEX LLC, 2017-2026. Licensed under the Apache License, Version 2.0. See LICENSE file for more details.
