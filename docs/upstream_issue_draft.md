# Draft: Informational Discussion Issue for catboost/catboost

> **Status:** STAGED — NOT YET POSTED. Awaiting completion of the trigger conditions
> recorded in `.claude/state/DECISIONS.md` § DEC-046 ("Trigger conditions before
> attempting an upstream PR"). This draft refreshed 2026-04-26 (Sprint 41) for
> post-S30/S40 reality.
>
> **Purpose**: this is a *discussion / FYI issue*, **not a pull request**. Per
> `CONTRIBUTING.md`, non-trivial contributions require a discussion issue first;
> this is that. We are not asking the upstream team to merge anything.
>
> **Target**: https://github.com/catboost/catboost/issues/new
>
> **History**: an earlier version of this draft was prepared in Sprint 15
> (commit `74f2ba63d4`, 2026-04-12) but never posted. The 2026-04-26 refresh
> aligns with v0.5.0 release content, the DEC-036/042/045/046 closures, and the
> "characterized-difference port" framing established in DEC-046.

---

## Title

[Discussion] MLX (Apple Silicon GPU) training backend — v0.5.0 ships as a fork; coordination-of-interest check

## Body

### TL;DR

We have shipped a working **MLX/Metal GPU training backend** for CatBoost on Apple
Silicon as a public fork (`RR-AMATOK/catboost-mlx`, v0.5.0, 2026-04-26). It is *not* a
byte-faithful port of CatBoost-CPU/CUDA — see "Honest characterization" below. We are
not requesting an upstream merge at this time. This issue is to **flag the work's
existence** and ask whether the upstream team has any interest in coordination,
ranging from "no thanks, please continue as a fork" to "let's talk about what
upstream-readiness would look like."

**Repository**: https://github.com/RR-AMATOK/catboost-mlx
**Release**: v0.5.0 (2026-04-26) — first public Release on the fork

### What it does

- Gradient boosted decision tree training on Apple Silicon GPU using Apple's MLX
  framework and Metal compute shaders. Replaces the CUDA backend with a Metal
  backend; not co-installed with CUDA (mutually exclusive at build time, safe
  because CUDA is unavailable on darwin-arm64).
- 12 loss functions (RMSE, MAE, Quantile, Huber, Poisson, Tweedie, MAPE, Logloss,
  CrossEntropy, MultiClass, PairLogit, YetiRank).
- 3 grow policies (SymmetricTree, Depthwise, Lossguide) — note that the upstream
  CUDA backend ships SymmetricTree only; Depthwise/Lossguide on GPU is new with this
  backend.
- All 3 grow policies support both `score_function=L2` and `score_function=Cosine`
  (closed in this fork's Sprint 33; see DEC-036 / DEC-042 in the fork's
  `.claude/state/DECISIONS.md` for the partition-state mechanism that took several
  sprints to root-cause).
- Categorical features with online (Ordered) CTR target encoding.
- Custom Metal kernels: histogram, split scoring, leaf accumulation, tree applier.
- Python bindings via nanobind (`_core.cpython-*.so`) — zero-copy numpy, GIL released
  during training.
- Standalone CLI binaries (`csv_train`, `csv_predict`).
- ONNX + CoreML model export.
- 1000+ pytest tests; CI on macos-14 (Apple Silicon M1) GitHub-hosted runners.

### What it does NOT do (honest characterization, see DEC-046)

This is the most important section of this draft. **The fork is positioned as a
"characterized-difference Apple Silicon CatBoost-Plain port", not a byte-faithful
replacement.**

Specifically:

- **`boosting_type='Ordered'` is not implemented.** Only `boosting_type='Plain'` is
  supported. Ordered Boosting is CatBoost's signature anti-leakage feature; we
  document its absence prominently in the README. This is the single largest
  feature-completeness gap.
- **`max_depth ≤ 6`** (kernel constraint — `MAX_LEAVES=64` private storage cap).
- **16M-row dataset ceiling** (`ComputePartitionLayout` int32 accumulator). Sufficient
  for most workloads; explicitly rejected above this.
- **`NewtonL2` and `NewtonCosine` score functions explicitly rejected** at the Python
  API; only `L2` and `Cosine` ship.
- **CTR rare-class behavior differs slightly** from upstream by a quantified amount.
  On the Kaggle Irrigation Need dataset (270k × 53 features, 8 categoricals,
  multiclass with rare class at 3.18%) at matched `RandomStrength=0` +
  `bootstrap_type=No`, the fork agrees with CatBoost-CPU on **99.917% of predictions**
  with a 0.28pp balanced-accuracy gap. The gap is decomposed in DEC-046 as 39% pure
  CPU seed-noise floor + 24% MLX architectural floor + **37% attributable to a
  specific identified mechanism** (CTR RNG ordering at `csv_train.cpp:2196-2206`).
  Numeric-only workloads (`cat_features=[]`) converge to within architectural floor:
  99.948% prediction agreement, mean absolute probability difference 2.2e-3, no
  rare-class skew.

We claim "characterized-difference port", which means: every divergence from upstream
semantics is empirically measured, decomposed, bounded, and documented in the README.
We do not claim byte-faithful CatBoost compatibility, and we are explicit with our own
users about which workloads converge to within architectural floor (numeric-only) vs
which carry a documented bounded gap (categorical-heavy with rare classes).

### Synthetic-anchor parity (where we are bit-identical)

At `RandomStrength=0` + `bootstrap_type=No` on synthetic anchors (the standard test
configuration), the fork agrees with CatBoost-CPU **bit-identically** at fp32 precision
on every test we have constructed:

- N=1k seed=42 LG+Cosine, 50 iters: feature- and border-aligned splits, identical
  RMSE
- 12/12 iter=1 + iter=2 splits at the F2 anchor: bit-identical (feature + border)
- DEC-045 closed the prior "small-N drift" investigation as a harness configuration
  artifact (`csv_train.cpp:599` `RandomStrength = 1.0f` default vs CPU's `0.0`); with
  matched RS=0 the drift collapses to zero

Default `RandomStrength=1.0` produces a bounded RNG-implementation bias of mean
−4.08% (95% CI [−4.78%, −3.39%]) at the canonical small-N anchor. This is documented
in the README as a known and *bounded* difference; it is not a correctness issue.

### What we would like to discuss (if anything)

We are explicitly not requesting an upstream merge. The fork is shipping
independently, with PyPI as the planned primary distribution channel. That said:

1. **Coordination of interest**: is upstream open to any form of coordination, even
   loose? E.g., a `awesome-catboost` listing, a link from upstream README to the fork
   for Apple Silicon users, a shared issue tracker for cross-runtime questions.
2. **Build system**: we recognize that any path toward an upstream-merged backend
   would require `ya.make` integration that we cannot author from outside the
   monorepo (per the warning in `CMakeLists.darwin-arm64.txt:1-13`). Would the team
   ever consider authoring `ya.make` for an externally-developed backend, or is that
   fundamentally out of scope?
3. **`IModelTrainer` interface stability**: we currently implement `IModelTrainer`
   and register via `TTrainerFactory` under `ETaskType::GPU`. Is this interface
   considered stable / guaranteed across upstream releases, so that a future
   coordination scenario would not be invalidated by an interface change?
4. **CI plausibility**: any path to upstream-merged would require Apple Silicon CI
   on upstream's side. We currently use GitHub-hosted macos-14 runners (Apple M1).
   Is this a hard blocker for upstream maintenance burden, or is there precedent
   for self-hosted contributor-provided runners?

We are happy to receive **"thanks, please continue as a fork"** as the answer. The
purpose of this issue is mostly to flag the work's existence and avoid surprising
the team if it gets traction in the Apple Silicon ML community.

### Trigger conditions for any future PR submission

For visibility, our project's standing decision (DEC-046) lists five trigger
conditions all of which must hold before we would attempt an actual upstream PR:

1. Ordered Boosting shipped (currently not implemented)
2. Numeric-path byte-identity (already ships) AND categorical gap < 99.99% closed or
   CI-bounded (currently 99.917% on the irrigation reference workload — bounded but
   not closed)
3. CMake option fully self-contained — zero impact when MLX off (largely true today;
   verified)
4. MLX test suite green in upstream's CI matrix on a borrowed M-series runner (not
   yet attempted)
5. ≥3 external PyPI users who would co-sign the PR (PyPI publish not yet executed)

These are not negotiation positions; they are *our* internal gates for not wasting
the upstream team's review time. Until they hold, an actual PR would not be
appropriate.

### Files

```
catboost/mlx/                         # ~1.1 MB / 42 source files
  kernels/                            # Metal compute shaders + JIT sources
  gpu_data/                           # GPU data layout, transfer, dataset builder
  methods/                            # Tree search (histogram, scoring, boosting)
  targets/                            # Loss functions (pointwise + pairwise)
  train_lib/                          # IModelTrainer + non-symmetric tree export
  tests/                              # csv_train.cpp, csv_predict.cpp, smoke tests

python/                               # nanobind bindings + sklearn-shaped API
  catboost_mlx/                       # _core.so + Python facade
  tests/                              # 1000+ pytest tests

.github/workflows/                    # mlx-build.yaml, mlx-test.yaml,
                                      # mlx-perf-regression.yaml
benchmarks/                           # MLX vs CPU harness + 18-config baseline
docs/sprint{16-40}/                   # decision history (PROBE-A through PROBE-Q,
                                      # DEC-005 through DEC-046)
```

All code Apache 2.0. No modifications to existing CatBoost source files outside
`catboost/mlx/`. The fork's root `CMakeLists.txt` adds 7 lines (Python/Cython
opt-out for MLX-only builds); no other upstream files are touched.

### Performance (honest, current)

We are not making aggressive performance claims at this time. End-to-end wall-clock
on the canonical synthetic 50k×50 RMSE anchor (S16-baseline configuration, depth 6,
100 iters) on M3 Max:

- CatBoost-CPU: well-optimized; representative of "should-be-fast" baseline
- CatBoost-MLX: comparable order-of-magnitude on numeric-only workloads at 50k+;
  documented `predict()` subprocess slowdown on categorical workloads (~8× at 50k×12,
  scaling with feature count — see README § "Python API uses subprocess")

We are deliberately NOT submitting head-to-head benchmark tables in this informational
issue. A dedicated benchmark sprint is planned to produce upstream-comparable numbers
on the standard `catboost/benchmarks` datasets (Higgs, Epsilon, MSLR-WEB10K,
Yahoo-LTR, Amazon — defensible-runnable subset given our constraints). That work
will produce a separate `docs/benchmarks/v0.5.x-pareto.md` artifact.

---

## Issue template fields (per ISSUE_TEMPLATE.md)

**Problem:**
Discussion issue. The "problem" is "no GPU training support on Apple Silicon in
upstream"; we describe a working solution that ships independently, and ask whether
there is any interest in coordination.

**catboost version:**
Fork based on current master at the time of v0.5.0 release (synced occasionally; not
continuously rebased).

**Operating System:**
macOS 14+ (Sonoma and later).

**CPU:**
Apple Silicon (M1/M2/M3/M4 family).

**GPU:**
Apple Silicon integrated Metal GPU.

---

> **Reminder to self before posting**: this is an *informational* issue. Do NOT
> attach a PR. Do NOT request review. Tone: neutral, low-commitment, providing
> information and asking only whether the team has any interest in any form of
> coordination.
