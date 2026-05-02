# Draft: Informational Discussion Issue for catboost/catboost

> **Status:** STAGED — NOT YET POSTED. Awaiting completion of the trigger conditions
> recorded in `.claude/state/DECISIONS.md` § DEC-046 ("Trigger conditions before
> attempting an upstream PR"). This draft refreshed 2026-04-30 (Sprint 44) for
> v0.6.0 reality — full 5-dataset benchmark suite landed; "reproducibility-grade
> CatBoost on MLX" framing locked.
>
> **Purpose**: this is a *discussion / FYI issue*, **not a pull request**. Per
> `CONTRIBUTING.md`, non-trivial contributions require a discussion issue first;
> this is that. We are not asking the upstream team to merge anything.
>
> **Target**: https://github.com/catboost/catboost/issues/new
>
> **History**: an earlier version of this draft was prepared in Sprint 15
> (commit `74f2ba63d4`, 2026-04-12) but never posted. The 2026-04-26 refresh
> aligned with v0.5.0. The 2026-04-30 refresh (this version) updates all numbers
> to the v0.6.0 benchmark suite (`docs/benchmarks/v0.6.0-pareto.md`),
> incorporates the Axis C cross-over result, documents the Amazon uint8
> cat-aliasing finding, and adopts the "reproducibility-grade CatBoost on MLX"
> framing that replaced the prior "characterized-difference port" framing.

---

## Title

[Discussion] MLX (Apple Silicon GPU) training backend — v0.6.0 ships as a fork; coordination-of-interest check

## Body

### TL;DR

We have shipped a working **MLX/Metal GPU training backend** for CatBoost on Apple
Silicon as a public fork (`RR-AMATOK/catboost-mlx`, v0.6.0, 2026-04-30). The v0.6.0
positioning is **"reproducibility-grade CatBoost on MLX"**: on numeric workloads at
fair convergence, CatBoost-MLX produces statistically indistinguishable results from
CatBoost-CPU. We are not faster — training is 5–16× slower across five measured
datasets — but we are deterministic, seed-reproducible, and GPU-native on Apple Silicon.
We are not requesting an upstream merge at this time. This issue is to **flag the work's
existence** and ask whether the upstream team has any interest in coordination,
ranging from "no thanks, please continue as a fork" to "let's talk about what
upstream-readiness would look like."

**Repository**: https://github.com/RR-AMATOK/catboost-mlx
**Release**: v0.6.0 (2026-04-30) — full 5-dataset benchmark suite; "reproducibility-grade" framing locked

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

### What it does NOT do (honest characterization)

This is the most important section of this draft. **The fork is positioned as a
"reproducibility-grade CatBoost on MLX" for numeric workloads, not a byte-faithful
replacement for categorical-heavy pipelines.**

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
- **Categorical features with cardinality > 255 are silently aliased.** The training
  path in `csv_train.cpp` quantizes categorical bins to `uint8` via
  `static_cast<uint8_t>`. For features with cardinality > 255 (e.g., Amazon's
  `RESOURCE` column at cardinality 799), all values above 255 alias to the same bin.
  The model trains and predicts without error, but the aliased representation differs
  from what CatBoost-CPU sees. On the Amazon Employee Access benchmark, this produces
  a constant logloss of ~0.2195 across all seeds and all iter levels (std = 0.000000),
  with a +0.088 gap to CatBoost-CPU. This is a documented v0.6.x bug tracked as an
  open DEC item. The fix requires changing the bin representation from uint8 to uint16
  throughout the CSV training path. Users with categorical features having cardinality
  > 255 should use CatBoost-CPU until this is resolved.
- **CTR encoding asymmetry on lower-cardinality categoricals.** On the Kaggle
  Irrigation Need dataset (270k × 53 features, 8 categoricals with cardinality < 100)
  at matched `RandomStrength=0` + `bootstrap_type=No`, the fork agrees with
  CatBoost-CPU on **99.917% of predictions** with a 0.28pp balanced-accuracy gap.
  The gap is decomposed in DEC-046 as 39% pure CPU seed-noise floor + 24% MLX
  architectural floor + **37% attributable to a specific identified mechanism** (CTR
  RNG ordering at `csv_train.cpp:2196-2206`). Numeric-only workloads
  (`cat_features=[]`) converge to within architectural floor: 99.948% prediction
  agreement, mean absolute probability difference 2.2e-3, no rare-class skew.
- **Training is 5–16× slower than CatBoost-CPU** on every measured dataset. The
  5-dataset benchmark suite (v0.6.0) shows the MLX/CPU train ratio is structurally
  constant across 1M/11M rows and 200/1000 iters on Higgs (~5×). On the
  high-dimensional Epsilon dataset (2,000 features), the ratio is 14–16×. This is
  compute-throughput, not launch overhead; closing it requires kernel-level work
  estimated at 5–8 sprints and is explicitly out of scope for v0.6.0.

The v0.6.0 claim is: on **numeric workloads** at **fair convergence** (sufficient iter
count), CatBoost-MLX produces results statistically indistinguishable from CatBoost-CPU.
We do not claim byte-faithful CatBoost compatibility on categorical workloads, and we
are explicit with our own users about which workloads converge to within the
architectural floor (numeric-only) vs which carry a documented limitation
(categorical-heavy, high-cardinality cats).

### Numeric accuracy parity at fair convergence (v0.6.0 benchmark results)

The v0.6.0 benchmark suite measures five datasets across four frameworks (CatBoost-CPU,
CatBoost-MLX, LightGBM, XGBoost) at multiple iter levels, 3 seeds each.

On **numeric** workloads at fair convergence:

| Dataset | Features | iter | MLX-vs-CPU gap |
|---|---|---|---|
| Higgs-1M | 28 numeric | 1000 | **+0.0002** (fp32 noise) |
| Epsilon | 2000 numeric | 2000 | **+0.0006** (architectural floor) |
| Higgs-11M | 28 numeric | 200+ | +0.0013 (under-converged; converges at iter=1000+) |

The gap scales with feature dimensionality — consistent with O(ε_mach × T × √L)
accumulation of fp32 rounding per leaf-Newton step.

**Axis C cross-over finding (Epsilon iter=4000, 5 seeds):** We tested the mathematician's
variance-reduction hypothesis — that MLX's GPU-deterministic leaf statistics would yield
a measurable advantage at long horizons. Results: at iter=4000, the mean MLX-vs-CPU gap
reverses sign (MLX nominally −0.000126 ahead of CPU), but this is **not statistically
significant** at n=5 (paired-t: t = −0.968, df=4; two-tailed critical value 2.776 at
α=0.05). Seed 43 reverses sign; the other four seeds favor MLX. We report this as a
consistent trend that a well-powered test (n ≈ 25–30) would be needed to confirm. The
wall-clock cost at iter=4000 is 8,765s (~2.4 hours) for MLX vs 511s for CPU. We do not
claim a cross-over advantage in v0.6.0.

On **categorical** workloads:

- Adult Census (8 categoricals, cardinality < 100): gap +0.1695 logloss vs CatBoost-CPU.
  The gap does not close with more iterations. Decomposed per DEC-046: 39% architectural
  floor + 61% CTR-RNG ordering asymmetry.
- Amazon Employee Access (9 categoricals, max cardinality 799): gap +0.088 logloss due
  to uint8 bin aliasing (see "What it does NOT do" above). The categorical-encoding
  claim **does not apply** to Amazon; this is a training-path bug.

**Synthetic-anchor bit-identity:**

At `RandomStrength=0` + `bootstrap_type=No` on synthetic anchors, the fork agrees with
CatBoost-CPU **bit-identically** at fp32 precision on every constructed test:

- N=1k seed=42 LG+Cosine, 50 iters: feature- and border-aligned splits, identical RMSE
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

### Performance (honest, v0.6.0)

The v0.6.0 benchmark suite (`docs/benchmarks/v0.6.0-pareto.md`) measured wall-clock on
an Apple M3 Max, macOS 26.3.1. We are not making competitive performance claims.

Training speed (MLX/CatBoost-CPU train ratio):

| Dataset | Scale | Ratio |
|---|---|---|
| Adult | 32k rows | 1.2× |
| Amazon | 26k rows (all cat) | 2.4× |
| Higgs-1M | 1M rows | 5.3–5.4× |
| Higgs-11M | 11M rows | 5.2× |
| Epsilon | 400k rows, 2000 features | 14.7–15.9× |

The ratio is **structurally constant across scale** (5.4× at 1M vs 5.2× at 11M on
Higgs). GPU launch overhead is fully amortized at 1M rows; the gap is compute-throughput.
MLX is also not competitive with LightGBM or XGBoost on training time.

Predict latency (in-process path, post-S43-T3 dispatch fix):
- Adult (16k rows, OneHot-cat): 52ms MLX vs 16ms CPU (3.3×)
- Higgs-1M (100k rows, iter=1000): 831ms MLX vs 12ms CPU (67×)
- Epsilon (100k rows, iter=2000): 13,580ms MLX vs 97ms CPU (140×)

The predict gap widens with model size. GPU kernel launch overhead dominates at low
batch sizes; catboost-mlx is not the right tool for single-row or small-batch inference.

---

## Issue template fields (per ISSUE_TEMPLATE.md)

**Problem:**
Discussion issue. The "problem" is "no GPU training support on Apple Silicon in
upstream"; we describe a working solution that ships independently, and ask whether
there is any interest in coordination.

**catboost version:**
Fork based on current master at the time of v0.6.0 release (synced occasionally; not
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
