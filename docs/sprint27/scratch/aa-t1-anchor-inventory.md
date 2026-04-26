# S27-AA-T1 — Numeric Anchor Inventory

**Produced:** 2026-04-22 (Sprint 27, Track B)
**Last refreshed:** 2026-04-25 (Sprint 39) — added AN-019 through AN-023 (Sprints 28–38)
**Agent:** @qa-engineer
**Purpose:** Enumerate every committed numeric anchor for T2 re-run and T3 drift classification.
**Scope:** live assertions + documented canonical values that would drift if generating code changed.

---

## Anchor Table

| ID | Path | Line | Value | Kind | Harness-to-regen | Last-touched-sha | Captured-context |
|----|------|------|-------|------|-----------------|-----------------|-----------------|
| AN-001 | `python/tests/test_qa_round9_sprint4_partition_layout.py` | 91 | `0.306348` | RMSE | `pytest python/tests/test_qa_round9_sprint4_partition_layout.py::TestRegressionAnchor::test_rmse_final_loss_matches_sprint4_anchor` | `634a72134d` | csv_train path, N=100, 20 features, 10 iters, depth 4, seed 0, rs=1, max_bin=32; updated S26-D0-9 after DEC-028 RandomStrength fix (prior value 0.432032) |
| AN-002 | `python/tests/test_qa_round9_sprint4_partition_layout.py` | 107 | `0.317253` | prediction | `pytest python/tests/test_qa_round9_sprint4_partition_layout.py::TestRegressionAnchor::test_specific_predictions_match_anchor` | `634a72134d` | preds[0] from same Sprint 4 100-row config post-DEC-028; prior value 0.414606 |
| AN-003 | `python/tests/test_qa_round9_sprint4_partition_layout.py` | 108 | `-0.568259` | prediction | `pytest python/tests/test_qa_round9_sprint4_partition_layout.py::TestRegressionAnchor::test_specific_predictions_match_anchor` | `634a72134d` | preds[1] same config post-DEC-028; prior value -0.545893 |
| AN-004 | `python/tests/test_qa_round9_sprint4_partition_layout.py` | 109 | `1.598960` | prediction | `pytest python/tests/test_qa_round9_sprint4_partition_layout.py::TestRegressionAnchor::test_specific_predictions_match_anchor` | `634a72134d` | preds[99] same config post-DEC-028; prior value 1.356884 |
| AN-005 | `python/tests/test_qa_round9_sprint4_partition_layout.py` | 155 | `[0.37227302, 0.36382151, 0.26390547]` | prediction | `pytest python/tests/test_qa_round9_sprint4_partition_layout.py::TestRegressionAnchor::test_multiclass_k3_proba_anchor` | `634a72134d` | first-row softmax proba from Sprint 4 multiclass K=3 config, post-DEC-028; prior values [0.35687973, 0.36606121, 0.27705906] |
| AN-006 | `python/tests/test_qa_round10_sprint5_bench_and_scan.py` | 66 | `0.11909308` | logloss | `pytest python/tests/test_qa_round10_sprint5_bench_and_scan.py::TestBenchBoostingAnchors::test_binary_100k_anchor` (requires `/tmp/bench_boosting`) | `f804c56742` | bench_boosting binary: 100k × 50 × cls=2 × depth6 × 100iters × bins32 × seed42; updated BUG-002 fix (prior 0.69314516, then 1.07820153 pre-BUG-001) |
| AN-007 | `python/tests/test_qa_round10_sprint5_bench_and_scan.py` | 67 | `0.63507235` | logloss | `pytest python/tests/test_qa_round10_sprint5_bench_and_scan.py::TestBenchBoostingAnchors::test_multiclass_20k_anchor` (requires `/tmp/bench_boosting`) | `f804c56742` | bench_boosting: 20k × 30 × cls=3 × depth5 × 50iters × bins32 × seed42; updated BUG-002 fix (prior 1.09757149 pre-BUG-002) |
| AN-008 | `CHANGELOG.md` | 41 | `1.78561831` | logloss | Build bench_boosting then: `./bench_boosting --rows 20000 --features 30 --classes 10 --depth 5 --iters 50 --seed 42` | `fd04d34684` | K=10 multiclass, 20k × 30 × depth5 × 50iters; corrected Sprint 8 TODO-022 (prior stale value 2.22267818 from mismatched params) |
| AN-009 | `docs/sprint19/results.md` | 41 | `0.48231599` | RMSE | Build bench_boosting from tip then: `./bench_boosting --rows 10000 --features 50 --classes 1 --depth 6 --iters 50 --bins 128 --seed 42` | `b04b1efd2c` | Sprint 19 gate config 10k/RMSE/d6/128b, post-S19-13; DEC-023 parity anchor (Value A); config #8 is bimodal at ~50/50 with 0.48231912 (BUG pre-v5) |
| AN-010 | `docs/sprint19/results.md` | 47 | `0.47740927` | RMSE | Build bench_boosting from tip then: `./bench_boosting --rows 50000 --features 50 --classes 1 --depth 6 --iters 50 --bins 128 --seed 42` | `b04b1efd2c` | Sprint 19 gate config 50k/RMSE/d6/128b (primary determinism gate); 100/100 determinism confirmed S19, S22-D3, S24-D0 |
| AN-011 | `docs/sprint19/results.md` | 34–49 | 16 additional config losses (1k–50k × RMSE/Logloss/MultiClass × 32/128, excl. #8 and #14 already listed) | RMSE / logloss / multiclass | `./bench_boosting` with each {N, loss, bins} combination; same fixed params as AN-009/010 | `b04b1efd2c` | S19-04 18-config bit-exact parity sweep; these are the declared DEC-008 reference values post-S19-13. Note: values differ from sprint18 table (different kernel version). |
| AN-012 | `docs/sprint18/parity_results.md` | 31–48 | 18 config losses (e.g. `0.44764100`, `0.46951200`, etc.) | RMSE / logloss / multiclass | Build bench_boosting at commit `19fa5ce6cc` (sprint18 kernel) then run 18-config DEC-008 grid | `7ab4e8e804` | S18-04b parity results — reference for post-BUG-S18-001 fixed kernel; determinism anchor 0.48016092 at 10k/RMSE/128b (used in S18 gate); superseded at kernel level by S19 values (AN-011) |
| AN-013 | `docs/sprint17/parity_results.md` | 22–39 | 18 config losses captured under csv_train sprint16 vs sprint17 (e.g. RMSE 10k/128 = `0.496241`) | RMSE / logloss / multiclass | Run csv_train sprint16 binary vs sprint17 binary on the 18-config CSV grid | `ed0ec8221b` | S17-04 parity sweep; reference binary is `csv_train_sprint16` (serial reduction). All ULP=0. Uses csv_train+CSV path, not bench_boosting — different data synthesis. |
| AN-014 | `benchmarks/results/m3_max_128gb.md` | 13–30 | e.g. MLX RMSE 10k = `3.0407`, MLX Logloss 10k = `0.4099` (9 MLX loss values total) | RMSE / logloss / multiclass | Run `python benchmarks/bench_mlx_vs_cpu.py` with catboost_mlx 0.3.0 on M3 Max | `7b36f60a82` | Sprint 14 benchmark run; catboost_mlx 0.3.0 (pre-DEC-028/DEC-029 fixes). Not asserted in tests. Purely documentary. |
| AN-015 | `python/tests/test_qa_round12_sprint9.py` | 655, 661 | `0.59795737`, `0.95248461` | logloss | Requires `.github/workflows/mlx_test.yaml` to contain these strings (UNKNOWN — target file does not contain them; test skips when `mlx_test.yaml` missing) | `b8a0ab258a` | Sprint 9 Item H: asserts CI workflow YAML contains binary baseline 0.59795737 and multiclass K=3 baseline 0.95248461. The target file (`mlx_test.yaml`) does not currently contain these values — test effectively always skips. See AMBIGUOUS note below. |
| AN-016 | `docs/sprint26/d0/g1-g3-g4-report.md` | 101 | `0.19457837` | RMSE | Run `python benchmarks/sprint26/d0/g4_determinism.py` (S26 G5 determinism, 100 runs at N=10k/seed=1337/rs=0) | `cbbfc29257` | S26 G5 determinism: 100 runs of Python-path SymmetricTree, N=10k, seed=1337, rs=0; mean and median RMSE converge to 0.19457837 / 0.19457836. Not asserted in automated tests — docs-only anchor. |
| AN-017 | `benchmarks/sprint26/fu2/fu2-gate-report.md` | 101 | `0.17222003` | RMSE | Run `python benchmarks/sprint26/d0/g4_determinism.py` or analogous script at FU-2 config (N=10k, seed=1337, rs=0, iterations=50, depth=6, max_bin=128, grow_policy=SymmetricTree, FU-2 binary) | `2d806d0fa4` | FU-2 G5 determinism: 100 runs; mean/median RMSE 0.17222003 / 0.17222002. Not asserted in automated tests — docs-only anchor. Represents current production-tip Python-path SymmetricTree output. |
| AN-018 | `docs/sprint19/scratch/algorithmic/a1_empirical_drop.md` | 31 | `0.48047778` | RMSE | Build bench_boosting at commit `0f992cf863` or nearby S19 tip then: same 50k/RMSE/d6/128b/seed42 config | `0f992cf863` | Parity gate value at S19 intermediate tip (post-EvalAtBoundary removal, pre-T1); used as parity reference during A1 production port. Also appears in sprint21 d0_attribution.md run output. Not asserted in live tests. |
| AN-019 | `docs/sprint33/l1-determinism/verdict.md` | 104–105 | CPU RMSE seed=42: `0.1936264503`; MLX pre-fix RMSE seed=42: `0.2956260000` (pre-fix, superseded) | RMSE | Re-run requires: N=50k, 20 features, y=0.5·X[0]+0.3·X[1]+0.1·noise, np.random.default_rng(42), ST/Cosine/RMSE, depth=6, bins=128, l2=3, lr=0.03, random_strength=0.0. CPU value: catboost 1.2.10; MLX value: csv_train at S33 baseline tip `9dfd62ccc3` (pre-L4-fix) | `9dfd62ccc3` | Sprint 33 L1 determinism probe anchor — shared base config for all sprint33 probes (A through E) and the L0–L4 scaffold. CPU RMSE is the "ground truth" ST+Cosine N=50k target. MLX value captures pre-L4-fix state; superseded by the DEC-042 post-fix output (~0.214 range per S33 Commit 3b). Docs-only. |
| AN-020 | `docs/sprint33/l2-graft/verdict.md` | 124 | `0.29299000` | RMSE | Re-run requires: graft_snapshot_seed42.json (CPU iter=1 cursor injected into MLX run) at the AN-019 config; see run_l2_graft.py | `9dfd62ccc3` | Sprint 33 L2 graft diagnostic: MLX iter=50 RMSE after receiving CPU iter=1 cursor (graft). Used to isolate iter≥2 divergence source. Intermediate diagnostic value; superseded by L4-fix resolution. Docs-only. |
| AN-021 | `docs/sprint33/sprint-close/cr-report.md` | 89 | Pre-fix iter=49 loss: `0.479101`; post-fix iter=49 loss: `0.493401` | loss (one-hot smoke) | Run csv_train on 8k one-hot anchor: 1 cat feature × 4 levels + 2 numeric, ST+Cosine, 50 iters, depth=6, bins=32, lr=0.05, l2=3, seed=42, max-onehot-size=8. Requires S33-L4-FIX commits 3a/3b present. | `e1d72d64e8` (Commit 3a) | Sprint 33 CR one-hot smoke anchor: applied the per-side mask pattern to one-hot path (L1698) as a sanity check; loss increased ~3% post-patch, which drove the "investigate before fixing one-hot" Path B decision. Pre-fix value is the S33 baseline; post-fix value is the patched output. The patch was reverted — neither value reflects current production code for the one-hot path (joint-skip retained per S34-PROBE-F-LITE). Docs-only. Status: SUPERSEDED — patched branch was reverted; one-hot Cosine retains joint-skip per PROBE-F-LITE verdict. |
| AN-022 | `docs/sprint38/probe-g/data/anchor_n1000_seed42.csv` | (committed CSV file) | MD5: `0976a75cb621a74e55eb9480d35935b3`; 1000 rows × 20 features | data file | `cp docs/sprint38/probe-g/data/anchor_n1000_seed42.csv` to working dir; generate via: `np.random.default_rng(42)`, N=1000, 20 features, y=0.5·X[0]+0.3·X[1]+0.1·N(0,1) | `a481972529` (S38 FBSPP fix commit) | Sprint 38 canonical small-N probe data file. Shared by PROBE-G (phase 1), F2, PROBE-H, fix-verify, and PROBE-Q as the N=1k seed=42 training set. Config: same feature/target formula as AN-019 but at N=1k. The committed CSV is the byte-identical artifact. Docs-only (no live test asserts against this CSV directly). |
| AN-023 | `docs/sprint38/probe-q/PHASE-2-FINDING.md` | 17–21 | MLX RMSE (RS=0): `0.204238`; CPU RMSE (RS=0): `0.204238`; drift: `0.000%` | RMSE | Run `./csv_train docs/sprint38/f2/data/anchor_n1000_seed42.csv --iterations 50 --depth 6 --lr 0.03 --bins 128 --l2 3 --loss rmse --score-function Cosine --seed 42 --random-strength 0`; compare with `catboost.CatBoost(iterations=50, depth=6, learning_rate=0.03, border_count=128, l2_leaf_reg=3, loss_function='RMSE', score_function='Cosine', random_strength=0.0, random_seed=42)` | `a481972529` | DEC-045 resolution anchor: bit-identical RMSE at matched RS=0. LG+Cosine, N=1k seed=42, 50 iters. Demonstrates that the 13–44% small-N drift observed in Sprints 37–38 was entirely a harness configuration asymmetry (CPU at RS=0, MLX at default RS=1.0). With RS matched, drift collapses to zero. Docs-only. |

---

## Sprint 28–38 additions: notes on classification

**AN-019 (docs-only, sprint33):** The N=50k ST+Cosine probe anchor is shared across all sprint33 investigations (PROBE-A through PROBE-E, L0–L4). The CPU value (`0.1936264503`) is a stable reference; the MLX pre-fix value (`0.2956260000`) is historical. No live test asserts either value. The fix (DEC-042 via S33-L4-FIX) brought MLX inline; current production output at this config is not separately anchored in a live test.

**AN-020 (docs-only, sprint33):** Graft diagnostic RMSE `0.29299000` is an intermediate L2 diagnostic. Not a production-meaningful anchor — generated from an artificial hybrid run (CPU iter=1 cursor in MLX run). Class-d by DEC-031 Rule 4 (dead anchor: unreachable without reconstructing the L2 graft harness). Retained as a historical record.

**AN-021 (superseded, sprint33):** One-hot smoke values `0.479101` / `0.493401` are from a patched branch that was reverted. The post-patch value does not reflect any committed code path. Per DEC-031 class-c (documented-supersession): the one-hot Cosine path retains joint-skip by intent (S34-PROBE-F-LITE). Neither value is a valid regression baseline for current code.

**AN-022 (data-file anchor, sprint38):** The committed CSV `anchor_n1000_seed42.csv` is a new type of anchor (data file, not a scalar). Not yet wired to any live test assertion. DEC-031 Rule 1 applies: if this CSV is used as a gate anchor in future harnesses, the test must include a MD5 or row-count assertion. As of S38, all uses are in probe scripts under `docs/`.

**AN-023 (docs-only, sprint38):** The DEC-045 bit-identical RMSE (`0.204238` at RS=0) is the resolution anchor for the small-N investigation. It should become a live test assertion (DEC-031 Rule 1) in a future sprint: wire it to `python/tests/` to catch any regression in RS=0 parity at N=1k. Currently docs-only.

---

## Ambiguous / Needs-T3-Judgment entries

**AN-015 (HIGH AMBIGUITY):** `test_qa_round12_sprint9.py` asserts that `.github/workflows/mlx_test.yaml` contains the strings `0.59795737` and `0.95248461`. The workflow file is named `mlx-test.yaml` (with a hyphen, not underscore), so the fixture always skips. These values appear to be Sprint 9 bench_boosting baselines that were supposed to be embedded in CI but never were — the test guards a dead integration. T3 should decide: (a) these anchors never landed in CI — class-a stale-capture; (b) or they represent a real value that should be compared against current output by resolving the broken fixture path.

**AN-011 (BULK ENTRY):** 16 additional bench_boosting config losses from `docs/sprint19/results.md`. These are the DEC-008 reference values used for ULP-comparison gates (not numeric equality assertions in tests). They are "captured outputs" in the doc sense but the live gate mechanism compares pre vs post kernel ULP, not doc-vs-run. Treating as docs-only; T2 could re-run the 18-config sweep to check drift.

**AN-012 (SUPERSEDED KERNEL):** Sprint 18 parity values (`7ab4e8e804`) are from a different kernel version than v5. They are not asserted anywhere — they document the S18 parity sweep result. Not a live regression risk unless someone references them for a v5 comparison. T3: likely class-a documented-supersession.

**AN-013 (SUPERSEDED PATH):** Sprint 17 parity values use `csv_train` subprocess with CSV files, not the bench_boosting+synthetic-data pipeline. These anchor values are not asserted in any live test and are from a kernel predating DEC-016/T1. T3: class-a documented-supersession.

**AN-014 (CATBOOST_MLX 0.3.0):** `m3_max_128gb.md` losses were captured with catboost_mlx 0.3.0 (Sprint 14), predating DEC-028/DEC-029 by 12+ sprints. Not asserted in tests. T3: class-a stale-capture.

---

## Summary statistics

| Dimension | Count |
|-----------|-------|
| Total anchors | 23 (AN-001–023; AN-019–023 added S39 refresh) |
| Live test assertions (code throws on mismatch) | 9 (AN-001…007, AN-015 × 2 values, AN-006, AN-007) |
| Docs-only anchors (no automated enforcement) | 14 (AN-008…014, AN-016…023) |
| Superseded / retired anchors | AN-021 (one-hot patch reverted; superseded), AN-020 (graft diagnostic; class-d) |
| Data-file anchors (committed CSV, no scalar assertion) | 1 (AN-022) |
| RMSE kind | 13 (AN-001, AN-009, AN-010, AN-016, AN-017, AN-018, AN-019 ×2 values, AN-020, AN-022 data, AN-023 ×2 values) |
| Prediction / proba kind | 4 (AN-002–005) |
| Logloss / loss (binary/multiclass) | 5 (AN-006–008, AN-011 bulk, AN-021 loss) |
| Predating DEC-028 merge (`634a72134d`) | 13 (AN-006–018; all captured before S26-D0-9) |
| Added Sprint 28–38 | 5 (AN-019–023) |
| DEC-028 anchor-update group (AN-001–005) | 5 (updated AT 634a72134d, i.e. these ARE the DEC-028-aware values) |

---

## Top 5 highest-risk anchors

1. **AN-015** — `test_qa_round12_sprint9.py:655,661` (`0.59795737`, `0.95248461`). Fixture always skips due to `mlx_test.yaml` vs `mlx-test.yaml` naming. Zero enforcement for 15+ sprints. Unknown which pipeline configuration generated these values or whether current code produces them.

2. **AN-008** — `CHANGELOG.md:41` (`1.78561831`, K=10). Documented-only, not asserted in any test. K=10 multiclass exercises `ComputeLeafValues` fused path (DEC-019) and the S18 `kHistMultiByte` tile. Any regression in the multiclass fused leaf path would be invisible. Last touched SHA `fd04d34684` predates S19/S22/S24 kernel changes.

3. **AN-009** — `docs/sprint19/results.md:41` (`0.48231599`, config #8 bimodal). This is the DEC-023 Value A anchor. v5 kernel fixed the bimodality; this value should be deterministic on current master. However, it is not asserted in any live test — only appears in sprint docs and `.claude/state/`. A regression in the SIMD-shuffle path (e.g. from a Metal compiler version change) would go undetected.

4. **AN-006** — `python/tests/test_qa_round10_sprint5_bench_and_scan.py:66` (`0.11909308`). Tolerance `1e-6` (very tight). Requires `/tmp/bench_boosting` binary — test skips if not built. This is the only active bench_boosting loss assertion; captured post-BUG-002 (S5). Does not exercise the Python/nanobind path. If bench_boosting is not rebuilt from current source before running, a stale binary can produce a false pass.

5. **AN-017** — `benchmarks/sprint26/fu2/fu2-gate-report.md:101` (`0.17222003`). Most recent production-tip Python-path SymmetricTree loss anchor (S26-FU-2, `2d806d0fa4`). Not enforced by any automated test. This is exactly the kind of value needed for G2-AA gate validation — but it's only in a markdown table. It should become an AN that T2 re-runs.
