# S49-T3 Envelope Sweep — Orchestrator Runbook

**Branch:** `mlx/sprint-49-c6-engineering` HEAD `c323c7fe64`
**Authority:** DEC-008 + S49-T0c Q4 lock
**Gate:** 18-config {N, Loss, Bins} × 3 seeds = 54 trainings per pass × 2 passes = 108 total

---

## Prerequisites

1. Repo is on branch `mlx/sprint-49-c6-engineering` at HEAD `c323c7fe64`.
2. Working directory is repo root: `/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/catboost-mlx`
3. conda env is active (the one with catboost_mlx dependencies).
4. `python/` build dependencies available (`cmake`, `ninja`, Apple developer tools).

Verify:
```bash
git log --oneline -1
# Expected: c323c7fe64 [mlx] sprint-49: T2.7+T2.8+T2.9 — plumb UseHistogramSubtraction...

python3 -c "import catboost_mlx; print(catboost_mlx.__version__)"
# Should succeed — nanobind _core.so already built from T2
```

---

## Step-by-step execution

### Pass 1 — Baseline (C6 disabled)

```bash
# 1a. Apply Option B patch (forces UseHistogramSubtraction=false)
bash docs/sprint49/T3/envelope-sweep/apply_option_b_baseline.sh

# 1b. Verify patch
grep "Option B baseline patch" catboost/mlx/train_lib/train.cpp
# Expected: one match

# 1c. Rebuild _core.so
cd python
pip install -e . --no-build-isolation -q
cd ..

# 1d. Verify build picked up the patch (optional sanity check)
python3 -c "
import catboost_mlx, numpy as np
m = catboost_mlx.CatBoostMLXRegressor(iterations=2, verbose=False)
X = np.random.randn(100, 3).astype('f4')
y = np.random.randn(100).astype('f4')
m.fit(X, y)
print('Baseline build OK')
"

# 1e. Run baseline sweep (54 trainings ~20-40 min on M3 Max)
SWEEP_PASS=baseline python3 docs/sprint49/T3/envelope-sweep/run_t3_sweep.py
# Output: docs/sprint49/T3/envelope-sweep/results_baseline.json
```

### Pass 2 — C6 (loss-conditional dispatch active)

```bash
# 2a. Revert Option B patch
bash docs/sprint49/T3/envelope-sweep/revert_option_b_baseline.sh

# 2b. Verify revert
grep "lossFunction == ELossFunction::Logloss" catboost/mlx/train_lib/train.cpp
# Expected: one match (the original C6 3-way OR expression)

# 2c. Rebuild _core.so
cd python
pip install -e . --no-build-isolation -q
cd ..

# 2d. Verify C6 build (optional — should log "UseHistogramSubtraction=1" for logloss)
python3 -c "
import catboost_mlx, numpy as np, logging
logging.basicConfig(level=logging.INFO)
m = catboost_mlx.CatBoostMLXClassifier(iterations=2, loss='logloss', verbose=False)
X = np.random.randn(200, 3).astype('f4')
y = np.array([0]*100 + [1]*100, dtype='f4')
m.fit(X, y)
print('C6 build OK')
"

# 2e. Run C6 sweep (54 trainings ~20-40 min)
SWEEP_PASS=c6 python3 docs/sprint49/T3/envelope-sweep/run_t3_sweep.py
# Output: docs/sprint49/T3/envelope-sweep/results_c6.json
```

### Pass 3 — Analysis

```bash
# 3a. Generate analysis.md
python3 docs/sprint49/T3/envelope-sweep/run_t3_sweep.py --compare
# Output: docs/sprint49/T3/envelope-sweep/analysis.md

# 3b. Print summary
head -60 docs/sprint49/T3/envelope-sweep/analysis.md
```

---

## Expected gate outcomes

Per Q4 lock (S49-T0c):

| Gate result | Meaning | T4 path |
|---|---|---|
| ALL_PASS | RMSE bit-identical + Logloss/MultiClass ≤ ULP ceiling | Full ship — T4 unconstrained |
| RMSE_FAIL_OTHERS_PASS | C6 leaking into RMSE path | STOP — code bug. Must not happen. |
| NON_RMSE_FAIL | Logloss or MultiClass exceeds ceiling | STOP — investigate. @mathematician needed. |
| MULTI_CLASS_FAIL | Multiple loss classes fail | STOP — likely code bug. Halt T4. |

The most likely outcome is ALL_PASS (RMSE path is unconditionally production; C6 subtract adds at most 1 extra γ_1 step → ≤ 4 ULP on float32 for Logloss/MultiClass typical loss magnitudes).

---

## DEC-008 ULP ceilings

| Loss | Ceiling | Theoretical basis |
|---|---|---|
| RMSE | ulp ≤ 4 (expected: 0) | γ_8 ≈ 4.77e-7; C6 inactive for RMSE |
| Logloss | ulp ≤ 4 | γ_8 ≈ 4.77e-7; one extra subtract adds γ_1 ≈ 1.19e-7 |
| MultiClass | ulp ≤ 8 | γ_14 ≈ 8.3e-7; three-dim composition |

---

## Timing estimate

- N=10k: ~15-30s per training (100 iterations, depth 6, 20 features)
- N=50k: ~60-120s per training
- 54 trainings per pass × 2 passes = 108 total
- Estimated wall-clock: 3-6 hours (M3 Max, Metal GPU)

To run faster: use `--verbose` to confirm progress; consider running seeds in parallel
by splitting into three SEED= env-var runs (not currently supported — add if needed).

---

## Files produced

```
docs/sprint49/T3/envelope-sweep/
  run_t3_sweep.py              — sweep runner + comparison script
  apply_option_b_baseline.sh   — Option B patch applier
  revert_option_b_baseline.sh  — Option B patch reverter
  ORCHESTRATOR_RUNBOOK.md      — this file
  results_baseline.json        — 54-row baseline results (after Pass 1)
  results_c6.json              — 54-row C6 results (after Pass 2)
  analysis.md                  — full analysis + gate verdict (after Pass 3)
```

Stage `analysis.md` + both JSON files for orchestrator review after completion. No commits.

---

## Abort conditions

- If `apply_option_b_baseline.sh` fails: check git status; file may be modified. Use `git diff` to inspect.
- If rebuild fails: inspect `python/build/` for CMake errors. The patch is syntactically valid C++.
- If sweep fails mid-run: results up to the failure point are saved. Fix the error and re-run; already-saved results are overwritten from scratch (no partial-results resume).
- If a training run crashes (Metal OOM, etc.): the record is saved with `status: "error: ..."` and `final_loss: NaN`. The comparison will flag it as `MISSING`. Report to orchestrator.

---

*S49-T3 — QA gate before Bundle 2 measurement.*
