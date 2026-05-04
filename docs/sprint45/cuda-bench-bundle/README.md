# CatBoost-CUDA Cross-Class Benchmark Bundle

**Purpose:** Run CatBoost-CUDA on a Windows GPU box on the same datasets and hyperparameters that the M3 Max MLX sweep ran on, so we have a cross-class reference (MLX-on-M3-Max vs CUDA-on-Windows) for the v0.7.0 perf writeup.

This bundle is **self-contained** and ships with **4 of the 5 datasets pre-prepared inside `cache/`** (Adult + Amazon + Higgs-1M + Higgs-11M, ~8.2 GB total). Only Epsilon needs to download on the Windows side (~15 GB compressed from libsvm; auto-downloaded by `prepare.py`). Drop the `cuda-bench-bundle/` folder into `Downloads/` (or anywhere), follow the steps below, then bring back the `results/` folder + a `hardware.txt`. After that, delete the entire bundle folder.

---

## 0. Prerequisites

- **Windows 10/11** with an NVIDIA GPU (CUDA-capable)
- **NVIDIA driver** + CUDA-runtime libs (any reasonably recent version; CatBoost bundles its own CUDA stack via the wheel)
- **Python 3.10 or 3.11** (3.12 is fine; 3.13 may not have CatBoost wheels yet — check first)
- **Internet** (for pip install + Epsilon dataset auto-download)
- **~25–30 GB free disk** if you run all 5 datasets (8.2 GB bundled cache + ~15 GB for Epsilon raw + ~14 GB for Epsilon CSVs; the Epsilon raws can be deleted after the CSVs are written). Less if you skip Epsilon.

## 1. Record hardware

Open a terminal in the bundle directory. **Either Command Prompt or PowerShell works** — `python` calls work identically in both.

```
python record_hardware.py
```

This captures GPU model + CUDA driver, CPU, RAM, OS, Python version, CatBoost version into `hardware.txt`. Send it back with the results — needed for the writeup.

## 2. Install dependencies

```cmd
cd path\to\cuda-bench-bundle
pip install -r requirements.txt
```

Verify CatBoost can see CUDA:

```cmd
python -c "import catboost; from catboost import CatBoostClassifier; import numpy as np; m = CatBoostClassifier(iterations=2, task_type='GPU', verbose=False); m.fit(np.random.randn(100, 4), np.random.randint(0, 2, 100)); print('CUDA OK; CatBoost version:', catboost.__version__)"
```

If that errors, the most common fixes are:
- Update the NVIDIA driver
- Use `pip install catboost` (CPU-only) and report back — we still want the comparison vs CPU CUDA missing
- Set `task_type='CPU'` (defeats the purpose; only do this if CUDA truly won't work)

## 3. Prepare datasets

The bundle ships with **4 datasets pre-prepared inside `cache/`**:

| Dataset    | Status              | Size on disk |
|------------|---------------------|-------------:|
| Adult      | Pre-bundled         |        2.3 MB |
| Amazon     | Pre-bundled         |        2.0 MB |
| Higgs-1M   | Pre-bundled         |        766 MB |
| Higgs-11M  | Pre-bundled         |        7.5 GB |
| Epsilon    | **Auto-download**   |       ~14 GB after prep |

Run `prepare.py` to auto-download Epsilon (the only missing dataset). Adapters that find their dataset already in `cache/` skip themselves.

```cmd
python prepare.py
```

What this does for each dataset:
- **Adult, Amazon, Higgs-1M, Higgs-11M** — `meta.json` already in `cache/<name>/`; SKIPPED.
- **Epsilon** — auto-downloads `epsilon_normalized.bz2` (~12 GB) + `epsilon_normalized.t.bz2` (~3 GB) from the LIBSVM mirror, decompresses both, parses the libsvm format, writes `train.csv` (~7 GB) + `test.csv` (~7 GB). The mirror is **slow** (~5–15 MB/s typical) — expect 20–60 minutes for the train file alone. Disk needed: ~30 GB peak (15 GB compressed + 14 GB decompressed + 14 GB CSVs; you can delete the .bz2 and decompressed raws after the CSVs are written).

To skip Epsilon entirely (use only the 4 pre-bundled datasets):

```cmd
python prepare.py --skip epsilon
```

To re-prep one dataset from scratch (rare; only useful if a `cache/` file is corrupt):

```cmd
REM First delete the dataset's cache subdir, then:
python prepare.py --only higgs_1m
```

The cache lives **inside the bundle** at `cache/`, not in `%USERPROFILE%\.cache\`. To clean up after the run, just delete the bundle folder.

## 4. Run the sweep

```cmd
python run.py
```

This runs the full **5-dataset × iter-grid × 3-seed** sweep, matching the M3 Max methodology:

| Dataset    | Iters                       | Seeds         | Runs |
|------------|-----------------------------|---------------|-----:|
| Adult      | 200, 500, 1000, 2000        | 42, 43, 44    |   12 |
| Higgs-1M   | 200, 500, 1000, 2000        | 42, 43, 44    |   12 |
| Epsilon    | 200, 500, 1000, 2000        | 42, 43, 44    |   12 |
| Amazon     | 200, 500, 1000, 2000        | 42, 43, 44    |   12 |
| Higgs-11M  | 200                         | 42, 43, 44    |    3 |
| **Total**  |                             |               | **51** |

Wall-time estimates (very rough; depends heavily on GPU):
- A100 / H100: ~30–60 min total
- 4090 / 3090: ~1–2 h total
- 2080 Ti / 1080 Ti: ~4–8 h total

The runner is **cache-aware** — if you stop and restart, completed runs are skipped.

To skip slow workloads:

```cmd
python run.py --skip higgs_11m epsilon
```

Smoke test (one dataset, one iter, one seed):

```cmd
python run.py --only adult --iters 200 --seeds 42
```

Results land in `results/` (inside the bundle) as JSON files, one per (dataset, iter, seed) cell. Schema matches the M3 Max sweep so the existing aggregator can ingest both side-by-side.

## 5. What to send back

Two things, zipped together:

1. The entire `results/` folder (inside the bundle)
2. The `hardware.txt` file from step 1

Anything else (the bundle itself, including `cache/`) you can delete.

## 6. Cleanup

The bundle is fully self-contained — no files are written outside it. To clean up:

**In Command Prompt:**
```cmd
cd Downloads
rmdir /s /q cuda-bench-bundle
```

**In PowerShell:**
```powershell
cd Downloads
Remove-Item -Recurse -Force cuda-bench-bundle
```

**Or just drag the folder to the Recycle Bin in File Explorer** — same effect.

That removes everything: cached datasets (8.2 GB pre-bundled + 14 GB Epsilon CSVs if downloaded), results, scripts, the lot.

Done.

---

## Methodology notes

- **Hyperparameters** are fixed across all frameworks via `scripts/_runner_common.py:BENCH_HP`. Same `depth=6`, `learning_rate=0.1`, `l2_reg=3.0`, `random_strength=0.0`, `bootstrap_type='No'` as the M3 Max sweep.
- **`random_strength=0` + `bootstrap_type='No'`** removes per-run RNG injection so seed-to-seed variance reflects only initial split selection (CatBoost's deterministic-greedy contract).
- **iter-grid** matches `docs/sprint44/sprint-plan.md` and the v0.6.0 launch writeup.
- **Output JSON schema** matches the M3 Max sweep, so cross-class aggregation works without conversion.

## Troubleshooting

**Q: `task_type='GPU'` raises `Cuda error 100 (device not found)`.**
A: The bundled CUDA libs in the CatBoost wheel may not match your driver. Try `pip install --upgrade catboost`, or update the NVIDIA driver.

**Q: Out-of-memory on Epsilon or Higgs-11M.**
A: Epsilon (400k × 2000 features) needs ~6 GB GPU RAM at iter=2000. Higgs-11M needs ~12 GB. If your GPU is smaller, run with `--skip epsilon higgs_11m` and only return the workloads that fit.

**Q: Adapter says "manual download required" for Epsilon or Amazon.**
A: Follow the printed recipe. Epsilon needs PASCAL LSC libsvm files; Amazon needs Kaggle CLI auth.

**Q: One run fails, the rest are fine.**
A: Re-run `python run.py` — it skips completed cells and retries failed ones. If a cell consistently fails, send the JSON output (or whatever the runner printed) back with the results bundle.
