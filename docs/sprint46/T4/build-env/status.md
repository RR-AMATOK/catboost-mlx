# S46-T4 Option 2 — Phase 1: Build Environment Status

**Date:** 2026-05-05
**Branch:** `mlx/sprint-46-simd-shuffle-research`
**Author:** @ml-engineer (S46-T4 Option 2 execution)

---

## Summary

Phase 1 chose Route A (CMake integration). Code changes are complete and staged.
Build execution is blocked by shell access constraints in this session — all commands
are documented below for the operator to run.

---

## Root Cause of Prior Standalone Build Failure

MLX 0.31.2 (Homebrew, installed 2026-05-01) headers at `/opt/homebrew/include/mlx/`
use `using namespace std` that exposes private libc++ symbols globally. Darwin 25.3 SDK
(Apple Clang 21.0.0, macOS 15.4, libc++ 2100.43.0) moved `__make_tuple_indices`,
`__remove_cv_t`, and `__enable_if_t` into the `std::` namespace only. Any `clang++`
invocation that includes `/opt/homebrew/include/mlx/mlx.h` + stdlib headers fails with
20+ compile errors.

The Python `_core.so` extension builds successfully because its CMake `find_package(MLX)`
picks up headers from a conda/pip-installed MLX package (cached at
`/private/var/folders/7s/rvrgpkk50xlfkqyqt2p334g40000gn/T/build-env-z8sy340d/lib/python3.13/site-packages/mlx/include`)
which uses a compatible header set. The Homebrew dylib (`/opt/homebrew/lib/libmlx.dylib`)
is reused for linking.

---

## Route A: CMake Integration

**Strategy:** Add probe binary targets to the existing
`python/catboost_mlx/_core/CMakeLists.txt`. The `find_package(MLX)` block at line 24
already resolves the correct include paths via the Python-installed MLX headers.
Probe executables piggyback on this resolution without touching the production `_core` target.

**Changes made (staged, not committed):**

File: `python/catboost_mlx/_core/CMakeLists.txt` (lines 83–134 appended)

Three executable targets added under `option(BUILD_S46_PROBES "..." OFF)`:
- `bench_boosting_baseline` — production flags only (no probe define)
- `bench_boosting_probe_b` — compiled with `-DSIMD_SHUFFLE_PROBE_B`
- `bench_boosting_probe_d` — compiled with `-DSIMD_SHUFFLE_PROBE_D`

All three: `target_include_directories` uses `${CATBOOST_MLX_ROOT}` (repo root),
`target_link_libraries` uses `mlx ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK}` —
identical to the `_core` target except no nanobind.

---

## Build Commands (for operator)

```bash
REPO="/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/catboost-mlx"
BUILDDIR="$REPO/python/build/temp.macosx-11.1-arm64-cpython-313/catboost_mlx._core"

# Step 1: Reconfigure CMake with probes enabled.
# Uses the same cmake binary and MLX paths from the existing cache.
"$BUILDDIR/../../../.." 2>/dev/null || true  # no-op directory check
/private/var/folders/7s/rvrgpkk50xlfkqyqt2p334g40000gn/T/build-env-z8sy340d/lib/python3.13/site-packages/cmake/data/bin/cmake \
  -S "$REPO/python/catboost_mlx/_core" \
  -B "$BUILDDIR" \
  -DBUILD_S46_PROBES=ON \
  2>&1 | tail -20

# Step 2: Build the three probe targets.
cmake --build "$BUILDDIR" --target bench_boosting_baseline  2>&1 | tail -20
cmake --build "$BUILDDIR" --target bench_boosting_probe_b   2>&1 | tail -20
cmake --build "$BUILDDIR" --target bench_boosting_probe_d   2>&1 | tail -20

# Step 3: Verify binaries exist.
ls -lh "$BUILDDIR/bench_boosting_baseline" \
        "$BUILDDIR/bench_boosting_probe_b" \
        "$BUILDDIR/bench_boosting_probe_d"
```

If Step 1 fails because the temp cmake binary path has changed (it's inside a build-env
tmpdir), fall back to:
```bash
python3.13 -m pip install cmake --quiet  # ensures cmake is available
cmake -S "$REPO/python/catboost_mlx/_core" -B "$BUILDDIR" -DBUILD_S46_PROBES=ON
```

---

## Route B (documented for completeness, not recommended)

Three compilation flag combinations to try if CMake fails:

**Option B-1:** Use conda-installed MLX headers directly:
```bash
CONDA_MLX=$(python3.13 -c "import mlx; import os; print(os.path.dirname(mlx.__file__))")
clang++ -std=c++17 -O2 \
  -I"$REPO" \
  -I"$CONDA_MLX/include" \
  -I"$CONDA_MLX/include/metal_cpp" \
  -L"$CONDA_MLX" -lmlx \
  -framework Metal -framework Foundation \
  -Wno-c++20-extensions \
  -DSIMD_SHUFFLE_PROBE_B \
  "$REPO/catboost/mlx/tests/bench_boosting.cpp" \
  -o /tmp/bench_probe_b
```

**Option B-2:** Apply libc++ compat flag:
```bash
clang++ -std=c++17 -O2 \
  -I"$REPO" \
  -I/opt/homebrew/include \
  -L/opt/homebrew/lib -lmlx \
  -framework Metal -framework Foundation \
  -Wno-c++20-extensions \
  -D_LIBCPP_ENABLE_CXX17_REMOVED_FEATURES \
  -DSIMD_SHUFFLE_PROBE_B \
  "$REPO/catboost/mlx/tests/bench_boosting.cpp" \
  -o /tmp/bench_probe_b
```

**Option B-3:** Pin SDK to 14.5:
```bash
clang++ -std=c++17 -O2 \
  -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX14.5.sdk \
  -I"$REPO" \
  -I/opt/homebrew/include \
  -L/opt/homebrew/lib -lmlx \
  -framework Metal -framework Foundation \
  -Wno-c++20-extensions \
  -DSIMD_SHUFFLE_PROBE_B \
  "$REPO/catboost/mlx/tests/bench_boosting.cpp" \
  -o /tmp/bench_probe_b
```

---

## Status

- Route A CMakeLists changes: **WRITTEN, NOT YET BUILT**
- Phase 2 (D2 dispatcher rewrite): **COMPLETE** (separate changes to bench_boosting.cpp and kernel_sources.h)
- Phase 3 (measurements): **BLOCKED** pending Phase 1 build success

Once binaries are built, proceed to Phase 3 measurement commands in
`docs/sprint46/T4/build-env/phase3-commands.md`.
