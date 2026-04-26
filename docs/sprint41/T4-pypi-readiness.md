# S41-T4 — PyPI publish-readiness audit

**Date**: 2026-04-26  |  **Branch**: `mlx/sprint-41-polish`

## Goal

Verify the `catboost-mlx` Python package can be cleanly built and installed as a
distribution artifact (sdist + wheel) without source-code changes, in preparation for a
future PyPI publish. This audit *does not publish anything* — it confirms the package is
publishable on demand.

## Method

1. `python -m build --sdist --wheel --outdir /tmp/cmlx-build/` in `python/`.
2. Audit sdist tarball contents for accidental inclusions (secrets, build artifacts,
   internal state, etc.).
3. Install the wheel into a fresh venv with only the runtime deps (numpy, mlx).
4. Run the README's 30-second smoke test against the wheel-installed package.

## Results

### Build artifacts

```
/tmp/cmlx-build/catboost_mlx-0.5.0.tar.gz                        (382 KB sdist)
/tmp/cmlx-build/catboost_mlx-0.5.0-cp313-cp313-macosx_26_0_arm64.whl  (421 KB wheel)
```

Both built successfully.

### Sdist content audit (47 files)

What's bundled:
- Source: `catboost_mlx/{__init__,core,pool,_predict_utils,_tree_utils,_utils,export_onnx,export_coreml}.py`, `py.typed`
- Compiled binaries: `catboost_mlx/bin/{csv_train,csv_predict}` (bundled per `MANIFEST.in:5`)
- Docs: `LICENSE`, `CHANGELOG.md`, `README.md`
- Packaging: `pyproject.toml`, `setup.py`, `setup.cfg`, `MANIFEST.in`, egg-info
- Tests: `tests/test_basic.py`

What's NOT bundled (verified):
- ❌ `.git/`, `.gitignore` — clean
- ❌ `.env`, secrets, credentials — clean
- ❌ `.claude/state/`, `docs/sprint*/` — clean
- ❌ `.cache/` — clean

**Verdict**: sdist is hygienic.

### Wheel install in fresh venv

```bash
python3 -m venv /tmp/cmlx-venv
/tmp/cmlx-venv/bin/pip install 'numpy>=1.26' 'mlx>=0.18'
/tmp/cmlx-venv/bin/pip install /tmp/cmlx-build/catboost_mlx-0.5.0-cp313-cp313-macosx_26_0_arm64.whl
```

Install completed cleanly. README smoke test against this venv:

```
catboost_mlx version: 0.5.0
OK — first 5 predictions: [-0.403 -2.7   -0.729  0.47  -0.34 ]
```

Output bit-identical to dev-install run. Wheel functions correctly.

## Findings (must-fix before PyPI publish)

### F1 — Wheel macOS deployment target tag is too high

The wheel filename is `catboost_mlx-0.5.0-cp313-cp313-macosx_26_0_arm64.whl`. The
`macosx_26_0_arm64` tag means pip will refuse to install the wheel on any macOS below
26.0 (Tahoe). Most M-series users today run macOS 14 (Sonoma) or 15 (Sequoia), so the
current wheel would be installable on a vanishingly small fraction of the target audience.

**Root cause**: the build host is on macOS 26.3 (Tahoe); without a pinned deployment
target, the linker stamps the wheel with the host's OS version. The build warned:

> [WARNING] This wheel needs a higher macOS version than the version your Python
> interpreter is compiled against. To silence this warning, set
> MACOSX_DEPLOYMENT_TARGET to at least 26_0 or recreate these files with lower
> MACOSX_DEPLOYMENT_TARGET

**Fix**: set `MACOSX_DEPLOYMENT_TARGET=14.0` in the build environment for production
wheels. The README states macOS 14.0+ as the minimum supported version (§ Prerequisites
and § Installation & Quick Start), so 14.0 is the correct floor.

```bash
export MACOSX_DEPLOYMENT_TARGET=14.0
python -m build --sdist --wheel --outdir dist/
# Should produce catboost_mlx-X.Y.Z-cp313-cp313-macosx_14_0_arm64.whl
```

This needs to apply to BOTH the `_core.cpython-*.so` nanobind extension built during
`pip install -e python/` AND the `csv_train` / `csv_predict` binaries built earlier. The
existing csv_train build commands in `catboost/mlx/README.md` § Installation should also
add `-mmacosx-version-min=14.0` to the `clang++` invocation for symmetry.

### F2 — Linux/non-arm64 install path missing

The wheel is correctly platform-tagged as `arm64` only. There is no fallback wheel for
`x86_64` or Linux (correct — MLX is Apple Silicon only). However, a user accidentally
running `pip install catboost-mlx` on Linux today would get an error message that may be
unclear about *why* it can't install. A `setup.py`-level platform guard or a clear
error-message override could improve UX. Defer to a future polish pass — not blocking.

## Findings (nice-to-have)

- Sdist is 382 KB — within reasonable PyPI limits (PyPI cap is 60 MB per file). Plenty
  of headroom.
- Wheel is 421 KB — same.
- Bundled binaries (`csv_train` 1.0 MB, `csv_predict` 1.0 MB approx) inflate the wheel
  but are necessary for end-user functionality without a separate compile step.
- Per-Python-version wheels would need to be built for cp310, cp311, cp312, cp313 to
  cover the supported Python range. Currently only cp313 is built (the host's Python).

## Verdict

**T4: GREEN with one must-fix gate before PyPI publish.**

The package is mechanically publishable today. The macOS deployment target tag (F1) is
the only blocker — without it, the wheel is installable on essentially zero target
machines.

**Recommended actions**:
1. Add `MACOSX_DEPLOYMENT_TARGET=14.0` to any future release-build script.
2. Add `-mmacosx-version-min=14.0` to the C++ binary build commands in the README and in
   any CI workflow that builds release artifacts.
3. Build per-Python-version wheels (cp310-cp313) on a build matrix when publishing.
4. Defer Linux-fallback error-message improvement to a future sprint.

Not committing any code changes from this audit — F1 is a build-environment fix
applied at PyPI-publish time, not a source change. The audit is recorded here for
when that publish step is taken.
