cmake_minimum_required(VERSION 3.27)
project(catboost_mlx_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Locate Python ─────────────────────────────────────────────────────────────
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

# ── Locate nanobind (installed via pip) ───────────────────────────────────────
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# ── Locate MLX ────────────────────────────────────────────────────────────────
# MLX_DIR is set by mlx.extension.CMakeBuild via os.environ["MLX_DIR"].
# The cmake config lives at $MLX_DIR/share/cmake/MLX/MLXConfig.cmake.
if(DEFINED ENV{MLX_DIR})
  set(MLX_DIR "$ENV{MLX_DIR}/share/cmake/MLX")
endif()
find_package(MLX REQUIRED)

# ── Apple frameworks ──────────────────────────────────────────────────────────
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

# ── CatBoost-MLX project root ────────────────────────────────────────────────
# When invoked by setup.py the source tree root is three levels above _core/.
if(NOT DEFINED CATBOOST_MLX_ROOT)
  get_filename_component(CATBOOST_MLX_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../../.."
    ABSOLUTE)
endif()

# ── Build the nanobind extension ──────────────────────────────────────────────
nanobind_add_module(
  _core
  NB_STATIC
  STABLE_ABI
  LTO
  "${CMAKE_CURRENT_SOURCE_DIR}/bindings.cpp"
  "${CATBOOST_MLX_ROOT}/catboost/mlx/train_api.cpp"
)

target_include_directories(_core PRIVATE
  "${CATBOOST_MLX_ROOT}"
)

target_link_libraries(_core PRIVATE
  mlx
  ${METAL_FRAMEWORK}
  ${FOUNDATION_FRAMEWORK}
)

# Suppress warnings from included csv_train.cpp
target_compile_options(_core PRIVATE
  -Wno-c++20-extensions
  -Wno-unused-function
)

# Debug leaf instrumentation (set -DCATBOOST_MLX_DEBUG_LEAF=ON to enable)
option(CATBOOST_MLX_DEBUG_LEAF "Enable leaf-magnitude debug probes (P1-P8)" OFF)
if(CATBOOST_MLX_DEBUG_LEAF)
  target_compile_definitions(_core PRIVATE CATBOOST_MLX_DEBUG_LEAF)
  message(STATUS "CATBOOST_MLX_DEBUG_LEAF probes enabled")
endif()

# macOS ARM64 deployment target
if(APPLE)
  set_target_properties(_core PROPERTIES
    OSX_DEPLOYMENT_TARGET "13.0"
  )
endif()

# Place the built .so alongside the Python package
set_target_properties(_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
)

# ── S46 Probe binaries (Route A: piggyback on working MLX find_package) ──────
# Each target is a standalone C++ executable that links mlx via the same
# find_package(MLX) path that builds _core.so successfully.
# Guards select the probe variant; production _core.so is NEVER modified.
#
# Build:
#   cmake --build <build_dir> --target bench_boosting_baseline
#   cmake --build <build_dir> --target bench_boosting_probe_b
#   cmake --build <build_dir> --target bench_boosting_probe_d
#
# Binaries land in: <build_dir>/

option(BUILD_S46_PROBES "Build S46 simd_shuffle probe binaries" OFF)

if(BUILD_S46_PROBES)
  set(_BENCH_SRC
    "${CATBOOST_MLX_ROOT}/catboost/mlx/tests/bench_boosting.cpp"
    "${CATBOOST_MLX_ROOT}/catboost/mlx/methods/histogram_t2_impl.cpp"
  )
  set(_BENCH_COMMON_OPTS -Wno-c++20-extensions -Wno-unused-function)

  # Baseline: no probe guard — matches production build flags exactly.
  add_executable(bench_boosting_baseline ${_BENCH_SRC})
  target_include_directories(bench_boosting_baseline PRIVATE "${CATBOOST_MLX_ROOT}")
  target_link_libraries(bench_boosting_baseline PRIVATE mlx ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
  target_compile_options(bench_boosting_baseline PRIVATE ${_BENCH_COMMON_OPTS})
  if(APPLE)
    set_target_properties(bench_boosting_baseline PROPERTIES OSX_DEPLOYMENT_TARGET "13.0")
  endif()

  # Probe B: hierarchical reduction (per-lane register accumulation + butterfly).
  # Guard: SIMD_SHUFFLE_PROBE_B in kernel_sources.h and bench_boosting.cpp.
  add_executable(bench_boosting_probe_b ${_BENCH_SRC})
  target_include_directories(bench_boosting_probe_b PRIVATE "${CATBOOST_MLX_ROOT}")
  target_link_libraries(bench_boosting_probe_b PRIVATE mlx ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
  target_compile_definitions(bench_boosting_probe_b PRIVATE SIMD_SHUFFLE_PROBE_B)
  target_compile_options(bench_boosting_probe_b PRIVATE ${_BENCH_COMMON_OPTS})
  if(APPLE)
    set_target_properties(bench_boosting_probe_b PROPERTIES OSX_DEPLOYMENT_TARGET "13.0")
  endif()

  # Probe D: Split-K D2 (K=4 parallel partial dispatches + merge kernel).
  # Guard: SIMD_SHUFFLE_PROBE_D in kernel_sources.h and bench_boosting.cpp.
  # D2 dispatcher rewrite applied: K slices concatenated, not mx::add chained.
  add_executable(bench_boosting_probe_d ${_BENCH_SRC})
  target_include_directories(bench_boosting_probe_d PRIVATE "${CATBOOST_MLX_ROOT}")
  target_link_libraries(bench_boosting_probe_d PRIVATE mlx ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
  target_compile_definitions(bench_boosting_probe_d PRIVATE SIMD_SHUFFLE_PROBE_D)
  target_compile_options(bench_boosting_probe_d PRIVATE ${_BENCH_COMMON_OPTS})
  if(APPLE)
    set_target_properties(bench_boosting_probe_d PROPERTIES OSX_DEPLOYMENT_TARGET "13.0")
  endif()

  message(STATUS "S46 probe binaries enabled: bench_boosting_baseline, bench_boosting_probe_b, bench_boosting_probe_d")
endif()
