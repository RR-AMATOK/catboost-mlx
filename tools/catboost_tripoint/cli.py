"""
cli.py -- catboost-tripoint command-line entry point.

Usage
-----
    catboost-tripoint verify \\
        --model path/to/model.json \\
        --data  path/to/eval.parquet \\
        [--target-col target] \\
        [--backends cpu,mlx,cuda] \\
        [--batch-size 1000] \\
        [--report-json path/to/out.json]

Exit codes
----------
0  PASS   -- all pairwise prediction diffs within derived fp32 floor
1  FAIL   -- at least one pairwise diff exceeds the derived floor
2  ERROR  -- model load, data load, or other setup failure
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys

from . import __version__
from .verifier import VerifyResult, verify


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_TICK = "OK"
_CROSS = "NO"


def _fmt_avail(ok: bool) -> str:
    return "AVAILABLE" if ok else "NOT AVAILABLE"


def _fmt_ratio(wall: float, ref_wall: float) -> str:
    if ref_wall <= 0:
        return ""
    ratio = wall / ref_wall
    return f"  ({ratio:.1f}x CPU)"


def _print_summary(result: VerifyResult, *, file=sys.stdout) -> None:
    """Print the human-readable verification summary table."""
    p = lambda *a, **kw: print(*a, file=file, **kw)

    p(f"\ncatboost-tripoint v{__version__} -- cross-platform CatBoost verification")
    p()

    # Model metadata
    p(f"Model: {result.model_path}")
    leaves_est = 2 ** result.max_depth
    p(f"  Loss type:        {result.loss_type}")
    p(f"  Tree count:       {result.tree_count:,}")
    p(f"  Max depth:        {result.max_depth}")
    p(f"  Estimated leaves: {result.tree_count * leaves_est:,}")
    p()

    # Backend availability
    p("Backends tested:")
    ordered = ["cpu", "mlx", "cuda"]
    for key in ordered:
        if key not in result.backends:
            continue
        br = result.backends[key]
        tag = f"{_fmt_avail(br.available)}"
        status = "OK" if br.available else "NO"
        p(f"  {br.name:<25} [{status}] {tag}")
        if br.error:
            p(f"    note: {br.error}")
    p()

    # Data metadata
    p(f"Data: {result.data_path}")
    p(f"  Rows:     {result.n_rows:,}")
    p(f"  Features: {result.n_features}")
    p()

    # Prediction timings
    p("Predictions:")
    cpu_wall = result.backends.get("cpu", None)
    ref_wall = cpu_wall.wall_seconds if cpu_wall and cpu_wall.available else 0.0
    for key in ordered:
        if key not in result.backends:
            continue
        br = result.backends[key]
        if not br.available:
            p(f"  {br.name:<25} NOT AVAILABLE")
            continue
        ratio_str = ""
        if key != "cpu" and ref_wall > 0:
            ratio_str = _fmt_ratio(br.wall_seconds, ref_wall)
        p(f"  {br.name:<25} {result.n_rows:,} rows in {br.wall_seconds:.2f}s{ratio_str}")
    p()

    # Pairwise agreement
    p("Cross-backend agreement (max-abs prediction diff):")
    if not result.agreement:
        p("  No pairs to compare (need >= 2 available backends).")
    for pair in result.agreement:
        floor_tag = "within fp32 floor" if pair.within_floor else "EXCEEDS fp32 floor"
        above_str = f"  ({pair.n_diff_above_floor} rows above floor)" if pair.n_diff_above_floor > 0 else ""
        p(f"  {pair.backend_a} vs {pair.backend_b}:")
        p(f"    max-abs diff = {pair.max_abs_diff:.3e}   <- {floor_tag}{above_str}")
        p(f"    mean-abs diff = {pair.mean_abs_diff:.3e}")
    p()

    # Theoretical floor breakdown
    fi = result.floor_info
    p("Theoretical fp32 floor (eps_mach * T * sqrt(L)):")
    p(f"  eps_mach = {fi['epsilon_machine']:.2e}")
    p(f"  T (trees) = {fi['tree_count']}")
    p(f"  sqrt(L) = sqrt({fi['max_leaves']}) = {fi['sqrt_leaves']:.1f}")
    p(f"  Floor = {fi['epsilon_machine']:.2e} x {fi['tree_count']} x {fi['sqrt_leaves']:.1f}"
      f" = {fi['derived_floor']:.2e}")
    p(f"  Formula: {fi['formula']}")
    p()

    # Verdict
    p(f"Verdict: {result.verdict}")
    p()


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def _build_json_report(result: VerifyResult) -> dict:
    backends_out = {}
    for key, br in result.backends.items():
        backends_out[key] = {
            "name": br.name,
            "version": br.version,
            "available": br.available,
            "wall_seconds": br.wall_seconds,
            "predictions_sha256": br.predictions_sha256,
            "error": br.error,
        }

    agreement_out = {}
    for pair in result.agreement:
        label = f"{pair.backend_a.split('-')[-1]}_vs_{pair.backend_b.split('-')[-1]}"
        agreement_out[label] = {
            "max_abs_diff": pair.max_abs_diff,
            "mean_abs_diff": pair.mean_abs_diff,
            "n_diff_above_floor": pair.n_diff_above_floor,
            "within_floor": pair.within_floor,
        }

    fi = result.floor_info
    return {
        "version": __version__,
        "tool": "catboost-tripoint",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "model_metadata": {
            "path": result.model_path,
            "loss_type": result.loss_type,
            "tree_count": result.tree_count,
            "max_depth": result.max_depth,
            "estimated_leaves_total": result.tree_count * (2 ** result.max_depth),
        },
        "data_metadata": {
            "path": result.data_path,
            "n_rows": result.n_rows,
            "n_features": result.n_features,
        },
        "backends": backends_out,
        "agreement": agreement_out,
        "theoretical_floor": {
            "epsilon_machine": fi["epsilon_machine"],
            "tree_count": fi["tree_count"],
            "max_depth": fi["max_depth"],
            "max_leaves": fi["max_leaves"],
            "sqrt_leaves": fi["sqrt_leaves"],
            "derived_floor": fi["derived_floor"],
            "formula": fi["formula"],
        },
        "verdict": result.verdict,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_backends(raw: str) -> list[str]:
    """Parse a comma-separated backends string into a list, normalising names."""
    mapping = {
        "cpu": "cpu",
        "mlx": "mlx",
        "cuda": "cuda",
        "gpu": "cuda",
    }
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    out = []
    for p in parts:
        if p not in mapping:
            raise argparse.ArgumentTypeError(
                f"Unknown backend {p!r}. Choose from: cpu, mlx, cuda."
            )
        out.append(mapping[p])
    return out


def _cmd_verify(args: argparse.Namespace) -> int:
    """Execute the verify sub-command.  Returns exit code."""
    try:
        backends = _parse_backends(args.backends)
    except argparse.ArgumentTypeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        result = verify(
            model_path=args.model,
            data_path=args.data,
            target_col=args.target_col,
            backends=backends,
            batch_size=args.batch_size,
        )
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR (unexpected): {exc}", file=sys.stderr)
        return 2

    _print_summary(result)

    if args.report_json:
        report = _build_json_report(result)
        with open(args.report_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to: {args.report_json}")

    if result.verdict == "PASS":
        return 0
    elif result.verdict == "FAIL":
        return 1
    else:
        return 2


def main() -> None:
    """Entry point for the catboost-tripoint CLI."""
    parser = argparse.ArgumentParser(
        prog="catboost-tripoint",
        description="Cross-platform CatBoost prediction parity oracle.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    vp = sub.add_parser("verify", help="Verify cross-backend prediction agreement.")
    vp.add_argument("--model", required=True, metavar="PATH",
                    help="Path to model file (.json for catboost-mlx, .cbm for catboost CPU/CUDA).")
    vp.add_argument("--data", required=True, metavar="PATH",
                    help="Path to evaluation data (.parquet or .csv).")
    vp.add_argument("--target-col", default=None, metavar="COL",
                    help="Column name to drop from features (label column).")
    vp.add_argument("--backends", default="cpu,mlx", metavar="LIST",
                    help="Comma-separated list of backends to test: cpu,mlx,cuda. "
                         "Default: cpu,mlx. CUDA is tested only if catboost GPU is available.")
    vp.add_argument("--batch-size", type=int, default=1000, metavar="N",
                    help="Prediction batch size (default: 1000). Sketch uses full-batch.")
    vp.add_argument("--report-json", default=None, metavar="PATH",
                    help="Write structured JSON verification report to this path.")

    args = parser.parse_args()

    if args.command == "verify":
        sys.exit(_cmd_verify(args))
    else:
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
