"""Record hardware identification into hardware.txt.

Captures GPU model, CUDA driver version, VRAM, CPU model, RAM, OS, Python
version. Writes to ./hardware.txt next to this script. Cross-shell:
runs identically from cmd, PowerShell, Windows Terminal, Git Bash, etc.

Run: python record_hardware.py
"""
from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path


def section(title: str, body: str) -> str:
    return f"=== {title} ===\n{body.rstrip()}\n\n"


def safe_run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=30
        )
        if out.returncode == 0:
            return out.stdout
        return f"(exit {out.returncode})\nstdout: {out.stdout}\nstderr: {out.stderr}"
    except FileNotFoundError:
        return f"(command not found: {cmd[0]})"
    except subprocess.TimeoutExpired:
        return "(timeout)"
    except Exception as exc:
        return f"(error: {exc})"


def main() -> int:
    out_path = Path(__file__).resolve().parent / "hardware.txt"
    parts = []

    parts.append(section("Python", f"{sys.version}\nplatform: {platform.platform()}\nmachine: {platform.machine()}"))

    # GPU + CUDA
    parts.append(section("nvidia-smi", safe_run(["nvidia-smi"])))
    parts.append(section("nvcc --version", safe_run(["nvcc", "--version"])))

    # CPU
    if sys.platform == "win32":
        parts.append(section("CPU (wmic)", safe_run(["wmic", "cpu", "get", "name"])))
        parts.append(section("RAM (wmic)", safe_run(["wmic", "computersystem", "get", "totalphysicalmemory"])))
    elif sys.platform == "darwin":
        parts.append(section("CPU", safe_run(["sysctl", "-n", "machdep.cpu.brand_string"])))
        parts.append(section("RAM", safe_run(["sysctl", "-n", "hw.memsize"])))
    else:
        parts.append(section("CPU", safe_run(["lscpu"])))
        parts.append(section("RAM", safe_run(["cat", "/proc/meminfo"])))

    # CatBoost
    try:
        import catboost
        parts.append(section("CatBoost", f"version: {catboost.__version__}"))
    except ImportError:
        parts.append(section("CatBoost", "not importable; pip install catboost first"))

    out_path.write_text("".join(parts))
    print(f"Wrote {out_path}")
    print(f"  {out_path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
