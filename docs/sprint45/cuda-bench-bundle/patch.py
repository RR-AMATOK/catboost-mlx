"""Apply the Windows compatibility fix to scripts/_runner_common.py.

The original _runner_common.py imports the POSIX-only `resource` stdlib
module unconditionally; this fails on Windows with ModuleNotFoundError.
This script rewrites the import as a try/except and adds a psutil-based
fallback for peak_rss_bytes(). Idempotent: re-running on a patched file
prints "Already patched; nothing to do." and exits cleanly.

Run from the cuda-bench-bundle/ directory: python patch.py
"""
import re
from pathlib import Path

p = Path("scripts") / "_runner_common.py"
if not p.exists():
    raise SystemExit(
        f"Cannot find {p}. Run this from cuda-bench-bundle/ directory."
    )

s = p.read_text(encoding="utf-8")

if "_HAS_RESOURCE" in s:
    print("Already patched; nothing to do.")
    raise SystemExit(0)

# Fix 1: replace `import resource` with a try/except wrapper
s = s.replace(
    "import resource\n",
    "try:\n    import resource\n    _HAS_RESOURCE = True\n"
    "except ImportError:\n    resource = None\n    _HAS_RESOURCE = False\n",
    1,
)

# Fix 2: replace peak_rss_bytes() with a Windows-friendly version
new_fn = '''def peak_rss_bytes() -> int:
    """Peak RSS in bytes. Falls back to psutil on Windows; 0 if neither."""
    if _HAS_RESOURCE:
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(raw)
        return int(raw) * 1024
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except ImportError:
        return 0'''

s = re.sub(
    r"def peak_rss_bytes\(\) -> int:.*?return int\(raw\) \* 1024",
    new_fn,
    s,
    count=1,
    flags=re.DOTALL,
)

p.write_text(s, encoding="utf-8")

# Sanity check
verify = p.read_text(encoding="utf-8")
if "_HAS_RESOURCE" not in verify:
    raise SystemExit("FAILED: patch did not apply (marker not found)")
print("PATCHED: scripts/_runner_common.py")
print("Now run: python run.py")
