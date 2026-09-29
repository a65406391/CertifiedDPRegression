"""What every stored result should say about where it came from.

Results are compared across months and machines, so a results file that cannot say which commit,
which interpreter and which package versions produced it is evidence of nothing. `stamp()` returns
the block to store beside the numbers; `tools/compare_results.py` ignores these fields when it
checks a re-run, so adding them never breaks a comparison.

    from common.provenance import stamp
    payload = {"results": rows, "provenance": stamp(config=cfg)}
"""

from __future__ import annotations

import datetime as dt
import platform
import subprocess
import sys


def git_sha() -> str:
    """The commit the code was run from, or "unknown" outside a checkout."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    """True if the working tree had uncommitted changes, so the sha alone does not pin the code."""
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except Exception:
        return False


def stamp(config: dict | None = None) -> dict:
    """The provenance block: commit, cleanliness, time, interpreter, key package versions, config."""
    versions = {}
    for name in ("numpy", "torch", "scipy", "gurobipy"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            pass
    return {
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "versions": versions,
        "config": config,
    }
