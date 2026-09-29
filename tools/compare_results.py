"""Compare two results JSONs numerically, ignoring key order and provenance fields.

    PYTHONPATH=. python -m tools.compare_results reference.json current.json

Used when consolidating shared code: a refactor that only moves an implementation must leave every
number identical. Prints the largest relative difference and exits non-zero if anything moved.
"""

from __future__ import annotations

import json
import math
import sys

IGNORE = {"generated", "git_sha", "host", "started", "finished", "seconds", "wall_time_s", "timestamp"}


def walk(a, b, path="", out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in IGNORE:
                continue
            if k not in a or k not in b:
                out.append((f"{path}.{k}", "missing on one side", None))
            else:
                walk(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, f"length {len(a)} vs {len(b)}", None))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]", out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if math.isnan(a) and math.isnan(b):
            return out
        scale = max(abs(a), abs(b), 1e-300)
        rel = abs(a - b) / scale
        if rel > 1e-12:
            out.append((path, f"{a!r} vs {b!r}", rel))
    elif a != b:
        out.append((path, f"{a!r} vs {b!r}", None))
    return out


def main() -> int:
    ref, cur = (json.load(open(p)) for p in sys.argv[1:3])
    diffs = walk(ref, cur)
    numeric = [d for d in diffs if d[2] is not None]
    if not diffs:
        print("identical (every number, ignoring provenance fields)")
        return 0
    print(f"{len(diffs)} differences, {len(numeric)} numeric")
    for path, what, rel in diffs[:25]:
        print(f"  {path}: {what}" + (f"   rel {rel:.3e}" if rel else ""))
    if len(diffs) > 25:
        print(f"  ... and {len(diffs) - 25} more")
    if numeric:
        print(f"largest relative difference: {max(d[2] for d in numeric):.3e}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
