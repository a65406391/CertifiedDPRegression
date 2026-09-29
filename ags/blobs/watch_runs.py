"""Live status across every certified run: position, solver health, ETA."""

import glob
import json
import os
import sys
import time


def tail(p):
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            n = f.tell()
            f.seek(max(0, n - 8192))
            ls = f.read().decode(errors="replace").splitlines()
        for line in reversed(ls):
            try:
                return json.loads(line)
            except Exception:
                continue
    except Exception:
        pass
    return None


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "ags/blobs/runs/certified"
    dirs = sorted(d for d in glob.glob(f"{root}/*") if os.path.isdir(d))
    print(
        f"{'run':<26}{'k':>3}{'steps':>12}{'solves':>15}{'%':>6}{'opt/TL/dual':>14}{'ETA':>10}{'finish':>10}"
    )
    print("-" * 96)
    for d in dirs:
        p = tail(f"{d}/progress.jsonl")
        man = os.path.exists(f"{d}/MANIFEST.json")
        if p is None:
            print(f"{os.path.basename(d):<26}{'-':>3}{'(no heartbeat)':>12}")
            continue
        e = p.get("eta") or {}
        c = p.get("counts", {})
        st = f"{p['steps']}/{p['expected_steps']}"
        sv = f"{p['solves']}/{p['expected_solves']}"
        pct = 100 * e.get("frac", 0)
        hz = f"{c.get('optimal', 0)}/{c.get('time_limit', 0)}/{c.get('dual_fallback', 0)}"
        eta = e.get("eta_s")
        etas = "done" if man else (f"{eta / 60:.0f}m" if eta else "-")
        fin = "" if man else (e.get("eta_clock") or "")
        age = time.time() - p.get("epoch", 0)
        stale = " STALE" if age > 90 and not man else ""  # heartbeat is 30 s
        print(
            f"{os.path.basename(d):<26}{str(p.get('k')):>3}{st:>12}{sv:>15}{pct:>5.1f}%{hz:>14}{etas:>10}{fin:>10}{stale}"
        )


if __name__ == "__main__":
    main()
