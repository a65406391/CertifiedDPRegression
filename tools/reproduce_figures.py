"""Check that every paper figure can still be regenerated from stored results.

    PYTHONPATH=. python -m tools.reproduce_figures --check      # the gate: run producers, compare bytes
    PYTHONPATH=. python -m tools.reproduce_figures --list       # registry + figures with no live producer
    PYTHONPATH=. python -m tools.reproduce_figures --check -k blobs   # only producers matching a substring

Each producer below reads stored results and writes figures; none of them retrains or re-solves, so
the check is fast and needs no GPU or solver. A producer marked `heavy` does recompute and is skipped
unless --heavy is passed.

The gate fails if a producer errors, or if it stops writing a file it used to write. Byte differences
are reported but do not fail: matplotlib versions change PNG bytes without changing the picture, so
a difference is a prompt to look, not a verdict. `--strict` turns differences into failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WATCHED = ("agtr", "ags")  # figure trees scanned for writes
BASELINE = Path(__file__).with_name("figure_baseline.json")


@dataclass
class Producer:
    name: str
    module: str
    args: list[str] = field(default_factory=list)
    kind: str = "current"  # current | legacy | superseded
    heavy: bool = False  # retrains or re-solves; skipped unless --heavy
    note: str = ""


PRODUCERS = [
    # --- AGS, current ---------------------------------------------------------------------
    Producer("ags.blobs.headline", "ags.blobs.plot_headline", note="fig1, fig2"),
    Producer("ags.blobs.approx", "ags.blobs.plot_approx", note="fig3"),
    Producer("ags.blobs.combined", "ags.blobs.plot_combined", note="fig4"),
    Producer(
        "ags.blobs.tight",
        "ags.blobs.release_tight",
        ["--plot-only"],
        note="fig7: blobs re-released under the proven rules, from the stored envelopes",
    ),
    # the same three under the cross-radius certificate; combined reads the two JSONs written above it
    Producer(
        "ags.blobs.headline.cross_radius",
        "ags.blobs.plot_headline",
        ["--certificate", "cross_radius"],
        note="fig1 under the cross-radius certificate (fig2 needs boxes the sweep did not store)",
    ),
    Producer(
        "ags.blobs.approx.cross_radius",
        "ags.blobs.plot_approx",
        ["--certificate", "cross_radius"],
        note="fig3 under the cross-radius certificate",
    ),
    Producer(
        "ags.blobs.combined.cross_radius",
        "ags.blobs.plot_combined",
        ["--certificate", "cross_radius"],
        note="fig6: fig4 under the cross-radius certificate",
    ),
    # --- MNIST contractive comparison: draws from runs/contractive*/headline*.json only ---------------
    Producer(
        "ags.mnist.contractive10",
        "ags.mnist.plot_contractive",
        ["--task", "mnist10"],
        note="fig7: 10 classes, full data",
    ),
    Producer(
        "ags.mnist.contractive10.sweep",
        "ags.mnist.plot_contractive",
        ["--task", "mnist10", "--sweep"],
        note="fig7 on the 10k sweep subset",
    ),
    Producer(
        "ags.mnist.contractive10.dimension_free",
        "ags.mnist.plot_contractive",
        ["--task", "mnist10", "--beta-rule", "dimension_free"],
        note="fig7 under Corollary 4.1 as written",
    ),
    Producer(
        "ags.mnist.contractive10.width",
        "ags.mnist.plot_contractive",
        ["--task", "mnist10", "--certificate", "width"],
        note="fig7 with the reference width certificate",
    ),
    Producer(
        "ags.mnist.contractive10.aggregate",
        "ags.mnist.plot_contractive",
        ["--task", "mnist10", "--beta-rule", "aggregate"],
        note="fig7 under the earlier beta/P rule with the reference constants",
    ),
    Producer(
        "ags.mnist.contractive10.corollary_p",
        "ags.mnist.plot_contractive",
        ["--task", "mnist10", "--beta-rule", "corollary_p"],
        note="fig7 with Cauchy(6u/eps) at beta = eps/(6P)",
    ),
    Producer(
        "ags.mnist.contractive5",
        "ags.mnist.plot_contractive",
        ["--task", "mnist5"],
        note="fig8: digits 0-4, full data",
    ),
    # --- IMDB contractive comparison on sentence-encoder features: draws from runs/contractive/headline.json only
    Producer(
        "ags.imdb.contractive",
        "ags.imdb.plot_contractive",
        note="fig9: logistic and hidden-layer heads on mpnet features, cold start",
    ),
    Producer(
        "ags.imdb.contractive.warm",
        "ags.imdb.plot_contractive",
        ["--group", "warm"],
        note="fig9 warm-started at the public head",
    ),
    Producer(
        "ags.imdb.contractive.long",
        "ags.imdb.plot_contractive",
        ["--group", "long"],
        note="fig9 on the T = 3000 trajectory",
    ),
    Producer(
        "ags.imdb.contractive.dimension_free",
        "ags.imdb.plot_contractive",
        ["--beta-rule", "dimension_free"],
        note="fig9 under Corollary 4.1 as written",
    ),
    Producer(
        "ags.imdb.contractive.n2k",
        "ags.imdb.plot_contractive",
        ["--task", "imdb_n2k"],
        note="fig9 on 2k private reviews: the b-scaling",
    ),
    Producer(
        "ags.imdb.contractive.n2k.warm",
        "ags.imdb.plot_contractive",
        ["--task", "imdb_n2k", "--group", "warm"],
        note="fig9 on 2k private reviews, warm-started",
    ),
    Producer(
        "ags.imdb.contractive.width",
        "ags.imdb.plot_contractive",
        ["--certificate", "width"],
        note="fig9 with the reference width certificate",
    ),
    Producer(
        "ags.imdb.contractive.aggregate",
        "ags.imdb.plot_contractive",
        ["--beta-rule", "aggregate"],
        note="fig9 under the earlier beta/P rule",
    ),
    Producer(
        "ags.imdb.contractive.corollary_p",
        "ags.imdb.plot_contractive",
        ["--beta-rule", "corollary_p"],
        note="fig9 with Cauchy(6u/eps) at beta = eps/(6P)",
    ),
    Producer(
        "ags.mnist.paper5",
        "ags.mnist.plot_paper",
        note="fig8 for the paper: dense budgets to 100, fixed geometry shared with ags.imdb.paper",
    ),
    Producer(
        "ags.paper.combined",
        "ags.plot_paper_combined",
        note="the paper's AGS figure: 5-class MNIST and IMDB, pure and approx DP, in one row",
    ),
    Producer(
        "ags.paper.combined.no_hidden",
        "ags.plot_paper_combined",
        ["--no-hidden"],
        note="the same without IMDB's hidden-layer arms",
    ),
    Producer(
        "ags.imdb.paper",
        "ags.imdb.plot_paper",
        note="fig9 for the paper: dense budgets to 100, the twin subfigure of ags.mnist.paper5",
    ),
    Producer(
        "ags.adult.contractive",
        "ags.adult.plot_contractive",
        note="fig10: AGS alone on Adult, d = 16 (P = 34), against the non-private line",
    ),
    Producer(
        "ags.adult.contractive.d32",
        "ags.adult.plot_contractive",
        ["--task", "adult_d32"],
        note="fig10 at d = 32 (P = 66)",
    ),
    Producer(
        "ags.adult.windows",
        "ags.adult.plot_windows",
        note="fig11: the sampling-frequency ablation at d = 16 (n releases, eps / n each)",
    ),
    Producer(
        "ags.adult.windows.d32",
        "ags.adult.plot_windows",
        ["--task", "adult_d32"],
        note="fig11 at d = 32",
    ),
    Producer(
        "ags.adult.budget",
        "ags.adult.plot_budget",
        note="fig14: the per-release budget at a fixed (eps, 1e-5), basic against optimal composition",
    ),
    Producer(
        "ags.adult.windows.accounting_twin",
        "ags.adult.plot_windows",
        ["--variant", "accounting_twin"],
        note="fig16: fig15 with the per-release budget and its gain on one twin-axis panel",
    ),
    Producer(
        "ags.adult.windows.accounting_approx",
        "ags.adult.plot_windows",
        ["--variant", "accounting_approx"],
        note="fig15: approx DP to 2^8 releases, optimal against basic composition, with the budget and its gain",
    ),
    Producer(
        "ags.adult.windows.modes",
        "ags.adult.plot_windows",
        ["--variant", "modes"],
        note="fig13: the sampling ablation at d = 16, pure DP left and approximate DP right, one line per eps",
    ),
    Producer(
        "ags.adult.windows.accounting",
        "ags.adult.plot_windows",
        ["--variant", "accounting"],
        note="fig12: the ablation's accuracy beside the composed (eps', delta) guarantee of the n releases, d = 16",
    ),
    Producer(
        "ags.adult.windows.accounting.d32",
        "ags.adult.plot_windows",
        ["--task", "adult_d32", "--variant", "accounting"],
        note="fig12 at d = 32",
    ),
    Producer(
        "ags.sst2.hybrid",
        "ags.sst2.plot_hybrid",
        note="fig17: the hybrid on SST-2 (DP-LoRA encoder + AGS head) against end-to-end DP-LoRA and the frozen heads",
    ),
    Producer(
        "ags.mnist.contractive5.sweep",
        "ags.mnist.plot_contractive",
        ["--task", "mnist5", "--sweep"],
        note="fig8 on the 10k sweep subset",
    ),
    # --- MNIST optimized comparison: draws from runs/optimized/optimized_headline.json only ----------
    Producer(
        "ags.mnist.contractive5.aggregate",
        "ags.mnist.plot_contractive",
        ["--task", "mnist5", "--beta-rule", "aggregate"],
        note="fig8 under the earlier beta/P rule",
    ),
    Producer("ags.mnist.optimized", "ags.mnist.plot_optimized", note="fig5: sound whole-vector AGS release"),
    Producer(
        "ags.mnist.optimized.dimension_free",
        "ags.mnist.plot_optimized",
        ["--beta-rule", "dimension_free"],
        note="fig5 under Corollary 4.1 as written",
    ),
    Producer(
        "ags.mnist.optimized.pld",
        "ags.mnist.plot_optimized",
        ["--beta-rule", "pld_coordinate"],
        note="fig5, per-coordinate Laplace composed by PLD",
    ),
    Producer(
        "ags.mnist.optimized.width",
        "ags.mnist.plot_optimized",
        ["--certificate", "width"],
        note="fig5 with the reference width certificate",
    ),
    Producer("ags.mnist.headline", "ags.mnist.plot_headline", note="fig1, fig2"),
    Producer("ags.mnist.repro", "ags.mnist.plot_reproduce_paper", note="fig3, the P=650 correction"),
    # --- AGT-R, current -------------------------------------------------------------------
    Producer("agtr.linear.release", "agtr.linear.plot", note="cross-radius release"),
    Producer(
        "agtr.california.release.norm", "agtr.california.release", ["--variant", "norm_gamma1", "--plot-only"]
    ),
    Producer(
        "agtr.california.release.clamp",
        "agtr.california.release",
        ["--variant", "clamp_gamma1", "--plot-only"],
    ),
    Producer(
        "agtr.california.release.relu",
        "agtr.california.release",
        ["--variant", "relu_h64_clamp_gamma0.1", "--plot-only"],
        note="the working California configuration",
    ),
    Producer(
        "agtr.california.repro",
        "agtr.california.repro.repro_figure2",
        note="reproduction of the published Figure 2",
    ),
    Producer("agtr.condition_table", "agtr.condition_table", ["--combined"], note="Thm 4.3/4.4 feasibility"),
    Producer(
        "agtr.condition_table.paper",
        "agtr.condition_table",
        ["--paper-style"],
        note="the paper's version of the feasibility figure, (b) on 1 - alpha >= 0.8",
    ),
    Producer(
        "agtr.condition_table.paper_h80",
        "agtr.condition_table",
        ["--paper-style", "--height-scale", "0.8"],
        note="the same, 20 % lower (the supervisor asked for 30 %; 70 % was too squashed)",
    ),
    Producer(
        "agtr.pate.paper",
        "agtr.plot_pate_paper",
        note="the paper's AGT-R figure: linear and California matched PATE comparisons in one row",
    ),
    Producer(
        "agtr.pate.paper.no_gs", "agtr.plot_pate_paper", ["--no-gs"], note="the same without the GS arm"
    ),
    Producer(
        "agtr.pate.paper.no_gs_linear",
        "agtr.plot_pate_paper",
        ["--no-gs-linear"],
        note="the same with the GS arm left out of the Linear panels only (California keeps it)",
    ),
    Producer(
        "agtr.pate.paper.excess_linear",
        "agtr.plot_pate_paper",
        ["--excess-linear"],
        note="the same with the Linear panels as error above the non-private one and without the GS arm",
    ),
    Producer(
        "agtr.pate.paper.excess",
        "agtr.plot_pate_paper",
        ["--excess"],
        note="the same with every panel as error above the non-private one and without the GS arm",
    ),
    Producer(
        "agtr.ablation.paper",
        "agtr.plot_ablation_paper",
        note="the paper's AGT-R ablation row: Linear concretisation and California batch size, trade-off panels only",
    ),
    Producer(
        "agtr.pate.paper.two_row",
        "agtr.plot_pate_paper",
        ["--two-row"],
        note="the full figure as two rows of two panels (one dataset per row), for one column",
    ),
    Producer(
        "agtr.certificate.paper",
        "agtr.plot_certificate_paper",
        note="the appendix's certificate figure: the staircase panels of both ablations in one row",
    ),
    Producer(
        "agtr.pate.paper.linear_only",
        "agtr.plot_pate_paper",
        ["--group", "Linear"],
        note="the Linear pair of the combined figure on its own, every arm drawn",
    ),
    # --- matched PATE, current ------------------------------------------------------------
    Producer(
        "agtr.linear.pate.n4000",
        "agtr.linear.pate.release",
        ["--tag", "exact_n4000_e20_lr0.3_g1", "--plot-only"],
    ),
    Producer(
        "agtr.linear.pate.n40000",
        "agtr.linear.pate.release",
        ["--tag", "exact_n40000_e20_lr0.3_g1", "--plot-only"],
    ),
    Producer("agtr.california.pate", "agtr.california.pate.release", ["--plot-only"]),
]

# AGS ablation families, driven through the shared dispatcher. Only the pairs that have stored
# results. Halfmoons was archived in September 2026, the GPT-2 module removed on 25 September.
_ABLATIONS = {
    "ags_comparison": ("mnist",),
    "approx_dp": ("mnist",),
    "sampling_ablation": ("mnist",),
    "hybrid_ablation": ("mnist",),
}
for _family, _datasets in _ABLATIONS.items():
    for _ds in _datasets:
        PRODUCERS.append(
            Producer(
                f"ags.{_ds}.ablations.{_family}",
                "common.figures",
                [_family, _ds],
                note="reads ags/ablations/runs",
            )
        )

# Legacy producers: they still run, but they draw figures under the retired width/radius and
# fixed-beta conventions (see Notes/REPO_NOTES.md). Registered so the gate notices if they break,
# and so their figures are not mistaken for orphans.
PRODUCERS += [
    Producer("legacy.agtr.linear.baselines", "agtr.legacy.linear.plot_baselines", kind="legacy"),
    Producer("legacy.agtr.california.baselines", "agtr.legacy.california.plot_baselines", kind="legacy"),
    Producer("legacy.agtr.linear.ablation", "agtr.legacy.linear.plot_ablation", kind="legacy"),
    Producer("legacy.agtr.california.ablation", "agtr.legacy.california.plot_ablation", kind="legacy"),
    Producer("legacy.agtr.linear.combined", "agtr.legacy.linear.make_combined", kind="legacy"),
    Producer(
        "legacy.agtr.linear.data_ablation", "agtr.linear.original.plot_lin_data_ablation", kind="legacy"
    ),
    Producer("legacy.agtr.linear.dim_ablation", "agtr.linear.original.plot_lin_dim_ablation", kind="legacy"),
    Producer(
        "legacy.ags.accounting.status",
        "ags.accounting.plot_overnight_status",
        kind="legacy",
        note="one-shot report, date in the output path",
    ),
    Producer(
        "legacy.agtr.linear.final_figures",
        "agtr.linear.original.plot_final_linear_figures",
        kind="legacy",
        heavy=True,
        note="retrains before plotting",
    ),
    Producer(
        "legacy.agtr.linear.benchmark",
        "agtr.linear.original.plot_linear_benchmark",
        kind="legacy",
        heavy=True,
        note="retrains before plotting",
    ),
]


def figure_files() -> dict[Path, float]:
    out = {}
    for tree in WATCHED:
        for p in (ROOT / tree).rglob("*"):
            if p.is_file() and p.suffix == ".png" and "original" not in p.parts and "archive" not in p.parts:
                out[p] = p.stat().st_mtime
    return out


def sha(p: Path) -> str:
    """Content hash of a PNG (figures are PNG only; a PDF would carry its creation time and never
    share bytes with an identical picture)."""
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def committed_sha(p: Path) -> str | None:
    rel = p.relative_to(ROOT).as_posix()
    r = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True)
    if r.returncode != 0:
        return None
    return hashlib.sha256(r.stdout).hexdigest()[:12]


def run(pr: Producer, timeout: int) -> tuple[bool, str, list[Path]]:
    before = figure_files()
    stamp = time.time()
    time.sleep(0.01)
    env = dict(os.environ, PYTHONPATH=str(ROOT), MPLBACKEND="Agg")
    r = subprocess.run(
        [sys.executable, "-m", pr.module, *pr.args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    written = [
        p for p, m in figure_files().items() if p not in before or m > stamp or p.stat().st_mtime > stamp
    ]
    if r.returncode != 0:
        tail = (r.stderr.strip().splitlines() or ["(no stderr)"])[-1]
        return False, tail[:160], written
    return True, "", sorted(set(written))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="run producers and compare (the gate)")
    ap.add_argument(
        "--save-baseline",
        action="store_true",
        help="run producers and record their output hashes as the reference",
    )
    ap.add_argument("--list", action="store_true", help="registry and orphan figures")
    ap.add_argument("--heavy", action="store_true", help="also run producers that recompute")
    ap.add_argument("--strict", action="store_true", help="byte differences fail the gate")
    ap.add_argument("-k", dest="filter", default="", help="only producers matching this substring")
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    if not (a.check or a.list or a.save_baseline):
        ap.print_help()
        return 0

    selected = [p for p in PRODUCERS if a.filter in p.name and (a.heavy or not p.heavy)]

    if a.list:
        print(f"{len(PRODUCERS)} producers ({sum(p.heavy for p in PRODUCERS)} heavy)\n")
        for p in PRODUCERS:
            print(f"  {p.kind:11s} {p.name:38s} {p.module} {' '.join(p.args)}")
            if p.note:
                print(f"{'':13s}{'':38s} # {p.note}")
        return 0

    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    fresh_baseline: dict[str, str] = {}
    print(f"running {len(selected)} producers\n")
    failed, differed, new, produced = [], [], [], set()
    for p in selected:
        ok, err, written = run(p, a.timeout)
        produced |= set(written)
        if not ok:
            failed.append(p.name)
            print(f"  FAIL  {p.name:38s} {err}")
            continue
        regressed, unseen = [], []
        for w in written:
            rel = w.relative_to(ROOT).as_posix()
            h = sha(w)
            fresh_baseline[rel] = h
            if rel not in baseline:
                unseen.append(rel)
            elif baseline[rel] != h:
                regressed.append(rel)
        differed += regressed
        new += unseen
        bits = [f"{len(written) - len(regressed) - len(unseen)} match"]
        if regressed:
            bits.append(f"{len(regressed)} CHANGED")
        if unseen:
            bits.append(f"{len(unseen)} new")
        print(f"  ok    {p.name:38s} {len(written):2d} files, {', '.join(bits)}")

    orphans = sorted(
        {p.relative_to(ROOT).as_posix() for p in figure_files() if p.suffix == ".png"}
        - {p.relative_to(ROOT).as_posix() for p in produced}
    )
    if a.save_baseline:
        BASELINE.write_text(json.dumps(dict(sorted(fresh_baseline.items())), indent=1) + "\n")
        print(f"\nbaseline written: {len(fresh_baseline)} files -> {BASELINE.relative_to(ROOT)}")
    print(
        f"\n{len(selected) - len(failed)}/{len(selected)} producers ran; "
        f"{len(produced)} files written; {len(differed)} changed vs baseline; "
        f"{len(new)} not in baseline; "
        f"{len(orphans)} figures have no producer in the registry"
    )
    if orphans:
        print("\nfigures with no live producer (legacy, or a producer is missing from the registry):")
        for o in orphans[:40]:
            print(f"  {o}")
        if len(orphans) > 40:
            print(f"  ... and {len(orphans) - 40} more")
    if differed:
        print("\nCHANGED since the baseline - a refactor altered these:")
        for d in differed[:40]:
            print(f"  {d}")
        if len(differed) > 40:
            print(f"  ... and {len(differed) - 40} more")
    return 1 if failed or (a.strict and differed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
