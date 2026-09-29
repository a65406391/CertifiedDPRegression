"""The hybrid figure (fig17): DP-LoRA fine-tuning of the encoder, then an AGS head, on SST-2.

Reads the fine-tuning results (runs/finetune/<tag>/results.json: the end-to-end arm's accuracy with the
model's own head) and the contractive headlines of every encoder task (runs/contractive_<tag>/headline.json:
the AGS head at every head budget, the DP-SGD head baselines on the frozen encoder). Computes nothing but the
choice of split: at a total budget eps the hybrid is the encoder fine-tuned at eps_1 with the AGS head at
eps_2 = eps - eps_1, the pair chosen on the public dev set among the fine-tuned encoders. Composition:
(eps_1, 5e-6) for the encoder plus (eps_2, 5e-6) for a Laplace head or (eps_2, 0) for a Cauchy head is
(eps, 1e-5)-DP.

Left: the (eps, 1e-5)-DP pipelines: the hybrid with a Laplace head, end-to-end DP fine-tuning, and the
frozen encoder with an AGS head or an optimized DP-SGD head. Right: the same with pure-DP heads (Cauchy
for AGS, Laplace with basic composition for DP-SGD); end-to-end fine-tuning has no pure variant and is
drawn dashed for reference. Bands: the non-private fine-tuned model and the non-private frozen head.
Third panel: the same four arms at a total budget of 3 against the private set size (2k, 10k and all 62,349
sentences; the hybrid's split chosen among eps_1 in {0.5, 1, 2}), drawn when the data-size tasks have run.

    PYTHONPATH=. python -m ags.sst2.plot_hybrid
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

from ags.sst2.config import FINETUNE_EPS, PANEL_EPS1, PANEL_N, SST2, TASKS, finetuned
from common import plotting
from common.plotting import C_NOMINAL

TOTALS = [0.5, 1, 2, 3, 5, 8]
PANEL_EPS = 3.0  # the data-size panel's total budget
N_ALL = 62349  # the private set: the 67,349 GLUE training sentences less the 5,000 held out as the test set
VARIANT = "tight/cross_radius"
# a softer palette than the paper's DP-SGD orange, AGS navy and optimized purple (28 Sept, the supervisor's
# request: thicker lines, larger text, easier on the eyes); the hybrid stays the darkest so it reads first
C_HYBRID = "#2f3b52"  # slate
C_E2E = "#e07b54"  # soft coral
C_FROZEN_AGS = "#4a86c8"  # soft blue
C_FROZEN_DPSGD = "#9b7bbf"  # soft purple
LW = 2.2  # line width in points; the paper's combined figures use about 1.3
OUT = f"{SST2['figures_dir']}/{SST2['fig_stem']}.png"
SUMMARY = "ags/sst2/runs/hybrid_summary.json"


def ft_result(eps1, n_private=None):
    """The fine-tuning results at eps1 (0: non-private); a data-size arm carries its own DP-SGD setting."""
    tag = finetuned(eps1, n_private)["encoder"]
    p = f"ags/sst2/runs/finetune/{tag}/results.json"
    return json.load(open(p)) if os.path.exists(p) else None


def headline(task):
    C = TASKS.get(task) or finetuned(float(task.split("eps")[1].split("_")[0]))
    p = f"{C['runs_dir']}/headline.json"
    return json.load(open(p))["groups"]["logistic"] if os.path.exists(p) else None


def row(G, mode, eps):
    """The headline row at head budget eps (keys are str(e), ints and floats alike)."""
    for k, v in G["arms"][mode].items():
        if abs(float(k) - eps) < 1e-9:
            return v
    return None


def hybrid(eps, mode, tasks_by_eps1):
    """Best (eps_1, eps_2) on the public set: the AGS head on the encoder fine-tuned at eps_1, at eps_2."""
    best = None
    for eps1, G in tasks_by_eps1.items():
        eps2 = round(eps - eps1, 6)
        if eps2 <= 0 or G is None:
            continue
        r = row(G, mode, eps2)
        if r is None or r["ags"].get(VARIANT) is None:
            continue
        cell = r["ags"][VARIANT]
        cand = dict(
            eps1=eps1, eps2=eps2, select=cell["select"], test=cell["test"], mechanism=cell["mechanism"]
        )
        if best is None or cand["select"] > best["select"]:
            best = cand
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    frozen = headline("sst2")
    fts = {e: headline(f"sst2_ft_eps{e:g}") for e in FINETUNE_EPS}
    nonpriv = ft_result(0)
    S = dict(totals=TOTALS, arms={}, splits={})
    for mode in ("approx", "pure"):
        arms = S["arms"][mode] = dict(hybrid=[], end_to_end=[], frozen_ags=[], frozen_dpsgd=[])
        for eps in TOTALS:
            h = hybrid(eps, mode, fts)
            arms["hybrid"].append(h["test"] if h else None)
            S["splits"].setdefault(mode, {})[str(eps)] = h
            e2e = ft_result(eps)
            arms["end_to_end"].append(e2e["test"] if e2e else None)
            r = row(frozen, mode, eps) if frozen else None
            arms["frozen_ags"].append(r["ags"][VARIANT]["test"] if r and r["ags"].get(VARIANT) else None)
            o = r["dpsgd_optimized"] if r else None
            arms["frozen_dpsgd"].append(o["test"] if o else None)
    S["non_private"] = dict(
        fine_tuned=nonpriv["test"] if nonpriv else None,
        frozen_head=frozen["non_private_line"] if frozen else None,
    )
    P = S["panel"] = dict(
        eps=PANEL_EPS, n=[], hybrid=[], end_to_end=[], frozen_ags=[], frozen_dpsgd=[], splits=[]
    )
    for n in [*PANEL_N, None]:
        frz = frozen if n is None else headline(f"sst2_n{n}")
        fts_n = fts if n is None else {e: headline(f"sst2_ft_eps{e:g}_n{n}") for e in PANEL_EPS1}
        h = hybrid(PANEL_EPS, "approx", fts_n)
        e2e = ft_result(PANEL_EPS, n)
        r = row(frz, "approx", PANEL_EPS) if frz else None
        P["n"].append(n or N_ALL)
        P["hybrid"].append(h["test"] if h else None)
        P["splits"].append(h)
        P["end_to_end"].append(e2e["test"] if e2e else None)
        P["frozen_ags"].append(r["ags"][VARIANT]["test"] if r and r["ags"].get(VARIANT) else None)
        P["frozen_dpsgd"].append(r["dpsgd_optimized"]["test"] if r and r.get("dpsgd_optimized") else None)
    with_panel = any(v is not None for k in ("hybrid", "end_to_end", "frozen_ags") for v in P[k][:-1])
    os.makedirs(os.path.dirname(SUMMARY), exist_ok=True)
    json.dump(S, open(SUMMARY, "w"), indent=1)

    n_panels = 3 if with_panel else 2
    figsize = (7.16, 3.5) if with_panel else (7.0, 3.6)
    plotting.apply_combined(
        n_panels,
        figsize=figsize,
        page_width_in=7.16,
        print_target_pt=10,
        label_pt=10.5,
        tick_pt=9,
        legend_pt=9.5,
    )
    plt.rcParams["axes.titlesize"] = plt.rcParams["axes.labelsize"] * 1.3  # the titles read first
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, sharey=True, layout="constrained")
    fig.get_layout_engine().set(h_pad=0.08)
    handles = {}
    mk = dict(markeredgecolor="white", markeredgewidth=0.9, ms=7, lw=LW)
    panels = [
        (axes[0], "approx", r"Approx. $(\varepsilon, 10^{-5})$-DP", TOTALS, S["arms"]["approx"]),
        (axes[1], "pure", r"Pure-DP Head", TOTALS, S["arms"]["pure"]),
    ]
    if with_panel:
        panels.append((axes[2], "panel", r"Private Set Size at $(3, 10^{-5})$-DP", P["n"], P))
    for ax, mode, title, xvals, arms in panels:

        def draw(key, label, **sty):
            xs = [x for x, v in zip(xvals, arms[key]) if v is not None]
            ys = [v for v in arms[key] if v is not None]
            if xs:
                (h,) = ax.plot(xs, ys, **{**mk, **sty})
                handles.setdefault(label, h)

        draw(
            "hybrid",
            "hybrid: DP-LoRA encoder + AGS head",
            color=C_HYBRID,
            marker="*",
            ls="-",
            ms=11,
            zorder=6,
        )
        if mode in ("approx", "panel"):
            draw("end_to_end", "end-to-end DP-LoRA fine-tuning", color=C_E2E, marker="s", ls="-")
        else:
            xs = [e for e, v in zip(TOTALS, S["arms"]["approx"]["end_to_end"]) if v is not None]
            ys = [v for v in S["arms"]["approx"]["end_to_end"] if v is not None]
            if xs:
                ax.plot(xs, ys, color=C_E2E, marker="s", ls="--", alpha=0.6, **mk)
        draw("frozen_ags", "frozen encoder + AGS head", color=C_FROZEN_AGS, marker="o", ls="-")
        draw(
            "frozen_dpsgd",
            "frozen encoder + DP-SGD head (optimized)",
            color=C_FROZEN_DPSGD,
            marker="P",
            ls="-",
        )
        if S["non_private"]["fine_tuned"] is not None:
            b = ax.axhline(
                S["non_private"]["fine_tuned"], color=C_NOMINAL, ls="-", lw=7, alpha=0.28, zorder=1
            )
            handles.setdefault("non-private fine-tuning", b)
        if S["non_private"]["frozen_head"] is not None:
            l = ax.axhline(S["non_private"]["frozen_head"], color=C_NOMINAL, ls=":", lw=1.8, zorder=1)
            handles.setdefault("non-private frozen head", l)
        ax.set_xscale("log")
        ax.set_xticks(xvals)
        if mode == "panel":
            ax.set_xticklabels([f"{n // 1000}k" for n in xvals])
            ax.set_xlabel(r"Private Sentences $N$")
        else:
            ax.set_xticklabels([f"{e:g}" for e in xvals])
            ax.set_xlabel(r"Privacy Budget $\varepsilon$")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(title, pad=6)
    axes[0].set_ylabel("Test Accuracy")
    fig.legend(
        list(handles.values()),
        list(handles.keys()),
        loc="outside lower center",
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handlelength=1.8,
        handletextpad=0.5,
    )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out)
    plt.close(fig)
    print(f"wrote {a.out}; summary {SUMMARY}")
    for mode in ("approx", "pure"):
        print(
            f"  {mode}: "
            + "  ".join(
                f"eps={e}: " + ("-" if v is None else f"{v:.3f}")
                for e, v in zip(TOTALS, S["arms"][mode]["hybrid"])
            )
        )
    if with_panel:
        print(
            "  panel: "
            + "  ".join(f"N={n}: " + ("-" if v is None else f"{v:.3f}") for n, v in zip(P["n"], P["hybrid"]))
        )


if __name__ == "__main__":
    main()
