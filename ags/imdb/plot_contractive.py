"""The IMDB contractive figure: pure and approximate DP side by side, one shared legend.

Reads runs/contractive/headline.json (written by run_contractive.py release) and computes nothing.
The main arms come from the logistic head's group: non-private, PATE, DP-SGD (matched; in approx DP
both basic and exact Gaussian composition), DP-SGD (optimized), AGS. The hidden-layer head's group adds
two arms, AGS and matched DP-SGD on the same head. `--group warm` draws the warm-started groups
instead, with the public head's accuracy as a second reference line. The AGS variant is chosen as in
ags/mnist/plot_contractive.py:

    --certificate cross_radius | width
    --beta-rule aggregate | dimension_free

The default (cold, cross_radius, aggregate) is the sound whole-vector release and writes the stem
named in config; other variants carry a suffix.
"""

import argparse
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

from ags.imdb.run_contractive import TASKS
from common import plotting


def main(tasks=None, default_task="imdb"):
    tasks = TASKS if tasks is None else tasks
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", choices=tuple(tasks), default=default_task)
    ap.add_argument(
        "--group",
        choices=("cold", "warm", "long"),
        default="cold",
        help="cold: logistic + hidden-layer arms; warm: the warm-started groups; long: the T = 3000 trajectory",
    )
    ap.add_argument("--certificate", choices=("cross_radius", "width"), default="cross_radius")
    ap.add_argument(
        "--beta-rule",
        choices=("tight", "aggregate", "corollary_p", "dimension_free"),
        default="tight",
        dest="beta_rule",
        help="tight (default): the tightest proven release; aggregate: the earlier beta/P rule with the reference constants; corollary_p: Cauchy(6u/eps) at beta = eps/(6P); dimension_free: Corollary 4.1 as written, not established for P > 1",
    )
    ap.add_argument("--width", type=float, default=6.7)
    ap.add_argument("--height", type=float, default=5.2)
    ap.add_argument("--page-width", type=float, default=plotting.COLUMN_WIDTH_IN, dest="page_width")
    ap.add_argument(
        "--ncol", type=int, default=None, help="legend columns (default 3 with the hidden-layer arms, else 4)"
    )
    a = ap.parse_args()
    C = tasks[a.task]

    Hd = json.load(open(f"{C['runs_dir']}/{C.get('headline_file', 'headline.json')}"))
    eps = Hd["eps"]
    suffix = {"cold": "", "warm": "_warm", "long": "_long"}[a.group]
    main_group = Hd["groups"].get("logistic" + suffix)
    extra_group = Hd["groups"].get("mlp" + suffix)  # None for the long trajectory
    if main_group is None:
        raise SystemExit(f"headline.json has no {'logistic' + suffix} group")
    variant = f"{a.beta_rule}/{a.certificate}"
    stem = (
        C["fig_stem"]
        + suffix
        + ("" if a.certificate == "cross_radius" else "_width")
        + ("" if a.beta_rule == "tight" else f"_{a.beta_rule}")
    )

    figsize = (a.width, a.height)
    plotting.apply_combined(2, figsize=figsize, page_width_in=a.page_width)
    from common.plotting import (
        BLUES,
        C_DPSGD_COMPOSED,
        C_DPSGD_MATCHED,
        C_DPSGD_OPT,
        C_NOMINAL as C_NOM,
        C_PATE,
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True, layout="constrained")
    handles = {}
    pub = main_group.get("public_head")
    same = pub is not None and abs(pub["test"] - main_group["non_private_line"]) < 0.01
    NP_LABEL = "non-private (= public head)" if same else "non-private"

    def series(G, mode, getter):
        out = []
        for e in eps:
            try:
                v = getter(G["arms"][mode][str(e)])
            except (KeyError, TypeError):
                v = None
            out.append(v["test"] if v else float("nan"))
        return out

    def panel(ax, mode, title):
        arms = [
            (
                "AGS",
                series(main_group, mode, lambda r: r["ags"][variant]),
                dict(color=BLUES[-1], marker="o", ls="-"),
            ),
            (
                "DP-SGD (matched, basic)",
                series(main_group, mode, lambda r: r["dpsgd_matched"]["basic"]),
                dict(color=C_DPSGD_MATCHED, marker="s", ls="--"),
            ),
        ]
        if mode == "approx":
            arms.append(
                (
                    "DP-SGD (matched, composed)",
                    series(main_group, mode, lambda r: r["dpsgd_matched"]["composed"]),
                    dict(color=C_DPSGD_COMPOSED, marker="D", ls=(0, (1, 1.2)), lw=2.2),
                )
            )
        arms.append(
            (
                "DP-SGD (optimized)",
                series(main_group, mode, lambda r: r["dpsgd_optimized"]),
                dict(color=C_DPSGD_OPT, marker="P", ls="-"),
            )
        )
        arms.append(
            ("PATE", series(main_group, mode, lambda r: r["pate"]), dict(color=C_PATE, marker="^", ls="-"))
        )
        if extra_group is not None:
            arms.append(
                (
                    "AGS (hidden layer)",
                    series(extra_group, mode, lambda r: r["ags"][variant]),
                    dict(color=BLUES[-3], marker="v", ls="-."),
                )
            )
            arms.append(
                (
                    "DP-SGD (matched, hidden layer)",
                    series(extra_group, mode, lambda r: r["dpsgd_matched"]["basic"]),
                    dict(color=C_DPSGD_MATCHED, marker="v", ls="-.", alpha=0.75),
                )
            )
        for lab, y, sty in arms:
            (h,) = ax.plot(
                eps,
                y,
                zorder=6 if lab.startswith("AGS") else 5,
                markeredgecolor="white",
                markeredgewidth=1.1,
                **sty,
            )
            if any(v == v for v in y):  # an arm with no rows at all (Adult: AGS alone) gets no legend entry
                handles.setdefault(lab, h)
        band = ax.axhline(main_group["non_private_line"], color=C_NOM, ls="-", lw=7, alpha=0.28, zorder=1)
        handles.setdefault(NP_LABEL, band)
        if main_group.get("public_head") and not same:
            pub_line = ax.axhline(main_group["public_head"]["test"], color=C_NOM, ls="--", lw=1.4, zorder=2)
            handles.setdefault("public head", pub_line)
        ax.set_xscale("log")
        ax.set_xlim(0.9, 1100)
        ax.set_ylim(0.4, 1.0)
        ax.set_xticks([1, 10, 100, 1000])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel(r"Privacy Budget  $\varepsilon$")
        ax.set_title(title, pad=8)

    panel(axes[0], "pure", r"Pure $(\varepsilon, 0)$-DP")
    panel(axes[1], "approx", r"Approx. $(\varepsilon, \delta = 10^{-5})$-DP")
    axes[0].set_ylabel("Test Accuracy")
    fs = plt.rcParams["legend.fontsize"]
    order = [
        "AGS",
        "DP-SGD (matched, basic)",
        "DP-SGD (matched, composed)",
        "DP-SGD (optimized)",
        "PATE",
        "AGS (hidden layer)",
        "DP-SGD (matched, hidden layer)",
        NP_LABEL,
        "public head",
    ]
    order = [k for k in order if k in handles]
    ncol = a.ncol if a.ncol is not None else 3
    fig.legend(
        [handles[k] for k in order],
        order,
        loc="outside lower center",
        ncol=ncol,
        **plotting.LEGEND_BELOW_KW,
    )
    fig.savefig(f"{C['figures_dir']}/{stem}.png")
    plt.close(fig)
    print(
        f"wrote {C['figures_dir']}/{stem}.png  [task={a.task}, group={a.group}, certificate={a.certificate}, beta rule={a.beta_rule}]"
    )


if __name__ == "__main__":
    main()
