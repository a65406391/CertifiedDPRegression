"""The paper's IMDB figure: fig9 on the dense budget grid, fixed geometry shared with the MNIST twin.

Reads runs/contractive/headline_dense.json (written by `run_contractive release --task imdb_dense`) and
draws common.plotting.contractive_paper_figure with the logistic group's arms and the hidden-layer group's
AGS and matched arms as the two extra entries; the legend band holds four rows whatever the arm count, so
this figure and ags/mnist/plot_paper.py's have identical graphs as the two subfigures of one figure.

    PYTHONPATH=. python -m ags.imdb.plot_paper
"""

import argparse
import json

from ags.imdb.run_contractive import TASKS
from common import plotting


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", default="imdb_dense", choices=tuple(TASKS))
    ap.add_argument("--ylim", type=float, nargs=2, default=(0.45, 0.85))
    ap.add_argument("--no-hidden", action="store_true", help="leave out the hidden-layer arms")
    a = ap.parse_args()
    C = TASKS[a.task]
    Hd = json.load(open(f"{C['runs_dir']}/{C.get('headline_file', 'headline.json')}"))
    G = Hd["groups"]["logistic"]
    extra = None if a.no_hidden else Hd["groups"].get("mlp", {}).get("arms")
    stem = f"{C['figures_dir']}/{C['fig_stem']}" + ("_no_hidden" if a.no_hidden else "")
    plotting.contractive_paper_figure(
        G["arms"], Hd["eps"], stem, extra_arms=extra, non_private=G["non_private_line"], ylim=tuple(a.ylim)
    )
    print(f"wrote {stem}.png  [task={a.task}, eps<=100, non-private {G['non_private_line']:.3f}]")


if __name__ == "__main__":
    main()
