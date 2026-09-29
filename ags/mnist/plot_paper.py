"""The paper's 5-class MNIST figure: fig8 on the dense budget grid, fixed geometry shared with the IMDB twin.

Reads runs/contractive5/headline_dense.json (written by `run_contractive release --task mnist5_dense`)
and draws common.plotting.contractive_paper_figure: pure and approximate DP up to eps = 100, the legend
in a two-column band of fixed height, so this figure and ags/imdb/plot_paper.py's have identical graphs
when placed as the two subfigures of a double-column figure.

    PYTHONPATH=. python -m ags.mnist.plot_paper
"""

import argparse
import json

from ags.mnist.run_contractive import TASKS
from common import plotting


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", default="mnist5_dense", choices=tuple(TASKS))
    ap.add_argument("--ylim", type=float, nargs=2, default=(0.15, 0.95))
    a = ap.parse_args()
    C = TASKS[a.task]
    H = json.load(open(f"{C['runs_dir']}/{C.get('headline_file', 'headline.json')}"))
    stem = f"{C['figures_dir']}/{C['fig_stem']}"
    plotting.contractive_paper_figure(
        H["arms"], H["eps"], stem, non_private=H["non_private_line"], ylim=tuple(a.ylim)
    )
    print(f"wrote {stem}.png  [task={a.task}, eps<=100, non-private {H['non_private_line']:.3f}]")


if __name__ == "__main__":
    main()
