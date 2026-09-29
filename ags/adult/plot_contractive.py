"""The Adult contractive figure (fig10): ags/imdb/plot_contractive.py on the Adult headline.

PYTHONPATH=. python -m ags.adult.plot_contractive [--group warm] [--beta-rule aggregate] [--certificate width]
"""

from ags.adult.run_contractive import TASKS
from ags.imdb import plot_contractive as plotter


def main():
    plotter.main(tasks=TASKS, default_task="adult")


if __name__ == "__main__":
    main()
