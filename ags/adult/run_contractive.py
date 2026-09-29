"""The Adult contractive comparison: the IMDB driver on the Adult loader.

Stages, flags and outputs are those of ags/imdb/run_contractive.py (sweep, baselines --group, release;
resumable; several sweep workers share each configuration's radii); the data come from
ags/adult/shared.py through the config's `loader` hook. Settings: ags/adult/config.py.

    PYTHONPATH=. python -m ags.adult.run_contractive sweep --device cuda:0
    PYTHONPATH=. python -m ags.adult.run_contractive all --task smoke     # every path, minutes
"""

from ags.adult.config import ADULT, ADULT_D32, ADULT_SMOKE
from ags.imdb import run_contractive as driver

TASKS = {"adult": ADULT, "adult_d32": ADULT_D32, "smoke": ADULT_SMOKE}


def main():
    driver.main(tasks=TASKS, default_task="adult")


if __name__ == "__main__":
    main()
