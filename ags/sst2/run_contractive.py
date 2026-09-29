"""The SST-2 hybrid: the IMDB driver on the SST-2 loader, one task per encoder (ags/sst2/config.py).

PYTHONPATH=. python -m ags.sst2.run_contractive sweep --task sst2 --device cuda:0
PYTHONPATH=. python -m ags.sst2.run_contractive release --task sst2_ft_eps3
"""

from ags.imdb import run_contractive as driver
from ags.sst2.config import TASKS


def main():
    driver.main(tasks=TASKS, default_task="sst2")


if __name__ == "__main__":
    main()
