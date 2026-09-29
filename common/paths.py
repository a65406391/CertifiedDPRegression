"""Where experiment outputs go.

`get_repo_root` walks up to the directory holding pyproject.toml, so a script works from anywhere;
`get_output_dir` places ablation results under `ags/<dataset>/ablations/runs/<experiment>/`.

Most experiments still write to paths spelled out in their own modules, several of them date-stamped.
Routing those through here is unfinished work, noted in Notes/READABILITY_PLAN.md.
"""

from pathlib import Path


def get_repo_root() -> Path:
    """Find the root of the repository by looking for pyproject.toml."""
    # Start at the directory of this file and walk up
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent

    # Fallback to current working directory
    return Path.cwd()


def get_output_dir(dataset: str, experiment: str) -> str:
    """Return the output directory for a specific dataset and experiment type.

    Format: <repo_root>/ags/<dataset>/ablations/runs/<experiment>/
    """
    root = get_repo_root()
    out_dir = root / "ags" / dataset / "ablations" / "runs" / experiment
    return str(out_dir)
