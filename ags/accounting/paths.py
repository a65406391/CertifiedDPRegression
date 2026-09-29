"""Where each dataset keeps its accounting artifacts.

Accounting experiments live with their dataset, `ags/<dataset>/accounting/`, and write immutable
JSON into an `artifacts/` folder beside them. The few scripts here that read across datasets, the
overnight summaries and the status plotter, resolve a name through `artifact()`; every stored name
already carries its dataset as a prefix, so nothing else is needed.

Outputs that belong to no single dataset, the cross-dataset summaries and the accountant audit,
stay in `ags/accounting/artifacts/`.
"""

from pathlib import Path

from common.paths import get_repo_root

ROOT = get_repo_root()
BLOBS = ROOT / "ags" / "blobs" / "accounting" / "artifacts"
MNIST = ROOT / "ags" / "mnist" / "accounting" / "artifacts"
SHARED = ROOT / "ags" / "accounting" / "artifacts"

_MNIST_PREFIXES = ("pca_", "logistic_", "mnist_cnn_", "compact_cnn_", "lenet_", "smoke_", "INVALIDATED_")


def artifact(name: str) -> Path:
    """The stored artifact `name`, wherever its dataset keeps it."""
    if name.startswith("blobs_"):
        return BLOBS / name
    if name.startswith(_MNIST_PREFIXES):
        return MNIST / name
    return SHARED / name
