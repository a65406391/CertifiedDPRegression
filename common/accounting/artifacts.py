"""Immutable JSON output.

`write_immutable_json` writes canonically sorted JSON, rejects NaN, sets the file read-only and
refuses to overwrite an existing path, returning the SHA-256 of what it wrote.

Results are evidence. A rerun that silently replaced an artifact would make every number in the
notes unverifiable, so a rerun must choose a new filename instead.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


def write_immutable_json(path: Path, payload: dict[str, object]) -> str:
    """Create, never overwrite, a canonical JSON artifact and return SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return digest
