"""Exclusive per-project lock so the CLI, Pi calls and job runners never lose updates."""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from genome_agent.state.models import STATE_DIR

LOCK_FILE = "lock"


@contextmanager
def project_lock(project_dir: Path) -> Iterator[None]:
    path = project_dir / STATE_DIR / LOCK_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
