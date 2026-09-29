"""Append-only event log: `.genome-agent/provenance.jsonl`.

state.json holds the current picture; this log holds the history, one JSON
object per line, and is never rewritten.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from genome_agent.state.models import STATE_DIR

PROVENANCE_FILE = "provenance.jsonl"


class ProvenanceLog:
    def __init__(self, project_dir: Path) -> None:
        self.path = project_dir / STATE_DIR / PROVENANCE_FILE

    def append(self, event: str, **data: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": datetime.now(UTC).isoformat(), "event": event, **data}
        with self.path.open("a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines()]
