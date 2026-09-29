"""Machine-wide record of estimated vs. observed resource use, and safe corrections.

Every finished job with a measured peak RSS is appended to
`$GENOME_AGENT_DATA_DIR/observations.jsonl` (default
`~/.local/share/genome-agent/`), shared by all projects on this machine.

Correction policy: upward only. If a tool with identical parameters has ever
exceeded its estimate, later estimates are multiplied by the worst observed
ratio. Estimates are never lowered from observations, because small (e.g. toy)
runs cannot justify extrapolating lower memory for large genomes. Lowering
needs a proper, input-size-aware calibration.
"""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from genome_agent.resources.models import ResourceEstimate

OBSERVATIONS_FILE = "observations.jsonl"
DATA_DIR_ENV = "GENOME_AGENT_DATA_DIR"


def default_data_dir() -> Path:
    env = os.environ.get(DATA_DIR_ENV)
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_DATA_HOME")
    return (Path(xdg) if xdg else Path.home() / ".local" / "share") / "genome-agent"


def params_key(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True)


class ResourceObservation(BaseModel):
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    host: str = Field(default_factory=platform.node)
    tool: str
    params: dict[str, Any]
    cpus: int
    input_bytes: int
    estimated_ram_gb: float
    peak_rss_gb: float
    wall_time_s: float | None
    status: str
    project: str
    job_id: str

    @property
    def ratio(self) -> float:
        return self.peak_rss_gb / self.estimated_ram_gb if self.estimated_ram_gb > 0 else 0.0


class ObservationStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_data_dir() / OBSERVATIONS_FILE

    def append(self, obs: ResourceObservation) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(obs.model_dump_json() + "\n")

    def load(self) -> list[ResourceObservation]:
        if not self.path.exists():
            return []
        return [
            ResourceObservation.model_validate_json(line)
            for line in self.path.read_text().splitlines()
            if line.strip()
        ]

    def matching(self, tool: str, params: dict[str, Any]) -> list[ResourceObservation]:
        key = params_key(params)
        return [o for o in self.load() if o.tool == tool and params_key(o.params) == key]

    def correct(self, est: ResourceEstimate, tool: str, params: dict[str, Any]) -> ResourceEstimate:
        """Raise `est.ram_gb` if identical past runs exceeded their estimates."""
        same = self.matching(tool, params)
        worst = max((o.ratio for o in same), default=0.0)
        if worst <= 1.0:
            return est
        n = len(same)
        return est.model_copy(
            update={
                # ceil to 0.01 GB: an upward correction must never round down
                "ram_gb": math.ceil(est.ram_gb * worst * 100) / 100,
                "basis": f"{est.basis}; x{worst:.2f} upward correction: a past run with "
                f"these params exceeded its estimate ({n} observations on this machine)",
            }
        )

    def report(self) -> list[dict[str, Any]]:
        """Per (tool, params): how estimates compared with reality."""
        groups: dict[tuple[str, str], list[ResourceObservation]] = {}
        for o in self.load():
            groups.setdefault((o.tool, params_key(o.params)), []).append(o)
        rows = []
        for (tool, key), obs in sorted(groups.items()):
            ratios = [o.ratio for o in obs]
            rows.append(
                {
                    "tool": tool,
                    "params": json.loads(key),
                    "runs": len(obs),
                    "median_peak_over_estimate": round(statistics.median(ratios), 3),
                    "max_peak_over_estimate": round(max(ratios), 3),
                    "underestimates": sum(r > 1.0 for r in ratios),
                    "max_peak_rss_gb": max(o.peak_rss_gb for o in obs),
                    "max_input_gb": round(max(o.input_bytes for o in obs) / 1e9, 3),
                }
            )
        return rows
