"""Structured, persistent experiment state. The source of truth, not the chat log."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from genome_agent.resources.models import ResourceEstimate, ResourcePolicy, SystemResources

SCHEMA_VERSION = 1
STATE_DIR = ".genome-agent"
STATE_FILE = "state.json"


def _now() -> datetime:
    return datetime.now(UTC)


class ReadKind(StrEnum):
    HIFI = "pacbio_hifi"
    ONT = "ont"
    ILLUMINA = "illumina"
    HIC = "hic"
    RNASEQ = "rnaseq"
    REFERENCE = "reference_fasta"  # an assembly/reference used for checks, not reads
    OTHER = "other"


class BiologicalContext(BaseModel):
    species: str | None = None
    organism_type: str | None = None
    expected_ploidy: int | None = None
    genome_size_bp: int | None = None
    phasing_desired: bool = False
    annotation_desired: bool = False
    reference: str | None = None


class Dataset(BaseModel):
    path: str
    kind: ReadKind
    species: str | None = None  # as documented by the user; verify with read_origin_check
    size_bytes: int | None = None
    stats: dict[str, Any] = Field(default_factory=dict)


class JobStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class Job(BaseModel):
    """One requested tool run. Rejected requests are recorded too (benchmarking)."""

    id: str
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: list[str] = Field(default_factory=list)
    cpus: int
    ram_gb: float
    estimate: ResourceEstimate | None = None
    argv: list[str] = Field(default_factory=list)
    outdir: str | None = None
    runner_pid: int | None = None
    cancel_requested: bool = False
    enforcement: str | None = None  # OS limit backend used, None = advisory
    oom_killed: bool = False
    status: JobStatus = JobStatus.PLANNED
    rejection_reasons: list[str] = Field(default_factory=list)
    exit_code: int | None = None
    timed_out: bool = False
    error: str | None = None
    wall_time_s: float | None = None
    peak_rss_gb: float | None = None  # observed; compare with estimate.ram_gb
    stdout_path: str | None = None
    stderr_path: str | None = None
    created_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None


class Result(BaseModel):
    job_id: str
    kind: str
    path: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class DecisionRecord(BaseModel):
    decision: str
    reason: str
    timestamp: datetime = Field(default_factory=_now)
    evidence: dict[str, Any] = Field(default_factory=dict)
    resources: dict[str, Any] = Field(default_factory=dict)
    alternatives_considered: list[str] = Field(default_factory=list)
    actor: str = "harness"  # "harness", "planner:<name>", "llm:<model>"


class ProjectState(BaseModel):
    schema_version: int = SCHEMA_VERSION
    name: str
    created_at: datetime = Field(default_factory=_now)
    objective: str = ""
    biological_context: BiologicalContext = Field(default_factory=BiologicalContext)
    policy: ResourcePolicy = Field(default_factory=ResourcePolicy)
    system_resources: SystemResources | None = None
    datasets: list[Dataset] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    current_plan: list[str] = Field(default_factory=list)
    jobs: list[Job] = Field(default_factory=list)
    results: list[Result] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    decisions: list[DecisionRecord] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    # Tools the harness must refuse to run in this project (e.g. decision-only benchmarks)
    blocked_tools: list[str] = Field(default_factory=list)
    # With a registered reference, origin-sensitive tools (assemblers) only accept reads
    # whose read_origin_check verdict is "matches reference".
    require_origin_check: bool = True

    def origin_verdicts(self) -> dict[str, str]:
        """Latest read_origin_check verdict per reads file."""
        out: dict[str, str] = {}
        for r in self.results:
            if r.kind == "read_origin":
                for f, v in r.data.get("files", {}).items():
                    out[f] = v.get("verdict", "unknown")
        return out

    def has_reference(self) -> bool:
        return any(d.kind == ReadKind.REFERENCE for d in self.datasets)

    def measured_read_bases(self, paths: list[str]) -> int | None:
        """Total bases of `paths` if a read_stats result measured every one of them."""
        per_file: dict[str, int] = {}
        for r in self.results:
            if r.kind != "read_stats":
                continue
            for f, stats in r.data.get("files", {}).items():
                if isinstance(stats, dict) and isinstance(stats.get("sum_len"), int):
                    per_file[str(Path(f).resolve())] = stats["sum_len"]
        resolved = [str(Path(p).resolve()) for p in paths]
        if not resolved or any(p not in per_file for p in resolved):
            return None
        return sum(per_file[p] for p in resolved)

    @staticmethod
    def path_for(project_dir: Path) -> Path:
        return project_dir / STATE_DIR / STATE_FILE

    def save(self, project_dir: Path) -> Path:
        path = self.path_for(project_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(self.model_dump_json(indent=2))
        tmp.replace(path)  # atomic on POSIX: an interrupted save never corrupts state
        return path

    @classmethod
    def load(cls, project_dir: Path) -> ProjectState:
        path = cls.path_for(project_dir)
        if not path.exists():
            raise FileNotFoundError(f"no GenomeAgent project at {project_dir} (missing {path})")
        state = cls.model_validate_json(path.read_text())
        if state.schema_version != SCHEMA_VERSION:
            raise ValueError(f"state schema v{state.schema_version} != supported v{SCHEMA_VERSION}")
        return state
