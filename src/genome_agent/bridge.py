"""JSON tool protocol for external agent runtimes (Pi, or anything else).

    genome-agent tool <operation> --project DIR --actor "llm:provider/model" < args.json

stdout is always one JSON object: {"ok": true, "result": ...} or
{"ok": false, "error": "..."}. The caller supplies `--actor`; the model cannot
set it through the arguments. Arguments are validated with strict Pydantic
models, and anything that executes goes through Harness.run_tool.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from genome_agent.agent.simulation import load_registry
from genome_agent.executor.validation import DEFAULT_TIMEOUT_S, JobRequest
from genome_agent.harness import RUNS_DIR, Harness
from genome_agent.provenance.log import ProvenanceLog
from genome_agent.state.models import Dataset, DecisionRecord, JobStatus, ProjectState, ReadKind
from genome_agent.tools.feasibility import assess
from genome_agent.tools.registry import ToolInputs

MAX_LOG_LINES = 2000
DEFAULT_WAIT_S = 300.0
MAX_WAIT_S = 1800.0


class BridgeError(Exception):
    """An expected, model-facing error (bad arguments, missing file...)."""


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoArgs(_Args):
    pass


class AssessArgs(_Args):
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: list[str] | None = None  # default: all project datasets
    genome_size_bp: int | None = None  # default: biological context


class RunArgs(AssessArgs):
    wait_s: float = Field(
        default=DEFAULT_WAIT_S,
        ge=0,
        le=MAX_WAIT_S,
        description="seconds to wait for completion; the job keeps running after that",
    )
    cpus: int = Field(ge=1)
    ram_gb: float = Field(gt=0)
    timeout_s: float = Field(default=DEFAULT_TIMEOUT_S, gt=0)
    reason: str = Field(min_length=1)
    alternatives_considered: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)


class PlanSpec(_Args):
    """A run the agent intends (or would) launch; typed so plans are comparable."""

    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: list[str]
    cpus: int = Field(ge=1)
    ram_gb: float = Field(gt=0)


class DecisionArgs(_Args):
    decision: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    evidence: dict[str, Any] = Field(default_factory=dict)
    alternatives_considered: list[str] = Field(default_factory=list)
    plan: PlanSpec | None = None

    @model_validator(mode="after")
    def _plans_are_typed(self) -> DecisionArgs:
        if self.decision.endswith("_plan") and self.plan is None:
            raise ValueError(
                f"'{self.decision}' decisions must include the typed plan field "
                "(tool, params, inputs, cpus, ram_gb) so the harness can check it"
            )
        return self


class DatasetArgs(_Args):
    path: str
    kind: ReadKind
    species: str | None = None


class JobArgs(_Args):
    job_id: str


class WaitArgs(JobArgs):
    timeout_s: float = Field(default=DEFAULT_WAIT_S, gt=0, le=MAX_WAIT_S)


class LogArgs(_Args):
    job_id: str
    stream: Literal["stderr", "stdout"] = "stderr"
    tail_lines: int = Field(default=100, ge=1, le=MAX_LOG_LINES)


def _defaults(h: Harness, a: AssessArgs) -> tuple[list[str], int | None]:
    """Explicit inputs, else the registered datasets of kinds the tool accepts."""
    if a.inputs is not None:
        inputs = a.inputs
    else:
        try:
            adapter = h.registry.get(a.tool)
        except KeyError as exc:
            raise BridgeError(str(exc.args[0])) from None
        inputs = adapter.select_inputs({d.path: str(d.kind) for d in h.state.datasets})
    if not inputs:
        raise BridgeError(
            f"no inputs given and no registered dataset suits {a.tool} (use add_dataset)"
        )
    return inputs, a.genome_size_bp or h.state.effective_genome_size()[0]


def _budget(h: Harness) -> dict[str, Any]:
    """Policy ceiling, plus what is free now after running jobs' reservations."""
    assert h.state.system_resources is not None
    free, running = h.available_budget()
    return h.state.policy.apply(h.state.system_resources).model_dump() | {
        "free_now": free.model_dump(),
        "reserved_by_running_jobs": running,
    }


def inspect_system(h: Harness, _: NoArgs, actor: str) -> dict[str, Any]:
    res = h.state.system_resources
    assert res is not None
    return {
        "system_resources": res.model_dump(mode="json"),
        "policy": h.state.policy.model_dump(),
        "budget": _budget(h),
    }


def list_tools(h: Harness, _: NoArgs, actor: str) -> dict[str, Any]:
    res = h.state.system_resources
    assert res is not None
    return {
        "tools": [a.describe() | {"available": a.is_available(res)} for a in h.registry.adapters()],
        "note": "Order is the configured preference order. Only available tools can run.",
    }


def project_status(h: Harness, _: NoArgs, actor: str) -> dict[str, Any]:
    s = h.state
    return {
        "name": s.name,
        "objective": s.objective,
        "biological_context": s.biological_context.model_dump(),
        "genome_size_in_use": dict(zip(("bp", "source"), s.effective_genome_size(), strict=True)),
        "datasets": [d.model_dump() for d in s.datasets],
        "blocked_tools": s.blocked_tools,
        "budget": _budget(h),
        "jobs": [
            j.model_dump(
                mode="json",
                include={
                    "id",
                    "tool",
                    "status",
                    "exit_code",
                    "error",
                    "rejection_reasons",
                    "wall_time_s",
                    "cpus",
                    "ram_gb",
                    "peak_rss_gb",
                    "outdir",
                },
            )
            for j in s.jobs
        ],
        "results": [r.model_dump() for r in s.results],
        "failures": s.failures,
        "decisions": [
            d.model_dump(mode="json", include={"decision", "actor", "reason"})
            for d in s.decisions[-10:]
        ],
        "metrics": s.metrics,
    }


def assess_tool(h: Harness, a: AssessArgs, actor: str) -> dict[str, Any]:
    try:
        adapter = h.registry.get(a.tool)
    except KeyError as exc:
        raise BridgeError(str(exc.args[0])) from None
    assert h.state.system_resources is not None
    inputs, genome_size = _defaults(h, a)
    missing = [p for p in inputs if not Path(p).is_file()]
    if missing:
        raise BridgeError(f"inputs not found: {missing}")
    try:
        params = adapter.parse_params(a.params)
    except ValidationError as exc:
        raise BridgeError(f"invalid params for {a.tool}: {exc.errors()}") from None
    tool_inputs = ToolInputs.from_files(
        [Path(p) for p in inputs],
        {d.path: str(d.kind) for d in h.state.datasets},
        genome_size_bp=genome_size,
        read_bases=h.state.measured_read_bases(inputs),
    )
    budget, _ = h.available_budget()
    if problems := adapter.check_inputs(tool_inputs):
        raise BridgeError("; ".join(problems))
    result = assess(adapter, tool_inputs, params, budget, h.observations)
    return result.model_dump() | {
        "available": adapter.is_available(h.state.system_resources),
        "blocked_in_project": a.tool in h.state.blocked_tools,
    }


def _job_view(h: Harness, job_id: str) -> dict[str, Any]:
    job = h.job(job_id)
    view: dict[str, Any] = {
        "job": job.model_dump(mode="json", exclude={"params", "created_at", "finished_at"}),
        "results": [r.model_dump() for r in h.state.results if r.job_id == job.id],
    }
    if job.status == JobStatus.RUNNING:
        view["hint"] = "still running: call wait_job or job_status later, or cancel_job"
    return view


def run_tool(h: Harness, a: RunArgs, actor: str) -> dict[str, Any]:
    inputs, genome_size = _defaults(h, a)
    job = h.run_tool(
        JobRequest(
            **a.model_dump(exclude={"inputs", "genome_size_bp", "wait_s"}),
            inputs=inputs,
            genome_size_bp=genome_size,
            actor=actor,
        ),
        wait_s=a.wait_s,
    )
    return _job_view(h, job.id)


def _known_job(h: Harness, job_id: str) -> None:
    try:
        h.job(job_id)
    except KeyError as exc:
        raise BridgeError(str(exc.args[0])) from None


def job_status(h: Harness, a: JobArgs, actor: str) -> dict[str, Any]:
    _known_job(h, a.job_id)
    return _job_view(h, a.job_id)


def wait_job(h: Harness, a: WaitArgs, actor: str) -> dict[str, Any]:
    _known_job(h, a.job_id)
    h.wait_job(a.job_id, a.timeout_s)
    return _job_view(h, a.job_id)


def cancel_job(h: Harness, a: JobArgs, actor: str) -> dict[str, Any]:
    _known_job(h, a.job_id)
    job = h.cancel_job(a.job_id)
    h.provenance.append("job_cancel_requested", job_id=job.id, actor=actor)
    return _job_view(h, a.job_id)


def record_decision(h: Harness, a: DecisionArgs, actor: str) -> dict[str, Any]:
    evidence = dict(a.evidence)
    if a.plan:
        req = JobRequest(
            **a.plan.model_dump(),
            genome_size_bp=h.state.effective_genome_size()[0],
            reason=a.reason,
            actor=actor,
        )
        if problems := h.check_plan(req):
            raise BridgeError(
                "plan rejected by the harness (the same rules as run_tool): " + "; ".join(problems)
            )
        evidence |= {"plan": a.plan.model_dump(), "plan_checked_by_harness": True}
    h.record_decision(
        DecisionRecord(
            decision=a.decision,
            reason=a.reason,
            evidence=evidence,
            alternatives_considered=a.alternatives_considered,
            actor=actor,
        )
    )
    return {"recorded": a.decision, "total_decisions": len(h.state.decisions)}


def add_dataset(h: Harness, a: DatasetArgs, actor: str) -> dict[str, Any]:
    p = Path(a.path).expanduser().resolve()
    if not p.is_file():
        raise BridgeError(f"not a file: {p}")
    with h.transaction() as state:
        if str(p) in {d.path for d in state.datasets}:
            raise BridgeError(f"dataset already registered: {p}")
        state.datasets.append(
            Dataset(path=str(p), kind=a.kind, species=a.species, size_bytes=p.stat().st_size)
        )
    h.record_decision(
        DecisionRecord(
            decision="add_dataset",
            reason=f"registered {a.kind} dataset" + (f" ({a.species})" if a.species else ""),
            evidence={"path": str(p)},
            actor=actor,
        )
    )
    return {"datasets": [d.model_dump() for d in h.state.datasets]}


def read_job_log(h: Harness, a: LogArgs, actor: str) -> dict[str, Any]:
    """Tail of a job's stdout/stderr. Only logs recorded for this project's jobs."""
    _known_job(h, a.job_id)
    job = h.job(a.job_id)
    raw = job.stderr_path if a.stream == "stderr" else job.stdout_path
    if raw is None:
        raise BridgeError(f"job {a.job_id} has no {a.stream} log (status: {job.status})")
    path = Path(raw).resolve()
    if not path.is_relative_to(h.project_dir / RUNS_DIR):
        raise BridgeError(f"refusing to read a log outside the project runs/ directory: {path}")
    with path.open("rb") as fh:
        lines = fh.read().decode(errors="replace").splitlines()
    return {
        "job_id": a.job_id,
        "stream": a.stream,
        "total_lines": len(lines),
        "lines": lines[-a.tail_lines :],
    }


type Operation = Callable[[Harness, Any, str], dict[str, Any]]

OPERATIONS: dict[str, tuple[type[_Args], Operation]] = {
    "inspect_system": (NoArgs, inspect_system),
    "list_tools": (NoArgs, list_tools),
    "project_status": (NoArgs, project_status),
    "assess_tool": (AssessArgs, assess_tool),
    "run_tool": (RunArgs, run_tool),
    "record_decision": (DecisionArgs, record_decision),
    "add_dataset": (DatasetArgs, add_dataset),
    "read_job_log": (LogArgs, read_job_log),
    "job_status": (JobArgs, job_status),
    "wait_job": (WaitArgs, wait_job),
    "cancel_job": (JobArgs, cancel_job),
}


def call(project_dir: Path, operation: str, raw_args: dict[str, Any], actor: str) -> dict[str, Any]:
    """Dispatch one operation and log the call (read-only ones too) for benchmarking."""
    start = time.monotonic()
    response = _dispatch(project_dir, operation, raw_args, actor)
    if ProjectState.path_for(project_dir).exists():
        ProvenanceLog(project_dir).append(
            "bridge_call",
            operation=operation,
            actor=actor,
            ok=response["ok"],
            error=response.get("error"),
            args=raw_args,
            duration_s=round(time.monotonic() - start, 3),
        )
    return response


def _dispatch(
    project_dir: Path, operation: str, raw_args: dict[str, Any], actor: str
) -> dict[str, Any]:
    """Expected problems return ok=false; bugs raise."""
    if operation not in OPERATIONS:
        return {"ok": False, "error": f"unknown operation '{operation}'; known: {list(OPERATIONS)}"}
    args_model, fn = OPERATIONS[operation]
    try:
        args = args_model.model_validate(raw_args)
        harness = Harness(project_dir, load_registry(project_dir))
        return {"ok": True, "result": fn(harness, args, actor)}
    except ValidationError as exc:
        errs = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
        return {"ok": False, "error": f"invalid arguments: {errs}"}
    except (BridgeError, FileNotFoundError) as exc:
        return {"ok": False, "error": str(exc)}
