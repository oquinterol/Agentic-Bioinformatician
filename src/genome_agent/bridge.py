"""JSON tool protocol for external agent runtimes (Pi, or anything else).

    genome-agent tool <operation> --project DIR --actor "llm:provider/model" < args.json

stdout is always one JSON object: {"ok": true, "result": ...} or
{"ok": false, "error": "..."}. The caller supplies `--actor`; the model cannot
set it through the arguments. Arguments are validated with strict Pydantic
models, and anything that executes goes through Harness.run_tool.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from genome_agent.agent.simulation import load_registry
from genome_agent.executor.validation import DEFAULT_TIMEOUT_S, JobRequest
from genome_agent.harness import Harness
from genome_agent.state.models import Dataset, DecisionRecord, ReadKind
from genome_agent.tools.feasibility import assess
from genome_agent.tools.registry import ToolInputs


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
    cpus: int = Field(ge=1)
    ram_gb: float = Field(gt=0)
    timeout_s: float = Field(default=DEFAULT_TIMEOUT_S, gt=0)
    reason: str = Field(min_length=1)
    alternatives_considered: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)


class DecisionArgs(_Args):
    decision: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    evidence: dict[str, Any] = Field(default_factory=dict)
    alternatives_considered: list[str] = Field(default_factory=list)


class DatasetArgs(_Args):
    path: str
    kind: ReadKind


def _defaults(h: Harness, a: AssessArgs) -> tuple[list[str], int | None]:
    inputs = a.inputs if a.inputs is not None else [d.path for d in h.state.datasets]
    if not inputs:
        raise BridgeError("no inputs given and the project has no datasets (use add_dataset)")
    return inputs, a.genome_size_bp or h.state.biological_context.genome_size_bp


def _budget(h: Harness) -> dict[str, Any]:
    assert h.state.system_resources is not None
    return h.state.policy.apply(h.state.system_resources).model_dump()


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
        "datasets": [d.model_dump() for d in s.datasets],
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
    tool_inputs = ToolInputs.from_files([Path(p) for p in inputs], genome_size_bp=genome_size)
    budget = h.state.policy.apply(h.state.system_resources)
    result = assess(adapter, tool_inputs, params, budget)
    return result.model_dump() | {"available": adapter.is_available(h.state.system_resources)}


def run_tool(h: Harness, a: RunArgs, actor: str) -> dict[str, Any]:
    inputs, genome_size = _defaults(h, a)
    job = h.run_tool(
        JobRequest(
            **a.model_dump(exclude={"inputs", "genome_size_bp"}),
            inputs=inputs,
            genome_size_bp=genome_size,
            actor=actor,
        )
    )
    return {
        "job": job.model_dump(mode="json", exclude={"params", "created_at", "finished_at"}),
        "results": [r.model_dump() for r in h.state.results if r.job_id == job.id],
    }


def record_decision(h: Harness, a: DecisionArgs, actor: str) -> dict[str, Any]:
    h.record_decision(DecisionRecord(**a.model_dump(), actor=actor))
    return {"recorded": a.decision, "total_decisions": len(h.state.decisions)}


def add_dataset(h: Harness, a: DatasetArgs, actor: str) -> dict[str, Any]:
    p = Path(a.path).expanduser().resolve()
    if not p.is_file():
        raise BridgeError(f"not a file: {p}")
    if str(p) in {d.path for d in h.state.datasets}:
        raise BridgeError(f"dataset already registered: {p}")
    h.state.datasets.append(Dataset(path=str(p), kind=a.kind, size_bytes=p.stat().st_size))
    h.record_decision(
        DecisionRecord(
            decision="add_dataset",
            reason=f"registered {a.kind} reads",
            evidence={"path": str(p)},
            actor=actor,
        )
    )
    return {"datasets": [d.model_dump() for d in h.state.datasets]}


type Operation = Callable[[Harness, Any, str], dict[str, Any]]

OPERATIONS: dict[str, tuple[type[_Args], Operation]] = {
    "inspect_system": (NoArgs, inspect_system),
    "list_tools": (NoArgs, list_tools),
    "project_status": (NoArgs, project_status),
    "assess_tool": (AssessArgs, assess_tool),
    "run_tool": (RunArgs, run_tool),
    "record_decision": (DecisionArgs, record_decision),
    "add_dataset": (DatasetArgs, add_dataset),
}


def call(project_dir: Path, operation: str, raw_args: dict[str, Any], actor: str) -> dict[str, Any]:
    """Dispatch one operation. Expected problems return ok=false; bugs raise."""
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
