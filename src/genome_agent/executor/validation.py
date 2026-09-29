"""Turn an untrusted JobRequest (from a planner or an LLM) into a ValidatedJob.

Path model:
- inputs may live anywhere on the filesystem (existing data on this machine is
  the normal case); they must exist and be readable, and are only ever read.
- outputs always go to <project>/runs/<job_id>/, chosen by the harness.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from genome_agent.resources.models import ResourceBudget, ResourceEstimate, SystemResources
from genome_agent.tools.registry import EstimationError, ToolInputs, ToolRegistry

DEFAULT_TIMEOUT_S = 24 * 3600.0


class JobRequest(BaseModel):
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: list[str] = Field(default_factory=list)
    cpus: int = Field(ge=1)
    ram_gb: float = Field(gt=0)
    timeout_s: float = Field(default=DEFAULT_TIMEOUT_S, gt=0)
    genome_size_bp: int | None = None
    reason: str
    alternatives_considered: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    actor: str = "harness"


@dataclass(frozen=True)
class ValidatedJob:
    request: JobRequest
    params: Any
    inputs: ToolInputs
    estimate: ResourceEstimate
    argv: list[str]
    outdir: Path


class JobRejectedError(Exception):
    def __init__(self, reasons: list[str], estimate: ResourceEstimate | None = None) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons
        self.estimate = estimate


def _check_inputs(raw: list[str]) -> tuple[list[Path], list[str]]:
    paths, problems = [], []
    for s in raw:
        p = Path(s).expanduser().resolve()
        if not p.is_file():
            problems.append(f"input not found or not a file: {p}")
        elif not os.access(p, os.R_OK):
            problems.append(f"input not readable: {p}")
        else:
            paths.append(p)
    return paths, problems


def validate(
    req: JobRequest,
    registry: ToolRegistry,
    resources: SystemResources,
    budget: ResourceBudget,
    outdir: Path,
) -> ValidatedJob:
    try:
        adapter = registry.get(req.tool)
    except KeyError as exc:
        raise JobRejectedError([str(exc.args[0])]) from None
    if not adapter.is_available(resources):
        raise JobRejectedError([f"executable '{adapter.executable}' not detected on this machine"])
    try:
        params = adapter.parse_params(req.params)
    except ValidationError as exc:
        reasons = [f"invalid params: {e['loc']}: {e['msg']}" for e in exc.errors()]
        raise JobRejectedError(reasons) from None

    paths, problems = _check_inputs(req.inputs)
    if problems:
        raise JobRejectedError(problems)
    inputs = ToolInputs.from_files(paths, genome_size_bp=req.genome_size_bp)

    try:
        est = adapter.estimate(inputs, params, req.cpus)
    except EstimationError as exc:
        raise JobRejectedError([str(exc)]) from None

    reasons = []
    if req.cpus > budget.cpu_threads:
        reasons.append(f"requested {req.cpus} threads > budget {budget.cpu_threads}")
    if req.ram_gb > budget.ram_gb:
        reasons.append(f"requested {req.ram_gb:.1f} GB RAM > budget {budget.ram_gb:.1f} GB")
    if req.ram_gb < est.ram_gb:
        reasons.append(
            f"requested {req.ram_gb:.1f} GB RAM < estimated peak {est.ram_gb:.1f} GB ({est.basis})"
        )
    reasons += [v for v in budget.violations(est) if "threads" not in v]  # threads checked above
    if reasons:
        raise JobRejectedError(reasons, est)

    argv = adapter.build_command(inputs, params, outdir, req.cpus)
    if argv[0] in resources.tools:  # pin to the exact binary that was inventoried
        argv[0] = resources.tools[argv[0]].path
    return ValidatedJob(req, params, inputs, est, argv, outdir)
