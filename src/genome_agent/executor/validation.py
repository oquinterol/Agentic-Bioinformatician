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
from genome_agent.resources.observations import ObservationStore
from genome_agent.tools.feasibility import corrected_estimate
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


MATCH = "matches reference"
MISMATCH = "does not match reference"


@dataclass(frozen=True)
class OriginPolicy:
    verdicts: dict[str, str]  # reads path -> latest read_origin_check verdict
    reference_registered: bool
    require_check: bool


def _check_origin(
    tool: str,
    sensitive: bool,
    paths: list[Path],
    datasets: dict[str, str],
    origin: OriginPolicy | None,
) -> list[str]:
    """Safety rules that must not depend on the agent's judgement.

    1. Never feed an origin-sensitive tool reads that failed the origin check.
    2. If a reference is registered and the project requires it, reads must
       have been verified as matching before an origin-sensitive tool uses them.
    """
    if origin is None or not sensitive:
        return []
    problems = []
    for p in paths:
        kind = datasets.get(str(p))
        if kind is None or kind == "reference_fasta":
            continue
        verdict = origin.verdicts.get(str(p))
        if verdict == MISMATCH:
            problems.append(f"{p} failed read_origin_check ({MISMATCH}); {tool} must not use it")
        elif origin.reference_registered and origin.require_check and verdict != MATCH:
            state = "not verified" if verdict is None else f"verdict is '{verdict}'"
            problems.append(
                f"{p} is {state}: run read_origin_check against the registered reference "
                f"before {tool} (project requires verified read origin)"
            )
    return problems


def validate(
    req: JobRequest,
    registry: ToolRegistry,
    resources: SystemResources,
    budget: ResourceBudget,
    outdir: Path,
    datasets: dict[str, str] | None = None,
    observations: ObservationStore | None = None,
    read_bases: int | None = None,
    origin: OriginPolicy | None = None,
) -> ValidatedJob:
    """`datasets` maps registered dataset paths to their kind (ReadKind value);
    `observations` raises estimates that past identical runs exceeded."""
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
    if adapter.accepted_read_kinds is not None:
        kinds = datasets or {}
        wrong = [
            f"{adapter.name} accepts {sorted(adapter.accepted_read_kinds)} datasets; "
            f"{p} is {kinds.get(str(p), 'not a registered dataset')}"
            for p in paths
            if kinds.get(str(p)) not in adapter.accepted_read_kinds
        ]
        if wrong:
            raise JobRejectedError(wrong)
    if origin_problems := _check_origin(
        adapter.name, adapter.requires_verified_origin, paths, datasets or {}, origin
    ):
        raise JobRejectedError(origin_problems)
    inputs = ToolInputs.from_files(
        paths, datasets, genome_size_bp=req.genome_size_bp, read_bases=read_bases
    )
    if problems := adapter.check_inputs(inputs):
        raise JobRejectedError(problems)

    try:
        est = corrected_estimate(adapter, inputs, params, req.cpus, observations)
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
