"""Compare agent runs on identical projects, using only state.json and provenance.jsonl.

A benchmark is a pure function of what the harness recorded, so any backend
(deterministic planner, Pi + any model) is measured the same way.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from genome_agent.provenance.log import ProvenanceLog
from genome_agent.state.models import JobStatus, ProjectState


class RunSummary(BaseModel):
    project: str
    actor: str
    plan: dict[str, Any] | None
    plan_reason: str | None
    plan_risks: Any = None
    estimate_basis: str | None
    reservation_vs_budget: str | None
    jobs: list[dict[str, Any]]
    bridge_calls: int
    bridge_errors: int
    rejected_jobs: int
    decisions: int
    session_s: float | None


def _plan_decision(state: ProjectState) -> Any:
    for d in reversed(state.decisions):
        if d.decision == "assembly_plan":
            return d
    return None


def _extract_plan(evidence: dict[str, Any]) -> dict[str, Any] | None:
    """Plans are recorded by different actors; accept the common shapes.

    Preferred: evidence["plan"] (typed via record_decision's `plan`). Also
    accepted: top-level keys, or "chosen_*" keys with a resources dict.
    """
    keys = ("tool", "params", "inputs", "cpus", "ram_gb")
    plan = evidence.get("plan")
    if isinstance(plan, dict) and "tool" in plan:
        return {k: plan.get(k) for k in keys}
    if "tool" in evidence:
        return {k: evidence.get(k) for k in keys}
    if "chosen_tool" in evidence:
        resources: dict[str, Any] = next(
            (v for k, v in evidence.items() if k.startswith("chosen_resources")), {}
        )
        return {
            "tool": evidence["chosen_tool"],
            "params": evidence.get("chosen_params"),
            "inputs": evidence.get("chosen_inputs"),
            "cpus": resources.get("cpus"),
            "ram_gb": resources.get("ram_gb"),
        }
    return None


def summarize(project_dir: Path) -> RunSummary:
    state = ProjectState.load(project_dir)
    events = ProvenanceLog(project_dir).read()
    calls = [e for e in events if e["event"] == "bridge_call" and e.get("actor") != "user"]
    d = _plan_decision(state)
    plan = _extract_plan(d.evidence) if d else None
    budget = state.policy.apply(state.system_resources) if state.system_resources else None

    basis = None
    if d:
        basis = d.evidence.get("estimate_basis") or (d.evidence.get("plan", {}) or {}).get(
            "evidence", {}
        ).get("estimate_basis")
    reservation = None
    if plan and budget and isinstance(plan.get("ram_gb"), int | float):
        reservation = (
            f"{plan['ram_gb']} / {budget.ram_gb} GB RAM, {plan.get('cpus')} / "
            f"{budget.cpu_threads} threads"
        )
    times = [datetime.fromisoformat(e["ts"]) for e in events if e.get("actor") != "user"]
    return RunSummary(
        project=project_dir.name,
        actor=d.actor if d else "(no plan recorded)",
        plan=plan,
        plan_reason=d.reason if d else None,
        plan_risks=(d.evidence.get("risks") or d.evidence.get("main_risks")) if d else None,
        estimate_basis=basis,
        reservation_vs_budget=reservation,
        jobs=[
            {
                "id": j.id,
                "status": j.status,
                "wall_time_s": j.wall_time_s,
                "peak_rss_gb": j.peak_rss_gb,
                "estimated_ram_gb": j.estimate.ram_gb if j.estimate else None,
            }
            for j in state.jobs
        ],
        bridge_calls=len(calls),
        bridge_errors=sum(not c["ok"] for c in calls),
        rejected_jobs=sum(j.status == JobStatus.REJECTED for j in state.jobs),
        decisions=sum(x.decision not in ("init_project", "add_dataset") for x in state.decisions),
        session_s=(max(times) - min(times)).total_seconds() if len(times) > 1 else None,
    )


def render(summaries: list[RunSummary]) -> str:
    lines = []
    for s in summaries:
        lines += [
            f"== {s.project}  [{s.actor}]",
            f"   plan:        {s.plan}",
            f"   reservation: {s.reservation_vs_budget}",
            f"   reason:      {s.plan_reason}",
        ]
        if s.plan_risks:
            lines.append(f"   risks:       {s.plan_risks}")
        for j in s.jobs:
            lines.append(
                f"   job {j['id']}: {j['status']}, {j['wall_time_s']} s, "
                f"peak {j['peak_rss_gb']} GB (est {j['estimated_ram_gb']})"
            )
        lines.append(
            f"   calls={s.bridge_calls} errors={s.bridge_errors} rejected={s.rejected_jobs} "
            f"decisions={s.decisions} session={s.session_s} s"
        )
    return "\n".join(lines)
