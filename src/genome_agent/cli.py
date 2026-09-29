"""Command-line entry point: `genome-agent inspect | init | status | simulate`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from genome_agent import __version__
from genome_agent.resources.inspector import inspect_system
from genome_agent.resources.models import ResourcePolicy, SystemResources
from genome_agent.state.models import DecisionRecord, ProjectState

PROJECT_SUBDIRS = ("data", "runs", "results")


def render_inspect(res: SystemResources, policy: ResourcePolicy) -> str:
    b = policy.apply(res)
    cores = f"{res.cpu_physical_cores} cores / " if res.cpu_physical_cores else ""
    lines = [
        "GenomeAgent Environment",
        "",
        f"OS:   {res.os} (kernel {res.kernel})",
        f"CPU:  {res.cpu_model}",
        f"      {cores}{res.cpu_threads} threads, {b.cpu_threads} usable under policy",
        f"RAM:  {res.ram_total_gb:.1f} GB total, {res.ram_available_gb:.1f} GB available, "
        f"{b.ram_gb:.1f} GB usable under policy",
        f"Swap: {res.swap_free_gb:.1f} / {res.swap_total_gb:.1f} GB free",
        f"Disk: {res.disk_free_gb:.0f} GB free at {res.workspace}, "
        f"{b.disk_gb:.0f} GB usable under policy",
        f"Tmp:  {res.tmp_free_gb:.0f} GB free at {res.tmp_dir}",
        f"GPU:  {', '.join(g.name for g in res.gpus) or 'none'}",
        f"Containers: {', '.join(res.container_runtimes) or 'none'}",
        f"Schedulers: {', '.join(res.schedulers) or 'none'}",
        "",
        "Detected tools:",
    ]
    if not res.tools:
        lines.append("  (none)")
    for t in sorted(res.tools.values(), key=lambda t: t.name.lower()):
        lines.append(f"  {t.name:<14} {t.version or '?':<14} {t.path}")
        if t.error:
            lines.append(f"  {'':<14} ! {t.error}")
    if res.notes:
        lines += ["", "Notes:", *(f"  ! {n}" for n in res.notes)]
    return "\n".join(lines)


def cmd_inspect(args: argparse.Namespace) -> int:
    res = inspect_system(Path(args.workspace))
    policy = ResourcePolicy()
    if args.json:
        payload = {
            "system_resources": res.model_dump(mode="json"),
            "policy": policy.model_dump(),
            "budget": policy.apply(res).model_dump(),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(render_inspect(res, policy))
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    project = Path(args.path).resolve()
    if ProjectState.path_for(project).exists():
        print(f"error: project already initialised at {project}", file=sys.stderr)
        return 1
    for sub in PROJECT_SUBDIRS:
        (project / sub).mkdir(parents=True, exist_ok=True)
    res = inspect_system(project)
    state = ProjectState(name=project.name, objective=args.objective or "", system_resources=res)
    state.decisions.append(
        DecisionRecord(
            decision="init_project",
            reason="project created; machine inventory snapshotted",
            evidence={"cpu_threads": res.cpu_threads, "ram_available_gb": res.ram_available_gb},
            resources=state.policy.apply(res).model_dump(),
        )
    )
    path = state.save(project)
    print(f"Initialised GenomeAgent project '{state.name}' at {project}\nState: {path}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    project = Path(args.path).resolve()
    try:
        state = ProjectState.load(project)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    res = state.system_resources
    print(f"Project:    {state.name}  (schema v{state.schema_version})")
    print(f"Objective:  {state.objective or '(not set)'}")
    print(f"Datasets:   {len(state.datasets)}")
    print(f"Jobs:       {len(state.jobs)}  failures: {len(state.failures)}")
    print(f"Decisions:  {len(state.decisions)}")
    if res:
        b = state.policy.apply(res)
        print(
            f"Budget:     {b.cpu_threads} threads, {b.ram_gb:.1f} GB RAM, {b.disk_gb:.0f} GB disk"
            f"  (snapshot {res.inspected_at:%Y-%m-%d %H:%M} UTC)"
        )
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    import tempfile

    from genome_agent.agent.simulation import Scenario, simulate

    sc = Scenario.load(Path(args.scenario))
    project = Path(args.project) if args.project else Path(tempfile.mkdtemp(prefix="ga-sim-"))
    outcome, harness = simulate(sc, project.resolve())
    state = harness.state
    budget = state.policy.apply(state.system_resources) if state.system_resources else None
    print(f"Scenario:  {sc.name} — {sc.description}")
    if budget:
        print(
            f"Budget:    {budget.cpu_threads} threads, {budget.ram_gb:.1f} GB RAM, "
            f"{budget.disk_gb:.0f} GB disk"
        )
    print()
    jobs = {j.id: j for j in state.jobs}
    for d in state.decisions:
        print(f"[{d.actor}] {d.decision}: {d.reason}")
        job = jobs.get(str(d.evidence.get("job_id", "")))
        if job:
            print(
                f"    -> {job.id}: {job.status}"
                + (f" (exit {job.exit_code})" if job.exit_code else "")
            )
    for tool, why in state.decisions[-1].evidence.get("rejected", {}).items():
        print(f"    x {tool}: {'; '.join(why)}")
    print()
    print(f"Outcome:   {'ACHIEVED' if outcome.achieved else 'STOPPED'} — {outcome.reason}")
    print(
        f"Agent:     {outcome.iterations} iterations, {outcome.replans} replans, "
        f"{outcome.jobs_failed} failed, {outcome.jobs_rejected} rejected"
    )
    print(f"Project:   {project}")
    problems = sc.check(outcome, harness)
    for p in problems:
        print(f"MISMATCH:  {p}", file=sys.stderr)
    return 1 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="genome-agent", description=__doc__)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("inspect", help="inventory this machine")
    s.add_argument("--json", action="store_true", help="machine-readable output")
    s.add_argument("--workspace", default=".", help="directory whose disk is measured")
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("init", help="create a project directory")
    s.add_argument("path")
    s.add_argument("--objective", help="scientific objective, free text")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("simulate", help="run a mock scenario with the deterministic planner")
    s.add_argument("scenario", help="scenario TOML file (see examples/)")
    s.add_argument("--project", help="keep the simulated project here (default: temp dir)")
    s.set_defaults(func=cmd_simulate)

    s = sub.add_parser("status", help="summarise project state")
    s.add_argument("path", nargs="?", default=".")
    s.set_defaults(func=cmd_status)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
