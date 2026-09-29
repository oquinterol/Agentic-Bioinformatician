"""Command-line entry point for GenomeAgent (see `genome-agent --help`)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from genome_agent import __version__
from genome_agent.harness import Harness
from genome_agent.resources.inspector import inspect_system
from genome_agent.resources.models import ResourcePolicy, SystemResources
from genome_agent.state.models import BiologicalContext, DecisionRecord, ProjectState, ReadKind
from genome_agent.tools.registry import DataType

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
        f"Limits:     {', '.join(res.enforcement) or 'advisory only (no cgroup backend)'}",
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
    policy = ResourcePolicy(
        ram_fraction=args.ram_fraction,
        reserved_threads=args.reserved_threads,
        disk_fraction=args.disk_fraction,
        enforcement=args.enforcement,
    )
    state = ProjectState(
        name=project.name,
        objective=args.objective or "",
        system_resources=res,
        policy=policy,
        biological_context=BiologicalContext(
            species=args.species,
            organism_type=args.organism_type,
            expected_ploidy=args.ploidy,
            genome_size_bp=args.genome_size,
        ),
        blocked_tools=args.block_tool or [],
    )
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

    from genome_agent.agent.simulation import Scenario, build_project, simulate

    sc = Scenario.load(Path(args.scenario))
    project = Path(args.project) if args.project else Path(tempfile.mkdtemp(prefix="ga-sim-"))
    if args.setup_only:
        build_project(sc, project.resolve())
        print(f"Simulated project '{sc.name}' ready at {project} (no planner run)")
        return 0
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


def cmd_tool(args: argparse.Namespace) -> int:
    from genome_agent.bridge import call

    raw = sys.stdin.read().strip() if not sys.stdin.isatty() else ""
    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        print(json.dumps({"ok": False, "error": f"stdin is not valid JSON: {exc}"}))
        return 2
    if not isinstance(payload, dict):
        print(json.dumps({"ok": False, "error": "arguments must be a JSON object"}))
        return 2
    response = call(Path(args.project).resolve(), args.operation, payload, args.actor)
    print(json.dumps(response, default=str))
    return 0 if response["ok"] else 2


def cmd_toy_data(args: argparse.Namespace) -> int:
    from genome_agent.toydata import make_toy_hifi

    ds = make_toy_hifi(
        Path(args.outdir),
        genome_size=args.genome_size,
        coverage=args.coverage,
        heterozygosity=args.heterozygosity,
        seed=args.seed,
    )
    print(f"Truth genome: {ds.genome_fasta} ({ds.genome_size_bp:,} bp per haplotype)")
    print(f"HiFi reads:   {ds.reads_fastq} ({ds.n_reads} reads, {ds.read_bases:,} bp)")
    return 0


def cmd_add_dataset(args: argparse.Namespace) -> int:
    from genome_agent.bridge import call

    resp = call(
        Path(args.project).resolve(), "add_dataset", {"path": args.path, "kind": args.kind}, "user"
    )
    if not resp["ok"]:
        print(f"error: {resp['error']}", file=sys.stderr)
        return 1
    print(f"Registered {args.kind} dataset {Path(args.path).resolve()}")
    return 0


def _plan_dry_run(harness: Harness, goal: DataType) -> int:
    """Record what the deterministic planner would run, without running it."""
    from genome_agent.agent.backend import Observation, RunTool
    from genome_agent.agent.planner import DeterministicPlanner

    planner = DeterministicPlanner()
    budget, _ = harness.available_budget()
    obs = Observation(harness.state, harness.registry, budget, goal, harness.observations)
    action = planner.next_action(obs)
    if isinstance(action, RunTool):
        req = action.request
        decision = DecisionRecord(
            decision="assembly_plan",
            reason=req.reason,
            actor=planner.name,
            evidence={"plan": req.model_dump(exclude={"actor", "reason"}), "dry_run": True},
            alternatives_considered=req.alternatives_considered,
        )
    else:
        decision = DecisionRecord(
            decision="stop", reason=action.reason, actor=planner.name, evidence=action.evidence
        )
    harness.record_decision(decision)
    print(json.dumps(decision.model_dump(mode="json"), indent=2))
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    from genome_agent.agent.loop import run_loop
    from genome_agent.agent.planner import DeterministicPlanner
    from genome_agent.agent.simulation import load_registry
    from genome_agent.harness import Harness
    from genome_agent.tools.registry import DataType

    project = Path(args.project).resolve()
    harness = Harness(project, load_registry(project))
    if args.dry_run:
        return _plan_dry_run(harness, DataType(args.goal))
    n_decisions = len(harness.state.decisions)
    outcome = run_loop(harness, DeterministicPlanner(), DataType(args.goal), args.max_iterations)
    jobs = {j.id: j for j in harness.state.jobs}
    for d in harness.state.decisions[n_decisions:]:
        print(f"[{d.actor}] {d.decision}: {d.reason}")
        job = jobs.get(str(d.evidence.get("job_id", "")))
        if job:
            rss = f", peak RSS {job.peak_rss_gb:.2f} GB" if job.peak_rss_gb is not None else ""
            est = f" (estimated {job.estimate.ram_gb:.2f} GB)" if job.estimate else ""
            print(f"    -> {job.id}: {job.status}, {job.wall_time_s}s{rss}{est}")
    for r in harness.state.results:
        print(f"Result {r.job_id} [{r.kind}]: {json.dumps(r.data)[:400]}")
    print(f"Outcome: {'ACHIEVED' if outcome.achieved else 'STOPPED'} — {outcome.reason}")
    return 0 if outcome.achieved else 2


def cmd_calibration(args: argparse.Namespace) -> int:
    from genome_agent.resources.observations import ObservationStore

    store = ObservationStore()
    rows = store.report()
    if args.json:
        print(json.dumps({"store": str(store.path), "groups": rows}, indent=2))
        return 0
    print(f"Observations: {store.path}")
    if not rows:
        print("  (none yet: they are recorded when jobs finish)")
    for r in rows:
        flag = "  <- UNDERESTIMATED" if r["underestimates"] else ""
        print(
            f"  {r['tool']:<14} {json.dumps(r['params'])}\n"
            f"      runs={r['runs']} peak/estimate median={r['median_peak_over_estimate']} "
            f"max={r['max_peak_over_estimate']} max_peak={r['max_peak_rss_gb']} GB "
            f"max_input={r['max_input_gb']} GB{flag}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="genome-agent", description=__doc__)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("calibration", help="estimated vs. observed RAM on this machine")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_calibration)

    s = sub.add_parser("inspect", help="inventory this machine")
    s.add_argument("--json", action="store_true", help="machine-readable output")
    s.add_argument("--workspace", default=".", help="directory whose disk is measured")
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("init", help="create a project directory")
    s.add_argument("path")
    s.add_argument("--objective", help="scientific objective, free text")
    d = ResourcePolicy()
    s.add_argument(
        "--ram-fraction",
        type=float,
        default=d.ram_fraction,
        help="share of available RAM jobs may use (default %(default)s)",
    )
    s.add_argument(
        "--reserved-threads",
        type=int,
        default=d.reserved_threads,
        help="threads kept free for the system (default %(default)s)",
    )
    s.add_argument(
        "--disk-fraction",
        type=float,
        default=d.disk_fraction,
        help="share of free disk jobs may use (default %(default)s)",
    )
    s.add_argument("--species")
    s.add_argument("--organism-type")
    s.add_argument("--ploidy", type=int)
    s.add_argument("--genome-size", type=int, help="expected haploid genome size (bp)")
    s.add_argument(
        "--block-tool",
        action="append",
        help="tool the harness must refuse to run in this project (repeatable)",
    )
    s.add_argument(
        "--enforcement",
        choices=["auto", "none", "systemd"],
        default=d.enforcement,
        help="OS-enforced job limits: auto uses systemd user scopes when available",
    )
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("simulate", help="run a mock scenario with the deterministic planner")
    s.add_argument("scenario", help="scenario TOML file (see examples/)")
    s.add_argument("--project", help="keep the simulated project here (default: temp dir)")
    s.add_argument(
        "--setup-only",
        action="store_true",
        help="build the project but let another backend (e.g. Pi) do the planning",
    )
    s.set_defaults(func=cmd_simulate)

    s = sub.add_parser("tool", help="JSON tool protocol for agent runtimes (args on stdin)")
    s.add_argument("operation")
    s.add_argument("--project", default=".")
    s.add_argument("--actor", default="harness", help='who is deciding, e.g. "llm:anthropic/..."')
    s.set_defaults(func=cmd_tool)

    s = sub.add_parser("toy-data", help="write a deterministic toy HiFi dataset")
    s.add_argument("outdir")
    s.add_argument("--genome-size", type=int, default=200_000)
    s.add_argument("--coverage", type=float, default=30.0)
    s.add_argument("--heterozygosity", type=float, default=0.0)
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(func=cmd_toy_data)

    s = sub.add_parser("add-dataset", help="register an existing reads file (read-only)")
    s.add_argument("project")
    s.add_argument("path")
    s.add_argument("--kind", required=True, choices=[k.value for k in ReadKind])
    s.set_defaults(func=cmd_add_dataset)

    s = sub.add_parser("plan", help="run the deterministic planner on a project")
    s.add_argument("project", nargs="?", default=".")
    s.add_argument("--goal", default="contigs_fasta")
    s.add_argument("--max-iterations", type=int, default=10)
    s.add_argument("--dry-run", action="store_true", help="record the plan, run nothing")
    s.set_defaults(func=cmd_plan)

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
