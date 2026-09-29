"""The single entry point through which anything gets executed.

run_tool: validate → record decision → launch a detached runner (job RUNNING,
saved) → optionally wait. Finished jobs are reconciled from the runner's
execution.json whenever the project is opened or refreshed. Every outcome —
rejected, failed, timed out, cancelled, succeeded — ends up in state.json and
provenance.jsonl.

Every state mutation happens inside `transaction()`: take the project lock,
reload state.json, mutate, save. So concurrent writers (CLI, Pi tool calls,
several harness instances) never overwrite each other.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from genome_agent.executor.local import ExecutionResult
from genome_agent.executor.runner import (
    JobSpec,
    enforced_argv,
    is_runner_alive,
    launch,
    read_execution,
)
from genome_agent.executor.validation import (
    JobRejectedError,
    JobRequest,
    OriginPolicy,
    ValidatedJob,
    validate,
)
from genome_agent.provenance.log import ProvenanceLog
from genome_agent.resources.models import ResourceBudget
from genome_agent.resources.observations import ObservationStore, ResourceObservation
from genome_agent.state.lock import project_lock
from genome_agent.state.models import DecisionRecord, Job, JobStatus, ProjectState, Result
from genome_agent.tools.registry import ToolRegistry

RUNS_DIR = "runs"
_POLL_S = 0.2
_CANCEL_GRACE_S = 30.0  # runner start-up can be slow on a busy machine


class Harness:
    def __init__(
        self,
        project_dir: Path,
        registry: ToolRegistry,
        observations: ObservationStore | None = None,
    ) -> None:
        self.project_dir = project_dir.resolve()
        self.registry = registry
        self.observations = observations or ObservationStore()
        self.provenance = ProvenanceLog(self.project_dir)
        self._tx_depth = 0
        self._children: dict[int, subprocess.Popen[bytes]] = {}
        self.state = ProjectState.load(self.project_dir)
        if self.state.system_resources is None:
            raise ValueError("project has no system resource snapshot; re-run `init`")
        self.refresh()

    # -- state -------------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[ProjectState]:
        """Lock, reload, yield the fresh state for mutation, save. Re-entrant."""
        if self._tx_depth:
            self._tx_depth += 1
            try:
                yield self.state
            finally:
                self._tx_depth -= 1
            return
        with project_lock(self.project_dir):
            self.state = ProjectState.load(self.project_dir)
            self._tx_depth = 1
            try:
                yield self.state
                self.state.save(self.project_dir)
            finally:
                self._tx_depth = 0

    def job(self, job_id: str) -> Job:
        for j in self.state.jobs:
            if j.id == job_id:
                return j
        raise KeyError(f"unknown job '{job_id}'; known: {[j.id for j in self.state.jobs]}")

    def available_budget(self) -> tuple[ResourceBudget, list[str]]:
        """Policy budget minus what RUNNING jobs have reserved."""
        res = self.state.system_resources
        assert res is not None
        budget = self.state.policy.apply(res)
        running = [j for j in self.state.jobs if j.status == JobStatus.RUNNING]
        if not running:
            return budget, []
        disk = sum(j.estimate.disk_gb + j.estimate.tmp_gb for j in running if j.estimate)
        return (
            ResourceBudget(
                cpu_threads=budget.cpu_threads - sum(j.cpus for j in running),
                ram_gb=round(budget.ram_gb - sum(j.ram_gb for j in running), 2),
                disk_gb=round(budget.disk_gb - disk, 2),
            ),
            [j.id for j in running],
        )

    def record_decision(self, decision: DecisionRecord) -> None:
        """Record a decision that does not launch a job (e.g. stop, replan)."""
        with self.transaction() as state:
            state.decisions.append(decision)
        self.provenance.append(
            "decision", decision=decision.decision, actor=decision.actor, reason=decision.reason
        )

    def update_metrics(self, metrics: dict[str, float]) -> None:
        with self.transaction() as state:
            state.metrics.update(metrics)

    # -- jobs --------------------------------------------------------------

    def _validate(
        self, state: ProjectState, req: JobRequest, budget: ResourceBudget, outdir: Path
    ) -> ValidatedJob:
        """The single set of rules shared by run_tool and check_plan."""
        res = state.system_resources
        assert res is not None
        return validate(
            req,
            self.registry,
            res,
            budget,
            outdir,
            datasets={d.path: d.kind for d in state.datasets},
            observations=self.observations,
            read_bases=state.measured_read_bases(req.inputs),
            origin=OriginPolicy(
                verdicts=state.origin_verdicts(),
                reference_registered=state.has_reference(),
                require_check=state.require_origin_check,
                pair_verdicts=state.consistency_verdicts(),
            ),
        )

    def check_plan(self, req: JobRequest) -> list[str]:
        """Problems that would make `req` be rejected if run now; [] if it would be
        accepted. Blocked tools are ignored: plans may target them."""
        budget, _ = self.available_budget()
        try:
            self._validate(self.state, req, budget, self.project_dir / RUNS_DIR / "plan-check")
        except JobRejectedError as exc:
            return exc.reasons
        return []

    def run_tool(self, req: JobRequest, wait_s: float | None = None) -> Job:
        """Validate and launch. Wait up to `wait_s` seconds (None = until done)."""
        with self.transaction() as state:
            res = state.system_resources
            assert res is not None
            job_id = f"{len(state.jobs) + 1:04d}-{req.tool}"
            outdir = self.project_dir / RUNS_DIR / job_id
            job = Job(
                id=job_id,
                tool=req.tool,
                params=req.params,
                inputs=req.inputs,
                cpus=req.cpus,
                ram_gb=req.ram_gb,
            )
            state.jobs.append(job)
            budget, running = self.available_budget()
            try:
                if req.tool in state.blocked_tools:
                    raise JobRejectedError(
                        [f"{req.tool} is blocked in this project (blocked_tools)"]
                    )
                try:
                    job.enforcement = state.policy.enforcement_backend(res)
                except ValueError as exc:
                    raise JobRejectedError([str(exc)]) from None
                vj = self._validate(state, req, budget, outdir)
            except JobRejectedError as exc:
                reasons = exc.reasons
                if running:
                    reasons = [*reasons, f"budget already excludes resources reserved by {running}"]
                job.status, job.rejection_reasons, job.estimate = (
                    JobStatus.REJECTED,
                    reasons,
                    exc.estimate,
                )
                job.finished_at = datetime.now(UTC)
                self.provenance.append(
                    "job_rejected", job_id=job_id, actor=req.actor, reasons=reasons
                )
                return job

            job.argv, job.estimate, job.outdir = vj.argv, vj.estimate, str(outdir)
            state.decisions.append(
                DecisionRecord(
                    decision=f"run_{req.tool}",
                    reason=req.reason,
                    actor=req.actor,
                    evidence={
                        **req.evidence,
                        "job_id": job_id,
                        "inputs": req.inputs,
                        "estimate": vj.estimate.model_dump(),
                    },
                    resources={"cpus": req.cpus, "ram_gb": req.ram_gb},
                    alternatives_considered=req.alternatives_considered,
                )
            )
            outdir.mkdir(parents=True, exist_ok=True)
            spec = JobSpec(
                argv=vj.argv,
                cpus=req.cpus,
                ram_gb=req.ram_gb,
                timeout_s=req.timeout_s,
                enforcement=job.enforcement,
            )
            wrapper = shlex.join(enforced_argv(spec)[: -len(vj.argv)])
            limits = (
                f"# limits enforced by the kernel: {wrapper}\n"
                if job.enforcement
                else "# limits were validated but not enforced (advisory)\n"
            )
            (outdir / "command.sh").write_text("#!/bin/sh\n" + limits + shlex.join(vj.argv) + "\n")
            proc = launch(outdir, spec)
            self._children[proc.pid] = proc
            job.runner_pid, job.status = proc.pid, JobStatus.RUNNING
            self.provenance.append("job_started", job_id=job_id, actor=req.actor, argv=vj.argv)

        return self.wait_job(job_id, wait_s)

    def wait_job(self, job_id: str, timeout_s: float | None = None) -> Job:
        """Poll until the job leaves RUNNING or `timeout_s` elapses; return it."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            self.refresh()
            job = self.job(job_id)
            if job.status != JobStatus.RUNNING:
                return job
            if deadline is not None and time.monotonic() >= deadline:
                return job
            time.sleep(_POLL_S)

    def cancel_job(self, job_id: str) -> Job:
        with self.transaction():
            job = self.job(job_id)
            if job.status != JobStatus.RUNNING:
                return job
            job.cancel_requested = True  # a runner that dies before handling it = cancelled
        if job.outdir and is_runner_alive(job.runner_pid, Path(job.outdir)):
            assert job.runner_pid is not None
            os.kill(job.runner_pid, signal.SIGTERM)
        return self.wait_job(job_id, _CANCEL_GRACE_S)

    def refresh(self) -> None:
        """Reconcile RUNNING jobs with their runners' results."""
        if not any(j.status == JobStatus.RUNNING for j in self.state.jobs):
            return
        with self.transaction() as state:
            for job in state.jobs:
                if job.status != JobStatus.RUNNING or job.outdir is None:
                    continue
                outdir = Path(job.outdir)
                result = read_execution(outdir)
                if result is not None:
                    self._reap(job.runner_pid)
                    self._finalize(state, job, result)
                elif not is_runner_alive(job.runner_pid, outdir):
                    self._reap(job.runner_pid)
                    if (result := read_execution(outdir)) is not None:  # finished meanwhile
                        self._finalize(state, job, result)
                        continue
                    if job.cancel_requested:
                        job.status = JobStatus.CANCELLED
                        job.error = "cancelled on request (before the tool started)"
                        job.finished_at = datetime.now(UTC)
                        self.provenance.append("job_finished", job_id=job.id, status=job.status)
                        continue
                    job.status = JobStatus.FAILED
                    job.error = (
                        f"runner process {job.runner_pid} disappeared without a result "
                        "(machine restart or killed?)"
                    )
                    job.finished_at = datetime.now(UTC)
                    state.failures.append(f"{job.id}: {job.error}")
                    self.provenance.append("job_lost", job_id=job.id, error=job.error)

    def _reap(self, pid: int | None) -> None:
        proc = self._children.pop(pid, None) if pid is not None else None
        if proc is not None:
            proc.wait()

    def _observe(self, state: ProjectState, job: Job) -> None:
        """Record estimated vs. observed RAM on this machine; flag underestimates."""
        if job.peak_rss_gb is None or job.estimate is None or job.status == JobStatus.CANCELLED:
            return
        params = self.registry.get(job.tool).parse_params(job.params).model_dump()
        self.observations.append(
            ResourceObservation(
                tool=job.tool,
                params=params,
                cpus=job.cpus,
                input_bytes=sum(Path(p).stat().st_size for p in job.inputs if Path(p).exists()),
                estimated_ram_gb=job.estimate.ram_gb,
                requested_ram_gb=job.ram_gb,
                oom_killed=job.oom_killed,
                peak_rss_gb=job.peak_rss_gb,
                wall_time_s=job.wall_time_s,
                status=job.status,
                project=state.name,
                job_id=job.id,
            )
        )
        if job.oom_killed or job.peak_rss_gb > job.estimate.ram_gb:
            self.provenance.append(
                "estimate_exceeded",
                job_id=job.id,
                estimated_ram_gb=job.estimate.ram_gb,
                requested_ram_gb=job.ram_gb,
                oom_killed=job.oom_killed,
                peak_rss_gb=job.peak_rss_gb,
                note="future estimates for this tool/params are raised accordingly",
            )

    def _finalize(self, state: ProjectState, job: Job, r: ExecutionResult) -> None:
        job.exit_code, job.wall_time_s, job.timed_out, job.error = (
            r.exit_code,
            round(r.wall_time_s, 3),
            r.timed_out,
            r.error,
        )
        job.stdout_path, job.stderr_path = str(r.stdout_path), str(r.stderr_path)
        job.peak_rss_gb = r.peak_rss_gb
        job.oom_killed = r.oom_killed
        job.finished_at = datetime.now(UTC)
        job.status = JobStatus.FAILED
        if r.cancelled:
            job.status = JobStatus.CANCELLED
        elif r.ok:
            adapter = self.registry.get(job.tool)
            assert job.outdir is not None
            try:
                data = adapter.parse_result(Path(job.outdir))
            except Exception as exc:  # recorded, not swallowed: the job is marked failed
                job.error = f"result parsing failed: {type(exc).__name__}: {exc}"
            else:
                job.status = JobStatus.SUCCEEDED
                for kind in sorted(adapter.output_types):
                    state.results.append(
                        Result(job_id=job.id, kind=kind, path=job.outdir, data=data)
                    )
        if job.status == JobStatus.FAILED:
            detail = job.error or f"exit code {job.exit_code}"
            state.failures.append(f"{job.id}: {detail} (stderr: {job.stderr_path})")
        self._observe(state, job)
        self.provenance.append(
            "job_finished",
            job_id=job.id,
            status=job.status,
            exit_code=job.exit_code,
            wall_time_s=job.wall_time_s,
            peak_rss_gb=job.peak_rss_gb,
            estimated_ram_gb=job.estimate.ram_gb if job.estimate else None,
            error=job.error,
        )
