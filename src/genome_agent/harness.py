"""The single entry point through which anything gets executed.

run_tool: validate → record decision → mark running (saved) → execute →
parse → update state (saved). Every outcome — rejected, failed, timed out,
succeeded — ends up in state.json and provenance.jsonl.
"""

from __future__ import annotations

import shlex
from datetime import UTC, datetime
from pathlib import Path

from genome_agent.executor.local import Executor, LocalExecutor
from genome_agent.executor.validation import JobRejectedError, JobRequest, validate
from genome_agent.provenance.log import ProvenanceLog
from genome_agent.state.models import DecisionRecord, Job, JobStatus, ProjectState, Result
from genome_agent.tools.registry import ToolRegistry

RUNS_DIR = "runs"


class Harness:
    def __init__(
        self,
        project_dir: Path,
        registry: ToolRegistry,
        executor: Executor | None = None,
    ) -> None:
        self.project_dir = project_dir.resolve()
        self.registry = registry
        self.executor = executor or LocalExecutor()
        self.state = ProjectState.load(self.project_dir)
        self.provenance = ProvenanceLog(self.project_dir)
        if self.state.system_resources is None:
            raise ValueError("project has no system resource snapshot; re-run `init`")

    def _save(self) -> None:
        self.state.save(self.project_dir)

    def record_decision(self, decision: DecisionRecord) -> None:
        """Record a decision that does not launch a job (e.g. stop, replan)."""
        self.state.decisions.append(decision)
        self.provenance.append(
            "decision", decision=decision.decision, actor=decision.actor, reason=decision.reason
        )
        self._save()

    def run_tool(self, req: JobRequest) -> Job:
        res = self.state.system_resources
        assert res is not None
        job_id = f"{len(self.state.jobs) + 1:04d}-{req.tool}"
        outdir = self.project_dir / RUNS_DIR / job_id
        job = Job(
            id=job_id,
            tool=req.tool,
            params=req.params,
            inputs=req.inputs,
            cpus=req.cpus,
            ram_gb=req.ram_gb,
        )
        self.state.jobs.append(job)

        try:
            vj = validate(
                req,
                self.registry,
                res,
                self.state.policy.apply(res),
                outdir,
                datasets={d.path: d.kind for d in self.state.datasets},
            )
        except JobRejectedError as exc:
            job.status, job.rejection_reasons, job.estimate = (
                JobStatus.REJECTED,
                exc.reasons,
                exc.estimate,
            )
            job.finished_at = datetime.now(UTC)
            self.provenance.append(
                "job_rejected", job_id=job_id, actor=req.actor, reasons=exc.reasons
            )
            self._save()
            return job

        job.argv, job.estimate, job.outdir = vj.argv, vj.estimate, str(outdir)
        self.state.decisions.append(
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
        (outdir / "command.sh").write_text("#!/bin/sh\n" + shlex.join(vj.argv) + "\n")
        job.status = JobStatus.RUNNING
        self.provenance.append("job_started", job_id=job_id, actor=req.actor, argv=vj.argv)
        self._save()  # an interrupted run is visible as RUNNING on resume

        r = self.executor.run(vj.argv, outdir, req.cpus, req.ram_gb, req.timeout_s)
        job.exit_code, job.wall_time_s, job.timed_out, job.error = (
            r.exit_code,
            round(r.wall_time_s, 3),
            r.timed_out,
            r.error,
        )
        job.stdout_path, job.stderr_path = str(r.stdout_path), str(r.stderr_path)
        job.peak_rss_gb = r.peak_rss_gb
        job.finished_at = datetime.now(UTC)

        if r.ok:
            adapter = self.registry.get(req.tool)
            try:
                data = adapter.parse_result(outdir)
            except Exception as exc:  # recorded, not swallowed: the job is marked failed
                job.error = f"result parsing failed: {type(exc).__name__}: {exc}"
            else:
                job.status = JobStatus.SUCCEEDED
                for kind in sorted(adapter.output_types):
                    self.state.results.append(
                        Result(job_id=job_id, kind=kind, path=str(outdir), data=data)
                    )
        if job.status != JobStatus.SUCCEEDED:
            job.status = JobStatus.FAILED
            detail = job.error or f"exit code {job.exit_code}"
            self.state.failures.append(f"{job_id}: {detail} (stderr: {job.stderr_path})")

        self.provenance.append(
            "job_finished",
            job_id=job_id,
            status=job.status,
            exit_code=job.exit_code,
            wall_time_s=job.wall_time_s,
            peak_rss_gb=job.peak_rss_gb,
            estimated_ram_gb=job.estimate.ram_gb if job.estimate else None,
            error=job.error,
        )
        self._save()
        return job
