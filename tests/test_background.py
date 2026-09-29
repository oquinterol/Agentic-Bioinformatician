import json
import os
import signal
import subprocess
import sys
import time

from genome_agent.agent.simulation import Scenario, build_project
from genome_agent.bridge import call
from genome_agent.executor.validation import JobRequest
from genome_agent.harness import Harness
from genome_agent.state.models import DecisionRecord, JobStatus, ProjectState

SCENARIO = {
    "name": "bg",
    "expect": "achieved",
    "machine": {"cpu_threads": 8, "ram_available_gb": 8, "disk_free_gb": 40},  # 6 thr, 6.4 GB
    "context": {"genome_size_bp": 1_000_000},
    "datasets": [{"name": "hifi", "kind": "pacbio_hifi"}],
    "tools": [
        {"name": "slow", "base_ram_gb": 4, "sleep_s": 30},
        {"name": "short", "base_ram_gb": 4, "sleep_s": 1.5},
        {"name": "quick", "base_ram_gb": 1},
    ],
}


def project(tmp_path):
    return build_project(Scenario.model_validate(SCENARIO), tmp_path / "p")


def req(h, tool, **kw):
    base = {
        "tool": tool,
        "inputs": [h.state.datasets[0].path],
        "genome_size_bp": 10**6,
        "cpus": 2,
        "ram_gb": 4,
        "reason": "t",
    }
    return JobRequest(**(base | kw))


def test_run_tool_can_return_while_job_runs(tmp_path):
    h = project(tmp_path)
    job = h.run_tool(req(h, "short"), wait_s=0)
    assert job.status == JobStatus.RUNNING and job.runner_pid
    other = Harness(h.project_dir, h.registry)  # e.g. a later CLI call
    assert other.job(job.id).status == JobStatus.RUNNING
    done = other.wait_job(job.id, timeout_s=30)
    assert done.status == JobStatus.SUCCEEDED and done.peak_rss_gb is not None
    assert ProjectState.load(h.project_dir).results[0].job_id == job.id


def test_running_jobs_reserve_budget(tmp_path):
    h = project(tmp_path)
    first = h.run_tool(req(h, "short", cpus=4), wait_s=0)
    second = h.run_tool(req(h, "quick", cpus=1, ram_gb=4))
    assert second.status == JobStatus.REJECTED
    assert any("RAM > budget 2.4" in r for r in second.rejection_reasons), second.rejection_reasons
    assert any(first.id in r for r in second.rejection_reasons)
    h.wait_job(first.id, 30)
    third = h.run_tool(req(h, "quick", cpus=4, ram_gb=4))
    assert third.status == JobStatus.SUCCEEDED


def test_cancel_kills_the_job(tmp_path):
    h = project(tmp_path)
    job = h.run_tool(req(h, "slow"), wait_s=0)
    t0 = time.monotonic()
    cancelled = h.cancel_job(job.id)
    assert cancelled.status == JobStatus.CANCELLED and time.monotonic() - t0 < 10
    assert "cancelled" in cancelled.error


def test_vanished_runner_is_detected(tmp_path):
    h = project(tmp_path)
    job = h.run_tool(req(h, "short"), wait_s=0)
    os.kill(job.runner_pid, signal.SIGKILL)
    lost = h.wait_job(job.id, 10)
    assert lost.status == JobStatus.FAILED and "disappeared" in lost.error
    assert any("disappeared" in f for f in h.state.failures)


def test_job_outlives_the_process_that_launched_it(tmp_path):
    h = project(tmp_path)
    args = {"tool": "short", "cpus": 1, "ram_gb": 4, "reason": "bg", "wait_s": 0}
    code = (
        "import json; from pathlib import Path; from genome_agent.bridge import call; "
        f"print(json.dumps(call(Path({str(h.project_dir)!r}), 'run_tool', {args!r}, 'llm:t')))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    job_id = json.loads(out.stdout)["result"]["job"]["id"]  # launcher process has exited
    resp = call(h.project_dir, "wait_job", {"job_id": job_id, "timeout_s": 30}, "llm:t")
    assert resp["ok"] and resp["result"]["job"]["status"] == "succeeded"
    assert resp["result"]["results"][0]["kind"] == "contigs_fasta"


def test_bridge_job_ops(tmp_path):
    h = project(tmp_path)
    args = {"tool": "slow", "cpus": 1, "ram_gb": 4, "reason": "x", "wait_s": 0}
    r = call(h.project_dir, "run_tool", args, "a")
    job_id = r["result"]["job"]["id"]
    assert "still running" in r["result"]["hint"]
    status = call(h.project_dir, "job_status", {"job_id": job_id}, "a")["result"]
    assert status["job"]["status"] == "running"
    cancel = call(h.project_dir, "cancel_job", {"job_id": job_id}, "a")["result"]
    assert cancel["job"]["status"] == "cancelled"
    assert not call(h.project_dir, "job_status", {"job_id": "nope"}, "a")["ok"]
    too_long = call(h.project_dir, "wait_job", {"job_id": job_id, "timeout_s": 99999}, "a")
    assert not too_long["ok"]


def test_concurrent_harnesses_do_not_lose_updates(tmp_path):
    a = project(tmp_path)
    b = Harness(a.project_dir, a.registry)
    a.record_decision(DecisionRecord(decision="from_a", reason="x"))
    b.record_decision(DecisionRecord(decision="from_b", reason="y"))
    a.record_decision(DecisionRecord(decision="from_a_again", reason="z"))
    names = [d.decision for d in ProjectState.load(a.project_dir).decisions]
    assert names == ["from_a", "from_b", "from_a_again"]
