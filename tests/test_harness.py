import os
import shutil
import sys
from pathlib import Path

import pytest

from genome_agent.executor.local import LocalExecutor
from genome_agent.executor.validation import JobRequest
from genome_agent.harness import Harness
from genome_agent.provenance.log import ProvenanceLog
from genome_agent.resources.models import ToolInfo
from genome_agent.state.models import JobStatus, ProjectState
from genome_agent.tools.adapters.mock import MockAssembler
from genome_agent.tools.adapters.seqkit import SeqkitStats
from genome_agent.tools.registry import ToolRegistry

from .conftest import make_resources

GENOME = 800_000_000


@pytest.fixture
def reads(tmp_path) -> Path:
    """Input data OUTSIDE the project directory, as on a real machine."""
    d = tmp_path / "sequencer_output"
    d.mkdir()
    f = d / "reads.fq"
    f.write_text("@r1\nACGTACGTAC\n+\nIIIIIIIIII\n@r2\nACGTA\n+\nIIIII\n")
    return f


def make_harness(tmp_path, *adapters, resources=None) -> Harness:
    project = tmp_path / "project"
    ProjectState(name="p", system_resources=resources or make_resources()).save(project)
    return Harness(project, ToolRegistry(list(adapters)))


def req(tool, reads, **kw) -> JobRequest:
    base = dict(
        tool=tool, inputs=[str(reads)], cpus=2, ram_gb=4, genome_size_bp=GENOME, reason="test"
    )
    return JobRequest(**(base | kw))


def test_success_records_everything(tmp_path, reads):
    h = make_harness(tmp_path, MockAssembler("asm", base_ram_gb=3, n50=77))
    job = h.run_tool(req("asm", reads, alternatives_considered=["other"]))

    assert job.status == JobStatus.SUCCEEDED and job.exit_code == 0
    assert Path(job.outdir).is_relative_to(h.project_dir / "runs")
    assert "done" in Path(job.stdout_path).read_text()
    assert (Path(job.outdir) / "command.sh").read_text().startswith("#!/bin/sh\n")

    state = ProjectState.load(h.project_dir)  # persisted, resumable
    assert state.results[0].data == {"contig_n50": 77, "n_contigs": 1}
    assert state.decisions[-1].alternatives_considered == ["other"]
    events = [e["event"] for e in ProvenanceLog(h.project_dir).read()]
    assert events == ["job_started", "job_finished"]


def test_nonzero_exit_is_a_recorded_failure(tmp_path, reads):
    h = make_harness(tmp_path, MockAssembler("asm", fail_exit_code=3))
    job = h.run_tool(req("asm", reads))
    assert job.status == JobStatus.FAILED and job.exit_code == 3
    assert "simulated failure" in Path(job.stderr_path).read_text()
    assert "exit code 3" in ProjectState.load(h.project_dir).failures[0]


def test_timeout_kills_the_job(tmp_path, reads):
    h = make_harness(tmp_path, MockAssembler("slow", sleep_s=30))
    job = h.run_tool(req("slow", reads, timeout_s=0.5))
    assert job.status == JobStatus.FAILED and job.timed_out
    assert job.wall_time_s < 10


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"tool": "nope"}, "unknown tool 'nope'"),
        ({"params": {"threads": 99}}, "invalid params"),
        ({"inputs": ["/definitely/not/here.fq"]}, "input not found"),
        ({"cpus": 7}, "requested 7 threads > budget 6"),
        ({"ram_gb": 7}, "requested 7.0 GB RAM > budget 6.4"),
        ({"ram_gb": 1}, "< estimated peak 3.0 GB"),
        ({"genome_size_bp": None}, "profile the reads first"),
    ],
)
def test_rejections(tmp_path, reads, overrides, expected):
    h = make_harness(tmp_path, MockAssembler("asm", base_ram_gb=3))
    job = h.run_tool(req(overrides.pop("tool", "asm"), reads, **overrides))
    assert job.status == JobStatus.REJECTED
    assert any(expected in r for r in job.rejection_reasons), job.rejection_reasons
    assert job.argv == [] and job.outdir is None  # nothing was run or created
    assert ProvenanceLog(h.project_dir).read()[0]["event"] == "job_rejected"


def test_estimate_over_budget_is_rejected_even_if_request_looks_small(tmp_path, reads):
    h = make_harness(tmp_path, MockAssembler("assembler_A", base_ram_gb=32))
    job = h.run_tool(req("assembler_A", reads))
    assert job.status == JobStatus.REJECTED
    assert any("32.0 GB RAM > budget" in r for r in job.rejection_reasons)


def test_undetected_executable_is_rejected(tmp_path, reads):
    h = make_harness(tmp_path, SeqkitStats())  # make_resources() detects no tools
    job = h.run_tool(req("seqkit_stats", reads, params={}))
    assert "not detected" in job.rejection_reasons[0]


def test_launch_failure_is_reported(tmp_path):
    r = LocalExecutor().run(["/no/such/binary"], tmp_path, 1, 1, 5)
    assert not r.ok and r.exit_code is None and r.error.startswith("launch:")


def test_executor_limits_openmp_threads(tmp_path):
    code = "import os; print(os.environ['OMP_NUM_THREADS'])"
    r = LocalExecutor().run([sys.executable, "-c", code], tmp_path, 3, 1, 5)
    assert r.ok and r.stdout_path.read_text().strip() == "3"


@pytest.mark.skipif(shutil.which("seqkit") is None, reason="seqkit not installed")
def test_real_seqkit_through_harness(tmp_path, reads):
    exe = shutil.which("seqkit")
    res = make_resources(tools={"seqkit": ToolInfo(name="seqkit", path=exe)})
    h = make_harness(tmp_path, SeqkitStats(), resources=res)
    job = h.run_tool(req("seqkit_stats", reads, params={"all_stats": True}, ram_gb=1))
    assert job.status == JobStatus.SUCCEEDED, job.error
    assert job.argv[0] == exe  # pinned to the inventoried binary
    stats = ProjectState.load(h.project_dir).results[0].data["files"][str(reads)]
    assert stats["num_seqs"] == 2 and stats["N50"] == 10
    assert os.access(reads, os.R_OK) and reads.read_text().startswith("@r1")  # input untouched
