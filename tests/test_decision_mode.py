import shutil

from genome_agent import cli
from genome_agent.bridge import call
from genome_agent.executor.validation import JobRequest
from genome_agent.harness import Harness
from genome_agent.resources.models import ToolInfo
from genome_agent.state.models import (
    BiologicalContext,
    Dataset,
    JobStatus,
    ProjectState,
    ReadKind,
    Result,
)
from genome_agent.tools.registry import default_registry

from .conftest import make_resources


def hifi_project(tmp_path, blocked=(), results=()):
    reads = tmp_path / "reads.fq"
    reads.write_bytes(b"x" * 4_000_000)  # file-size heuristic: 2 Mbp
    exe = shutil.which("hifiasm") or "/usr/bin/hifiasm"
    state = ProjectState(
        name="p",
        system_resources=make_resources(
            ram_available_gb=60, tools={"hifiasm": ToolInfo(name="hifiasm", path=exe)}
        ),
        datasets=[Dataset(path=str(reads), kind=ReadKind.HIFI)],
        biological_context=BiologicalContext(genome_size_bp=10**6),
        blocked_tools=list(blocked),
        results=list(results),
    )
    state.save(tmp_path / "proj")
    return Harness(tmp_path / "proj", default_registry()), reads


def test_blocked_tool_is_rejected_by_the_harness(tmp_path):
    h, reads = hifi_project(tmp_path, blocked=["hifiasm"])
    job = h.run_tool(
        JobRequest(
            tool="hifiasm",
            params={"bloom_bits": 0},
            inputs=[str(reads)],
            cpus=1,
            ram_gb=5,
            reason="x",
        )
    )
    assert job.status == JobStatus.REJECTED and "blocked" in job.rejection_reasons[0]
    status = call(h.project_dir, "project_status", {}, "t")["result"]
    assert status["blocked_tools"] == ["hifiasm"]
    a = call(h.project_dir, "assess_tool", {"tool": "hifiasm"}, "t")["result"]
    assert a["blocked_in_project"] is True  # assessment still works for planning


def test_measured_read_bases_replace_the_file_size_heuristic(tmp_path):
    stats = Result(
        job_id="0001-seqkit_stats",
        kind="read_stats",
        data={"files": {str(tmp_path / "reads.fq"): {"sum_len": 30_000_000_000}}},
    )
    h, reads = hifi_project(tmp_path, results=[stats])
    assert h.state.measured_read_bases([str(reads)]) == 30_000_000_000
    assert h.state.measured_read_bases([str(reads), "/other.fq"]) is None
    a = call(h.project_dir, "assess_tool", {"tool": "hifiasm", "params": {"bloom_bits": 0}}, "t")[
        "result"
    ]
    assert a["estimate"]["ram_gb"] == 30.5 and "measured read bases" in a["estimate"]["basis"]


def test_plan_dry_run_records_plan_and_runs_nothing(tmp_path, capsys):
    h, _ = hifi_project(tmp_path, blocked=["hifiasm"])
    assert cli.main(["plan", str(h.project_dir), "--dry-run"]) == 0
    state = ProjectState.load(h.project_dir)
    assert state.jobs == []
    plan = state.decisions[-1]
    assert plan.decision == "assembly_plan" and plan.actor == "planner:deterministic"
    assert plan.evidence["plan"]["tool"] == "hifiasm" and plan.evidence["dry_run"]


def test_init_records_biological_context_and_blocks(tmp_path, monkeypatch, small_machine):
    monkeypatch.setattr(cli, "inspect_system", lambda ws=None: small_machine)
    cli.main(
        [
            "init",
            str(tmp_path / "p"),
            "--species",
            "Solanum tuberosum Group Phureja",
            "--ploidy",
            "2",
            "--genome-size",
            "800000000",
            "--block-tool",
            "hifiasm",
        ]
    )
    s = ProjectState.load(tmp_path / "p")
    assert s.biological_context.expected_ploidy == 2
    assert s.biological_context.genome_size_bp == 800_000_000
    assert s.blocked_tools == ["hifiasm"]
