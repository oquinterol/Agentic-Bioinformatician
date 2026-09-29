import gzip
import io
import shutil

import pytest

from genome_agent.agent.loop import run_loop
from genome_agent.agent.planner import DeterministicPlanner
from genome_agent.bridge import call
from genome_agent.executor.validation import JobRequest
from genome_agent.harness import Harness
from genome_agent.resources.models import ToolInfo
from genome_agent.state.models import Dataset, JobStatus, ProjectState, ReadKind
from genome_agent.tools.adapters.origin import ReadOriginCheck, ReadOriginParams
from genome_agent.tools.origin_check import sample_reads, verdict
from genome_agent.tools.registry import DataType, ToolInputs, default_registry
from genome_agent.toydata import make_toy_hifi

from .conftest import make_resources

needs_tools = pytest.mark.skipif(
    not (shutil.which("minimap2") and shutil.which("hifiasm")), reason="minimap2/hifiasm missing"
)


def test_sample_reads_fastq_fasta_and_gzip(tmp_path):
    fq = tmp_path / "r.fq.gz"
    with gzip.open(fq, "wt") as fh:
        fh.write("@a\nACGT\n+\nIIII\n@b\nGGCC\n+\nIIII\n@c\nTTTT\n+\nIIII\n")
    out = io.StringIO()
    assert sample_reads(fq, 2, 7, out) == 2
    assert out.getvalue() == ">7|0\nACGT\n>7|1\nGGCC\n"
    fa = tmp_path / "r.fa"
    fa.write_text(">x\nAC\nGT\n>y\nTT\n")
    out = io.StringIO()
    assert sample_reads(fa, 10, 0, out) == 2
    assert out.getvalue() == ">0|0\nACGT\n>0|1\nTT\n"


def test_verdict_thresholds():
    assert verdict(1500, 2000) == "matches reference"
    assert verdict(0, 2000) == "does not match reference"
    assert verdict(400, 2000) == "ambiguous"


def test_check_inputs_requires_one_reference_and_reads(tmp_path):
    a = ReadOriginCheck()
    only_reads = ToolInputs(files=(tmp_path / "r",), input_bytes=1, kinds=("pacbio_hifi",))
    two_refs = ToolInputs(
        files=(tmp_path / "a", tmp_path / "b"),
        input_bytes=1,
        kinds=("reference_fasta", "reference_fasta"),
    )
    ok = ToolInputs(
        files=(tmp_path / "a", tmp_path / "r"),
        input_bytes=1,
        kinds=("reference_fasta", "pacbio_hifi"),
    )
    assert "exactly one" in a.check_inputs(only_reads)[0]
    assert len(a.check_inputs(two_refs)) == 2
    assert a.check_inputs(ok) == []
    argv = a.build_command(ok, ReadOriginParams(sample_reads=500), tmp_path, 3)
    assert argv[argv.index("--reference") + 1] == str(tmp_path / "a") and argv[-1] == str(
        tmp_path / "r"
    )


def mixed_project(tmp_path, ram_gb=8.0):
    """Reference of genome A; reads of A and of an unrelated genome B, BOTH documented as A."""
    a = make_toy_hifi(tmp_path / "A", genome_size=150_000, seed=11)
    b = make_toy_hifi(tmp_path / "B", genome_size=150_000, seed=22)
    tools = {
        t: ToolInfo(name=t, path=shutil.which(t) or f"/usr/bin/{t}")
        for t in ("minimap2", "hifiasm")
    }
    species = "Genome A"
    ProjectState(
        name="mixed",
        system_resources=make_resources(ram_available_gb=ram_gb, tools=tools),
        datasets=[
            Dataset(path=str(a.genome_fasta), kind=ReadKind.REFERENCE, species=species),
            Dataset(path=str(a.reads_fastq), kind=ReadKind.HIFI, species=species),
            Dataset(path=str(b.reads_fastq), kind=ReadKind.HIFI, species=species),  # mislabelled
        ],
    ).save(tmp_path / "proj")
    return Harness(tmp_path / "proj", default_registry()), a, b


def test_hifiasm_never_takes_the_reference_as_reads(tmp_path):
    h, a, _ = mixed_project(tmp_path)
    r = call(h.project_dir, "assess_tool", {"tool": "hifiasm", "params": {"bloom_bits": 0}}, "t")
    assert r["ok"]
    job = h.run_tool(
        JobRequest(tool="hifiasm", inputs=[str(a.genome_fasta)], cpus=1, ram_gb=5, reason="x")
    )
    assert job.status == JobStatus.REJECTED and "is reference_fasta" in job.rejection_reasons[0]


@needs_tools
def test_real_origin_check_flags_the_mislabelled_reads(tmp_path):
    h, a, b = mixed_project(tmp_path)
    r = call(
        h.project_dir,
        "run_tool",
        {
            "tool": "read_origin_check",
            "cpus": 2,
            "ram_gb": 2,
            "reason": "verify",
            "params": {"sample_reads": 200},
        },
        "t",
    )
    assert r["ok"] and r["result"]["job"]["status"] == "succeeded", r
    files = r["result"]["results"][0]["data"]["files"]
    assert files[str(a.reads_fastq)]["verdict"] == "matches reference"
    assert files[str(b.reads_fastq)]["verdict"] == "does not match reference"


@needs_tools
def test_planner_checks_origin_first_and_assembles_only_matching_reads(tmp_path):
    h, a, b = mixed_project(tmp_path)
    outcome = run_loop(h, DeterministicPlanner(), DataType.CONTIGS_FASTA)
    tools = [(j.tool, j.status) for j in h.state.jobs]
    assert tools == [("read_origin_check", JobStatus.SUCCEEDED), ("hifiasm", JobStatus.SUCCEEDED)]
    assert outcome.achieved
    assembled = h.state.jobs[1]
    assert assembled.inputs == [str(a.reads_fastq)]  # B was excluded
    decision = next(d for d in h.state.decisions if d.decision == "run_hifiasm")
    assert decision.evidence["excluded_by_origin_check"] == {
        str(b.reads_fastq): "does not match reference"
    }
    primary = h.state.results[-1].data["primary"]
    assert abs(primary["total_bp"] - 150_000) / 150_000 < 0.02  # A only, not A + B
