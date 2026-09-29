import shutil

import pytest

from genome_agent.agent.loop import run_loop
from genome_agent.agent.planner import DeterministicPlanner
from genome_agent.executor.validation import JobRequest
from genome_agent.harness import Harness
from genome_agent.resources.models import ToolInfo
from genome_agent.state.models import Dataset, JobStatus, ProjectState, ReadKind
from genome_agent.tools.adapters.consistency import ConsistencyParams, LibraryConsistencyCheck
from genome_agent.tools.consistency_check import verdict
from genome_agent.tools.registry import DataType, ToolInputs, default_registry
from genome_agent.toydata import make_toy_hifi

from .conftest import make_resources

needs_tools = pytest.mark.skipif(
    not (shutil.which("minimap2") and shutil.which("hifiasm")), reason="minimap2/hifiasm missing"
)
G = 150_000


def test_verdict_is_relative_to_the_self_baseline():
    assert verdict(0.30, 0.36) == "consistent"
    assert verdict(0.0, 0.36) == "inconsistent"
    assert verdict(0.10, 0.36) == "ambiguous"
    assert verdict(0.0, 0.01).startswith("undetermined")


def test_adapter_contract(tmp_path):
    a = LibraryConsistencyCheck()
    one = ToolInputs(files=(tmp_path / "a",), input_bytes=1, kinds=("pacbio_hifi",))
    assert "at least two" in a.check_inputs(one)[0]
    two = ToolInputs(
        files=(tmp_path / "a", tmp_path / "b"),
        input_bytes=1,
        kinds=("pacbio_hifi", "pacbio_hifi"),
        genome_size_bp=800_000_000,
    )
    est = a.estimate(two, ConsistencyParams(), 4)
    assert est.ram_gb == 5.0  # 1 GB + 5 B/base x 0.8 Gbp (1x of the genome), rounded up
    argv = a.build_command(two, ConsistencyParams(), tmp_path, 4)
    assert argv[argv.index("--target-bases") + 1] == "800000000"


def toy_same_genome_different_reads(tmp_path):
    a1 = make_toy_hifi(tmp_path / "A1", genome_size=G, seed=11, read_seed=101)
    a2 = make_toy_hifi(tmp_path / "A2", genome_size=G, seed=11, read_seed=202)
    b = make_toy_hifi(tmp_path / "B", genome_size=G, seed=22)
    assert a1.genome_fasta.read_text() == a2.genome_fasta.read_text()
    assert a1.reads_fastq.read_bytes() != a2.reads_fastq.read_bytes()
    return a1.reads_fastq, a2.reads_fastq, b.reads_fastq


def no_reference_project(tmp_path, libs):
    tools = {
        t: ToolInfo(name=t, path=shutil.which(t) or f"/usr/bin/{t}")
        for t in ("minimap2", "hifiasm")
    }
    ProjectState(
        name="denovo",
        system_resources=make_resources(ram_available_gb=8, tools=tools),
        datasets=[Dataset(path=str(p), kind=ReadKind.HIFI, species="Genome A") for p in libs],
    ).save(tmp_path / "proj")
    p = ProjectState.load(tmp_path / "proj")
    p.biological_context.genome_size_bp = G
    p.save(tmp_path / "proj")
    return Harness(tmp_path / "proj", default_registry())


def test_pooling_without_reference_requires_a_consistency_check(tmp_path):
    a1, a2, _ = toy_same_genome_different_reads(tmp_path)
    h = no_reference_project(tmp_path, [a1, a2])
    job = h.run_tool(
        JobRequest(
            tool="hifiasm",
            params={"bloom_bits": 0},
            inputs=[str(a1), str(a2)],
            cpus=2,
            ram_gb=6,
            reason="pool",
        )
    )
    assert job.status == JobStatus.REJECTED
    assert "requires library_consistency_check" in job.rejection_reasons[0]


def test_single_library_de_novo_is_not_blocked(tmp_path):
    a1, _, _ = toy_same_genome_different_reads(tmp_path)
    h = no_reference_project(tmp_path, [a1])
    job = h.run_tool(
        JobRequest(
            tool="hifiasm",
            params={"bloom_bits": 0},
            inputs=[str(a1)],
            cpus=2,
            ram_gb=6,
            reason="nothing to cross-check",
        )
    )
    assert not any("consistency" in r for r in job.rejection_reasons)


@needs_tools
def test_real_check_separates_same_genome_from_other_genome(tmp_path):
    a1, a2, b = toy_same_genome_different_reads(tmp_path)
    h = no_reference_project(tmp_path, [a1, a2, b])
    job = h.run_tool(
        JobRequest(
            tool="library_consistency_check",
            inputs=[str(a1), str(a2), str(b)],
            genome_size_bp=G,
            cpus=2,
            ram_gb=2,
            reason="check",
            params={"sample_reads": 200},
        )
    )
    assert job.status == JobStatus.SUCCEEDED, job.error or job.rejection_reasons
    v = h.state.consistency_verdicts()
    assert v[frozenset((str(a1), str(a2)))] == {"consistent"}
    assert v[frozenset((str(a1), str(b)))] == {"inconsistent"}
    rejected = h.run_tool(
        JobRequest(
            tool="hifiasm",
            params={"bloom_bits": 0},
            inputs=[str(a1), str(b)],
            cpus=2,
            ram_gb=6,
            reason="pool",
        )
    )
    assert "must not pool them" in rejected.rejection_reasons[0]


@needs_tools
def test_planner_checks_then_pools_consistent_libraries(tmp_path):
    a1, a2, _ = toy_same_genome_different_reads(tmp_path)
    h = no_reference_project(tmp_path, [a1, a2])
    outcome = run_loop(h, DeterministicPlanner(), DataType.CONTIGS_FASTA)
    assert [(j.tool, j.status) for j in h.state.jobs] == [
        ("library_consistency_check", JobStatus.SUCCEEDED),
        ("hifiasm", JobStatus.SUCCEEDED),
    ]
    assert outcome.achieved and sorted(h.state.jobs[1].inputs) == sorted([str(a1), str(a2)])


@needs_tools
def test_planner_stops_when_libraries_disagree_and_no_reference(tmp_path):
    a1, _, b = toy_same_genome_different_reads(tmp_path)
    h = no_reference_project(tmp_path, [a1, b])
    outcome = run_loop(h, DeterministicPlanner(), DataType.CONTIGS_FASTA)
    assert not outcome.achieved
    assert [j.tool for j in h.state.jobs] == ["library_consistency_check"]  # never assembled
    reasons = h.state.decisions[-1].evidence["rejected"]["hifiasm"][0]
    assert "inconsistent" in reasons and "register a reference" in reasons
