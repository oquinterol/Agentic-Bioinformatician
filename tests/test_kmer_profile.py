import shutil
from pathlib import Path

import pytest

from genome_agent.agent.backend import Observation, RunTool
from genome_agent.agent.planner import DeterministicPlanner
from genome_agent.executor.validation import JobRequest
from genome_agent.harness import Harness
from genome_agent.resources.models import ResourcePolicy, ToolInfo
from genome_agent.state.models import Dataset, JobStatus, ProjectState, ReadKind
from genome_agent.tools.adapters.kmer import KmerProfile, KmerProfileParams
from genome_agent.tools.kmer_profile import parse_summary
from genome_agent.tools.registry import DataType, ToolInputs, default_registry
from genome_agent.toydata import make_toy_hifi

from .conftest import make_resources

DATA = Path(__file__).parent / "data"
needs_kmer_tools = pytest.mark.skipif(
    not (shutil.which("jellyfish") and shutil.which("genomescope.R")),
    reason="jellyfish/genomescope.R missing",
)


def test_parse_real_genomescope_summary():
    m = parse_summary((DATA / "genomescope_summary.txt").read_text())
    assert m["genome_haploid_length"] == {"min": 1940522.0, "max": 1970113.0}
    assert m["heterozygous_ab"]["min"] == pytest.approx(0.998618)
    assert m["read_error_rate"]["max"] == pytest.approx(0.10005)


def test_estimate_scales_with_genome_and_error_kmers():
    k = KmerProfile()
    small = ToolInputs(
        files=(Path("a.fq"),), input_bytes=0, genome_size_bp=2_000_000, read_bases=60_000_000
    )
    potato = ToolInputs(
        files=(Path("a.fq"),), input_bytes=0, genome_size_bp=800_000_000, read_bases=28_000_000_000
    )
    assert k.estimate(small, KmerProfileParams(), 2).ram_gb < 1
    unknown = ToolInputs(files=(Path("a.fq"),), input_bytes=0, read_bases=60_000_000)
    assert k.estimate(unknown, KmerProfileParams(), 2).ram_gb < 1  # capped by read bases
    big = k.estimate(potato, KmerProfileParams(), 2)
    assert 12 < big.ram_gb < 15  # 1.6 G genome entries + 0.59 G error k-mers at 6.5 B
    assert "jellyfish hash" in big.basis


def diploid_project(tmp_path, genome_size_declared):
    reads = make_toy_hifi(
        tmp_path / "seq", genome_size=2_000_000, coverage=30, heterozygosity=0.01, seed=5
    ).reads_fastq
    tools = {
        t: ToolInfo(name=t, path=shutil.which(t) or f"/usr/bin/{t}")
        for t in ("jellyfish", "genomescope.R", "hifiasm")
    }
    s = ProjectState(
        name="p",
        system_resources=make_resources(ram_available_gb=8, tools=tools),
        datasets=[Dataset(path=str(reads), kind=ReadKind.HIFI)],
    )
    s.biological_context.genome_size_bp = genome_size_declared
    s.biological_context.expected_ploidy = 2
    s.save(tmp_path / "proj")
    return Harness(tmp_path / "proj", default_registry()), reads


def test_planner_profiles_first_when_genome_size_is_unknown(tmp_path):
    h, _ = diploid_project(tmp_path, None)
    obs = Observation(
        h.state,
        h.registry,
        ResourcePolicy().apply(h.state.system_resources),
        DataType.CONTIGS_FASTA,
    )
    action = DeterministicPlanner().next_action(obs)
    assert isinstance(action, RunTool) and action.request.tool == "kmer_profile"
    assert action.request.params == {"ploidy": 2}


def test_declared_genome_size_skips_profiling(tmp_path):
    h, _ = diploid_project(tmp_path, 2_000_000)
    assert h.state.effective_genome_size() == (2_000_000, "declared")
    obs = Observation(
        h.state,
        h.registry,
        ResourcePolicy().apply(h.state.system_resources),
        DataType.CONTIGS_FASTA,
    )
    assert DeterministicPlanner().next_action(obs).request.tool == "hifiasm"


@needs_kmer_tools
def test_real_profile_recovers_known_genome(tmp_path):
    h, reads = diploid_project(tmp_path, None)
    job = h.run_tool(
        JobRequest(
            tool="kmer_profile",
            params={"ploidy": 2},
            inputs=[str(reads)],
            cpus=2,
            ram_gb=2,
            reason="measure genome size",
        )
    )
    assert job.status == JobStatus.SUCCEEDED, job.error or job.rejection_reasons
    prof = h.state.results[-1].data["files"][str(reads)]
    assert abs(prof["haploid_length_bp"] - 2_000_000) / 2_000_000 < 0.05
    assert 0.8 < prof["heterozygosity_pct"] < 1.5
    size, source = h.state.effective_genome_size()
    assert size == prof["haploid_length_bp"] and source.startswith("kmer_profile")
