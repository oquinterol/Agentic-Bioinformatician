import shutil
import sys
from pathlib import Path

import pytest

from genome_agent.agent.backend import Observation, RunTool
from genome_agent.agent.planner import DeterministicPlanner
from genome_agent.executor.local import LocalExecutor
from genome_agent.harness import Harness
from genome_agent.resources.models import ResourcePolicy, ToolInfo
from genome_agent.state.models import Dataset, JobStatus, ProjectState, ReadKind
from genome_agent.tools.adapters.hifiasm import Hifiasm, HifiasmParams, estimate_read_bases
from genome_agent.tools.registry import DataType, ToolInputs, default_registry
from genome_agent.tools.seqstats import gfa_to_fasta, length_stats
from genome_agent.toydata import make_toy_hifi

from .conftest import make_resources

GIB = 1024**3


def test_bloom_filter_term_matches_manual():
    h = Hifiasm()
    tiny = ToolInputs(files=(), input_bytes=0, read_bases=0)
    default = h.estimate(tiny, HifiasmParams(), cpus=4)
    no_bloom = h.estimate(tiny, HifiasmParams(bloom_bits=0), cpus=4)
    assert default.ram_gb == pytest.approx(0.5 + 16.0)  # 2^(37-3) B = 16 GiB
    assert no_bloom.ram_gb == pytest.approx(0.5)
    assert h.estimate(tiny, HifiasmParams(bloom_bits=38), 4).ram_gb == pytest.approx(32.5)
    assert "verified" in default.basis and "1 real run" in default.basis
    assert "UNCALIBRATED" in no_bloom.basis  # -f0 counting at scale is not measured yet


def test_model_reproduces_the_real_phureja_run():
    """hifiasm 0.25, 28.24 Gbp HiFi, -f37, observed peak 24.93 GB (max of phases, not a sum)."""
    inputs = ToolInputs(files=(), input_bytes=10_372_360_661, read_bases=28_241_525_595)
    est = Hifiasm().estimate(inputs, HifiasmParams(), 14)
    assert 24.93 <= est.ram_gb <= 24.93 * 1.10  # conservative, within 10 %


def test_read_bases_measured_or_inferred(tmp_path):
    fq, gz = tmp_path / "r.fq", tmp_path / "r.fq.gz"
    fq.write_bytes(b"x" * 2_000_000)
    gz.write_bytes(b"x" * 2_000_000)
    assert estimate_read_bases(ToolInputs.from_files([fq]))[0] == 1_000_000  # ~2 B/base
    assert estimate_read_bases(ToolInputs.from_files([gz]))[0] == 3_000_000  # gz ~3x
    measured = Hifiasm().estimate(
        ToolInputs.from_files([fq], read_bases=10**9), HifiasmParams(bloom_bits=0), 1
    )
    assert measured.ram_gb == pytest.approx(1.5) and "measured" in measured.basis


def test_command_flags(tmp_path):
    inputs = ToolInputs(files=(Path("/data/a.fq"), Path("/data/b.fq")), input_bytes=1)
    p = HifiasmParams(bloom_bits=0, purge_level=1, primary=True, hom_cov=40)
    argv = Hifiasm().build_command(inputs, p, tmp_path, cpus=6)
    assert argv == [
        "hifiasm",
        "-o",
        str(tmp_path / "asm"),
        "-t",
        "6",
        "-f",
        "0",
        "-l",
        "1",
        "--primary",
        "--hom-cov",
        "40",
        "/data/a.fq",
        "/data/b.fq",
    ]


def test_params_are_bounded():
    with pytest.raises(ValueError):
        HifiasmParams(purge_level=7)


def test_length_stats():
    assert length_stats([10, 20, 30, 40]) == {
        "n_contigs": 4,
        "total_bp": 100,
        "n50_bp": 30,
        "largest_bp": 40,
    }
    assert length_stats([])["n50_bp"] == 0


def test_parse_result_from_gfa(tmp_path):
    (tmp_path / "asm.bp.p_ctg.gfa").write_text(
        "H\tVN:Z:1.0\nS\tptg1\tACGTACGT\tLN:i:8\nS\tptg2\tAC\n"
    )
    (tmp_path / "asm.bp.hap1.p_ctg.gfa").write_text("S\th1\tACGT\n")
    data = Hifiasm().parse_result(tmp_path)
    assert data["primary"]["n_contigs"] == 2 and data["primary"]["n50_bp"] == 8
    assert "hap2" not in data and data["hap1"]["total_bp"] == 4
    assert Path(data["primary"]["fasta"]).read_text() == ">ptg1\nACGTACGT\n>ptg2\nAC\n"


def test_parse_result_without_contigs_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="no primary contig GFA"):
        Hifiasm().parse_result(tmp_path)


def test_gfa_to_fasta_ignores_links(tmp_path):
    gfa = tmp_path / "x.gfa"
    gfa.write_text("S\ta\tAAA\nL\ta\t+\tb\t-\t0M\nS\tb\tCC\n")
    assert gfa_to_fasta(gfa, tmp_path / "x.fa")["total_bp"] == 5


def test_describe_advertises_low_memory_variant():
    assert {"bloom_bits": 0} in Hifiasm().describe()["param_variants"]


def small_hifiasm_project(tmp_path, ram_gb=8.0):
    reads = make_toy_hifi(tmp_path / "seq").reads_fastq
    exe = shutil.which("hifiasm") or "/usr/bin/hifiasm"
    res = make_resources(
        ram_available_gb=ram_gb, tools={"hifiasm": ToolInfo(name="hifiasm", path=exe)}
    )
    proj = tmp_path / "proj"
    ProjectState(
        name="p",
        system_resources=res,
        datasets=[Dataset(path=str(reads), kind=ReadKind.HIFI)],
    ).save(proj)
    return Harness(proj, default_registry())


def test_planner_disables_bloom_filter_on_8gb_machine(tmp_path):
    h = small_hifiasm_project(tmp_path)
    budget = ResourcePolicy().apply(h.state.system_resources)
    action = DeterministicPlanner().next_action(
        Observation(h.state, h.registry, budget, DataType.CONTIGS_FASTA)
    )
    assert isinstance(action, RunTool)
    assert action.request.tool == "hifiasm" and action.request.params == {"bloom_bits": 0}
    assert any(
        "16.50 GB RAM" in r or "16.5" in r for r in action.request.evidence["rejected_variants"]
    )


def test_peak_rss_is_measured(tmp_path):
    code = "b = bytearray(300 * 1024 * 1024); b[::4096] = b'x' * len(b[::4096])"
    r = LocalExecutor().run([sys.executable, "-c", code], tmp_path, 1, 1, 30)
    assert r.ok and 0.25 < r.peak_rss_gb < 1.0


@pytest.mark.skipif(shutil.which("hifiasm") is None, reason="hifiasm not installed")
def test_real_hifiasm_on_toy_genome_within_8gb_budget(tmp_path):
    from genome_agent.agent.loop import run_loop

    h = small_hifiasm_project(tmp_path)
    outcome = run_loop(h, DeterministicPlanner(), DataType.CONTIGS_FASTA)
    job = h.state.jobs[0]
    assert outcome.achieved and job.status == JobStatus.SUCCEEDED
    assert job.params == {"bloom_bits": 0} and "-f" in job.argv
    assert job.peak_rss_gb < 1.0  # -f0 really avoids the 16 GiB bloom filter
    assert job.peak_rss_gb <= job.estimate.ram_gb  # estimate was conservative
    primary = h.state.results[0].data["primary"]
    assert primary["n_contigs"] == 1
    assert abs(primary["total_bp"] - 200_000) / 200_000 < 0.01


def test_toy_data_is_deterministic_and_diploid(tmp_path):
    a = make_toy_hifi(tmp_path / "a", genome_size=20_000, coverage=5, read_length=4_000)
    b = make_toy_hifi(tmp_path / "b", genome_size=20_000, coverage=5, read_length=4_000)
    assert a.reads_fastq.read_bytes() == b.reads_fastq.read_bytes()
    d = make_toy_hifi(
        tmp_path / "d", genome_size=20_000, coverage=5, read_length=4_000, heterozygosity=0.01
    )
    haps = d.genome_fasta.read_text().split("\n")[1::2][:2]
    diffs = sum(x != y for x, y in zip(*haps, strict=True))
    assert diffs == 200  # 1 % of 20 kb


def project_with(tmp_path, datasets: list[tuple[str, ReadKind]]):
    exe = shutil.which("hifiasm") or "/usr/bin/hifiasm"
    res = make_resources(ram_available_gb=60, tools={"hifiasm": ToolInfo(name="hifiasm", path=exe)})
    ds = []
    for name, kind in datasets:
        f = tmp_path / name
        f.write_text("@r\nACGT\n+\nIIII\n")
        ds.append(Dataset(path=str(f), kind=kind))
    ProjectState(name="p", system_resources=res, datasets=ds).save(tmp_path / "proj")
    return Harness(tmp_path / "proj", default_registry())


def test_hifiasm_rejects_non_hifi_and_unregistered_inputs(tmp_path):
    from genome_agent.executor.validation import JobRequest

    h = project_with(tmp_path, [("ont.fq", ReadKind.ONT)])
    stray = tmp_path / "stray.fq"
    stray.write_text("@r\nA\n+\nI\n")
    for path, expected in [(tmp_path / "ont.fq", "is ont"), (stray, "not a registered dataset")]:
        job = h.run_tool(
            JobRequest(tool="hifiasm", inputs=[str(path)], cpus=1, ram_gb=20, reason="x")
        )
        assert job.status == JobStatus.REJECTED
        assert any(expected in r for r in job.rejection_reasons), job.rejection_reasons


def test_planner_uses_only_hifi_datasets_and_explains_missing_kind(tmp_path):
    h = project_with(tmp_path, [("hifi.fq", ReadKind.HIFI), ("hic.fq", ReadKind.HIC)])
    budget = ResourcePolicy().apply(h.state.system_resources)
    obs = Observation(h.state, h.registry, budget, DataType.CONTIGS_FASTA)
    action = DeterministicPlanner().next_action(obs)
    assert isinstance(action, RunTool) and action.request.inputs == [str(tmp_path / "hifi.fq")]

    h2 = (
        project_with(tmp_path / "b", [("ont.fq", ReadKind.ONT)])
        if (tmp_path / "b").mkdir() is None
        else None
    )
    obs2 = Observation(h2.state, h2.registry, budget, DataType.CONTIGS_FASTA)
    stop = DeterministicPlanner().next_action(obs2)
    assert not isinstance(stop, RunTool)
    assert (
        "no registered dataset of kind ['pacbio_hifi']" in stop.evidence["rejected"]["hifiasm"][0]
    )


def test_bridge_defaults_to_accepted_kinds(tmp_path):
    from genome_agent.bridge import call

    h = project_with(tmp_path, [("hifi.fq", ReadKind.HIFI), ("ont.fq", ReadKind.ONT)])
    r = call(h.project_dir, "assess_tool", {"tool": "hifiasm"}, "t")
    assert r["ok"]
    r2 = call(h.project_dir, "assess_tool", {"tool": "seqkit_stats"}, "t")  # accepts any
    assert r2["ok"]
    assert (
        "pacbio_hifi"
        in call(h.project_dir, "list_tools", {}, "t")["result"]["tools"][0]["accepted_read_kinds"]
    )
