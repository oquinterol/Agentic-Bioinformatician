import subprocess

from genome_agent.resources.models import ResourcePolicy
from genome_agent.tools.adapters.mock import MockAssembler, MockAssemblerParams
from genome_agent.tools.feasibility import assess
from genome_agent.tools.registry import ToolInputs

INPUTS = ToolInputs(files=(), input_bytes=0, genome_size_bp=800_000_000)
P = MockAssemblerParams()


def test_design_scenario_rejects_32gb_assembler_accepts_6gb(small_machine):
    budget = ResourcePolicy().apply(small_machine)  # 6 threads, 6.4 GB
    a = assess(MockAssembler("assembler_A", base_ram_gb=32), INPUTS, P, budget)
    b = assess(MockAssembler("assembler_B", base_ram_gb=6), INPUTS, P, budget)
    assert not a.fits and "32.0 GB RAM > budget 6.4" in a.reasons[0]
    assert b.fits and b.estimate.ram_gb == 6


def test_fewer_threads_is_tried_before_rejecting(small_machine):
    budget = ResourcePolicy().apply(small_machine)
    tool = MockAssembler("per_thread", base_ram_gb=2, ram_gb_per_thread=1)  # 2 + cpus
    res = assess(tool, INPUTS, P, budget)
    assert res.fits and res.estimate.cpus == 4  # 6 GB <= 6.4; 5 threads would be 7 GB


def test_unknown_genome_size_is_a_knowledge_gap(small_machine):
    budget = ResourcePolicy().apply(small_machine)
    no_size = ToolInputs(files=(), input_bytes=0)
    res = assess(MockAssembler("x"), no_size, P, budget)
    assert not res.fits and "profile the reads first" in res.reasons[0]


def test_mock_assembler_runs_and_fails_for_real(tmp_path):
    ok, bad = MockAssembler("ok", n50=42), MockAssembler("bad", fail_exit_code=3)
    subprocess.run(ok.build_command(INPUTS, P, tmp_path, 1), check=True)
    assert ok.parse_result(tmp_path) == {"contig_n50": 42, "n_contigs": 1}
    proc = subprocess.run(bad.build_command(INPUTS, P, tmp_path / "b", 1), capture_output=True)
    assert proc.returncode == 3 and b"simulated failure" in proc.stderr
