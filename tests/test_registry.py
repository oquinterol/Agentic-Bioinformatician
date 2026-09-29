import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from genome_agent.resources.models import ToolInfo
from genome_agent.tools.adapters.mock import MockAssembler
from genome_agent.tools.adapters.seqkit import OUTPUT_FILE, SeqkitStats, SeqkitStatsParams
from genome_agent.tools.registry import DataType, ToolInputs, ToolRegistry, default_registry

from .conftest import make_resources

DATA = Path(__file__).parent / "data"


def test_registry_rejects_duplicates_and_explains_unknown():
    reg = ToolRegistry([MockAssembler("a")])
    with pytest.raises(ValueError, match="already registered"):
        reg.register(MockAssembler("a"))
    with pytest.raises(KeyError, match="registered: \\['a'\\]"):
        reg.get("nope")


def test_available_depends_on_detected_executables(small_machine):
    reg = default_registry()
    assert reg.available(small_machine) == []
    with_seqkit = make_resources(tools={"seqkit": ToolInfo(name="seqkit", path="/bin/seqkit")})
    assert [a.name for a in reg.producing(DataType.READ_STATS, with_seqkit)] == ["seqkit_stats"]


def test_params_reject_unknown_fields():
    # The LLM cannot smuggle extra flags through params.
    with pytest.raises(ValidationError):
        SeqkitStats().parse_params({"all_stats": True, "extra_args": "; rm -rf /"})


def test_describe_exposes_json_schema():
    d = SeqkitStats().describe()
    assert d["params_schema"]["properties"]["all_stats"]["type"] == "boolean"
    assert d["output_types"] == ["read_stats"]


def test_seqkit_command_is_argv_not_shell(tmp_path):
    inputs = ToolInputs(files=(tmp_path / "a.fq",), input_bytes=100)
    argv = SeqkitStats().build_command(inputs, SeqkitStatsParams(), tmp_path, cpus=3)
    assert argv[:2] == ["seqkit", "stats"]
    assert argv[argv.index("--threads") + 1] == "3"
    assert argv[-1] == str(tmp_path / "a.fq")


def test_seqkit_estimate_caps_threads_and_scales_ram():
    small = ToolInputs(files=(DATA,), input_bytes=10)
    big = ToolInputs(files=(DATA,) * 8, input_bytes=50 * 10**9)  # 50 GB of reads
    s = SeqkitStats()
    assert s.estimate(small, SeqkitStatsParams(), cpus=16).cpus == 1
    assert s.estimate(big, SeqkitStatsParams(), cpus=16).cpus == 4
    assert s.estimate(big, SeqkitStatsParams(), 4).ram_gb == pytest.approx(8.5)
    assert s.estimate(big, SeqkitStatsParams(all_stats=False), 4).ram_gb == 0.5


def test_seqkit_parse_fixture(tmp_path):
    shutil.copy(DATA / "seqkit_stats.tsv", tmp_path / OUTPUT_FILE)
    stats = SeqkitStats().parse_result(tmp_path)["files"]["/w/t.fq"]
    assert stats["num_seqs"] == 2 and stats["N50"] == 10 and stats["GC(%)"] == 46.67


@pytest.mark.skipif(shutil.which("seqkit") is None, reason="seqkit not installed")
def test_seqkit_real_roundtrip(tmp_path):
    fq = tmp_path / "t.fq"
    fq.write_text("@r1\nACGTACGTAC\n+\nIIIIIIIIII\n@r2\nACGTA\n+\nIIIII\n")
    s = SeqkitStats()
    inputs = ToolInputs.from_files([fq])
    subprocess.run(s.build_command(inputs, SeqkitStatsParams(), tmp_path, 1), check=True)
    stats = s.parse_result(tmp_path)["files"][str(fq)]
    assert (stats["num_seqs"], stats["sum_len"], stats["N50"]) == (2, 15, 10)
