import subprocess

import pytest

from genome_agent.resources.inspector import (
    detect_gpus,
    detect_tools,
    parse_cpuinfo,
    parse_meminfo,
    parse_version,
)
from genome_agent.tools.catalog import ToolProbe

MEMINFO = """MemTotal:       65536000 kB
MemFree:         1000000 kB
MemAvailable:   32768000 kB
SwapTotal:       8388608 kB
SwapFree:        4194304 kB
"""

CPUINFO = "\n".join(
    f"processor\t: {i}\nmodel name\t: Fake Xeon\n"
    f"physical id\t: {i // 4 % 2}\ncore id\t\t: {i % 2}\n"
    for i in range(8)
)


def test_parse_meminfo_gib():
    m = parse_meminfo(MEMINFO)
    assert m == {"MemTotal": 62.5, "MemAvailable": 31.25, "SwapTotal": 8.0, "SwapFree": 4.0}


def test_parse_meminfo_missing_field_is_an_error():
    with pytest.raises(ValueError, match="MemAvailable"):
        parse_meminfo("MemTotal: 100 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n")


def test_parse_cpuinfo_counts_unique_physical_cores():
    model, cores = parse_cpuinfo(CPUINFO)
    assert model == "Fake Xeon"
    assert cores == 4  # 2 sockets x 2 cores, hyperthreaded to 8


def test_parse_cpuinfo_without_topology_returns_none():
    assert parse_cpuinfo("processor: 0\nmodel name: X\n") == ("X", None)


@pytest.mark.parametrize(
    ("out", "expected"),
    [
        ("samtools 1.24\nUsing htslib 1.24", "1.24"),
        ("2.31-r1302\n", "2.31-r1302"),
        ("0.25.0-r726", "0.25.0-r726"),
        ("seqkit v2.13.0", "2.13.0"),
        ("FastQC v0.12.1", "0.12.1"),
        ("usage: nothing useful", None),
    ],
)
def test_parse_version(out, expected):
    assert parse_version(out) == expected


def test_detect_tools_uses_injected_which_and_runner():
    probes = [ToolProbe("hifiasm"), ToolProbe("flye"), ToolProbe("merqury.sh", ())]
    paths = {"hifiasm": "/bin/hifiasm", "merqury.sh": "/bin/merqury.sh"}

    def run(argv, timeout):
        assert argv == ["/bin/hifiasm", "--version"]
        return 0, "0.25.0-r726\n"

    tools = detect_tools(probes, which=paths.get, run=run)
    assert set(tools) == {"hifiasm", "merqury.sh"}  # flye absent
    assert tools["hifiasm"].version == "0.25.0-r726"
    assert tools["merqury.sh"].version is None and tools["merqury.sh"].error is None


def test_detect_tools_records_probe_failure_instead_of_hiding_it():
    def run(argv, timeout):
        raise subprocess.TimeoutExpired(argv, timeout)

    tools = detect_tools([ToolProbe("busco")], which=lambda n: "/bin/busco", run=run)
    assert tools["busco"].version is None
    assert "TimeoutExpired" in tools["busco"].error


def test_broken_nvidia_smi_means_no_gpu_plus_note():
    gpus, notes = detect_gpus(
        which=lambda n: "/bin/nvidia-smi",
        run=lambda a, t: (9, "NVIDIA-SMI has failed because it couldn't communicate"),
    )
    assert gpus == []
    assert "exit 9" in notes[0]


def test_working_nvidia_smi_parses_gpus():
    gpus, notes = detect_gpus(
        which=lambda n: "/bin/nvidia-smi", run=lambda a, t: (0, "NVIDIA A100, 40960\n")
    )
    assert notes == []
    assert gpus[0].name == "NVIDIA A100" and gpus[0].memory_gb == 40.0
