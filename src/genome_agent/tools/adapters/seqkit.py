"""seqkit stats: read-set / assembly summary statistics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from genome_agent.resources.models import ResourceEstimate
from genome_agent.tools.registry import DataType, ToolAdapter, ToolInputs, ToolParams

OUTPUT_FILE = "seqkit_stats.tsv"
_MAX_USEFUL_THREADS = 4
# With --all, seqkit keeps every sequence length in memory (for N50/quartiles).
# Conservative assumption: >= 50 input bytes per record, 8 bytes kept per record.
_BYTES_PER_RECORD = 50
_BYTES_KEPT_PER_RECORD = 8


class SeqkitStatsParams(ToolParams):
    all_stats: bool = True  # quartiles, N50, Q20/Q30, GC


class SeqkitStats(ToolAdapter[SeqkitStatsParams]):
    name = "seqkit_stats"
    executable = "seqkit"
    purpose = "Summary statistics (count, total length, N50, GC, quality) of FASTA/FASTQ files"
    input_types = frozenset({DataType.READS_FASTQ, DataType.SEQUENCES_FASTA})
    output_types = frozenset({DataType.READ_STATS})
    params_model = SeqkitStatsParams

    def estimate(
        self, inputs: ToolInputs, params: SeqkitStatsParams, cpus: int
    ) -> ResourceEstimate:
        threads = max(1, min(cpus, _MAX_USEFUL_THREADS, len(inputs.files)))
        ram = 0.5
        basis = "static: streaming parser, ~0.5 GB baseline"
        if params.all_stats:
            ram += inputs.input_bytes / _BYTES_PER_RECORD * _BYTES_KEPT_PER_RECORD / 1e9
            basis += "; --all keeps 8 B/record, assuming >=50 B/record of input"
        return ResourceEstimate(cpus=threads, ram_gb=round(ram, 2), disk_gb=0.01, basis=basis)

    def build_command(
        self, inputs: ToolInputs, params: SeqkitStatsParams, outdir: Path, cpus: int
    ) -> list[str]:
        argv = [self.executable, "stats", "--tabular", "--threads", str(cpus)]
        if params.all_stats:
            argv.append("--all")
        return [*argv, "--out-file", str(outdir / OUTPUT_FILE), *map(str, inputs.files)]

    def parse_result(self, outdir: Path) -> dict[str, Any]:
        with (outdir / OUTPUT_FILE).open(newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        return {"files": {row.pop("file"): {k: _number(v) for k, v in row.items()} for row in rows}}


def _number(value: str) -> int | float | str:
    for cast in (int, float):
        try:
            return cast(value.replace(",", ""))
        except ValueError:
            pass
    return value
