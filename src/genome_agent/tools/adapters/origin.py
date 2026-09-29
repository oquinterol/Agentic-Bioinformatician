"""read_origin_check: verify that reads come from the organism of a registered reference.

Memory model: minimap2 -x map-hifi index. Measured on this project's 1.58 Gb
potato draft: 4.86 GiB peak (~3.3 bytes per reference base); the estimate uses
4 bytes/base + 1 GB, rounded up.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from pydantic import Field

from genome_agent.resources.models import ResourceEstimate
from genome_agent.tools.registry import DataType, ToolAdapter, ToolInputs, ToolParams

OUTPUT_FILE = "origin_check.json"
REFERENCE_KIND = "reference_fasta"
READ_KINDS = frozenset({"pacbio_hifi", "ont", "illumina", "hic", "rnaseq"})
_BYTES_PER_REF_BASE = 4.0
_BASE_GB = 1.0


class ReadOriginParams(ToolParams):
    sample_reads: int = Field(
        default=2000, ge=100, le=100_000, description="first N reads mapped per file"
    )


class ReadOriginCheck(ToolAdapter[ReadOriginParams]):
    name = "read_origin_check"
    executable = "minimap2"
    purpose = (
        "Verify that each reads file comes from the organism of a registered reference "
        "(maps a sample with minimap2; verdict per file: matches / does not match / ambiguous)"
    )
    input_types = frozenset({DataType.READS_FASTQ, DataType.SEQUENCES_FASTA})
    output_types = frozenset({DataType.READ_ORIGIN})
    params_model = ReadOriginParams
    accepted_read_kinds = READ_KINDS | {REFERENCE_KIND}

    def check_inputs(self, inputs: ToolInputs) -> list[str]:
        refs = inputs.of_kind(REFERENCE_KIND)
        problems = []
        if len(refs) != 1:
            problems.append(
                f"read_origin_check needs exactly one registered reference_fasta input "
                f"(got {len(refs)})"
            )
        if not inputs.of_kind(*READ_KINDS):
            problems.append("read_origin_check needs at least one registered reads input")
        return problems

    def estimate(self, inputs: ToolInputs, params: ReadOriginParams, cpus: int) -> ResourceEstimate:
        refs = inputs.of_kind(REFERENCE_KIND)
        ref_bytes = sum(r.stat().st_size for r in refs if r.exists())
        ram = _BASE_GB + _BYTES_PER_REF_BASE * ref_bytes / 1e9
        return ResourceEstimate(
            cpus=cpus,
            ram_gb=math.ceil(ram * 100) / 100,
            disk_gb=0.1,
            basis=f"minimap2 index ~{_BYTES_PER_REF_BASE} B per reference base "
            f"(measured 3.3 B/base on a 1.58 Gb reference) + {_BASE_GB} GB",
        )

    def build_command(
        self, inputs: ToolInputs, params: ReadOriginParams, outdir: Path, cpus: int
    ) -> list[str]:
        [ref] = inputs.of_kind(REFERENCE_KIND)
        reads = inputs.of_kind(*READ_KINDS)
        return [
            sys.executable, "-m", "genome_agent.tools.origin_check",
            "--reference", str(ref),
            "--sample", str(params.sample_reads),
            "--threads", str(cpus),
            "--out", str(outdir / OUTPUT_FILE),
            *map(str, reads),
        ]  # fmt: skip

    def parse_result(self, outdir: Path) -> dict[str, Any]:
        data: dict[str, Any] = json.loads((outdir / OUTPUT_FILE).read_text())
        return data
