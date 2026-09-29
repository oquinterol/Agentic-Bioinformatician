"""library_consistency_check: reference-free test that read libraries share an organism.

Memory model: one minimap2 -x map-hifi index of `target_bases` of reads at a
time. Measured: 3.48 GiB for 0.84 Gbp of HiFi reads (~4.2 B/base); the
estimate uses 5 B/base + 1 GB.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from pydantic import Field

from genome_agent.resources.models import ResourceEstimate
from genome_agent.tools.adapters.origin import READ_KINDS
from genome_agent.tools.registry import DataType, ToolAdapter, ToolInputs, ToolParams

OUTPUT_FILE = "consistency.json"
_BYTES_PER_TARGET_BASE = 5.0
_BASE_GB = 1.0
_DEFAULT_TARGET_BASES = 1_000_000_000


class ConsistencyParams(ToolParams):
    target_coverage: float = Field(
        default=1.0,
        gt=0,
        le=3,
        description="reads indexed per library, as x of the genome size (1x is enough)",
    )
    sample_reads: int = Field(default=2000, ge=200, le=50_000)


class LibraryConsistencyCheck(ToolAdapter[ConsistencyParams]):
    name = "library_consistency_check"
    executable = "minimap2"
    purpose = (
        "Reference-free check that two or more read libraries come from the same organism "
        "(cross-maps read samples; verdict per pair: consistent / inconsistent / ambiguous). "
        "Use it before pooling libraries when no reference genome is available."
    )
    input_types = frozenset({DataType.READS_FASTQ})
    output_types = frozenset({DataType.LIBRARY_CONSISTENCY})
    params_model = ConsistencyParams
    accepted_read_kinds = READ_KINDS

    def check_inputs(self, inputs: ToolInputs) -> list[str]:
        if len(inputs.of_kind(*READ_KINDS)) < 2:
            return ["library_consistency_check needs at least two registered read libraries"]
        return []

    def _target_bases(self, inputs: ToolInputs, params: ConsistencyParams) -> int:
        size = inputs.genome_size_bp or _DEFAULT_TARGET_BASES
        return int(size * params.target_coverage)

    def estimate(
        self, inputs: ToolInputs, params: ConsistencyParams, cpus: int
    ) -> ResourceEstimate:
        target = self._target_bases(inputs, params)
        return ResourceEstimate(
            cpus=cpus,
            ram_gb=math.ceil((_BASE_GB + _BYTES_PER_TARGET_BASE * target / 1e9) * 100) / 100,
            disk_gb=round(2.2 * target * len(inputs.files) / 1e9 + 0.1, 2),
            basis=f"minimap2 index of {target / 1e9:.2f} Gbp of reads at "
            f"{_BYTES_PER_TARGET_BASE} B/base (measured 4.2) + {_BASE_GB} GB",
        )

    def build_command(
        self, inputs: ToolInputs, params: ConsistencyParams, outdir: Path, cpus: int
    ) -> list[str]:
        return [
            sys.executable, "-m", "genome_agent.tools.consistency_check",
            "--target-bases", str(self._target_bases(inputs, params)),
            "--sample", str(params.sample_reads),
            "--threads", str(cpus),
            "--out", str(outdir / OUTPUT_FILE),
            *map(str, inputs.of_kind(*READ_KINDS)),
        ]  # fmt: skip

    def parse_result(self, outdir: Path) -> dict[str, Any]:
        data: dict[str, Any] = json.loads((outdir / OUTPUT_FILE).read_text())
        return data
