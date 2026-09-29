"""Configurable fake assembler for planning tests and simulations.

The command it builds is a real subprocess (`python -m genome_agent.tools.mock_exec`),
so the executor and failure recovery can be tested end to end without a genome.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from genome_agent.resources.models import ResourceEstimate, SystemResources
from genome_agent.tools.registry import (
    DataType,
    EstimationError,
    ToolAdapter,
    ToolInputs,
    ToolParams,
)


class MockAssemblerParams(ToolParams):
    pass


class MockAssembler(ToolAdapter[MockAssemblerParams]):
    """RAM model: base_ram_gb + ram_gb_per_gbp * genome Gbp + ram_gb_per_thread * cpus."""

    params_model = MockAssemblerParams
    input_types = frozenset({DataType.READS_FASTQ})
    output_types = frozenset({DataType.CONTIGS_FASTA})

    def __init__(
        self,
        name: str,
        *,
        base_ram_gb: float = 1.0,
        ram_gb_per_gbp: float = 0.0,
        ram_gb_per_thread: float = 0.0,
        disk_gb: float = 1.0,
        fail_exit_code: int = 0,
        n50: int = 1_000_000,
    ) -> None:
        self.name = name
        self.executable = sys.executable
        self.purpose = f"Mock assembler '{name}' (testing only)"
        self.base_ram_gb = base_ram_gb
        self.ram_gb_per_gbp = ram_gb_per_gbp
        self.ram_gb_per_thread = ram_gb_per_thread
        self.disk_gb = disk_gb
        self.fail_exit_code = fail_exit_code
        self.n50 = n50

    def is_available(self, res: SystemResources) -> bool:
        return True

    def estimate(
        self, inputs: ToolInputs, params: MockAssemblerParams, cpus: int
    ) -> ResourceEstimate:
        if inputs.genome_size_bp is None:
            raise EstimationError(f"{self.name}: genome size unknown; profile the reads first")
        gbp = inputs.genome_size_bp / 1e9
        ram = self.base_ram_gb + self.ram_gb_per_gbp * gbp + self.ram_gb_per_thread * cpus
        return ResourceEstimate(
            cpus=cpus,
            ram_gb=round(ram, 2),
            disk_gb=self.disk_gb,
            basis=f"mock linear model ({self.base_ram_gb} + {self.ram_gb_per_gbp}/Gbp"
            f" + {self.ram_gb_per_thread}/thread)",
        )

    def build_command(
        self, inputs: ToolInputs, params: MockAssemblerParams, outdir: Path, cpus: int
    ) -> list[str]:
        return [
            self.executable, "-m", "genome_agent.tools.mock_exec",
            "--outdir", str(outdir),
            "--n50", str(self.n50),
            "--exit-code", str(self.fail_exit_code),
        ]  # fmt: skip

    def parse_result(self, outdir: Path) -> dict[str, Any]:
        metrics: dict[str, Any] = json.loads((outdir / "metrics.json").read_text())
        return metrics
