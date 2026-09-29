"""kmer_profile: genome size, heterozygosity and error rate per library (reference-free).

Memory model (jellyfish hash, sized up front with -s): distinct k-mers are
expected from the genome (both haplotypes, x2 margin) plus HiFi error k-mers
(~0.1 % error x k per base). Measured: -s 50M took 0.26 GiB, ~5.5 B/entry;
the estimate uses 6.5 B/entry + 0.5 GB.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from pydantic import Field

from genome_agent.resources.models import ResourceEstimate
from genome_agent.tools.adapters.hifiasm import estimate_read_bases
from genome_agent.tools.adapters.origin import READ_KINDS
from genome_agent.tools.registry import DataType, ToolAdapter, ToolInputs, ToolParams

OUTPUT_FILE = "kmer_profile.json"
_BYTES_PER_ENTRY = 6.5
_BASE_GB = 0.5
_ERROR_KMER_RATE = 0.001  # HiFi per-base error rate
_UNKNOWN_GENOME = 1_000_000_000


class KmerProfileParams(ToolParams):
    k: int = Field(default=21, ge=15, le=31, description="k-mer length (21 is standard)")
    ploidy: int = Field(default=2, ge=1, le=6, description="ploidy assumed by the model")


class KmerProfile(ToolAdapter[KmerProfileParams]):
    name = "kmer_profile"
    executable = "jellyfish"
    purpose = (
        "Reference-free k-mer profile per read library (jellyfish + GenomeScope2): estimated "
        "haploid genome size, heterozygosity and error rate. Profiles that disagree between "
        "libraries, or with the declared species, flag mislabelled or contaminated data."
    )
    input_types = frozenset({DataType.READS_FASTQ})
    output_types = frozenset({DataType.GENOME_PROFILE})
    params_model = KmerProfileParams
    accepted_read_kinds = READ_KINDS

    def _hash_size(self, inputs: ToolInputs, params: KmerProfileParams) -> int:
        genome = inputs.genome_size_bp or _UNKNOWN_GENOME
        bases, _ = estimate_read_bases(inputs)
        per_lib_bases = bases / max(1, len(inputs.files))
        expected = 2 * genome + _ERROR_KMER_RATE * params.k * per_lib_bases
        # Hard bound: a library cannot hold more distinct k-mers than it has bases.
        # It also keeps the estimate sane when the genome size is still unknown.
        return int(min(expected, per_lib_bases)) if per_lib_bases else int(expected)

    def estimate(
        self, inputs: ToolInputs, params: KmerProfileParams, cpus: int
    ) -> ResourceEstimate:
        entries = self._hash_size(inputs, params)
        ram = _BASE_GB + _BYTES_PER_ENTRY * entries / 1e9
        return ResourceEstimate(
            cpus=cpus,
            ram_gb=math.ceil(ram * 100) / 100,
            disk_gb=math.ceil(_BYTES_PER_ENTRY * entries / 1e9 * 100) / 100,
            basis=f"jellyfish hash of {entries / 1e9:.2f} G entries (2x genome + HiFi error "
            f"k-mers, capped by read bases) at {_BYTES_PER_ENTRY} B/entry (measured ~5.5) + "
            f"{_BASE_GB} GB; libraries are profiled one at a time",
        )

    def build_command(
        self, inputs: ToolInputs, params: KmerProfileParams, outdir: Path, cpus: int
    ) -> list[str]:
        return [
            sys.executable, "-m", "genome_agent.tools.kmer_profile",
            "--k", str(params.k),
            "--ploidy", str(params.ploidy),
            "--hash-size", str(self._hash_size(inputs, params)),
            "--threads", str(cpus),
            "--out", str(outdir / OUTPUT_FILE),
            *map(str, inputs.of_kind(*READ_KINDS)),
        ]  # fmt: skip

    def parse_result(self, outdir: Path) -> dict[str, Any]:
        data: dict[str, Any] = json.loads((outdir / OUTPUT_FILE).read_text())
        return data
