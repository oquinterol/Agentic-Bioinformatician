"""Tool registry: every bioinformatics tool is described by one adapter.

An adapter owns everything tool-specific: what it consumes and produces, its
parameter schema, how it estimates resources, how it builds argv, and how it
parses outputs. Nothing outside `tools/` should know tool-specific details.

The LLM never supplies argv. It supplies `params`, which are validated against
the adapter's Pydantic model (extra fields forbidden) and turned into argv here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from genome_agent.resources.models import ResourceEstimate, SystemResources


class DataType(StrEnum):
    READS_FASTQ = "reads_fastq"
    SEQUENCES_FASTA = "sequences_fasta"
    READ_STATS = "read_stats"
    CONTIGS_FASTA = "contigs_fasta"
    ASSEMBLY_METRICS = "assembly_metrics"
    READ_ORIGIN = "read_origin"


class ToolParams(BaseModel):
    """Base for adapter parameter models. Unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True)
class ToolInputs:
    """What an adapter may know about its inputs when estimating and building."""

    files: tuple[Path, ...]
    input_bytes: int
    genome_size_bp: int | None = None
    read_bases: int | None = None
    kinds: tuple[str | None, ...] = ()  # dataset kind per file (None = unregistered)

    @classmethod
    def from_files(
        cls, files: list[Path], datasets: dict[str, str] | None = None, **kw: Any
    ) -> ToolInputs:
        """`datasets` (path -> kind) fills `kinds` for registered files."""
        paths = tuple(f.resolve() for f in files)
        kinds = tuple((datasets or {}).get(str(p)) for p in paths)
        return cls(files=paths, input_bytes=sum(p.stat().st_size for p in paths), kinds=kinds, **kw)

    def of_kind(self, *kinds: str) -> list[Path]:
        return [f for f, k in zip(self.files, self.kinds, strict=False) if k in kinds]


class EstimationError(ValueError):
    """Raised when an estimate needs information the state does not have yet
    (e.g. genome size before k-mer profiling). This is a knowledge gap, not a bug."""


class ToolAdapter[P: ToolParams](ABC):
    name: str
    executable: str
    purpose: str
    input_types: frozenset[DataType]
    output_types: frozenset[DataType]
    params_model: type[P]
    # Dataset kinds (ReadKind values) this tool is valid for; None = any input file.
    accepted_read_kinds: frozenset[str] | None = None
    # True for tools whose result is invalid if reads come from the wrong organism
    requires_verified_origin: bool = False

    def is_available(self, res: SystemResources) -> bool:
        return self.executable in res.tools

    def parse_params(self, raw: dict[str, Any]) -> P:
        return self.params_model.model_validate(raw)

    def check_inputs(self, inputs: ToolInputs) -> list[str]:
        """Problems with this combination of inputs (e.g. 'needs one reference'); [] = ok."""
        return []

    def param_variants(self) -> list[dict[str, Any]]:
        """Parameter sets to try, most preferred first. Later entries trade speed
        or features for lower resource use (e.g. hifiasm without its bloom filter)."""
        return [{}]

    @abstractmethod
    def estimate(self, inputs: ToolInputs, params: P, cpus: int) -> ResourceEstimate:
        """Predicted peak usage when run with `cpus` threads."""

    @abstractmethod
    def build_command(self, inputs: ToolInputs, params: P, outdir: Path, cpus: int) -> list[str]:
        """argv[0] is `self.executable`; the executor resolves it to the detected path."""

    @abstractmethod
    def parse_result(self, outdir: Path) -> dict[str, Any]:
        """Structured metrics from the files the command wrote to `outdir`."""

    def describe(self) -> dict[str, Any]:
        """Tool description for an LLM backend (JSON Schema for params)."""
        return {
            "name": self.name,
            "purpose": self.purpose,
            "input_types": sorted(self.input_types),
            "output_types": sorted(self.output_types),
            "params_schema": self.params_model.model_json_schema(),
            "param_variants": self.param_variants(),
            "requires_verified_origin": self.requires_verified_origin,
            "accepted_read_kinds": (
                sorted(self.accepted_read_kinds) if self.accepted_read_kinds is not None else "any"
            ),
        }

    def select_inputs(self, datasets: dict[str, str]) -> list[str]:
        """Default inputs from registered datasets (path -> kind) that this tool accepts."""
        return [
            p
            for p, kind in datasets.items()
            if self.accepted_read_kinds is None or kind in self.accepted_read_kinds
        ]


type AnyAdapter = ToolAdapter[Any]


class ToolRegistry:
    def __init__(self, adapters: list[AnyAdapter] | None = None) -> None:
        self._adapters: dict[str, AnyAdapter] = {}
        for a in adapters or []:
            self.register(a)

    def register(self, adapter: AnyAdapter) -> None:
        if adapter.name in self._adapters:
            raise ValueError(f"tool '{adapter.name}' already registered")
        self._adapters[adapter.name] = adapter

    def get(self, name: str) -> AnyAdapter:
        try:
            return self._adapters[name]
        except KeyError:
            raise KeyError(f"unknown tool '{name}'; registered: {self.names()}") from None

    def names(self) -> list[str]:
        return sorted(self._adapters)

    def adapters(self) -> list[AnyAdapter]:
        """Every registered adapter, in registration (= preference) order."""
        return list(self._adapters.values())

    def available(self, res: SystemResources) -> list[AnyAdapter]:
        return [a for a in self._adapters.values() if a.is_available(res)]

    def producing(self, output: DataType, res: SystemResources) -> list[AnyAdapter]:
        """Available tools that can produce `output` — the candidate set for a goal."""
        return [a for a in self.available(res) if output in a.output_types]


def default_registry() -> ToolRegistry:
    from genome_agent.tools.adapters.hifiasm import Hifiasm
    from genome_agent.tools.adapters.origin import ReadOriginCheck
    from genome_agent.tools.adapters.seqkit import SeqkitStats

    return ToolRegistry([Hifiasm(), SeqkitStats(), ReadOriginCheck()])
