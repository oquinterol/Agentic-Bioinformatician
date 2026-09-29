"""Known executables and how to ask them for a version.

This only covers detection. Full adapters (command builder, estimator, parser)
will live next to this in the tool registry.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolProbe:
    name: str
    version_args: tuple[str, ...] = ("--version",)


BIOINFORMATICS_TOOLS: tuple[ToolProbe, ...] = (
    ToolProbe("samtools"),
    ToolProbe("minimap2"),
    ToolProbe("seqkit", ("version",)),
    ToolProbe("fastqc"),
    ToolProbe("NanoPlot"),
    ToolProbe("meryl", ("--version",)),
    ToolProbe("genomescope2", ("--version",)),
    ToolProbe("hifiasm"),
    ToolProbe("flye"),
    ToolProbe("verkko"),
    ToolProbe("busco"),
    ToolProbe("merqury.sh", ()),  # no version flag
    ToolProbe("quast"),
    ToolProbe("gfastats"),
    ToolProbe("purge_dups", ()),
    ToolProbe("yahs", ("--version",)),
    ToolProbe("ragtag.py", ("--version",)),
)

CONTAINER_RUNTIMES: tuple[str, ...] = ("docker", "podman", "apptainer", "singularity")
SCHEDULERS: tuple[str, ...] = ("sbatch", "qsub", "bsub")
