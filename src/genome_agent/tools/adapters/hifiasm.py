"""hifiasm: haplotype-resolved de novo assembly of PacBio HiFi reads.

Memory model:
- Verified: the initial k-mer bloom filter takes 2^(f-3) bytes (`-f`, default
  37 = 16 GiB). The man page recommends -f0 for small genomes. Measured on this
  project's toy data: -f37 peaks at 16.08 GiB and -f0 at 0.09 GiB.
- Uncalibrated: the part that scales with the input. `_GB_PER_GBP_READS` is a
  deliberately conservative placeholder until observed peak RSS from real runs
  (Job.peak_rss_gb) is used to calibrate it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from genome_agent.resources.models import ResourceEstimate
from genome_agent.tools.registry import DataType, ToolAdapter, ToolInputs, ToolParams
from genome_agent.tools.seqstats import gfa_to_fasta

PREFIX = "asm"
_BASE_RAM_GB = 0.5
_GB_PER_GBP_READS = 1.0  # UNCALIBRATED placeholder, see module docstring
_DISK_PER_INPUT_BYTE = 3.0  # ec/ovlp bins + GFAs; heuristic


def estimate_read_bases(inputs: ToolInputs) -> tuple[int, str]:
    """Read bases, measured if known, else inferred from file sizes."""
    if inputs.read_bases is not None:
        return inputs.read_bases, "measured read bases"
    total = 0.0
    for f in inputs.files:
        size = f.stat().st_size if f.exists() else 0
        # FASTQ: ~2 bytes/base (seq + qual); gzip assumed ~3x compression.
        total += size * 1.5 if f.suffix == ".gz" else size / 2
    return int(total), "read bases inferred from file size (FASTQ ~2 B/base, gz ~3x)"


class HifiasmParams(ToolParams):
    bloom_bits: int = Field(
        default=37,
        description="-f: k-mer bloom filter bits, uses 2^(f-3) bytes; 0 disables it "
        "(recommended for small genomes), 37 for human-sized, 38-39 for larger",
    )
    purge_level: int | None = Field(
        default=None, ge=0, le=3, description="-l: duplicate purging 0=none .. 3=aggressive"
    )
    primary: bool = Field(default=False, description="--primary: primary/alternate output")
    hom_cov: int | None = Field(default=None, ge=1, description="--hom-cov: homozygous coverage")


class Hifiasm(ToolAdapter[HifiasmParams]):
    name = "hifiasm"
    executable = "hifiasm"
    purpose = "De novo assembly of PacBio HiFi reads into (haplotype-resolved) contigs"
    input_types = frozenset({DataType.READS_FASTQ})
    output_types = frozenset({DataType.CONTIGS_FASTA})
    params_model = HifiasmParams
    # ONT needs --ont and Hi-C/UL reads are auxiliary inputs; neither is exposed yet.
    accepted_read_kinds = frozenset({"pacbio_hifi"})

    def param_variants(self) -> list[dict[str, Any]]:
        # Tool default first; then without the 16 GiB bloom filter (slower
        # k-mer counting, same assembly: advised by the manual for small genomes).
        return [{}, {"bloom_bits": 0}]

    def estimate(self, inputs: ToolInputs, params: HifiasmParams, cpus: int) -> ResourceEstimate:
        bases, how = estimate_read_bases(inputs)
        bloom_gb = 2 ** (params.bloom_bits - 3) / 1024**3 if params.bloom_bits else 0.0
        data_gb = _GB_PER_GBP_READS * bases / 1e9
        in_bytes = inputs.input_bytes
        return ResourceEstimate(
            cpus=cpus,
            ram_gb=round(_BASE_RAM_GB + bloom_gb + data_gb, 2),
            disk_gb=round(_DISK_PER_INPUT_BYTE * in_bytes / 1e9 + 0.05, 2),
            basis=f"bloom filter 2^({params.bloom_bits}-3) B = {bloom_gb:.2f} GiB (verified); "
            f"{_GB_PER_GBP_READS} GB per Gbp of reads (UNCALIBRATED) x {bases / 1e9:.3f} Gbp "
            f"({how}); base {_BASE_RAM_GB} GB",
        )

    def build_command(
        self, inputs: ToolInputs, params: HifiasmParams, outdir: Path, cpus: int
    ) -> list[str]:
        argv = [self.executable, "-o", str(outdir / PREFIX), "-t", str(cpus)]
        argv += ["-f", str(params.bloom_bits)]
        if params.purge_level is not None:
            argv += ["-l", str(params.purge_level)]
        if params.primary:
            argv.append("--primary")
        if params.hom_cov is not None:
            argv += ["--hom-cov", str(params.hom_cov)]
        return [*argv, *map(str, inputs.files)]

    def parse_result(self, outdir: Path) -> dict[str, Any]:
        """Convert primary (and haplotype) contig GFAs to FASTA and report stats."""
        candidates = {
            "primary": [f"{PREFIX}.p_ctg.gfa", f"{PREFIX}.bp.p_ctg.gfa"],
            "alternate": [f"{PREFIX}.a_ctg.gfa"],
            "hap1": [f"{PREFIX}.bp.hap1.p_ctg.gfa", f"{PREFIX}.hap1.p_ctg.gfa"],
            "hap2": [f"{PREFIX}.bp.hap2.p_ctg.gfa", f"{PREFIX}.hap2.p_ctg.gfa"],
        }
        out: dict[str, Any] = {}
        for label, names in candidates.items():
            gfa = next((outdir / n for n in names if (outdir / n).exists()), None)
            if gfa is None:
                continue
            fasta = gfa.with_suffix(".fa")
            out[label] = gfa_to_fasta(gfa, fasta) | {"fasta": str(fasta)}
        if "primary" not in out:
            raise FileNotFoundError(f"no primary contig GFA in {outdir}")
        return out
