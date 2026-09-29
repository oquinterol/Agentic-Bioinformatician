"""Deterministic toy HiFi datasets for integration tests and demos.

A random genome (optionally diploid, with SNPs between haplotypes) and
HiFi-like reads: ~12 kb, ~99.9 % accurate (substitutions only), random strand.
Small enough that real assemblers finish in seconds. The truth FASTA is written
alongside, so assemblies can be checked.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


@dataclass(frozen=True)
class ToyDataset:
    genome_fasta: Path
    reads_fastq: Path
    genome_size_bp: int
    n_reads: int
    read_bases: int


def _random_seq(rng: random.Random, n: int, gc: float) -> str:
    at = (1 - gc) / 2
    return "".join(rng.choices("ACGT", weights=(at, gc / 2, gc / 2, at), k=n))


def _mutate(rng: random.Random, seq: str, rate: float) -> str:
    if rate <= 0:
        return seq
    s = list(seq)
    for pos in rng.sample(range(len(s)), k=int(len(s) * rate)):
        s[pos] = rng.choice([b for b in "ACGT" if b != s[pos]])
    return "".join(s)


def make_toy_hifi(
    outdir: Path,
    *,
    genome_size: int = 200_000,
    coverage: float = 30.0,
    read_length: int = 12_000,
    error_rate: float = 0.001,
    heterozygosity: float = 0.0,
    gc: float = 0.4,
    seed: int = 1,
    read_seed: int | None = None,
) -> ToyDataset:
    """Write genome.fa (truth) and reads.fq to `outdir`.

    `seed` fixes the genome; `read_seed` (default: seed) the reads, so several
    libraries of the same genome can be simulated.
    """
    if read_length >= genome_size:
        raise ValueError("read_length must be smaller than genome_size")
    rng = random.Random(seed)
    outdir.mkdir(parents=True, exist_ok=True)
    hap1 = _random_seq(rng, genome_size, gc)
    haps = [hap1] if heterozygosity <= 0 else [hap1, _mutate(rng, hap1, heterozygosity)]

    genome = outdir / "genome.fa"
    genome.write_text("".join(f">hap{i + 1}\n{h}\n" for i, h in enumerate(haps)))

    if read_seed is not None:
        rng = random.Random(read_seed)
    n_reads = int(coverage * genome_size / read_length)
    read_bases = 0
    reads = outdir / "reads.fq"
    with reads.open("w") as fh:
        for i in range(n_reads):
            length = max(1_000, min(genome_size, int(rng.gauss(read_length, read_length / 6))))
            start = rng.randrange(0, genome_size - length + 1)
            seq = _mutate(rng, rng.choice(haps)[start : start + length], error_rate)
            if rng.random() < 0.5:
                seq = seq.translate(_COMPLEMENT)[::-1]
            fh.write(f"@toy_{i}\n{seq}\n+\n{'I' * len(seq)}\n")
            read_bases += len(seq)
    return ToyDataset(genome, reads, genome_size, n_reads, read_bases)
