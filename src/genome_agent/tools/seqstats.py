"""Contig statistics shared by assembler adapters."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def length_stats(lengths: Iterable[int]) -> dict[str, int]:
    ls = sorted(lengths, reverse=True)
    total = sum(ls)
    n50 = 0
    acc = 0
    for n in ls:
        acc += n
        if acc * 2 >= total:
            n50 = n
            break
    return {
        "n_contigs": len(ls),
        "total_bp": total,
        "n50_bp": n50,
        "largest_bp": ls[0] if ls else 0,
    }


def gfa_to_fasta(gfa: Path, fasta: Path) -> dict[str, int]:
    """Write the segments of a GFA as FASTA; return their length stats."""
    lengths = []
    with gfa.open() as src, fasta.open("w") as dst:
        for line in src:
            if not line.startswith("S\t"):
                continue
            _, name, seq, *_ = line.rstrip("\n").split("\t")
            dst.write(f">{name}\n{seq}\n")
            lengths.append(len(seq))
    return length_stats(lengths)
