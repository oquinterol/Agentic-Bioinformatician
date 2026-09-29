"""Reference-free check that several read libraries come from the same organism.

    python -m genome_agent.tools.consistency_check --target-bases N --sample N \
        --threads T --out consistency.json LIB1 LIB2 [LIB ...]

For every library L: its first `target-bases` of reads (ideally ~1x of the
genome) become a minimap2 target; a sample of reads taken from *after* that
region in every library (L itself included) is mapped to it. A read "overlaps"
if it aligns >= 2 kb at >= 98 % identity (HiFi vs HiFi of the same genome is
~99.8 %).

The self fraction (L's own later reads vs L's target) is the baseline for
"same organism" at this coverage and ploidy; a pair's verdict uses
cross/self, not an absolute threshold. Validated on real data: same library
35.7 % vs a mislabelled other-species library 0.0 % against 0.84 Gbp of targets.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import IO

MIN_IDENTITY = 0.98
MIN_OVERLAP = 2000
CONSISTENT_RATIO = 0.5
INCONSISTENT_RATIO = 0.05
MIN_SELF_FRACTION = 0.02  # below this the baseline is too weak to judge anything


def _open(path: Path) -> IO[str]:
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def _records(path: Path) -> Iterator[str]:
    """Yield sequences from single-line FASTQ or (multi-line) FASTA."""
    with _open(path) as fh:
        first = fh.readline()
        if first.startswith("@"):
            line = first
            while line:
                read = fh.readline().strip()
                fh.readline()
                fh.readline()
                yield read
                line = fh.readline()
        else:
            seq: list[str] = []
            for line in [first, *fh]:
                if line.startswith(">"):
                    if seq:
                        yield "".join(seq)
                    seq = []
                else:
                    seq.append(line.strip())
            if seq:
                yield "".join(seq)


def split_library(
    path: Path, target_bases: int, sample: int, target: Path, query: Path, tag: str
) -> tuple[int, int]:
    """Write the first `target_bases` to `target`, the next `sample` reads to `query`."""
    bases = reads_q = 0
    with target.open("w") as t, query.open("a") as q:
        for i, seq in enumerate(_records(path)):
            if bases < target_bases:
                t.write(f">t{i}\n{seq}\n")
                bases += len(seq)
            elif reads_q < sample:
                q.write(f">{tag}|{reads_q}\n{seq}\n")
                reads_q += 1
            else:
                break
    return bases, reads_q


def verdict(cross: float, self_fraction: float) -> str:
    if self_fraction < MIN_SELF_FRACTION:
        return "undetermined (weak self baseline: increase target bases)"
    ratio = cross / self_fraction
    if ratio >= CONSISTENT_RATIO:
        return "consistent"
    if ratio < INCONSISTENT_RATIO:
        return "inconsistent"
    return "ambiguous"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target-bases", type=int, required=True)
    p.add_argument("--sample", type=int, required=True)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("libs", type=Path, nargs="+")
    a = p.parse_args(argv)
    minimap2 = shutil.which("minimap2")
    if minimap2 is None:
        print("minimap2 not found on PATH", file=sys.stderr)
        return 2
    work = a.out.parent
    queries = work / "consistency_queries.fa"
    queries.write_text("")
    targets, sampled, target_bases = [], [], []
    for i, lib in enumerate(a.libs):
        t = work / f"consistency_target_{i}.fa"
        b, n = split_library(lib, a.target_bases, a.sample, t, queries, str(i))
        targets.append(t)
        sampled.append(n)
        target_bases.append(b)

    fractions: dict[int, dict[int, float]] = {}
    for i, t in enumerate(targets):
        paf = work / f"consistency_{i}.paf"
        with paf.open("w") as out:
            subprocess.run(
                [minimap2, "-x", "map-hifi", "-t", str(a.threads), "--secondary=no",
                 str(t), str(queries)],
                stdout=out, check=True,
            )  # fmt: skip
        hit: set[str] = set()
        for line in paf.open():
            f = line.split("\t")
            alen = int(f[10])
            if alen and int(f[9]) / alen >= MIN_IDENTITY and int(f[3]) - int(f[2]) >= MIN_OVERLAP:
                hit.add(f[0])
        fractions[i] = {
            j: (sum(q.split("|", 1)[0] == str(j) for q in hit) / sampled[j]) if sampled[j] else 0.0
            for j in range(len(a.libs))
        }

    pairs = []
    per_file: dict[str, list[str]] = {str(lib): [] for lib in a.libs}
    for i in range(len(a.libs)):
        for j in range(len(a.libs)):
            if i == j:
                continue
            v = verdict(fractions[i][j], fractions[i][i])
            pairs.append({
                "target": str(a.libs[i]), "query": str(a.libs[j]),
                "cross_fraction": round(fractions[i][j], 4),
                "self_fraction": round(fractions[i][i], 4),
                "verdict": v,
            })  # fmt: skip
            if v == "inconsistent":
                per_file[str(a.libs[j])].append(str(a.libs[i]))
    report = {
        "target_bases": dict(zip(map(str, a.libs), target_bases, strict=True)),
        "pairs": pairs,
        "inconsistent_with": per_file,
    }
    a.out.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
