"""Read-origin check: do these reads come from the organism of the reference?

    python -m genome_agent.tools.origin_check --reference REF.fa --sample N \
        --out origin_check.json --threads T READS [READS ...]

Takes the first N reads of each file (plain or gzipped FASTQ/FASTA), maps them
together with minimap2 (-x map-hifi, one index build), and reports per file how
many reads align over >= 80 % of their length at >= 95 % identity. The first N
reads are a convenience sample, not a random one: fine for species identity,
not for estimating rare contamination.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import IO

MIN_QUERY_COVERAGE = 0.8
MIN_IDENTITY = 0.95
MATCH_FRACTION = 0.5  # >= this share of well-aligned reads -> "matches reference"
MISMATCH_FRACTION = 0.05  # < this share -> "does not match reference"


def _open(path: Path) -> IO[str]:
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def sample_reads(path: Path, n: int, tag: int, out: IO[str]) -> int:
    """Copy the first `n` records of `path` to `out` as FASTA named `<tag>|<i>`."""
    written = 0
    with _open(path) as fh:
        first = fh.readline()
        fastq = first.startswith("@")
        line = first
        while line and written < n:
            if not line.startswith(("@", ">")):
                line = fh.readline()
                continue
            seq_lines = []
            line = fh.readline()
            while line and not line.startswith(("+", ">")) and not (fastq and line.startswith("@")):
                seq_lines.append(line.strip())
                line = fh.readline()
            if fastq and line.startswith("+"):
                fh.readline()  # quality line (HiFi FASTQ is single-line)
                line = fh.readline()
            out.write(f">{tag}|{written}\n{''.join(seq_lines)}\n")
            written += 1
    return written


def verdict(well_aligned: int, sampled: int) -> str:
    frac = well_aligned / sampled if sampled else 0.0
    if frac >= MATCH_FRACTION:
        return "matches reference"
    if frac < MISMATCH_FRACTION:
        return "does not match reference"
    return "ambiguous"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--sample", type=int, required=True)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("reads", type=Path, nargs="+")
    a = p.parse_args(argv)

    minimap2 = shutil.which("minimap2")
    if minimap2 is None:
        print("minimap2 not found on PATH", file=sys.stderr)
        return 2
    sample_fa = a.out.parent / "origin_sample.fa"
    counts = []
    with sample_fa.open("w") as out:
        for i, reads in enumerate(a.reads):
            counts.append(sample_reads(reads, a.sample, i, out))

    paf = a.out.parent / "origin_sample.paf"
    with paf.open("w") as out:
        subprocess.run(
            [minimap2, "-x", "map-hifi", "-t", str(a.threads), "--secondary=no",
             str(a.reference), str(sample_fa)],
            stdout=out, check=True,
        )  # fmt: skip

    best: dict[str, tuple[float, float]] = {}
    for line in paf.open():
        f = line.split("\t")
        qlen, alen = int(f[1]), int(f[10])
        qcov = (int(f[3]) - int(f[2])) / qlen if qlen else 0.0
        ident = int(f[9]) / alen if alen else 0.0
        if f[0] not in best or qcov > best[f[0]][0]:
            best[f[0]] = (qcov, ident)

    files: dict[str, dict[str, object]] = {}
    report = {"reference": str(a.reference), "files": files}
    for i, reads in enumerate(a.reads):
        hits = [v for k, v in best.items() if k.split("|", 1)[0] == str(i)]
        good = sum(q >= MIN_QUERY_COVERAGE and ident >= MIN_IDENTITY for q, ident in hits)
        files[str(reads)] = {
            "sampled_reads": counts[i],
            "reads_with_any_hit": len(hits),
            "well_aligned_reads": good,
            "well_aligned_fraction": round(good / counts[i], 4) if counts[i] else 0.0,
            "median_identity_of_hits": round(statistics.median(x for _, x in hits), 4)
            if hits
            else 0.0,
            "verdict": verdict(good, counts[i]),
        }
    a.out.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
