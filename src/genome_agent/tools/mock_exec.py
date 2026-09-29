"""Fake assembler process used by MockAssembler. Writes a tiny FASTA + metrics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--n50", type=int, required=True)
    p.add_argument("--exit-code", type=int, default=0)
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--alloc-mb", type=int, default=0, help="touch this much RAM")
    a = p.parse_args(argv)
    ballast = bytearray(a.alloc_mb * 1024 * 1024)
    ballast[::4096] = b"x" * len(ballast[::4096])  # touch every page so RSS really grows
    time.sleep(a.sleep)
    if a.exit_code:
        print(f"mock assembler: simulated failure (exit {a.exit_code})", file=sys.stderr)
        return int(a.exit_code)
    a.outdir.mkdir(parents=True, exist_ok=True)
    (a.outdir / "assembly.fa").write_text(">ctg1\nACGT\n")
    (a.outdir / "metrics.json").write_text(json.dumps({"contig_n50": a.n50, "n_contigs": 1}))
    print("mock assembler: done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
