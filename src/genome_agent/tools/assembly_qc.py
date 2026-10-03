"""Assembly QC runners: compleasm (gene completeness) and meryl + Merqury (QV).

    python -m genome_agent.tools.assembly_qc completeness --lineage L --library DIR \
        --threads T --out qc.json ASM [ASM ...]
    python -m genome_agent.tools.assembly_qc qv --reads READS --k 21 --memory-gb M \
        --threads T --out qc.json ASM [ASM]

compleasm and merqury.sh are external (see README); their outputs are parsed
here, never re-implemented.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_SUMMARY = re.compile(r"^(?P<cat>[SDFIM]):(?P<pct>[\d.]+)%,\s*(?P<n>\d+)")
_TOTAL = re.compile(r"^N:(?P<n>\d+)")


def parse_compleasm_summary(text: str) -> dict[str, Any]:
    """compleasm summary.txt -> {S, D, F, I, M: {pct, n}, N}."""
    out: dict[str, Any] = {}
    for line in text.splitlines():
        if m := _SUMMARY.match(line.strip()):
            out[m["cat"]] = {"pct": float(m["pct"]), "n": int(m["n"])}
        elif m := _TOTAL.match(line.strip()):
            out["N"] = int(m["n"])
    if "S" in out and "D" in out:
        out["complete_pct"] = round(out["S"]["pct"] + out["D"]["pct"], 2)
    return out


def parse_merqury_qv(text: str) -> dict[str, Any]:
    """<out>.qv rows: name, asm-only k-mers, total k-mers, QV, error rate."""
    rows = {}
    for line in text.splitlines():
        f = line.split("\t")
        if len(f) >= 5:
            rows[f[0]] = {"qv": float(f[3]), "error_rate": float(f[4])}
    return rows


def parse_merqury_completeness(text: str) -> dict[str, Any]:
    """<out>.completeness.stats rows: name, set, solid k-mers in asm, in reads, %."""
    rows = {}
    for line in text.splitlines():
        f = line.split("\t")
        if len(f) >= 5:
            rows[f[0]] = {"kmer_completeness_pct": float(f[4])}
    return rows


def _need(exe: str) -> str:
    path = shutil.which(exe)
    if path is None:
        raise RuntimeError(f"{exe} not found on PATH")
    return path


def completeness(args: argparse.Namespace) -> dict[str, Any]:
    compleasm = _need("compleasm")
    args.library.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"lineage": args.lineage, "assemblies": {}}
    for i, asm in enumerate(args.assemblies):
        out = args.out.parent / f"compleasm_{i}"
        subprocess.run(
            [compleasm, "run", "-a", str(asm), "-o", str(out), "-t", str(args.threads),
             "-l", args.lineage, "-L", str(args.library)],
            check=True,
        )  # fmt: skip
        summary = out / "summary.txt"
        report["assemblies"][str(asm)] = parse_compleasm_summary(summary.read_text()) | {
            "summary": str(summary)
        }
    return report


def qv(args: argparse.Namespace) -> dict[str, Any]:
    meryl = _need("meryl")
    merqury = _need("merqury.sh")
    work = args.out.parent
    db = work / "reads.meryl"
    subprocess.run(
        [meryl, f"k={args.k}", f"threads={args.threads}", f"memory={args.memory_gb}",
         "count", "output", str(db), *map(str, args.reads)],
        check=True,
    )  # fmt: skip
    prefix = "merqury"
    subprocess.run(
        [merqury, str(db), *map(str, args.assemblies), prefix], cwd=work, check=True
    )
    qv_rows = parse_merqury_qv((work / f"{prefix}.qv").read_text())
    comp_rows = parse_merqury_completeness((work / f"{prefix}.completeness.stats").read_text())
    names = {Path(a).name.removesuffix(".fa").removesuffix(".fasta"): str(a) for a in args.assemblies}
    report: dict[str, Any] = {"k": args.k, "reads": list(map(str, args.reads)), "assemblies": {}}
    for name, row in qv_rows.items():
        key = names.get(name, name)  # "both" covers the pair
        report["assemblies"][key] = row | comp_rows.get(name, {})
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("completeness")
    c.add_argument("--lineage", required=True)
    c.add_argument("--library", type=Path, required=True)
    q = sub.add_parser("qv")
    q.add_argument("--reads", type=Path, nargs="+", required=True)
    q.add_argument("--k", type=int, default=21)
    q.add_argument("--memory-gb", type=int, required=True)
    for s in (c, q):
        s.add_argument("--threads", type=int, default=1)
        s.add_argument("--out", type=Path, required=True)
        s.add_argument("assemblies", type=Path, nargs="+")
    a = p.parse_args(argv)
    try:
        report = completeness(a) if a.mode == "completeness" else qv(a)
    except (RuntimeError, subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"assembly QC failed: {exc}", file=sys.stderr)
        return 1
    a.out.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
