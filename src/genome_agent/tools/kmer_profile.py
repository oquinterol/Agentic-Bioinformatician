"""k-mer profile per read library: jellyfish histogram + GenomeScope2 model.

    python -m genome_agent.tools.kmer_profile --k 21 --ploidy 2 --hash-size N \
        --threads T --out kmer_profile.json LIB [LIB ...]

Each library is profiled separately, so libraries can be compared (a
mislabelled library shows a different genome size, heterozygosity or
ploidy). Gzipped inputs are streamed through jellyfish generators. If
genomescope.R is not on PATH, the histogram is still produced and the model
fields are null (reported, not invented).

Validated on a known toy genome (2.0 Mb, 1 % het, 0.1 % error): GenomeScope2
reported 1.94-1.97 Mb, 1.0-1.35 % het, 0.100 % error.
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

_ROW = re.compile(r"^(?P<name>[A-Za-z ()]+?)\s{2,}(?P<min>\S+(?: bp)?)\s+(?P<max>\S+(?: bp)?)\s*$")


def _num(value: str) -> float | None:
    v = value.replace(",", "").replace("bp", "").replace("%", "").strip()
    try:
        return float(v)
    except ValueError:
        return None


def parse_summary(text: str) -> dict[str, Any]:
    """GenomeScope2 summary.txt -> {property: {min, max}} (percent fields as %)."""
    out: dict[str, Any] = {}
    for line in text.splitlines():
        m = _ROW.match(line.strip())
        if not m or m["name"].strip() == "property":
            continue
        key = m["name"].strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
        out[key] = {"min": _num(m["min"]), "max": _num(m["max"])}
    return out


def profile(
    lib: Path, work: Path, k: int, ploidy: int, hash_size: int, threads: int
) -> dict[str, Any]:
    jellyfish = shutil.which("jellyfish")
    if jellyfish is None:
        raise RuntimeError("jellyfish not found on PATH")
    work.mkdir(parents=True, exist_ok=True)
    generator = work / "generators"
    reader = "zcat" if lib.suffix == ".gz" else "cat"
    generator.write_text(f"{reader} '{lib}'\n")
    counts, histo = work / "counts.jf", work / "kmers.histo"
    subprocess.run(
        [jellyfish, "count", "-C", "-m", str(k), "-s", str(hash_size), "-t", str(threads),
         "-g", str(generator), "-G", "1", "-o", str(counts)],
        check=True,
    )  # fmt: skip
    with histo.open("w") as out:
        subprocess.run(
            [jellyfish, "histo", "-t", str(threads), str(counts)], stdout=out, check=True
        )
    counts.unlink()  # large; the histogram is what matters

    result: dict[str, Any] = {"histogram": str(histo), "k": k, "ploidy": ploidy, "model": None}
    gs = shutil.which("genomescope.R")
    if gs is None:
        result["model_error"] = "genomescope.R not on PATH; histogram only"
        return result
    gs_out = work / "genomescope"
    proc = subprocess.run(
        [gs, "-i", str(histo), "-o", str(gs_out), "-k", str(k), "-p", str(ploidy)],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    summary = gs_out / "summary.txt"
    if proc.returncode != 0 or not summary.exists():
        result["model_error"] = f"genomescope failed ({proc.returncode}): {proc.stderr[-500:]}"
        return result
    model = parse_summary(summary.read_text())
    result["model"] = model
    result["converged"] = "Model converged" in proc.stdout + proc.stderr
    length = model.get("genome_haploid_length", {})
    het = model.get("heterozygous_ab", {})
    if length.get("min") and length.get("max"):
        result["haploid_length_bp"] = int((length["min"] + length["max"]) / 2)
    if het.get("min") is not None and het.get("max") is not None:
        result["heterozygosity_pct"] = round((het["min"] + het["max"]) / 2, 3)
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=21)
    p.add_argument("--ploidy", type=int, default=2)
    p.add_argument("--hash-size", type=int, required=True)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("libs", type=Path, nargs="+")
    a = p.parse_args(argv)
    report: dict[str, Any] = {"files": {}}
    for i, lib in enumerate(a.libs):
        try:
            report["files"][str(lib)] = profile(
                lib, a.out.parent / f"lib{i}", a.k, a.ploidy, a.hash_size, a.threads
            )
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"profiling {lib} failed: {exc}", file=sys.stderr)
            return 1
    a.out.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
