"""Local machine inventory.

The parsing functions are pure, so tests can feed them synthetic /proc
contents. `inspect_system` is the only function that touches the real machine.
Detection problems that are not fatal go into `SystemResources.notes`. They are
reported and never silently dropped.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from genome_agent.resources.models import SYSTEMD_SCOPE, GpuInfo, SystemResources, ToolInfo
from genome_agent.tools.catalog import (
    BIOINFORMATICS_TOOLS,
    CONTAINER_RUNTIMES,
    SCHEDULERS,
    ToolProbe,
)

GIB = 1024**3
KIB_PER_GIB = 1024**2
PROBE_TIMEOUT_S = 15.0

# (argv, timeout) -> (returncode, combined output). Raises OSError / TimeoutExpired.
Runner = Callable[[list[str], float], tuple[int, str]]
Which = Callable[[str], str | None]

_VERSION_RE = re.compile(r"v?(\d+(?:\.\d+)+(?:[-+][\w.]+)?)")


class UnsupportedPlatformError(RuntimeError):
    pass


def _run(argv: list[str], timeout: float) -> tuple[int, str]:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def parse_meminfo(text: str) -> dict[str, float]:
    """Return MemTotal/MemAvailable/SwapTotal/SwapFree in GiB."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            values[key.strip()] = int(parts[0]) / KIB_PER_GIB
    required = ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")
    missing = [k for k in required if k not in values]
    if missing:
        raise ValueError(f"/proc/meminfo missing fields: {missing}")
    return {k: round(values[k], 2) for k in required}


def parse_cpuinfo(text: str) -> tuple[str, int | None]:
    """Return (model name, physical core count or None if undeterminable)."""
    model = "unknown"
    cores: set[tuple[str, str]] = set()
    phys = core = None
    for line in [*text.splitlines(), ""]:
        if not line.strip():
            if phys is not None and core is not None:
                cores.add((phys, core))
            phys = core = None
            continue
        key, _, value = (s.strip() for s in line.partition(":"))
        if key == "model name" and model == "unknown":
            model = value
        elif key == "physical id":
            phys = value
        elif key == "core id":
            core = value
    return model, (len(cores) or None)


def parse_version(output: str) -> str | None:
    m = _VERSION_RE.search(output)
    return m.group(1) if m else None


def detect_tools(
    probes: Iterable[ToolProbe], which: Which = shutil.which, run: Runner = _run
) -> dict[str, ToolInfo]:
    found = [(p, path) for p in probes if (path := which(p.name))]

    def probe(item: tuple[ToolProbe, str]) -> ToolInfo:
        p, path = item
        if not p.version_args:
            return ToolInfo(name=p.name, path=path)
        try:
            _, out = run([path, *p.version_args], PROBE_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return ToolInfo(name=p.name, path=path, error=f"{type(exc).__name__}: {exc}")
        version = parse_version(out)
        err = None if version else f"could not parse version from: {out.strip()[:120]!r}"
        return ToolInfo(name=p.name, path=path, version=version, error=err)

    with ThreadPoolExecutor(max_workers=8) as pool:
        return {t.name: t for t in pool.map(probe, found)}


def detect_gpus(which: Which = shutil.which, run: Runner = _run) -> tuple[list[GpuInfo], list[str]]:
    exe = which("nvidia-smi")
    if not exe:
        return [], []
    argv = [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
    try:
        code, out = run(argv, PROBE_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], [f"nvidia-smi present but failed: {exc}"]
    if code != 0:
        return [], [f"nvidia-smi present but unusable (exit {code}): {out.strip()[:160]}"]
    gpus = []
    for line in out.strip().splitlines():
        name, _, mem = (s.strip() for s in line.rpartition(","))
        gpus.append(GpuInfo(name=name, memory_gb=round(float(mem) / 1024, 2) if mem else None))
    return gpus, []


SYSTEMD_PROBE = [
    "systemd-run", "--user", "--scope", "--quiet", "--collect",
    "-p", "MemoryMax=64M", "-p", "MemorySwapMax=0", "-p", "CPUQuota=100%", "--", "true",
]  # fmt: skip


def detect_enforcement(
    which: Which = shutil.which, run: Runner = _run
) -> tuple[list[str], list[str]]:
    """Verify, by actually trying it, that per-job cgroup limits can be applied."""
    if not which("systemd-run"):
        return [], []
    try:
        code, out = run(SYSTEMD_PROBE, PROBE_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], [f"systemd-run present but limit probe failed: {exc}"]
    if code != 0:
        msg = f"systemd-run user scopes unavailable (exit {code}): {out.strip()[:160]}"
        return [], [msg + "; resource limits will be advisory"]
    return [SYSTEMD_SCOPE], []


def inspect_system(workspace: Path | None = None) -> SystemResources:
    if platform.system() != "Linux":
        raise UnsupportedPlatformError(
            f"{platform.system()} is not supported yet; the MVP reads /proc (Linux only)"
        )
    workspace = (workspace or Path.cwd()).resolve()
    mem = parse_meminfo(Path("/proc/meminfo").read_text())
    cpu_model, physical = parse_cpuinfo(Path("/proc/cpuinfo").read_text())
    tmp = Path(tempfile.gettempdir())
    gpus, notes = detect_gpus()
    enforcement, enforcement_notes = detect_enforcement()

    return SystemResources(
        inspected_at=datetime.now(UTC),
        os=platform.freedesktop_os_release().get("PRETTY_NAME", "Linux"),
        kernel=platform.release(),
        cpu_model=cpu_model,
        cpu_physical_cores=physical,
        cpu_threads=len(os.sched_getaffinity(0)),
        ram_total_gb=mem["MemTotal"],
        ram_available_gb=mem["MemAvailable"],
        swap_total_gb=mem["SwapTotal"],
        swap_free_gb=mem["SwapFree"],
        workspace=str(workspace),
        disk_free_gb=round(shutil.disk_usage(workspace).free / GIB, 2),
        tmp_dir=str(tmp),
        tmp_free_gb=round(shutil.disk_usage(tmp).free / GIB, 2),
        gpus=gpus,
        tools=detect_tools(BIOINFORMATICS_TOOLS),
        container_runtimes=[r for r in CONTAINER_RUNTIMES if shutil.which(r)],
        schedulers=[s for s in SCHEDULERS if shutil.which(s)],
        enforcement=enforcement,
        notes=notes + enforcement_notes,
    )
