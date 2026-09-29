"""Detached job runner: `python -m genome_agent.executor.runner <outdir>`.

The harness writes `<outdir>/job.json` and starts this process in its own
session, so a job outlives the CLI call or Pi tool call that launched it. The
runner executes the command, then writes `<outdir>/execution.json` atomically.
It never touches state.json; the harness reconciles results when it next opens
the project. SIGTERM cancels: the tool's process group is killed and the
result is recorded as cancelled.
"""

from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path
from types import FrameType

from pydantic import BaseModel

from genome_agent.executor.local import ExecutionResult, LocalExecutor

JOB_SPEC_FILE = "job.json"
EXECUTION_FILE = "execution.json"
RUNNER_LOG = "runner.log"
RUNNER_MODULE = "genome_agent.executor.runner"


class JobSpec(BaseModel):
    argv: list[str]
    cpus: int
    ram_gb: float
    timeout_s: float


def launch(outdir: Path, spec: JobSpec) -> subprocess.Popen[bytes]:
    """Write the spec and start a detached runner for it."""
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / JOB_SPEC_FILE).write_text(spec.model_dump_json(indent=2))
    with (outdir / RUNNER_LOG).open("ab") as log:
        return subprocess.Popen(
            [sys.executable, "-m", RUNNER_MODULE, str(outdir)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )


def read_execution(outdir: Path) -> ExecutionResult | None:
    path = outdir / EXECUTION_FILE
    return ExecutionResult.model_validate_json(path.read_text()) if path.exists() else None


def is_runner_alive(pid: int | None, outdir: Path) -> bool:
    """True if `pid` is still the runner for `outdir` (guards against pid reuse)."""
    if pid is None:
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        status = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return status != "Z" and RUNNER_MODULE.encode() in cmdline and str(outdir).encode() in cmdline


def main(argv: list[str] | None = None) -> int:
    executor = LocalExecutor()

    def on_term(signum: int, frame: FrameType | None) -> None:
        executor.cancel()

    signal.signal(signal.SIGTERM, on_term)  # first, so an early cancel is never lost
    outdir = Path((argv or sys.argv[1:])[0]).resolve()
    spec = JobSpec.model_validate_json((outdir / JOB_SPEC_FILE).read_text())
    result = executor.run(spec.argv, outdir, spec.cpus, spec.ram_gb, spec.timeout_s)
    if executor.cancel_requested:
        result = result.model_copy(update={"cancelled": True, "error": "cancelled on request"})
    tmp = outdir / (EXECUTION_FILE + ".tmp")
    tmp.write_text(result.model_dump_json(indent=2))
    tmp.replace(outdir / EXECUTION_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
