"""Run one already-validated command as a local process.

No shell: argv goes straight to the OS. stdout/stderr stream to files (tools
like hifiasm log for hours; never buffer that in memory). On timeout the whole
process group is killed, so child processes do not outlive the job.

`Executor` is the seam for future backends (systemd-run with MemoryMax/CPUQuota,
containers, SLURM): they receive the same reservation and return the same result.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

STDOUT_FILE = "stdout.log"
STDERR_FILE = "stderr.log"


@dataclass(frozen=True)
class ExecutionResult:
    exit_code: int | None  # None if the process never started
    wall_time_s: float
    stdout_path: Path
    stderr_path: Path
    timed_out: bool = False
    error: str | None = None  # launch failure or timeout description

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and self.error is None


class Executor(Protocol):
    def run(
        self, argv: list[str], outdir: Path, cpus: int, ram_gb: float, timeout_s: float
    ) -> ExecutionResult: ...


class LocalExecutor:
    """Advisory limits only: cpus/ram were validated, not enforced by the OS."""

    def run(
        self, argv: list[str], outdir: Path, cpus: int, ram_gb: float, timeout_s: float
    ) -> ExecutionResult:
        outdir.mkdir(parents=True, exist_ok=True)
        out_path, err_path = outdir / STDOUT_FILE, outdir / STDERR_FILE
        env = os.environ | {"OMP_NUM_THREADS": str(cpus)}
        start = time.monotonic()
        with out_path.open("wb") as out, err_path.open("wb") as err:
            try:
                proc = subprocess.Popen(
                    argv, stdout=out, stderr=err, cwd=outdir, env=env, start_new_session=True
                )
            except OSError as exc:
                return ExecutionResult(
                    None, time.monotonic() - start, out_path, err_path, error=f"launch: {exc}"
                )
            try:
                code = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                code = proc.wait()
                return ExecutionResult(
                    code,
                    time.monotonic() - start,
                    out_path,
                    err_path,
                    timed_out=True,
                    error=f"timed out after {timeout_s:g} s",
                )
        return ExecutionResult(code, time.monotonic() - start, out_path, err_path)
