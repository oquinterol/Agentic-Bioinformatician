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
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

STDOUT_FILE = "stdout.log"
STDERR_FILE = "stderr.log"
_POLL_S = 0.1


class ExecutionResult(BaseModel):
    """Outcome of one process; also the on-disk `execution.json` written by the runner."""

    model_config = ConfigDict(frozen=True)

    exit_code: int | None  # None if the process never started
    wall_time_s: float
    stdout_path: Path
    stderr_path: Path
    timed_out: bool = False
    cancelled: bool = False
    error: str | None = None  # launch failure, timeout or cancellation description
    peak_rss_gb: float | None = None  # observed, from wait4 (None if never started)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.cancelled and not self.error


class Executor(Protocol):
    def run(
        self, argv: list[str], outdir: Path, cpus: int, ram_gb: float, timeout_s: float
    ) -> ExecutionResult: ...


def _wait(pid: int, timeout_s: float) -> tuple[int, float, bool]:
    """Wait for `pid`; return (exit code, peak RSS GiB, timed_out).

    wait4 gives the child's peak RSS (on Linux, the max over the child and the
    descendants it reaped), which is how estimates get checked against reality.
    """
    deadline = time.monotonic() + timeout_s
    timed_out = False
    while True:
        wpid, status, usage = os.wait4(pid, os.WNOHANG)
        if wpid == pid:
            break
        if time.monotonic() >= deadline:
            os.killpg(pid, signal.SIGKILL)
            _, status, usage = os.wait4(pid, 0)
            timed_out = True
            break
        time.sleep(_POLL_S)
    return os.waitstatus_to_exitcode(status), usage.ru_maxrss / 1024**2, timed_out


class LocalExecutor:
    """Advisory limits only: cpus/ram were validated, not enforced by the OS."""

    def __init__(self) -> None:
        self.current_pid: int | None = None  # process group to kill on cancellation
        self.cancel_requested = False

    def cancel(self) -> None:
        """Safe to call from a signal handler at any point before, during or after run()."""
        self.cancel_requested = True
        if self.current_pid is not None:
            os.killpg(self.current_pid, signal.SIGKILL)

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
                    exit_code=None,
                    wall_time_s=time.monotonic() - start,
                    stdout_path=out_path,
                    stderr_path=err_path,
                    error=f"launch: {exc}",
                )
            self.current_pid = proc.pid
            if self.cancel_requested:  # cancelled while the process was being started
                os.killpg(proc.pid, signal.SIGKILL)
            code, peak, timed_out = _wait(proc.pid, timeout_s)
            proc.returncode = code  # reaped by wait4; keep Popen consistent
            self.current_pid = None
        return ExecutionResult(
            exit_code=code,
            wall_time_s=time.monotonic() - start,
            stdout_path=out_path,
            stderr_path=err_path,
            timed_out=timed_out,
            error=f"timed out after {timeout_s:g} s" if timed_out else None,
            peak_rss_gb=round(peak, 3),
        )
