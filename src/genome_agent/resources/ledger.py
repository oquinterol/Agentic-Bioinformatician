"""Machine-wide reservation ledger shared by every project on this machine.

Per-project budgets alone let two projects each believe they own the whole
machine (e.g. a k-mer profile launched while an assembly runs elsewhere).
Every running job is therefore also recorded in
`$GENOME_AGENT_DATA_DIR/reservations.json`, and new jobs are validated
against min(project free budget, machine free budget).

Lock order is always project lock -> ledger lock; the ledger never takes a
project lock, so the two cannot deadlock. Entries whose runner is gone are
garbage-collected on every read.
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, TypeAdapter

from genome_agent.resources.models import ResourceBudget, ResourcePolicy
from genome_agent.resources.observations import default_data_dir

LEDGER_FILE = "reservations.json"

# policy -> (threads, ram_gb) the machine offers to jobs in total
Capacity = Callable[[ResourcePolicy], tuple[int, float]]


def live_capacity(policy: ResourcePolicy) -> tuple[int, float]:
    """Read the machine now (not a project snapshot)."""
    threads = len(os.sched_getaffinity(0))
    mem_total_kib = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemTotal:")
    )
    ram = mem_total_kib / 1024**2 * policy.ram_fraction
    return max(1, threads - policy.reserved_threads), round(ram, 2)


class Reservation(BaseModel):
    key: str  # "<project dir>::<job id>"
    project: str
    job_id: str
    outdir: str
    runner_pid: int
    cpus: int
    ram_gb: float
    started: datetime = Field(default_factory=lambda: datetime.now(UTC))


_LIST = TypeAdapter(list[Reservation])


class MachineLedger:
    def __init__(self, path: Path | None = None, capacity: Capacity = live_capacity) -> None:
        self.path = path or default_data_dir() / LEDGER_FILE
        self.capacity = capacity
        self._depth = 0

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Exclusive ledger lock (re-entrant within this object)."""
        if self._depth:
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            self._depth = 1
            try:
                yield
            finally:
                self._depth = 0
                fcntl.flock(fh, fcntl.LOCK_UN)

    def _read(self) -> list[Reservation]:
        if not self.path.exists():
            return []
        return _LIST.validate_json(self.path.read_text() or "[]")

    def _write(self, entries: list[Reservation]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_bytes(_LIST.dump_json(entries, indent=2))
        tmp.replace(self.path)

    def live(self) -> list[Reservation]:
        """Current reservations; entries whose runner is gone are dropped."""
        from genome_agent.executor.runner import is_runner_alive

        with self.locked():
            entries = self._read()
            alive = [e for e in entries if is_runner_alive(e.runner_pid, Path(e.outdir))]
            if len(alive) != len(entries):
                self._write(alive)
            return alive

    def reserve(self, r: Reservation) -> None:
        with self.locked():
            entries = [e for e in self._read() if e.key != r.key]
            self._write([*entries, r])

    def release(self, key: str) -> None:
        with self.locked():
            entries = self._read()
            kept = [e for e in entries if e.key != key]
            if len(kept) != len(entries):
                self._write(kept)

    def free(self, policy: ResourcePolicy) -> tuple[ResourceBudget, list[str]]:
        """What the whole machine still offers to a new job, and who holds the rest."""
        threads, ram = self.capacity(policy)
        live = self.live()
        return (
            ResourceBudget(
                cpu_threads=threads - sum(e.cpus for e in live),
                ram_gb=round(ram - sum(e.ram_gb for e in live), 2),
                disk_gb=float("inf"),  # disk is per filesystem; checked per project
            ),
            [e.key for e in live],
        )
