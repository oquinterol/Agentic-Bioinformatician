"""Cheap, deterministic scenarios: a fake machine + mock tools + tiny fake reads.

Mock tools still run as real subprocesses through the real harness, so a
simulation exercises validation, execution, failure recording and replanning,
without any sequencing data or bioinformatics software.
"""

from __future__ import annotations

import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from genome_agent.agent.backend import AgentBackend
from genome_agent.agent.loop import LoopOutcome, run_loop
from genome_agent.agent.planner import DeterministicPlanner
from genome_agent.harness import Harness
from genome_agent.resources.models import ResourcePolicy, SystemResources
from genome_agent.state.models import BiologicalContext, Dataset, ProjectState, ReadKind
from genome_agent.tools.adapters.mock import MockAssembler
from genome_agent.tools.registry import DataType, ToolRegistry

_FAKE_READS = "@r1\nACGTACGTACGTACGT\n+\nIIIIIIIIIIIIIIII\n"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MachineSpec(_Strict):
    cpu_threads: int = Field(ge=1)
    ram_available_gb: float = Field(gt=0)
    ram_total_gb: float | None = None
    disk_free_gb: float = Field(gt=0)


class MockToolSpec(_Strict):
    name: str
    base_ram_gb: float = 1.0
    ram_gb_per_gbp: float = 0.0
    ram_gb_per_thread: float = 0.0
    disk_gb: float = 1.0
    fail_exit_code: int = 0
    n50: int = 1_000_000


class DatasetSpec(_Strict):
    name: str
    kind: ReadKind


class Scenario(_Strict):
    name: str
    description: str = ""
    goal: DataType = DataType.CONTIGS_FASTA
    expect: Literal["achieved", "stopped"]
    expect_tool: str | None = None  # tool whose job should produce the goal
    machine: MachineSpec
    context: BiologicalContext = Field(default_factory=BiologicalContext)
    policy: ResourcePolicy = Field(default_factory=ResourcePolicy)
    datasets: list[DatasetSpec] = Field(min_length=1)
    tools: list[MockToolSpec] = Field(min_length=1)  # order = preference

    @classmethod
    def load(cls, path: Path) -> Scenario:
        with path.open("rb") as fh:
            return cls.model_validate(tomllib.load(fh))

    def check(self, outcome: LoopOutcome, harness: Harness) -> list[str]:
        """Differences between what happened and what the scenario expects."""
        problems = []
        got = "achieved" if outcome.achieved else "stopped"
        if got != self.expect:
            problems.append(f"expected '{self.expect}', got '{got}' ({outcome.reason})")
        if self.expect_tool is not None:
            tool_of = {j.id: j.tool for j in harness.state.jobs}
            producers = {tool_of[r.job_id] for r in harness.state.results if r.kind == self.goal}
            if self.expect_tool not in producers:
                problems.append(f"expected {self.expect_tool} to produce the goal, got {producers}")
        return problems


def _fake_machine(m: MachineSpec, workspace: Path) -> SystemResources:
    return SystemResources(
        inspected_at=datetime.now(UTC),
        os="simulated",
        kernel="simulated",
        cpu_model="simulated",
        cpu_physical_cores=None,
        cpu_threads=m.cpu_threads,
        ram_total_gb=m.ram_total_gb or m.ram_available_gb,
        ram_available_gb=m.ram_available_gb,
        swap_total_gb=0,
        swap_free_gb=0,
        workspace=str(workspace),
        disk_free_gb=m.disk_free_gb,
        tmp_dir=str(workspace),
        tmp_free_gb=m.disk_free_gb,
    )


def build_project(sc: Scenario, project_dir: Path) -> Harness:
    if ProjectState.path_for(project_dir).exists():
        raise FileExistsError(f"{project_dir} already contains a GenomeAgent project")
    data = project_dir / "data"
    data.mkdir(parents=True, exist_ok=True)
    datasets = []
    for d in sc.datasets:
        f = data / f"{d.name}.fq"
        f.write_text(_FAKE_READS)
        datasets.append(Dataset(path=str(f), kind=d.kind, size_bytes=f.stat().st_size))
    ProjectState(
        name=sc.name,
        objective=sc.description,
        biological_context=sc.context,
        policy=sc.policy,
        system_resources=_fake_machine(sc.machine, project_dir),
        datasets=datasets,
    ).save(project_dir)
    registry = ToolRegistry([MockAssembler(**t.model_dump()) for t in sc.tools])
    return Harness(project_dir, registry)


def simulate(
    sc: Scenario, project_dir: Path, backend: AgentBackend | None = None
) -> tuple[LoopOutcome, Harness]:
    harness = build_project(sc, project_dir)
    outcome = run_loop(harness, backend or DeterministicPlanner(), sc.goal)
    return outcome, harness
