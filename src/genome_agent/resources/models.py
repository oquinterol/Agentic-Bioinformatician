"""Authoritative description of the machine and the policy applied to it."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class GpuInfo(BaseModel):
    name: str
    memory_gb: float | None = None


class ToolInfo(BaseModel):
    """A bioinformatics executable found on PATH."""

    name: str
    path: str
    version: str | None = None
    error: str | None = None  # probe failed; tool present but version unknown


class SystemResources(BaseModel):
    """Snapshot of the local machine. Produced only by the inspector, never by the LLM."""

    inspected_at: datetime
    os: str
    kernel: str
    cpu_model: str
    cpu_physical_cores: int | None
    cpu_threads: int  # logical CPUs usable by this process (affinity-aware)
    ram_total_gb: float
    ram_available_gb: float
    swap_total_gb: float
    swap_free_gb: float
    workspace: str
    disk_free_gb: float
    tmp_dir: str
    tmp_free_gb: float
    gpus: list[GpuInfo] = Field(default_factory=list)
    tools: dict[str, ToolInfo] = Field(default_factory=dict)
    container_runtimes: list[str] = Field(default_factory=list)
    schedulers: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)  # non-fatal detection problems


class ResourcePolicy(BaseModel):
    """Limits on how much of the machine jobs may use. Configurable per project."""

    ram_fraction: float = Field(default=0.8, gt=0, le=1)
    reserved_threads: int = Field(default=2, ge=0)
    disk_fraction: float = Field(default=0.8, gt=0, le=1)

    def apply(self, res: SystemResources) -> ResourceBudget:
        return ResourceBudget(
            cpu_threads=max(1, res.cpu_threads - self.reserved_threads),
            ram_gb=round(res.ram_available_gb * self.ram_fraction, 2),
            disk_gb=round(res.disk_free_gb * self.disk_fraction, 2),
        )


class ResourceBudget(BaseModel):
    """Hard ceiling for any single job request."""

    cpu_threads: int
    ram_gb: float
    disk_gb: float

    def violations(self, est: ResourceEstimate) -> list[str]:
        """Human-readable reasons why `est` does not fit; empty if it fits."""
        out = []
        if est.cpus > self.cpu_threads:
            out.append(f"needs {est.cpus} threads > budget {self.cpu_threads}")
        if est.ram_gb > self.ram_gb:
            out.append(f"needs {est.ram_gb:.1f} GB RAM > budget {self.ram_gb:.1f} GB")
        if est.disk_gb + est.tmp_gb > self.disk_gb:
            out.append(
                f"needs {est.disk_gb + est.tmp_gb:.1f} GB disk (incl. tmp) "
                f"> budget {self.disk_gb:.1f} GB"
            )
        return out


class ResourceEstimate(BaseModel):
    """Predicted peak usage of one job. `basis` says how it was derived (auditable)."""

    cpus: int = Field(ge=1)
    ram_gb: float = Field(ge=0)
    disk_gb: float = Field(default=0, ge=0)  # final outputs
    tmp_gb: float = Field(default=0, ge=0)  # scratch, freed after the job
    wall_time_h: float | None = None
    basis: str
