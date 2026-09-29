"""The contract between the loop and whatever does the reasoning.

A backend sees an Observation and returns one AgentAction. It never executes
anything itself: RunTool goes through Harness.run_tool (validation, budget,
provenance). Deterministic planners and LLM backends (Pi, raw APIs, local
models) implement the same protocol, so they can be benchmarked on identical
scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from genome_agent.executor.validation import JobRequest
from genome_agent.resources.models import ResourceBudget
from genome_agent.state.models import ProjectState
from genome_agent.tools.registry import DataType, ToolRegistry


@dataclass(frozen=True)
class Observation:
    state: ProjectState
    registry: ToolRegistry
    budget: ResourceBudget
    goal: DataType


@dataclass(frozen=True)
class RunTool:
    request: JobRequest


@dataclass(frozen=True)
class Stop:
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


type AgentAction = RunTool | Stop


class AgentBackend(Protocol):
    name: str

    def next_action(self, obs: Observation) -> AgentAction: ...
