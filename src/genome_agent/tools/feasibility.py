"""Does a tool fit the budget, and with how many threads?

Deterministic and LLM-free, so both the simulated planner and an LLM backend
get the same authoritative answer.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from genome_agent.resources.models import ResourceBudget, ResourceEstimate
from genome_agent.tools.registry import EstimationError, ToolAdapter, ToolInputs


class Assessment(BaseModel):
    tool: str
    fits: bool
    estimate: ResourceEstimate | None = None  # at the chosen (or minimum) thread count
    reasons: list[str]  # why it does not fit, or which knowledge is missing


def assess(
    adapter: ToolAdapter[Any], inputs: ToolInputs, params: Any, budget: ResourceBudget
) -> Assessment:
    """Try the most threads the budget allows, then fewer, until the estimate fits.

    Fewer threads is the first fallback because per-thread memory is common
    (e.g. hifiasm, minimap2 index buffers).
    """
    try:
        est = None
        for cpus in range(budget.cpu_threads, 0, -1):
            est = adapter.estimate(inputs, params, cpus)
            if not budget.violations(est):
                return Assessment(tool=adapter.name, fits=True, estimate=est, reasons=[])
    except EstimationError as exc:
        return Assessment(tool=adapter.name, fits=False, reasons=[str(exc)])
    assert est is not None  # budget.cpu_threads >= 1 guarantees one iteration
    return Assessment(tool=adapter.name, fits=False, estimate=est, reasons=budget.violations(est))
