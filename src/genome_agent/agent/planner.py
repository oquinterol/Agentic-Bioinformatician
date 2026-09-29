"""Deterministic, LLM-free backend used for tests, simulations and as a baseline.

Policy (deliberately simple — scientific judgement is the LLM's job later):
1. candidates = available tools producing the goal, in registry order
   (registry order encodes preference);
2. drop tools that already failed or were rejected for this goal;
3. assess each remaining one against the budget (with thread fallback);
4. run the first that fits; if none fits, stop and explain every rejection.
"""

from __future__ import annotations

from pathlib import Path

from genome_agent.agent.backend import AgentAction, Observation, RunTool, Stop
from genome_agent.executor.validation import JobRequest
from genome_agent.state.models import JobStatus
from genome_agent.tools.feasibility import assess
from genome_agent.tools.registry import ToolInputs


class DeterministicPlanner:
    name = "planner:deterministic"

    def next_action(self, obs: Observation) -> AgentAction:
        state = obs.state
        res = state.system_resources
        assert res is not None
        spent = {j.tool for j in state.jobs if j.status in (JobStatus.FAILED, JobStatus.REJECTED)}
        candidates = obs.registry.producing(obs.goal, res)
        if not candidates:
            return Stop(f"no available tool on this machine produces '{obs.goal}'")

        paths = [Path(d.path) for d in state.datasets]
        if missing := [str(p) for p in paths if not p.is_file()]:
            return Stop("dataset files are missing", evidence={"missing": missing})
        inputs = ToolInputs.from_files(
            paths,
            genome_size_bp=state.biological_context.genome_size_bp,
        )
        rejected: dict[str, list[str]] = {
            j.tool: [
                f"previous attempt {j.id} {j.status}: "
                + (j.error or "; ".join(j.rejection_reasons) or f"exit code {j.exit_code}")
            ]
            for j in state.jobs
            if j.tool in spent
        }
        for adapter in candidates:
            if adapter.name in spent:
                continue
            a = assess(adapter, inputs, adapter.parse_params({}), obs.budget)
            if not a.fits:
                rejected[adapter.name] = a.reasons
                continue
            assert a.estimate is not None
            others = [c.name for c in candidates if c.name != adapter.name]
            return RunTool(
                JobRequest(
                    tool=adapter.name,
                    inputs=[d.path for d in state.datasets],
                    cpus=a.estimate.cpus,
                    ram_gb=a.estimate.ram_gb,
                    genome_size_bp=state.biological_context.genome_size_bp,
                    reason=_reason(adapter.name, obs, rejected),
                    alternatives_considered=others,
                    evidence={"rejected_alternatives": rejected, "budget": obs.budget.model_dump()},
                    actor=self.name,
                )
            )
        return Stop(
            f"no candidate for '{obs.goal}' fits this machine or all have failed",
            evidence={"rejected": rejected, "budget": obs.budget.model_dump()},
        )


def _reason(tool: str, obs: Observation, rejected: dict[str, list[str]]) -> str:
    text = f"{tool} is the most preferred tool producing '{obs.goal}' that fits the budget"
    if rejected:
        text += "; skipped: " + "; ".join(f"{t} ({', '.join(r)})" for t, r in rejected.items())
    return text
