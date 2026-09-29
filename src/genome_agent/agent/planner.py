"""Deterministic, LLM-free backend used for tests, simulations and as a baseline.

Policy (deliberately simple — scientific judgement is the LLM's job later):
1. candidates = available tools producing the goal, in registry order
   (registry order encodes preference);
2. drop tools that already failed or were rejected for this goal;
3. assess each remaining one against the budget (with thread fallback);
   each adapter's param_variants() are tried in order (e.g. hifiasm -f0);
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

        if missing := [d.path for d in state.datasets if not Path(d.path).is_file()]:
            return Stop("dataset files are missing", evidence={"missing": missing})
        datasets = {d.path: str(d.kind) for d in state.datasets}
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
            selected = adapter.select_inputs(datasets)
            if not selected:
                rejected[adapter.name] = [
                    f"no registered dataset of kind {sorted(adapter.accepted_read_kinds or [])}"
                ]
                continue
            inputs = ToolInputs.from_files(
                [Path(p) for p in selected],
                genome_size_bp=state.biological_context.genome_size_bp,
            )
            chosen, reasons = None, []
            for raw in adapter.param_variants():
                a = assess(adapter, inputs, adapter.parse_params(raw), obs.budget)
                if a.fits:
                    chosen = (raw, a)
                    break
                reasons += [f"params {raw or 'default'}: {r}" for r in a.reasons]
            if chosen is None:
                rejected[adapter.name] = reasons
                continue
            raw, a = chosen
            assert a.estimate is not None
            others = [c.name for c in candidates if c.name != adapter.name]
            if reasons:  # a lower-resource variant was needed
                others.append(f"{adapter.name} with default params (does not fit)")
            return RunTool(
                JobRequest(
                    tool=adapter.name,
                    params=raw,
                    inputs=selected,
                    cpus=a.estimate.cpus,
                    ram_gb=a.estimate.ram_gb,
                    genome_size_bp=state.biological_context.genome_size_bp,
                    reason=_reason(
                        adapter.name, obs, rejected | ({adapter.name: reasons} if reasons else {})
                    ),
                    alternatives_considered=others,
                    evidence={
                        "rejected_alternatives": rejected,
                        "rejected_variants": reasons,
                        "estimate_basis": a.estimate.basis,
                        "budget": obs.budget.model_dump(),
                    },
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
