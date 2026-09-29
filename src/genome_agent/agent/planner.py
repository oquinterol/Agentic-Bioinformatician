"""Deterministic, LLM-free backend used for tests, simulations and as a baseline.

Policy (deliberately simple — scientific judgement is the LLM's job later):
1. candidates = available tools producing the goal, in registry order
   (registry order encodes preference);
2. drop tools that already failed or were rejected for this goal;
3. assess each remaining one against the budget (with thread fallback);
   each adapter's param_variants() are tried in order (e.g. hifiasm -f0);
4. run the first that fits; if none fits, stop and explain every rejection.

Knowledge-gap rule: if a reference is registered and read_origin_check is
available, reads that an assembler would use are checked first, and reads
whose verdict is "does not match reference" are never assembled.
"""

from __future__ import annotations

from pathlib import Path

from genome_agent.agent.backend import AgentAction, Observation, RunTool, Stop
from genome_agent.executor.validation import JobRequest
from genome_agent.state.models import JobStatus
from genome_agent.tools.feasibility import assess
from genome_agent.tools.registry import AnyAdapter, ToolInputs


class DeterministicPlanner:
    name = "planner:deterministic"

    def next_action(self, obs: Observation) -> AgentAction:
        state = obs.state
        res = state.system_resources
        assert res is not None
        spent = {
            j.tool
            for j in state.jobs
            if j.status in (JobStatus.FAILED, JobStatus.REJECTED, JobStatus.CANCELLED)
        }
        candidates = obs.registry.producing(obs.goal, res)
        if not candidates:
            return Stop(f"no available tool on this machine produces '{obs.goal}'")

        if missing := [d.path for d in state.datasets if not Path(d.path).is_file()]:
            return Stop("dataset files are missing", evidence={"missing": missing})
        datasets = {d.path: str(d.kind) for d in state.datasets}
        verdicts = state.origin_verdicts()
        if (
            check := self._origin_check_first(obs, candidates, datasets, verdicts, spent)
        ) is not None:
            return check
        excluded = {p: v for p, v in verdicts.items() if v == MISMATCH}
        datasets = {p: k for p, k in datasets.items() if p not in excluded}
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
                datasets,
                genome_size_bp=state.biological_context.genome_size_bp,
                read_bases=state.measured_read_bases(selected),
            )
            chosen, reasons = None, []
            for raw in adapter.param_variants():
                a = assess(adapter, inputs, adapter.parse_params(raw), obs.budget, obs.observations)
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
                        "excluded_by_origin_check": excluded,
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

    def _origin_check_first(
        self,
        obs: Observation,
        candidates: list[AnyAdapter],
        datasets: dict[str, str],
        verdicts: dict[str, str],
        spent: set[str],
    ) -> AgentAction | None:
        res = obs.state.system_resources
        assert res is not None
        refs = [p for p, k in datasets.items() if k == "reference_fasta"]
        if len(refs) != 1 or ORIGIN_TOOL in spent or ORIGIN_TOOL not in obs.registry.names():
            return None
        checker = obs.registry.get(ORIGIN_TOOL)
        if not checker.is_available(res):
            return None
        to_check = sorted(
            {
                p
                for a in candidates
                for p in a.select_inputs(datasets)
                if datasets[p] != "reference_fasta"
            }
            - set(verdicts)
        )
        if not to_check:
            return None
        files = [refs[0], *to_check]
        inputs = ToolInputs.from_files([Path(p) for p in files], datasets)
        a = assess(checker, inputs, checker.parse_params({}), obs.budget, obs.observations)
        if not a.fits or a.estimate is None:
            return None  # cannot check here; assembling proceeds on documented metadata
        return RunTool(
            JobRequest(
                tool=ORIGIN_TOOL,
                inputs=files,
                cpus=a.estimate.cpus,
                ram_gb=a.estimate.ram_gb,
                reason="verify that the reads come from the organism of the registered reference "
                "before assembling them (dataset metadata is not verified)",
                evidence={"unchecked_reads": to_check, "reference": refs[0]},
                actor=self.name,
            )
        )


def _reason(tool: str, obs: Observation, rejected: dict[str, list[str]]) -> str:
    text = f"{tool} is the most preferred tool producing '{obs.goal}' that fits the budget"
    if rejected:
        text += "; skipped: " + "; ".join(f"{t} ({', '.join(r)})" for t, r in rejected.items())
    return text


MISMATCH = "does not match reference"
ORIGIN_TOOL = "read_origin_check"
