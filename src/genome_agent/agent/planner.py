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
from genome_agent.state.models import JobStatus, ProjectState
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
        if (profile := self._profile_first(obs, datasets, spent)) is not None:
            return profile
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
            if adapter.requires_verified_origin and (bad := _inconsistent_pairs(state, selected)):
                rejected[adapter.name] = [
                    f"libraries {sorted(pair)} are inconsistent (different organisms?) and no "
                    "reference can tell which one is the declared species: register a reference "
                    "or marker sequences, or exclude one library explicitly"
                    for pair in bad
                ]
                continue
            if not selected:
                rejected[adapter.name] = [
                    f"no registered dataset of kind {sorted(adapter.accepted_read_kinds or [])}"
                ]
                continue
            inputs = ToolInputs.from_files(
                [Path(p) for p in selected],
                datasets,
                genome_size_bp=state.effective_genome_size()[0],
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
                    genome_size_bp=state.effective_genome_size()[0],
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
        why = "; ".join(f"{tool}: {reasons[0]}" for tool, reasons in rejected.items() if reasons)
        return Stop(
            f"no candidate for '{obs.goal}' can run responsibly here ({why or 'none available'})",
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
        if not refs:
            return self._consistency_check_first(obs, candidates, datasets, spent)
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

    def _profile_first(
        self, obs: Observation, datasets: dict[str, str], spent: set[str]
    ) -> AgentAction | None:
        """Unknown genome size is a knowledge gap: measure it before planning around it."""
        state = obs.state
        res = state.system_resources
        assert res is not None
        if state.effective_genome_size()[0] is not None or PROFILE_TOOL in spent:
            return None
        if PROFILE_TOOL not in obs.registry.names():
            return None
        profiler = obs.registry.get(PROFILE_TOOL)
        libs = profiler.select_inputs(datasets)
        if not libs or not profiler.is_available(res):
            return None
        ploidy = state.biological_context.expected_ploidy or 2
        inputs = ToolInputs.from_files([Path(p) for p in libs], datasets)
        params = {"ploidy": ploidy}
        a = assess(profiler, inputs, profiler.parse_params(params), obs.budget, obs.observations)
        if not a.fits or a.estimate is None:
            return None
        return RunTool(
            JobRequest(
                tool=PROFILE_TOOL,
                params=params,
                inputs=libs,
                cpus=a.estimate.cpus,
                ram_gb=a.estimate.ram_gb,
                reason="genome size is unknown: estimate it (and heterozygosity) from k-mers "
                "before choosing or sizing an assembly",
                evidence={"assumed_ploidy": ploidy},
                actor=self.name,
            )
        )

    def _consistency_check_first(
        self,
        obs: Observation,
        candidates: list[AnyAdapter],
        datasets: dict[str, str],
        spent: set[str],
    ) -> AgentAction | None:
        """No reference: libraries an assembler would pool must be cross-checked first."""
        res = obs.state.system_resources
        assert res is not None
        if CONSISTENCY_TOOL in spent or CONSISTENCY_TOOL not in obs.registry.names():
            return None
        checker = obs.registry.get(CONSISTENCY_TOOL)
        if not checker.is_available(res):
            return None
        known = obs.state.consistency_verdicts()
        for adapter in candidates:
            if not adapter.requires_verified_origin:
                continue
            libs = adapter.select_inputs(datasets)
            unchecked = [
                (x, y)
                for i, x in enumerate(libs)
                for y in libs[i + 1 :]
                if frozenset((x, y)) not in known
            ]
            if not unchecked:
                continue
            inputs = ToolInputs.from_files(
                [Path(p) for p in libs],
                datasets,
                genome_size_bp=obs.state.effective_genome_size()[0],
            )
            a = assess(checker, inputs, checker.parse_params({}), obs.budget, obs.observations)
            if not a.fits or a.estimate is None:
                return None
            return RunTool(
                JobRequest(
                    tool=CONSISTENCY_TOOL,
                    inputs=libs,
                    genome_size_bp=obs.state.effective_genome_size()[0],
                    cpus=a.estimate.cpus,
                    ram_gb=a.estimate.ram_gb,
                    reason=f"no reference is registered: check that the libraries {adapter.name} "
                    "would pool come from the same organism before assembling them",
                    evidence={"unchecked_pairs": [list(p) for p in unchecked]},
                    actor=self.name,
                )
            )
        return None


def _reason(tool: str, obs: Observation, rejected: dict[str, list[str]]) -> str:
    text = f"{tool} is the most preferred tool producing '{obs.goal}' that fits the budget"
    if rejected:
        text += "; skipped: " + "; ".join(f"{t} ({', '.join(r)})" for t, r in rejected.items())
    return text


MISMATCH = "does not match reference"
ORIGIN_TOOL = "read_origin_check"
CONSISTENCY_TOOL = "library_consistency_check"
PROFILE_TOOL = "kmer_profile"


def _inconsistent_pairs(state: ProjectState, libs: list[str]) -> list[frozenset[str]]:
    known = state.consistency_verdicts()
    return [
        frozenset((x, y))
        for i, x in enumerate(libs)
        for y in libs[i + 1 :]
        if "inconsistent" in known.get(frozenset((x, y)), set())
    ]
