"""OBSERVE → DECIDE → EXECUTE → EVALUATE → (continue | replan | stop).

The loop owns termination and bookkeeping; the backend owns the choice. Agent
behaviour metrics (iterations, replans, failed jobs) are written to
state.metrics under stable `agent_*` keys for later benchmarking.
"""

from __future__ import annotations

from pydantic import BaseModel

from genome_agent.agent.backend import AgentBackend, Observation, RunTool, Stop
from genome_agent.harness import Harness
from genome_agent.state.models import DecisionRecord, JobStatus
from genome_agent.tools.registry import DataType

DEFAULT_MAX_ITERATIONS = 10


class LoopOutcome(BaseModel):
    achieved: bool
    reason: str
    iterations: int
    replans: int
    jobs_succeeded: int
    jobs_failed: int
    jobs_rejected: int


def goal_reached(harness: Harness, goal: DataType) -> bool:
    return any(r.kind == goal for r in harness.state.results)


def run_loop(
    harness: Harness,
    backend: AgentBackend,
    goal: DataType,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> LoopOutcome:
    state = harness.state
    res = state.system_resources
    assert res is not None
    budget = state.policy.apply(res)
    iterations = replans = 0
    reason = f"iteration limit ({max_iterations}) reached"
    achieved = False

    while iterations < max_iterations:
        if goal_reached(harness, goal):
            achieved, reason = True, f"goal '{goal}' reached"
            break
        iterations += 1
        action = backend.next_action(Observation(state, harness.registry, budget, goal))
        if isinstance(action, Stop):
            reason = action.reason
            harness.record_decision(
                DecisionRecord(
                    decision="stop",
                    reason=action.reason,
                    evidence=action.evidence,
                    actor=backend.name,
                )
            )
            break
        assert isinstance(action, RunTool)
        job = harness.run_tool(action.request)
        if job.status != JobStatus.SUCCEEDED:
            replans += 1  # the next iteration must choose differently
    else:
        achieved = goal_reached(harness, goal)
        if achieved:
            reason = f"goal '{goal}' reached"

    counts = {s: sum(j.status == s for j in state.jobs) for s in JobStatus}
    outcome = LoopOutcome(
        achieved=achieved,
        reason=reason,
        iterations=iterations,
        replans=replans,
        jobs_succeeded=counts[JobStatus.SUCCEEDED],
        jobs_failed=counts[JobStatus.FAILED],
        jobs_rejected=counts[JobStatus.REJECTED],
    )
    state.metrics.update(
        {
            "agent_iterations": iterations,
            "agent_replans": replans,
            "agent_jobs_failed": outcome.jobs_failed,
            "agent_jobs_rejected": outcome.jobs_rejected,
            "agent_goal_achieved": float(achieved),
        }
    )
    state.save(harness.project_dir)
    return outcome
