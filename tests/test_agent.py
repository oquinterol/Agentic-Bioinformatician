from pathlib import Path

import pytest

from genome_agent import cli
from genome_agent.agent.backend import Observation, RunTool, Stop
from genome_agent.agent.loop import run_loop
from genome_agent.agent.simulation import Scenario, build_project, simulate
from genome_agent.state.models import JobStatus, ProjectState
from genome_agent.tools.registry import DataType

EXAMPLES = Path(__file__).parent.parent / "examples"


def scenario(**overrides) -> Scenario:
    base = {
        "name": "t",
        "expect": "achieved",
        "machine": {"cpu_threads": 8, "ram_available_gb": 8, "disk_free_gb": 40},
        "context": {"genome_size_bp": 800_000_000},
        "datasets": [{"name": "hifi", "kind": "pacbio_hifi"}],
        "tools": [{"name": "asm"}],
    }
    return Scenario.model_validate(base | overrides)


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.toml")), ids=lambda p: p.stem)
def test_example_scenarios_meet_expectations(path, tmp_path):
    sc = Scenario.load(path)
    outcome, harness = simulate(sc, tmp_path / "proj")
    assert sc.check(outcome, harness) == []


def test_rejection_is_explained_in_the_run_decision(tmp_path):
    sc = Scenario.load(EXAMPLES / "simple_assembly.toml")
    _, h = simulate(sc, tmp_path / "p")
    d = ProjectState.load(h.project_dir).decisions[-1]
    assert d.decision == "run_assembler_B" and d.actor == "planner:deterministic"
    assert d.alternatives_considered == ["assembler_A"]
    assert "32.0 GB RAM" in d.evidence["rejected_alternatives"]["assembler_A"][0]


def test_impossible_stop_is_recorded_with_every_reason(tmp_path):
    _, h = simulate(Scenario.load(EXAMPLES / "impossible.toml"), tmp_path / "p")
    state = ProjectState.load(h.project_dir)
    assert state.jobs == []  # nothing was launched
    stop = state.decisions[-1]
    assert stop.decision == "stop" and set(stop.evidence["rejected"]) == {
        "assembler_A",
        "assembler_B",
    }
    assert state.metrics["agent_goal_achieved"] == 0.0


def test_recovery_does_not_retry_the_failed_tool(tmp_path):
    outcome, h = simulate(Scenario.load(EXAMPLES / "recover_after_failure.toml"), tmp_path / "p")
    assert [(j.tool, j.status) for j in h.state.jobs] == [
        ("assembler_B", JobStatus.FAILED),
        ("assembler_C", JobStatus.SUCCEEDED),
    ]
    assert outcome.replans == 1 and h.state.metrics["agent_replans"] == 1


def test_all_candidates_fail_then_stop(tmp_path):
    sc = scenario(
        expect="stopped",
        tools=[{"name": "a", "fail_exit_code": 2}, {"name": "b", "fail_exit_code": 3}],
    )
    outcome, _ = simulate(sc, tmp_path / "p")
    assert not outcome.achieved and outcome.jobs_failed == 2
    assert "all have failed" in outcome.reason


def test_unknown_genome_size_stops_with_knowledge_gap(tmp_path):
    outcome, h = simulate(scenario(expect="stopped", context={}), tmp_path / "p")
    assert not outcome.achieved
    assert "profile the reads first" in h.state.decisions[-1].evidence["rejected"]["asm"][0]


def test_iteration_limit_is_respected(tmp_path):
    class Stubborn:  # keeps asking for a tool that is always rejected
        name = "stubborn"

        def next_action(self, obs: Observation) -> RunTool | Stop:
            from genome_agent.executor.validation import JobRequest

            return RunTool(JobRequest(tool="nope", cpus=1, ram_gb=1, reason="x", actor=self.name))

    h = build_project(scenario(), tmp_path / "p")
    outcome = run_loop(h, Stubborn(), DataType.CONTIGS_FASTA, max_iterations=3)
    assert not outcome.achieved and outcome.iterations == 3 and outcome.jobs_rejected == 3


def test_scenario_rejects_unknown_keys():
    with pytest.raises(ValueError, match="typo_field"):
        scenario(typo_field=1)


def test_build_project_refuses_existing_project(tmp_path):
    build_project(scenario(), tmp_path / "p")
    with pytest.raises(FileExistsError):
        build_project(scenario(), tmp_path / "p")


def test_cli_simulate(tmp_path, capsys):
    rc = cli.main(["simulate", str(EXAMPLES / "impossible.toml"), "--project", str(tmp_path / "p")])
    out = capsys.readouterr().out
    assert (
        rc == 0 and "STOPPED" in out and "x assembler_A: params default: needs 35.0 GB RAM" in out
    )


def test_cli_simulate_reports_mismatch(tmp_path, capsys):
    bad = tmp_path / "bad.toml"
    bad.write_text((EXAMPLES / "impossible.toml").read_text().replace('"stopped"', '"achieved"'))
    assert cli.main(["simulate", str(bad), "--project", str(tmp_path / "p")]) == 1
    assert "MISMATCH" in capsys.readouterr().err
