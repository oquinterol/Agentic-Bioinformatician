from pathlib import Path

from genome_agent import cli
from genome_agent.agent.simulation import Scenario, build_project
from genome_agent.benchmark import _extract_plan, summarize
from genome_agent.bridge import call
from genome_agent.state.models import ProjectState

EXAMPLES = Path(__file__).parent.parent / "examples"


def project(tmp_path, name="llm"):
    h = build_project(Scenario.load(EXAMPLES / "simple_assembly.toml"), tmp_path / name)
    return h, h.state.datasets[0].path


def plan_args(reads, **plan_overrides):
    plan = {"tool": "assembler_B", "params": {}, "inputs": [reads], "cpus": 6, "ram_gb": 6.2}
    return {
        "decision": "assembly_plan",
        "reason": "fits",
        "evidence": {"risks": ["ram"]},
        "plan": plan | plan_overrides,
    }


def test_summarize_llm_style_plan(tmp_path, capsys):
    h, reads = project(tmp_path)
    call(h.project_dir, "assess_tool", {"tool": "assembler_B"}, "llm:x/y")
    call(h.project_dir, "assess_tool", {"tool": "ghost"}, "llm:x/y")  # an error
    assert call(h.project_dir, "record_decision", plan_args(reads), "llm:x/y")["ok"]
    s = summarize(h.project_dir)
    assert s.actor == "llm:x/y" and s.plan["ram_gb"] == 6.2
    assert s.reservation_vs_budget == "6.2 / 6.4 GB RAM, 6 / 6 threads"
    assert (s.bridge_calls, s.bridge_errors) == (3, 1) and s.plan_risks == ["ram"]
    assert cli.main(["compare", str(h.project_dir)]) == 0
    assert "[llm:x/y]" in capsys.readouterr().out


def test_summarize_without_plan(tmp_path):
    h = build_project(Scenario.load(EXAMPLES / "impossible.toml"), tmp_path / "p")
    s = summarize(h.project_dir)
    assert s.plan is None and s.actor == "(no plan recorded)"


def test_plans_are_checked_with_run_tool_rules(tmp_path):
    h, reads = project(tmp_path)
    too_big = call(h.project_dir, "record_decision", plan_args(reads, ram_gb=20), "llm:a")
    assert not too_big["ok"] and "RAM > budget" in too_big["error"]
    infeasible = call(h.project_dir, "record_decision", plan_args(reads, tool="assembler_A"), "a")
    assert not infeasible["ok"] and "estimated peak 32.0" in infeasible["error"]
    ok = call(h.project_dir, "record_decision", plan_args(reads), "llm:a")
    assert ok["ok"]
    d = ProjectState.load(h.project_dir).decisions[-1]
    assert d.evidence["plan_checked_by_harness"] is True


def test_plan_decisions_must_be_typed(tmp_path):
    h, _ = project(tmp_path)
    resp = call(
        h.project_dir,
        "record_decision",
        {"decision": "assembly_plan", "reason": "r", "evidence": {"tool": "x"}},
        "a",
    )
    assert not resp["ok"] and "must include the typed plan" in resp["error"]
    other = call(h.project_dir, "record_decision", {"decision": "stop", "reason": "r"}, "a")
    assert other["ok"]  # only *_plan decisions need a typed plan


def test_extract_plan_shapes():
    typed = {"plan": {"tool": "t", "params": {}, "inputs": ["a"], "cpus": 1, "ram_gb": 2}}
    assert _extract_plan(typed)["cpus"] == 1
    chosen = {
        "chosen_tool": "hifiasm",
        "chosen_params": {"bloom_bits": 0},
        "chosen_inputs": ["a"],
        "chosen_resources_x": {"cpus": 14, "ram_gb": 48.07},
    }
    assert _extract_plan(chosen) == {
        "tool": "hifiasm",
        "params": {"bloom_bits": 0},
        "inputs": ["a"],
        "cpus": 14,
        "ram_gb": 48.07,
    }
    assert _extract_plan({"nothing": 1}) is None
