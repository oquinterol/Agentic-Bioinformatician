from pathlib import Path

from genome_agent import cli
from genome_agent.agent.simulation import Scenario, build_project
from genome_agent.benchmark import summarize
from genome_agent.bridge import call

EXAMPLES = Path(__file__).parent.parent / "examples"


def test_summarize_llm_style_plan(tmp_path, capsys):
    h = build_project(Scenario.load(EXAMPLES / "simple_assembly.toml"), tmp_path / "llm")
    call(h.project_dir, "assess_tool", {"tool": "assembler_B"}, "llm:x/y")
    call(h.project_dir, "assess_tool", {"tool": "ghost"}, "llm:x/y")  # an error
    plan = {"tool": "assembler_B", "params": {}, "inputs": ["r.fq"], "cpus": 6, "ram_gb": 6.2}
    call(
        h.project_dir,
        "record_decision",
        {"decision": "assembly_plan", "reason": "fits", "evidence": plan | {"risks": ["ram"]}},
        "llm:x/y",
    )
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


def test_typed_plan_and_chosen_keys(tmp_path):
    h = build_project(Scenario.load(EXAMPLES / "simple_assembly.toml"), tmp_path / "p")
    plan = {"tool": "assembler_B", "inputs": ["r.fq"], "cpus": 2, "ram_gb": 6.0}
    ok = call(
        h.project_dir,
        "record_decision",
        {"decision": "assembly_plan", "reason": "r", "plan": plan},
        "llm:a",
    )
    assert ok["ok"] and summarize(h.project_dir).plan["cpus"] == 2
    bad = call(
        h.project_dir,
        "record_decision",
        {"decision": "assembly_plan", "reason": "r", "plan": {"tool": "x"}},
        "llm:a",
    )
    assert not bad["ok"]  # typed: incomplete plans are rejected
    call(
        h.project_dir,
        "record_decision",
        {
            "decision": "assembly_plan",
            "reason": "r",
            "evidence": {
                "chosen_tool": "hifiasm",
                "chosen_params": {"bloom_bits": 0},
                "chosen_inputs": ["a"],
                "chosen_resources_x": {"cpus": 14, "ram_gb": 48.07},
            },
        },
        "llm:b",
    )
    s = summarize(h.project_dir)
    assert s.plan == {
        "tool": "hifiasm",
        "params": {"bloom_bits": 0},
        "inputs": ["a"],
        "cpus": 14,
        "ram_gb": 48.07,
    }
