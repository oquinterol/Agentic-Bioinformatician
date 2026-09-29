import io
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from genome_agent import cli
from genome_agent.agent.simulation import Scenario, build_project
from genome_agent.bridge import OPERATIONS, call
from genome_agent.provenance.log import ProvenanceLog
from genome_agent.state.models import ProjectState

EXAMPLES = Path(__file__).parent.parent / "examples"
ACTOR = "llm:test/model-x"


@pytest.fixture
def project(tmp_path) -> Path:
    p = tmp_path / "proj"
    build_project(Scenario.load(EXAMPLES / "simple_assembly.toml"), p)
    return p


def ok(resp):
    assert resp["ok"], resp
    return resp["result"]


def test_list_tools_includes_mocks_in_preference_order_and_real_adapters(project):
    tools = ok(call(project, "list_tools", {}, ACTOR))["tools"]
    names = [t["name"] for t in tools]
    assert names[:2] == ["assembler_A", "assembler_B"] and "seqkit_stats" in names
    seqkit = next(t for t in tools if t["name"] == "seqkit_stats")
    assert seqkit["available"] is False  # simulated machine has no seqkit


def test_inspect_system_returns_authoritative_budget(project):
    r = ok(call(project, "inspect_system", {}, ACTOR))
    assert r["budget"] == {"cpu_threads": 6, "ram_gb": 6.4, "disk_gb": 32.0}


def test_assess_uses_project_defaults(project):
    a = ok(call(project, "assess_tool", {"tool": "assembler_A"}, ACTOR))
    b = ok(call(project, "assess_tool", {"tool": "assembler_B"}, ACTOR))
    assert not a["fits"] and "32.0 GB RAM" in a["reasons"][0]
    assert b["fits"] and b["estimate"]["ram_gb"] == 6.0


def test_run_tool_records_actor_from_caller_not_model(project):
    args = {
        "tool": "assembler_B",
        "cpus": 6,
        "ram_gb": 6,
        "reason": "A does not fit",
        "alternatives_considered": ["assembler_A"],
    }
    r = ok(call(project, "run_tool", args, ACTOR))
    assert r["job"]["status"] == "succeeded" and r["results"][0]["kind"] == "contigs_fasta"
    decision = ProjectState.load(project).decisions[-1]
    assert decision.actor == ACTOR and decision.alternatives_considered == ["assembler_A"]


def test_model_cannot_spoof_actor(project):
    args = {"tool": "assembler_B", "cpus": 1, "ram_gb": 6, "reason": "x", "actor": "harness"}
    resp = call(project, "run_tool", args, ACTOR)
    assert not resp["ok"] and "actor" in resp["error"]


def test_rejected_run_is_ok_response_with_rejected_job(project):
    args = {"tool": "assembler_A", "cpus": 2, "ram_gb": 6, "reason": "try"}
    job = ok(call(project, "run_tool", args, ACTOR))["job"]
    assert job["status"] == "rejected" and job["rejection_reasons"]


def test_record_decision_and_status(project):
    ok(call(project, "record_decision", {"decision": "stop", "reason": "done"}, ACTOR))
    status = ok(call(project, "project_status", {}, ACTOR))
    assert status["decisions"][-1] == {"decision": "stop", "actor": ACTOR, "reason": "done"}
    assert ProvenanceLog(project).read()[-1]["actor"] == ACTOR


def test_add_dataset_accepts_files_anywhere(project, tmp_path):
    reads = tmp_path / "elsewhere" / "ont.fq"
    reads.parent.mkdir()
    reads.write_text("@r\nA\n+\nI\n")
    r = ok(call(project, "add_dataset", {"path": str(reads), "kind": "ont"}, ACTOR))
    assert str(reads) in [d["path"] for d in r["datasets"]]
    again = call(project, "add_dataset", {"path": str(reads), "kind": "ont"}, ACTOR)
    assert not again["ok"] and "already registered" in again["error"]


@pytest.mark.parametrize(
    ("op", "args", "fragment"),
    [
        ("nope", {}, "unknown operation"),
        ("list_tools", {"surprise": 1}, "Extra inputs"),
        ("assess_tool", {"tool": "ghost"}, "unknown tool 'ghost'"),
        ("assess_tool", {"tool": "assembler_B", "inputs": ["/no/file"]}, "inputs not found"),
        ("assess_tool", {"tool": "assembler_B", "params": {"k": 1}}, "invalid params"),
        ("add_dataset", {"path": "/no/file", "kind": "ont"}, "not a file"),
        ("add_dataset", {"path": "/etc/hostname", "kind": "sanger"}, "kind"),
    ],
)
def test_expected_errors_are_model_facing(project, op, args, fragment):
    resp = call(project, op, args, ACTOR)
    assert not resp["ok"] and fragment in resp["error"], resp


def test_missing_project_is_an_error(tmp_path):
    resp = call(tmp_path, "project_status", {}, ACTOR)
    assert not resp["ok"] and "no GenomeAgent project" in resp["error"]


def test_cli_tool_reads_stdin_and_sets_exit_code(project, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO('{"tool": "assembler_A"}'))
    assert cli.main(["tool", "assess_tool", "--project", str(project)]) == 0
    assert json.loads(capsys.readouterr().out)["result"]["fits"] is False
    monkeypatch.setattr("sys.stdin", io.StringIO("[1, 2]"))
    assert cli.main(["tool", "assess_tool", "--project", str(project)]) == 2
    monkeypatch.setattr("sys.stdin", io.StringIO("{not json"))
    assert cli.main(["tool", "assess_tool", "--project", str(project)]) == 2


def test_cli_simulate_setup_only_runs_nothing(tmp_path):
    p = tmp_path / "p"
    assert (
        cli.main(
            [
                "simulate",
                str(EXAMPLES / "simple_assembly.toml"),
                "--project",
                str(p),
                "--setup-only",
            ]
        )
        == 0
    )
    assert ProjectState.load(p).jobs == []


def test_every_operation_is_exposed_by_the_pi_extension():
    ext = (Path(__file__).parent.parent / "integrations/pi/genome-agent.ts").read_text()
    for op in OPERATIONS:
        assert f'"{op}"' in ext, f"operation {op} missing from Pi extension"


@pytest.mark.skipif(shutil.which("pi") is None, reason="Pi not installed")
def test_pi_selftest_loads_bridge_without_prompting_a_model(project):
    launcher = Path(__file__).parent.parent / "integrations/pi/genome-pi"
    proc = subprocess.run(
        [str(launcher), "--project", str(project), "--selftest"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    # In RPC mode Pi keeps stdout for the protocol and routes extension output to stderr.
    out = proc.stdout + proc.stderr
    line = next((ln for ln in out.splitlines() if ln.startswith("GENOME_AGENT_SELFTEST ")), None)
    assert line, f"no selftest report\nstdout={proc.stdout[-2000:]}\nstderr={proc.stderr[-2000:]}"
    report = json.loads(line.split(" ", 1)[1])
    assert report["only_genome_agent_tools"] and report["builtin_tools_absent"]
    assert sorted(report["active_tools"]) == sorted(OPERATIONS)
    assert report["harness_call"]["result"]["budget"]["ram_gb"] == 6.4
