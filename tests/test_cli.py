import json

import pytest

from genome_agent import cli
from genome_agent.state.models import ProjectState


@pytest.fixture(autouse=True)
def fake_inspect(monkeypatch, small_machine):
    monkeypatch.setattr(cli, "inspect_system", lambda ws=None: small_machine)


def test_inspect_text(capsys):
    assert cli.main(["inspect"]) == 0
    out = capsys.readouterr().out
    assert "6 usable under policy" in out and "6.4 GB usable" in out


def test_inspect_json(capsys):
    cli.main(["inspect", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["budget"] == {"cpu_threads": 6, "ram_gb": 6.4, "disk_gb": 32.0}


def test_init_then_status(tmp_path, capsys):
    proj = tmp_path / "example_project"
    assert cli.main(["init", str(proj), "--objective", "best assembly"]) == 0
    assert (proj / "runs").is_dir()
    state = ProjectState.load(proj)
    assert state.decisions[0].decision == "init_project"
    assert cli.main(["status", str(proj)]) == 0
    assert "best assembly" in capsys.readouterr().out


def test_init_refuses_to_overwrite(tmp_path):
    cli.main(["init", str(tmp_path)])
    assert cli.main(["init", str(tmp_path)]) == 1


def test_status_outside_project_fails(tmp_path):
    assert cli.main(["status", str(tmp_path)]) == 1
