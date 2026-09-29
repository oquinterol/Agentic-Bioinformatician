import json

import pytest

from genome_agent.state.models import Dataset, DecisionRecord, ProjectState, ReadKind


def test_state_roundtrip(tmp_path, small_machine):
    s = ProjectState(name="p", objective="chromosome-scale", system_resources=small_machine)
    s.datasets.append(Dataset(path="data/reads.fq.gz", kind=ReadKind.HIFI))
    s.decisions.append(DecisionRecord(decision="run_x", reason="r", alternatives_considered=["y"]))
    s.save(tmp_path)
    assert ProjectState.load(tmp_path) == s


def test_load_missing_project_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="no GenomeAgent project"):
        ProjectState.load(tmp_path)


def test_load_rejects_unknown_schema(tmp_path):
    path = ProjectState(name="p").save(tmp_path)
    data = json.loads(path.read_text()) | {"schema_version": 999}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="schema"):
        ProjectState.load(tmp_path)
