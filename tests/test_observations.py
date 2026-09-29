import json

from genome_agent import cli
from genome_agent.agent.simulation import Scenario, build_project
from genome_agent.executor.validation import JobRequest
from genome_agent.provenance.log import ProvenanceLog
from genome_agent.resources.models import ResourceBudget, ResourceEstimate
from genome_agent.resources.observations import ObservationStore, ResourceObservation
from genome_agent.state.models import JobStatus
from genome_agent.tools.adapters.mock import MockAssembler, MockAssemblerParams
from genome_agent.tools.feasibility import assess
from genome_agent.tools.registry import ToolInputs

EST = ResourceEstimate(cpus=2, ram_gb=4.0, basis="model")


def obs(tool="t", params=None, est=4.0, peak=2.0, **kw) -> ResourceObservation:
    base = dict(
        tool=tool,
        params=params or {},
        cpus=2,
        input_bytes=10**9,
        estimated_ram_gb=est,
        peak_rss_gb=peak,
        wall_time_s=1.0,
        status="succeeded",
        project="p",
        job_id="0001-t",
    )
    return ResourceObservation(**(base | kw))


def test_no_history_means_no_correction(tmp_path):
    assert ObservationStore(tmp_path / "o.jsonl").correct(EST, "t", {}) == EST


def test_overestimates_never_lower_the_estimate(tmp_path):
    store = ObservationStore(tmp_path / "o.jsonl")
    store.append(obs(peak=0.1))  # 40x overestimate on a small run
    assert store.correct(EST, "t", {}).ram_gb == 4.0


def test_underestimate_raises_by_worst_ratio_for_same_params_only(tmp_path):
    store = ObservationStore(tmp_path / "o.jsonl")
    store.append(obs(peak=5.0))  # 1.25x
    store.append(obs(peak=6.0))  # 1.5x  <- worst
    store.append(obs(params={"x": 1}, peak=40.0))  # other params: ignored
    corrected = store.correct(EST, "t", {})
    assert corrected.ram_gb == 6.0
    assert "x1.50 upward correction" in corrected.basis and "2 observations" in corrected.basis
    assert store.correct(EST, "other_tool", {}).ram_gb == 4.0


def test_assess_uses_correction_and_can_flip_to_not_fitting(tmp_path):
    store = ObservationStore(tmp_path / "o.jsonl")
    tool = MockAssembler("asm", base_ram_gb=4)
    inputs = ToolInputs(files=(), input_bytes=0, genome_size_bp=10**6)
    budget = ResourceBudget(cpu_threads=4, ram_gb=6.0, disk_gb=100)
    assert assess(tool, inputs, MockAssemblerParams(), budget, store).fits
    store.append(obs(tool="asm", est=4.0, peak=8.0))  # it really needed 2x
    a = assess(tool, inputs, MockAssemblerParams(), budget, store)
    assert not a.fits and "8.0 GB RAM > budget 6.0" in a.reasons[0]


def test_assess_with_no_free_threads(tmp_path):
    tool = MockAssembler("asm")
    inputs = ToolInputs(files=(), input_bytes=0, genome_size_bp=10**6)
    a = assess(
        tool, inputs, MockAssemblerParams(), ResourceBudget(cpu_threads=0, ram_gb=9, disk_gb=9)
    )
    assert not a.fits and "reserved by running jobs" in a.reasons[0]


def test_harness_records_observations_and_flags_underestimates(tmp_path):
    sc = Scenario.model_validate(
        {
            "name": "o",
            "expect": "achieved",
            "machine": {"cpu_threads": 4, "ram_available_gb": 8, "disk_free_gb": 10},
            "context": {"genome_size_bp": 10**6},
            "datasets": [{"name": "hifi", "kind": "pacbio_hifi"}],
            "tools": [{"name": "tiny", "base_ram_gb": 0.05, "alloc_mb": 200}],  # really ~0.2 GB
        }
    )
    h = build_project(sc, tmp_path / "p")
    job = h.run_tool(
        JobRequest(
            tool="tiny",
            inputs=[h.state.datasets[0].path],
            genome_size_bp=10**6,
            cpus=1,
            ram_gb=0.05,
            reason="x",
        )
    )
    assert job.status == JobStatus.SUCCEEDED
    [o] = ObservationStore().load()
    assert o.tool == "tiny" and o.peak_rss_gb > o.estimated_ram_gb and o.job_id == job.id
    events = [e["event"] for e in ProvenanceLog(h.project_dir).read()]
    assert "estimate_exceeded" in events
    # The next request with the same too-small RAM is now rejected by the corrected estimate.
    again = h.run_tool(
        JobRequest(
            tool="tiny",
            inputs=[h.state.datasets[0].path],
            genome_size_bp=10**6,
            cpus=1,
            ram_gb=0.05,
            reason="x",
        )
    )
    assert again.status == JobStatus.REJECTED
    assert any("upward correction" in r for r in again.rejection_reasons)


def test_cli_calibration_report(capsys):
    store = ObservationStore()
    store.append(obs(tool="hifiasm", params={"bloom_bits": 0}, est=0.51, peak=0.1))
    store.append(obs(tool="hifiasm", params={"bloom_bits": 37}, est=16.51, peak=17.0))
    assert cli.main(["calibration"]) == 0
    out = capsys.readouterr().out
    assert "UNDERESTIMATED" in out and '"bloom_bits": 37' in out
    cli.main(["calibration", "--json"])
    groups = json.loads(capsys.readouterr().out)["groups"]
    assert {g["underestimates"] for g in groups} == {0, 1}
