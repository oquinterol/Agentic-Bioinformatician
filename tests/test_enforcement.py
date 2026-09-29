import subprocess
from pathlib import Path

import pytest

from genome_agent.agent.simulation import Scenario, build_project
from genome_agent.executor.runner import JobSpec, enforced_argv
from genome_agent.executor.validation import JobRequest
from genome_agent.resources.inspector import SYSTEMD_PROBE, detect_enforcement
from genome_agent.resources.models import SYSTEMD_SCOPE, ResourcePolicy
from genome_agent.resources.observations import ObservationStore
from genome_agent.state.lock import project_lock
from genome_agent.state.models import JobStatus, ProjectState
from genome_agent.tools.adapters.mock import MockAssemblerParams
from genome_agent.tools.feasibility import assess
from genome_agent.tools.registry import ToolInputs

from .conftest import make_resources


def _systemd_scopes_work() -> bool:
    try:
        return subprocess.run(SYSTEMD_PROBE, capture_output=True, timeout=15).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


needs_systemd = pytest.mark.skipif(not _systemd_scopes_work(), reason="no systemd user scopes")


def test_detect_enforcement_by_probing():
    assert detect_enforcement(which=lambda n: None) == ([], [])
    ok = detect_enforcement(which=lambda n: "/bin/systemd-run", run=lambda a, t: (0, ""))
    assert ok == ([SYSTEMD_SCOPE], [])
    backends, notes = detect_enforcement(
        which=lambda n: "/bin/systemd-run", run=lambda a, t: (1, "Failed to connect to bus")
    )
    assert backends == [] and "advisory" in notes[0] and "Failed to connect" in notes[0]


def test_policy_backend_selection():
    with_scope = make_resources(enforcement=[SYSTEMD_SCOPE])
    without = make_resources()
    assert ResourcePolicy().enforcement_backend(with_scope) == SYSTEMD_SCOPE
    assert ResourcePolicy().enforcement_backend(without) is None  # auto degrades
    assert ResourcePolicy(enforcement="none").enforcement_backend(with_scope) is None
    with pytest.raises(ValueError, match="not available"):
        ResourcePolicy(enforcement="systemd").enforcement_backend(without)


def test_enforced_argv():
    spec = JobSpec(
        argv=["hifiasm", "-t", "4"], cpus=4, ram_gb=2.5, timeout_s=10, enforcement=SYSTEMD_SCOPE
    )
    argv = enforced_argv(spec)
    assert argv[-4:] == ["--", "hifiasm", "-t", "4"]
    assert "MemoryMax=2560M" in argv and "MemorySwapMax=0" in argv and "CPUQuota=400%" in argv
    assert enforced_argv(spec.model_copy(update={"enforcement": None})) == spec.argv


def project(tmp_path, alloc_mb, enforcement=None, policy_enforcement="auto"):
    sc = Scenario.model_validate(
        {
            "name": "enf",
            "expect": "achieved",
            "machine": {"cpu_threads": 4, "ram_available_gb": 8, "disk_free_gb": 10},
            "context": {"genome_size_bp": 10**6},
            "datasets": [{"name": "hifi", "kind": "pacbio_hifi"}],
            "tools": [{"name": "hog", "base_ram_gb": 0.1, "alloc_mb": alloc_mb}],
        }
    )
    h = build_project(sc, tmp_path / "p")
    with project_lock(h.project_dir):
        state = ProjectState.load(h.project_dir)
        state.system_resources.enforcement = enforcement or []
        state.policy.enforcement = policy_enforcement
        state.save(h.project_dir)
    h.state = ProjectState.load(h.project_dir)
    return h


def run(h, ram_gb=0.2):
    return h.run_tool(
        JobRequest(
            tool="hog",
            inputs=[h.state.datasets[0].path],
            genome_size_bp=10**6,
            cpus=1,
            ram_gb=ram_gb,
            reason="x",
        )
    )


def test_required_systemd_unavailable_rejects(tmp_path):
    h = project(tmp_path, alloc_mb=0, policy_enforcement="systemd")
    job = run(h)
    assert job.status == JobStatus.REJECTED and "not available" in job.rejection_reasons[0]


def test_advisory_jobs_say_so_in_command_sh(tmp_path):
    h = project(tmp_path, alloc_mb=0)
    job = run(h)
    assert job.status == JobStatus.SUCCEEDED and job.enforcement is None
    assert "not enforced (advisory)" in (Path(job.outdir) / "command.sh").read_text()


@needs_systemd
def test_job_within_limit_succeeds_under_enforcement(tmp_path):
    h = project(tmp_path, alloc_mb=50, enforcement=[SYSTEMD_SCOPE])
    job = run(h, ram_gb=0.2)
    assert job.status == JobStatus.SUCCEEDED, job.error
    assert job.enforcement == SYSTEMD_SCOPE and 0.04 < job.peak_rss_gb < 0.2
    assert "MemoryMax=204M" in (Path(job.outdir) / "command.sh").read_text()


@needs_systemd
def test_kernel_kills_job_over_limit_and_next_estimate_rises(tmp_path):
    h = project(tmp_path, alloc_mb=600, enforcement=[SYSTEMD_SCOPE])
    job = run(h, ram_gb=0.2)  # estimate 0.1 GB; the process really wants ~0.6 GB
    assert job.status == JobStatus.FAILED and job.oom_killed
    assert "enforced memory limit" in job.error
    assert job.peak_rss_gb <= 0.25  # the kernel stopped it at the limit

    [o] = ObservationStore().load()
    assert o.oom_killed and o.requested_ram_gb == 0.2
    tool = h.registry.get("hog")
    inputs = ToolInputs(files=(), input_bytes=0, genome_size_bp=10**6)
    a = assess(tool, inputs, MockAssemblerParams(), h.available_budget()[0], ObservationStore())
    assert a.estimate.ram_gb >= 0.3  # at least 1.5x the limit that failed
