from datetime import UTC, datetime

from genome_agent.agent.simulation import Scenario, build_project
from genome_agent.executor.validation import JobRequest
from genome_agent.harness import Harness
from genome_agent.resources.ledger import MachineLedger, Reservation, live_capacity
from genome_agent.resources.models import ResourcePolicy
from genome_agent.state.models import JobStatus


def fixed(threads, ram):
    return lambda policy: (threads, ram)


def project(tmp_path, name, ledger):
    sc = Scenario.model_validate(
        {
            "name": name,
            "expect": "achieved",
            # each project alone would see 48 GB and 14 threads
            "machine": {"cpu_threads": 16, "ram_available_gb": 60, "disk_free_gb": 100},
            "context": {"genome_size_bp": 10**6},
            "datasets": [{"name": "hifi", "kind": "pacbio_hifi"}],
            "tools": [
                {"name": "assembly_like", "base_ram_gb": 6, "sleep_s": 30},
                {"name": "profile_like", "base_ram_gb": 6},
            ],
        }
    )
    h = build_project(sc, tmp_path / name)
    return Harness(h.project_dir, h.registry, ledger=ledger)


def req(h, tool, **kw):
    base = {
        "tool": tool,
        "inputs": [h.state.datasets[0].path],
        "genome_size_bp": 10**6,
        "cpus": 2,
        "ram_gb": 6,
        "reason": "t",
    }
    return JobRequest(**(base | kw))


def test_projects_share_the_machine(tmp_path):
    ledger = MachineLedger(tmp_path / "reservations.json", capacity=fixed(8, 10.0))
    a = project(tmp_path, "assembly", ledger)
    b = project(tmp_path, "profiling", ledger)

    running = a.run_tool(req(a, "assembly_like"), wait_s=0)
    assert running.status == JobStatus.RUNNING
    budget, holders = b.available_budget()
    assert budget.ram_gb == 4.0 and budget.cpu_threads == 6  # machine minus A's job
    assert any(h.endswith(f"::{running.id}") for h in holders)

    blocked = b.run_tool(req(b, "profile_like"))
    assert blocked.status == JobStatus.REJECTED
    assert any("RAM > budget 4.0" in r for r in blocked.rejection_reasons)
    assert any("assembly" in r for r in blocked.rejection_reasons)  # says who holds it

    a.cancel_job(running.id)
    assert ledger.live() == []  # released
    assert b.run_tool(req(b, "profile_like")).status == JobStatus.SUCCEEDED


def test_stale_reservations_are_garbage_collected(tmp_path):
    ledger = MachineLedger(tmp_path / "reservations.json", capacity=fixed(8, 10.0))
    ledger.reserve(
        Reservation(
            key="/gone::0001-x",
            project="/gone",
            job_id="0001-x",
            outdir="/gone/runs/0001-x",
            runner_pid=2**22 - 1,
            cpus=4,
            ram_gb=9,
            started=datetime.now(UTC),
        )
    )
    budget, holders = ledger.free(ResourcePolicy())
    assert holders == [] and budget.ram_gb == 10.0


def test_finished_jobs_release_their_reservation(tmp_path):
    ledger = MachineLedger(tmp_path / "reservations.json", capacity=fixed(8, 10.0))
    a = project(tmp_path, "a", ledger)
    assert a.run_tool(req(a, "profile_like")).status == JobStatus.SUCCEEDED
    assert ledger.live() == []


def test_live_capacity_reads_this_machine():
    threads, ram = live_capacity(ResourcePolicy())
    assert threads >= 1 and ram > 0
