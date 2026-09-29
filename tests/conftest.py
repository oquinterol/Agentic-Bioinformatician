from datetime import UTC, datetime

import pytest

from genome_agent.resources.models import SystemResources


def make_resources(**overrides) -> SystemResources:
    base = dict(
        inspected_at=datetime(2026, 1, 1, tzinfo=UTC),
        os="TestOS",
        kernel="0",
        cpu_model="Fake",
        cpu_physical_cores=4,
        cpu_threads=8,
        ram_total_gb=8.0,
        ram_available_gb=8.0,
        swap_total_gb=0,
        swap_free_gb=0,
        workspace="/w",
        disk_free_gb=40.0,
        tmp_dir="/tmp",
        tmp_free_gb=40.0,
    )
    return SystemResources(**(base | overrides))


@pytest.fixture
def small_machine() -> SystemResources:
    """The 8 CPU / 8 GB / 40 GB machine from the design scenarios."""
    return make_resources()
