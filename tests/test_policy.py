import pytest
from pydantic import ValidationError

from genome_agent.resources.models import ResourcePolicy

from .conftest import make_resources


def test_default_policy_on_small_machine(small_machine):
    b = ResourcePolicy().apply(small_machine)
    assert (b.cpu_threads, b.ram_gb, b.disk_gb) == (6, 6.4, 32.0)


def test_policy_uses_available_not_total_ram():
    b = ResourcePolicy().apply(make_resources(ram_total_gb=64, ram_available_gb=10))
    assert b.ram_gb == 8.0


def test_policy_never_gives_zero_threads():
    assert ResourcePolicy(reserved_threads=4).apply(make_resources(cpu_threads=2)).cpu_threads == 1


@pytest.mark.parametrize(
    "bad", [{"ram_fraction": 0}, {"ram_fraction": 1.5}, {"reserved_threads": -1}]
)
def test_policy_rejects_nonsense(bad):
    with pytest.raises(ValidationError):
        ResourcePolicy(**bad)
