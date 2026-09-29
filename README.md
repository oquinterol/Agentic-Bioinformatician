# GenomeAgent

A resource-aware autonomous agent for genome assembly. Given reads, biological
context and an objective, it looks for the best scientifically defensible
strategy that fits **this** machine:
OBSERVE → PLAN → ESTIMATE → EXECUTE → EVALUATE → REPLAN.

- **Brain:** [Pi Coding Agent](https://github.com/earendil-works/pi-mono), run with built-in tools
  disabled and given typed GenomeAgent tools only (Phase 8).
- **Harness:** this Python package. It is the authority on resources, policy,
  state, validation and provenance.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/ROADMAP.md](docs/ROADMAP.md).

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/genome-agent inspect            # machine inventory + policy budget
.venv/bin/genome-agent inspect --json     # same, machine-readable
.venv/bin/genome-agent init example_project --objective "best chromosome-scale assembly"
.venv/bin/genome-agent status example_project
.venv/bin/pytest && .venv/bin/ruff check src tests && .venv/bin/mypy src
```

The MVP is Linux-only (it reads `/proc`).
