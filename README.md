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

## No sandbox — by design

Neither Pi nor GenomeAgent runs inside a sandbox. That is intentional: the
agent must work with sequencing data **already on this machine** (sequencer
output directories, shared storage, existing references) without copying
hundreds of GB into a container.

What protects you instead is the harness:

- **The LLM gets no shell.** Pi runs with `--no-builtin-tools`; the model only
  sees typed GenomeAgent tools, and tool parameters reject unknown fields.
- **Inputs are read anywhere, only read.** Any readable file can be an input;
  adapters never write to input paths.
- **Outputs only go to the project.** Every job writes to `<project>/runs/<job_id>/`,
  a directory chosen by the harness, not the model.
- **Resource validation before launch.** CPU/RAM/disk requests are checked
  against the policy budget; limits are advisory (not OS-enforced) until cgroup
  support lands.
- **Everything is recorded** in `.genome-agent/state.json` and the append-only
  `.genome-agent/provenance.jsonl`, with a `command.sh` per job.

Consequence: whatever the harness allows runs with **your user's permissions**.
Run GenomeAgent as a user that has read access to the data it needs and write
access only where you are happy for results to land. If you need hard
isolation (untrusted data, shared servers), run the whole thing in a container
or VM and mount the data read-only.
