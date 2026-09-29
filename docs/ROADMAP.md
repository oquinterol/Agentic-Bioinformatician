# Roadmap

Each phase ends with passing tests. No phase assembles a real genome before Phase 9.

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Recon: environment, Pi 0.84.4 API reviewed | done |
| 1 | `ARCHITECTURE.md`, `ROADMAP.md` | done |
| 2 | `pyproject.toml`, package layout, CLI skeleton, pytest/ruff/mypy | done |
| 3 | Resource inspector (`genome-agent inspect [--json]`) + policy | done |
| 4 | State models + `init` / `status` | minimal (models + persistence) |
| 5 | Tool registry: adapter protocol, mock assemblers, `seqkit stats` adapter, feasibility (thread fallback) | done |
| 6 | Safe executor: `JobSpec` validation, timeout, capture, provenance | next |
| 7 | Deterministic planner + `genome-agent simulate scenario.yaml` (select / reject / recover) | |
| 8 | `genome-agent tool <name> --json` protocol + Pi extension bridge (`integrations/pi/`) | |
| 9 | First real workflow on a tiny dataset (seqkit → hifiasm on toy reads) | |
| 10 | cgroup enforcement, observed peak RAM, estimator calibration | |
| 11 | Benchmark harness: same scenario × N models | |

## Phase 7 acceptance scenarios

1. The machine has 8 GB of RAM. `assembler_A` needs 32 GB and `assembler_B` needs 6 GB. → The agent chooses B and records a decision that rejects A.
2. Every candidate exceeds the budget. → The agent stops gracefully and gives an explanation.
3. The chosen job fails with a non-zero exit. → The failure is recorded and the agent replans using an alternative.
