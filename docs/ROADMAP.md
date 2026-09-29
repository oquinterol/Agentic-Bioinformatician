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
| 6 | Safe executor: request validation, timeout, capture, provenance (`Harness.run_tool`) | done |
| 7 | Deterministic planner, agent loop, `genome-agent simulate scenario.toml` (select / reject / recover) | done |
| 8 | `genome-agent tool <op>` JSON protocol + Pi bridge (`integrations/pi/`), token-free selftest, `read_job_log`, provenance for every call | done |
| 9 | Real workflow on toy data: hifiasm adapter (verified bloom-filter memory model), param variants, observed peak RSS, `toy-data`/`add-dataset`/`plan` | done |
| 10 | Estimator calibration from observed peak RSS, cgroup enforcement, background jobs (non-blocking `run_tool`) | next |
| 11 | Benchmark harness: same scenario × N models | |

## Phase 7 acceptance scenarios (`examples/*.toml`, all covered by tests)

1. The machine has 8 GB of RAM. `assembler_A` needs 32 GB and `assembler_B` needs 6 GB. → The agent chooses B and records a decision that rejects A.
2. Every candidate exceeds the budget. → The agent stops gracefully and gives an explanation.
3. The chosen job fails with a non-zero exit. → The failure is recorded and the agent replans using an alternative.

## Findings from live runs (gpt-6-sol via Pi, 2026-09-29)

- Mock scenarios: the model matched the deterministic planner on all three scenarios. It also flagged toy inputs
  and mock tools as "not biologically validated", and refused to lower the genome size to make an estimate fit.
- Real toy data, 6.4 GB budget: it profiled the reads (seqkit), chose hifiasm `-f0`, avoided unmeasured
  ploidy/coverage flags, validated the contigs with seqkit, and stopped. The result was 1 contig of 199,639 bp for a 200,000 bp genome.
- Real toy data, 48.6 GB budget: it **still** chose `-f0` ("needless 16 GiB bloom filter for 6 Mb of reads").
  The deterministic planner takes the first variant that fits, so it would allocate 16 GiB. Being feasible is
  not the same as being efficient, and this is a benchmark dimension to keep.
- Estimates vs. observed: hifiasm `-f37` was estimated at 16.51 GB and peaked at 16.08 GB. `-f0` was estimated
  at 0.51 GB and peaked at 0.08–0.11 GB. The per-Gbp term is still uncalibrated.
- `run_tool` blocks until the job ends. Long assemblies need background jobs plus status polling.
