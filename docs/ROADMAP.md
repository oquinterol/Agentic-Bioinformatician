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
| 10 | a: read-kind coherence · b: background jobs, project lock, reservations · c: observations + upward-only correction · d: cgroup enforcement | done |
| 11 | Benchmark harness: same scenario × N models (deterministic baseline vs Pi models) | next |

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

## Real-data decision benchmark: Colombian creole potato (2026-09-29)

Data: two PacBio HiFi cells documented as *S. tuberosum* Group Phureja (m64140: 19.3 Gbp; m84100: 28.2 Gbp).
Machine: 16 threads (Xeon E5520), 48.4 GB RAM budget. The assembler was blocked, so each backend had to record an `assembly_plan`.

**Finding 1: the metadata was wrong.** The user suspected one cell was quinoa. Mapping 2,000 reads per cell
to the existing Phureja draft gave: m84100 1,998 of 2,000 mapped, median identity 0.981; m64140 0 of 2,000 well aligned.
The GC content is 35.4 % for m84100 and 36.7 % for m64140. This led to `read_origin_check` and `Dataset.species`.

| Round | Deterministic | gpt-6-sol | gpt-5.5 |
|---|---|---|---|
| 1 (no reference registered) | both cells, `-f0`, 33.4 GB (inferred bases) | m84100 only, `-f34 --primary`, 30.7 GB (for read length and headroom, *not* species) | both cells, `-f0 --primary`, 48.07 GB |
| 2 (reference + origin tool) | origin check (rule) → m84100, default mode, 32.1 GB (inferred) | **profiled + origin check on its own** → m84100, `-f0 --primary`, 28.7 GB | profiled, **skipped the origin check** ("not run in this decision-only pass") → both cells again |

**Finding 2: profiling changes feasibility.** The file-size heuristic inferred 32.9 Gbp, while seqkit measured 47.6 Gbp.
Plans based on inferred bases under-reserve memory, which under kernel enforcement means a late OOM kill.

**Finding 3: a model that has the right tool can still decide not to use it.** Safety-critical checks must not depend on
model judgement. Proposed next harness change: reject assembler inputs whose verdict is "does not match reference", and
(policy, on by default) require an origin verdict for reads when a reference is registered.

The final assembly was chosen by the user: m84100, hifiasm default mode, 14 threads, 48 GB reserved and
kernel-enforced. It runs in `bench/phureja_assembly`, with a memory curve logged every minute for estimator calibration.

**Round 3 (after the origin safeguard, commit e1a0844)** used gpt-5.5 only, with the same prompt, data and machine snapshot as round 2.
gpt-5.5 profiled the reads, **ran read_origin_check on its own**, excluded m64140 and recorded the same plan as gpt-6-sol:
m84100, `-f0 --primary`, 28.74 GB. That plan passed the harness plan check. The safeguard was *not* triggered, because nothing
was rejected. The visible change in its environment was `requires_verified_origin: true` in `list_tools`, which likely guided it.
This is one run, so model variance cannot be excluded. Either way, the harness now guarantees the outcome
whatever the model decides. (seqkit took 1,109 s instead of 657 s because it shared the CPU with the running assembly.)
