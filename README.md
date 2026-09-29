# GenomeAgent

A resource-aware autonomous agent for genome assembly. Given reads, biological
context and an objective, it looks for the best scientifically defensible
strategy that fits **this** machine:
OBSERVE → PLAN → ESTIMATE → EXECUTE → EVALUATE → REPLAN.

- **Brain:** [Pi Coding Agent](https://github.com/earendil-works/pi), run with built-in tools
  disabled and given typed GenomeAgent tools only (`integrations/pi/`).
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
.venv/bin/genome-agent simulate examples/simple_assembly.toml   # mock scenario, no genome needed
.venv/bin/pytest && .venv/bin/ruff check src tests && .venv/bin/mypy src
```

The MVP is Linux-only (it reads `/proc`).

## A real assembly on toy data

The tools used here (hifiasm, seqkit) must be installed. Everything finishes in seconds.

```bash
GA=.venv/bin/genome-agent
$GA toy-data /tmp/seq                       # 200 kb random genome + 30x HiFi-like reads
$GA init /tmp/toy --objective "Best contig assembly"
$GA add-dataset /tmp/toy /tmp/seq/reads.fq --kind pacbio_hifi   # data stays where it is
$GA plan /tmp/toy                           # deterministic planner -> real hifiasm
# Or constrain the machine and watch it switch to hifiasm -f0 (no 16 GiB bloom filter):
$GA init /tmp/toy8 --ram-fraction 0.1 && $GA add-dataset /tmp/toy8 /tmp/seq/reads.fq --kind pacbio_hifi
$GA plan /tmp/toy8
```

Each job records its estimated RAM and its **observed peak RSS**, so estimates can be checked against reality.

## Sample identity: with or without a reference

Mislabelled or contaminated reads silently ruin assemblies (this happened with real data here). Assemblers
(`requires_verified_origin`) are therefore gated by the harness, not by the model's judgement:

| Situation | Evidence the harness accepts | Tool |
|---|---|---|
| A reference or draft of the species is registered (`--kind reference_fasta`) | reads verified as "matches reference" | `read_origin_check` (minimap2 on a read sample) |
| De novo, no reference, 2+ libraries pooled | every pair "consistent" | `library_consistency_check` (cross-maps ~1x of each library, normalised by a self baseline) |
| Any case | genome size, heterozygosity and error rate per library, to compare with the declared organism | `kmer_profile` (jellyfish + GenomeScope2) |

Reads that fail a check are never assembled. When two libraries disagree and there is no reference, the
harness cannot tell which one is the declared species. The deterministic planner then stops and asks for
evidence instead of guessing. `ProjectState.require_origin_check` (default on) can be switched off per project.
Extra tools: `jellyfish` (BioArchLinux) and GenomeScope2 (R user library, `~/.local/bin/genomescope.R`).

## Running with Pi (LLM brain)

Pi uses whatever authentication you already configured (subscriptions or API
keys), so switching models is just a Pi flag.

```bash
# 1. A project: a real one (genome-agent init) or a simulated scenario
.venv/bin/genome-agent simulate examples/simple_assembly.toml --project /tmp/sim1 --setup-only

# 2. Check the bridge. This spends no tokens and never prompts a model.
integrations/pi/genome-pi --project /tmp/sim1 --selftest

# 3. Let a model do the planning (interactive, or one-shot with -p)
integrations/pi/genome-pi --project /tmp/sim1 --model <provider/model> \
  -p "Produce the best contig assembly this machine allows."

# 4. Inspect what it decided and compare with the deterministic baseline
.venv/bin/genome-agent status /tmp/sim1
cat /tmp/sim1/.genome-agent/provenance.jsonl
```

For scripted or background runs, close stdin with `</dev/null`. When stdin is
not a terminal, `pi -p` reads it as extra prompt input and waits until it closes.

`genome-pi` starts Pi with `--no-builtin-tools --no-extensions --no-skills
--no-context-files`, loads `integrations/pi/genome-agent.ts`, and appends
`integrations/pi/SYSTEM.md`. The model sees exactly eleven harness operations: `inspect_system`,
`list_tools`, `project_status`, `assess_tool`, `run_tool`, `record_decision`,
`add_dataset`, `read_job_log`, `job_status`, `wait_job` and `cancel_job`. Bioinformatics tools
(hifiasm, seqkit, read_origin_check, library_consistency_check, kmer_profile) are run through `run_tool`. Each tool forwards its arguments to `genome-agent tool <op>`, and
every decision is recorded with `actor = llm:<provider>/<model>`. Every call,
read-only calls included, is logged as a `bridge_call` event in `provenance.jsonl`.

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
- **Resource validation before launch, and kernel enforcement.** CPU, RAM and disk requests are
  checked against the budget left by running jobs. Where systemd user scopes work (`genome-agent
  inspect` shows `Limits: systemd-user-scope`), each job runs under `MemoryMax`, `MemorySwapMax=0`
  and `CPUQuota` equal to its reservation. A job that exceeds its reservation is killed and recorded
  as OOM, and later estimates for it are raised. Without systemd user scopes the limits are advisory,
  and `command.sh` says so.
- **Everything is recorded** in `.genome-agent/state.json` and the append-only
  `.genome-agent/provenance.jsonl`, with a `command.sh` per job.

Consequence: whatever the harness allows runs with **your user's permissions**.
Run GenomeAgent as a user that has read access to the data it needs and write
access only where you are happy for results to land. If you need hard
isolation (untrusted data, shared servers), run the whole thing in a container
or VM and mount the data read-only.
