# GenomeAgent — Architecture

GenomeAgent is a resource-aware scientific agent for genome assembly. It plans,
executes, evaluates and replans analyses **within the limits of the machine it
actually runs on**.

## 1. Layers

```
┌──────────────────────────────┐
│ Brain  (Pi Coding Agent)     │  LLM reasoning, conversation, UI, sessions
│  - no built-in bash/edit     │  (pi --no-builtin-tools)
│  - typed tools only          │
└──────────────┬───────────────┘
               │ JSON over subprocess:  genome-agent tool <op>
┌──────────────▼───────────────┐
│ Harness  (Python, this repo) │  THE AUTHORITY: resources, policy, state,
│  resources/  state/  tools/  │  validation, provenance, permissions
│  provenance/ executor/       │
└──────────────┬───────────────┘
               │ validated JobSpec only
┌──────────────▼───────────────┐
│ Executor                     │  local subprocess (MVP) → systemd-run/cgroups,
│                              │  containers, SLURM later
└──────────────────────────────┘
```

### Why Pi as the Brain

Verified against Pi 0.84.4 and re-verified against 0.87.1 (`dist/core/extensions/types.d.ts`,
`examples/extensions/`):

| Pi capability | How we use it |
|---|---|
| `pi.registerTool()` with TypeBox schemas | Each harness operation becomes a typed LLM tool |
| `--no-builtin-tools` / `--tools <list>` | Remove `bash`, `write`, `edit`; the LLM never gets a shell |
| `pi.on("tool_call")` can `block` | Defence in depth; the real check is still in Python |
| `pi.appendEntry()` / sessions | Conversation history only, **not** scientific state |
| `--mode json` / `--mode rpc`, multi-provider models | Headless runs and benchmarking across models |

Conflicts, and how we handle them:

- **Pi has no sandbox** (`docs/security.md`). Extensions run with full user
  permissions. The TypeScript extension must stay a thin, logic-free bridge.
  Every validation lives in Python, where pytest covers it.
- **Pi is TypeScript, and the harness is Python.** The only contract between them
  is the CLI JSON protocol (`genome-agent tool <op>`). Nothing else is shared, so Pi can
  be swapped for another backend (CaveAgent, a raw API loop, a local model) by
  reimplementing only the bridge.
- **Session history is not state.** Pi's session JSONL is useful for audits. The
  authoritative record is always `.genome-agent/state.json`.

The Python `AgentBackend` protocol (`agent/`) exists so that deterministic
planners, used in tests and simulations, and LLM backends are interchangeable.

### Bridge protocol (`bridge.py`, `integrations/pi/`)

```
Pi tool call ──► genome-agent tool <op> --project DIR --actor llm:<provider>/<model>
                 stdin:  JSON arguments (strict Pydantic model, extra keys rejected)
                 stdout: {"ok": true, "result": …} | {"ok": false, "error": "…"}
                 exit:   0 ok, 2 expected error, other = bug (traceback on stderr)
```

- The **caller** sets `--actor`, so the model cannot claim that a decision came from someone else.
- `ok: false` is thrown as a failed tool result in Pi, so the model sees the error text.
- A rejected job is still `ok: true`, with `job.status = "rejected"` and the reasons. A rejection is information for the model, not a protocol error.
- `simulate --setup-only` builds a scenario project, including its mock tools in
  `.genome-agent/mock_tools.json`, so an LLM and the deterministic planner can be compared on identical inputs.
- `genome-pi --selftest` loads the extension in RPC mode with no input, checks the active
  tool set, makes one harness call, and exits. It spends no tokens and also runs in pytest.

## 2. Module boundaries (`src/genome_agent/`)

| Module | Responsibility | Depends on |
|---|---|---|
| `resources/` | Machine inventory (`SystemResources`) and `ResourcePolicy` → `ResourceBudget` | stdlib |
| `state/` | Pydantic models: `ProjectState`, `Dataset`, `Job`, `Result`, `DecisionRecord`; JSON persistence | resources |
| `tools/` | Tool registry: probes, adapters (command builder, estimator, parser) | resources |
| `executor/` | Runs a validated `JobSpec`: timeout, capture, exit code, wall time | tools, state |
| `provenance/` | Append-only decision and command log | state |
| `agent/` | `AgentBackend` protocol (Observation → RunTool / Stop), `DeterministicPlanner`, `run_loop`, TOML scenarios | all above |
| `harness.py` | `Harness.run_tool` / `record_decision`: the only path to execution | all above |
| `bridge.py` | JSON tool protocol (`genome-agent tool <op>`) for Pi or any other runtime | all above |
| `cli.py` | `inspect`, `init`, `status`, `simulate`, `tool` | all above |

Rule: `resources/` and `state/` never import from `agent/`. The LLM layer sits on
top and can be removed.

## 3. Resource model

- `SystemResources` is a snapshot taken by `inspect_system()`. It is **authoritative**:
  the LLM reads it and never writes it.
- `ResourcePolicy` (configurable in `genome-agent.toml`) has these defaults:
  - RAM: 80 % of *available* RAM
  - CPU: logical threads − 2 (minimum 1)
  - Disk: 80 % of free space in the workspace
- `ResourceBudget = policy.apply(resources)` sets the hard ceiling for every job request.
- Tool adapters will expose `estimate(inputs, params) → ResourceEstimate`. The
  estimation strategy can be swapped (static rules now, then learned from job history).

## 4. Tool execution and security model

1. The LLM calls a typed tool, such as `run_tool(tool="hifiasm", params={...}, cpus=12, ram_gb=24)`.
2. The harness resolves the **adapter**. The adapter builds argv from the params.
   The LLM does not write raw command lines.
3. The validator checks that:
   - the tool is registered and detected
   - the parameters match the schema
   - input paths exist and are readable. They may be **anywhere** on the filesystem, because
     using existing local data is the main use case, and they are only ever read.
   - the output directory is always `<project>/runs/<job_id>/`, chosen by the harness
   - the CPU, RAM and disk requests are ≤ budget, and the requested RAM is ≥ the adapter's estimated peak
4. The executor runs the command with `subprocess` (no shell), a timeout, a new
   process group (a timeout kills children too) and `OMP_NUM_THREADS=cpus`. It
   captures stdout and stderr to files and records the exit code and wall time.
   A non-zero exit is recorded as a failure. Failures are never swallowed.
5. Every command, and every decision that led to it, is written to state and provenance.
   Each job also records its **observed peak RSS** (`os.wait4`) next to the estimate.

Adapters may offer `param_variants()`: parameter sets ordered from preferred to
lower-resource, such as hifiasm `{}` and then `{"bloom_bits": 0}`. The deterministic
planner tries them before rejecting a tool, and LLM backends see them in `list_tools`.

### Job lifecycle (background runners)

```
run_tool ─► validate against FREE budget (policy − RUNNING reservations)
         ─► state: RUNNING, runner_pid          (under project lock)
         ─► detached `python -m genome_agent.executor.runner runs/<id>`
                 └─ runs the tool, measures peak RSS, writes runs/<id>/execution.json
refresh  ─► on every project open/poll: execution.json → SUCCEEDED/FAILED/CANCELLED
            runner gone without a result → FAILED ("disappeared"), or CANCELLED if requested
```

- Runners never write `state.json`. Only the harness does, inside `transaction()`: flock, reload, mutate, save.
- `run_tool(wait_s=…)` returns while the job keeps running. `wait_job`, `job_status` and `cancel_job` follow it up.
- A job survives the CLI or Pi call that launched it. Cancellation (SIGTERM) is race-safe: it works even before the tool process exists.

### Machine-wide reservations

A project's budget comes from its own snapshot, so two projects could each believe they own the machine.
Every RUNNING job is therefore also written to `~/.local/share/genome-agent/reservations.json`
(`$GENOME_AGENT_DATA_DIR`, with its own flock). A new job is validated against **min(project budget
minus its running jobs, live machine capacity minus all reservations)**. Rejections name the jobs
holding the resources, including those of other projects. Validation, launch and reservation happen
under project lock → ledger lock, which is a fixed order, so it cannot deadlock. Reservations are released on
finish, loss or cancellation, and entries whose runner is gone are garbage-collected.

### Estimated vs. observed resources

Every finished job appends a `ResourceObservation` (tool, full params, estimate, peak RSS, input size)
to `~/.local/share/genome-agent/observations.jsonl` (`$GENOME_AGENT_DATA_DIR`). Estimates are **only
ever raised** from this history: if identical tool+params once exceeded their estimate, later estimates
are multiplied by the worst ratio, rounded up. They are never lowered, because small runs cannot justify
lower memory for large inputs. `genome-agent calibration` reports estimate accuracy per tool and params.

### Enforcement

`ResourcePolicy.enforcement` is `auto`, `none` or `systemd`. `inspect` verifies, by running a probe,
that `systemd-run --user --scope` can apply limits. When it can, the runner wraps each tool in a
transient scope with `MemoryMax` set to the reserved RAM, `MemorySwapMax=0` and `CPUQuota` set to
`cpus`×100 %. A SIGKILL under enforcement is recorded as `oom_killed`. Its observation counts as
needing at least 1.5× the failed limit, which is a heuristic step, so a replan requests more.
`auto` falls back to advisory limits and notes it. `systemd` refuses to run without them.

## 5. State model

```
ProjectState (.genome-agent/state.json)
├── objective, biological_context
├── datasets[]           Dataset(path, kind, platform, size, stats)
├── system_resources     SystemResources snapshot
├── policy               ResourcePolicy
├── hypotheses[], current_plan[]
├── jobs[]               Job(spec, status, exit_code, wall_time, log paths)
├── results[], metrics{}
├── decisions[]          DecisionRecord(decision, reason, evidence, resources, alternatives)
├── failures[]
└── artifacts[]
```

The state serializes to JSON and includes a schema version. Resuming a run means
loading the state again. It never means replaying the chat.

## 6. Benchmarking hooks

Every `Job` records the requested and observed resources and the wall time.
Every loop iteration is a `DecisionRecord`. Scientific metrics go into `metrics{}`
under stable keys (`n50`, `busco_complete`, `qv`, …). A benchmark is therefore
a function of `state.json`, and the same dataset and machine can be replayed
with different backends and models.
