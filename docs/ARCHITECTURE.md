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
               │ JSON over subprocess:  genome-agent tool <name> --json
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

Verified against Pi 0.84.4 (`@earendil-works/pi-coding-agent`, `docs/extensions.md`):

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
  is the CLI JSON protocol (`genome-agent tool …`). Nothing else is shared, so Pi can
  be swapped for another backend (CaveAgent, a raw API loop, a local model) by
  reimplementing only the bridge.
- **Session history is not state.** Pi's session JSONL is useful for audits. The
  authoritative record is always `.genome-agent/state.json`.

The Python `AgentBackend` protocol (`agent/`) exists so that deterministic
planners, used in tests and simulations, and LLM backends are interchangeable.

## 2. Module boundaries (`src/genome_agent/`)

| Module | Responsibility | Depends on |
|---|---|---|
| `resources/` | Machine inventory (`SystemResources`) and `ResourcePolicy` → `ResourceBudget` | stdlib |
| `state/` | Pydantic models: `ProjectState`, `Dataset`, `Job`, `Result`, `DecisionRecord`; JSON persistence | resources |
| `tools/` | Tool registry: probes, adapters (command builder, estimator, parser) | resources |
| `executor/` | Runs a validated `JobSpec`: timeout, capture, exit code, wall time | tools, state |
| `provenance/` | Append-only decision and command log | state |
| `agent/` | `AgentBackend` protocol, deterministic planner, OBSERVE→…→REPLAN loop | all above |
| `cli.py` | `inspect`, `init`, `status` (later `simulate`, `tool`) | all above |

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
   - all paths resolve inside the project workspace
   - the CPU, RAM and disk requests are ≤ budget
   - the output directory is allowed
4. The executor runs the command with `subprocess` (no shell) and a timeout. It
   captures stdout and stderr to files and records the exit code and wall time.
   A non-zero exit is recorded as a failure. Failures are never swallowed.
5. Every command, and every decision that led to it, is written to state and provenance.

Enforcement is currently **advisory** (validation before launch). The `Executor`
interface will accept `systemd-run --scope -p MemoryMax=… -p CPUQuota=…` or
container limits without API changes.

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
