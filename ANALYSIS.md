# MiniAgent — Repository Analysis

> Scope: a static, read-only analysis of the repository at the workspace root.
> Every non-obvious claim cites the file (and line, where it pins the claim) it
> came from. Claims derived from reading code rather than running it are marked
> *(static)*. Anything not verified is marked *(unverified)*.

## 1. Overview — what this repository does

MiniAgent is a **lightweight, terminal-first coding agent**: a CLI you run inside a
repository that reads, edits and tests code through tools, driven by an LLM.

- It is a *harness* around a model, not a framework, and talks to any
  OpenAI-compatible chat endpoint (`README.md:1-11`).
- Orchestration is an explicit **LangGraph state machine**:
  `build_context -> token_guard -> llm -> permission -> act -> terminate`
  (`harness/orchestration/graph.py:8-21`).
- Capability comes from plain Markdown on disk: `skills/*/SKILL.md` and
  `subagents/*/AGENT.md` (`README.md:18-19`).
- The Python package is named `harness/` for import stability even though the
  product is MiniAgent (`README.md:9-11`, `pyproject.toml:2`).

The project name in packaging is `miniagent` (`pyproject.toml:2`); the internal
package is `harness` (`harness/__init__.py:1`).

## 2. Repository layout

Top level (from `list_dir`):

```text
harness/            the Python package (all implementation)
  agent/            state, DTOs, events, termination guard
  cli/              event-driven terminal front end
  context/          context building, token budget, compaction
  inference/        ModelGateway abstraction + OpenAI-compatible adapter
  infra/            config, logging, checkpoint adapters
  memory/           long-term semantic memory (Qdrant) + embeddings
  orchestration/    LangGraph state machine (nodes, edges, routing)
  permission/       the permission gate (sandbox, rules, approval)
  skills/           filesystem skill discovery + progressive disclosure
  subagents/        context-isolated delegation
  tools/            MCP tool protocol, runtime, built-in tools
  core.py           AgentHarness: the single composition root
  main.py           `python -m harness.main` entry point
  __main__.py       `python -m harness` entry point
skills/             4 SKILL.md files (code_review, debugging, repo_exploration, testing)
subagents/          2 AGENT.md files (explorer, planner)
tests/              15 test modules + helpers.py
config.yaml         runtime configuration
pyproject.toml      packaging + pytest config
README.md, Agent_Harness_Design.md, CLI_Design.md, Detail.md, PERMISSIONS.md   design docs
miniagent.sh, run.sh   launch wrappers
```

The package's own docstring is the authoritative one-line-per-package map
(`harness/__init__.py:3-13`).

## 3. Architecture

### 3.1 Component table

| Component | Role | Evidence |
| --- | --- | --- |
| LangGraph orchestration | Runs the loop as a state graph, including checkpointing and conditional routing. | `harness/orchestration/graph.py:1-21` |
| `act` node | Executes every pending call of a turn (skills, delegation, plain tools) in the model's order. | `harness/orchestration/graph.py:109-115` |
| ModelGateway | Provider-agnostic chat interface; responses are normalized. | `harness/inference/gateway.py:15-28` |
| MCP tool protocol | Tools are real MCP tools; the same schemas drive the in-process runtime. | `README.md:28` |
| SQLite checkpoint | `langgraph-checkpoint-sqlite` persists run state per thread. | `pyproject.toml:11`, `harness/infra/checkpoint.py` |
| Qdrant long-term memory | Embedded Qdrant for durable cross-task facts, pluggable embeddings. | `harness/memory/service.py:1-5`, `config.yaml:34-46` |
| Filesystem skills | `SKILL.md` discovered on disk; only metadata injected until `load_skill`. | `harness/skills/loader.py:1-6` |
| Context builder / compact | Deterministic prompt assembly within a token budget, plus compaction. | `harness/context/manager.py:1-15` |
| Permission layer | One gate between the model and execution: sandbox, rules, approval memory. | `harness/permission/__init__.py:1-32` |
| Termination guard | Stops the loop on final answer, iteration cap, repeated calls, fatal errors. | `harness/agent/termination.py` |
| Subagents | `Planner` and `Explorer` run as separate graphs with a read-only tool ceiling. | `harness/subagents/registry.py:1-14`, `harness/core.py:46` |

### 3.2 The loop

```text
build_context -> token_guard -+-> compact -----+
                              |                |
                              +-> llm <--------+ (compact loops back)
                                    |
                                    v
                                permission  (ALLOW / DENY / ASK per call)
                                    |
                                    v
                                  act  (skills + delegate + tools, in order)
                                    |
                                    v
                              build_context (loop)
```

Source: `harness/orchestration/graph.py:8-21`; edges wired in `build()` at
`harness/orchestration/graph.py:118-143`.

Key structural facts:

- **LangGraph owns exactly four things**: state schema, nodes, edges, and
  checkpointing. Node bodies live in `orchestration/nodes.py`, routing decisions
  in `orchestration/routing.py`, termination in `agent/termination.py`
  (`harness/orchestration/graph.py:1-6`).
- The **permission node is optional**: it is only added when a `permission_gate`
  is supplied (`harness/orchestration/graph.py:98-108`).
- Routing is **termination-first**: `after_llm` checks the termination policy
  before dispatching any pending call (`harness/orchestration/routing.py:69-86`).
- All pending calls of a turn go to **one** `act` node, so every `tool_call_id`
  is answered even when a message mixes skills, delegation and plain tools
  (`harness/orchestration/routing.py:70-76`).
- Per-turn working data (prepared request, pending calls, token-guard verdict)
  lives in a `ContextVar` (`SCRATCH`), deliberately *not* in `AgentState`,
  because the checkpointer serializes state and a prepared `ModelRequest` holds
  callbacks (`harness/orchestration/nodes.py:33-43`).

## 4. Main modules (`harness/` subpackages)

- **`orchestration`** — LangGraph state machine: nodes, edges, checkpointing.
  `graph.py` defines `Orchestrator` and `build_orchestrator`
  (`harness/orchestration/graph.py:65-247`).
- **`inference`** — `ModelGateway` protocol over OpenAI-compatible APIs
  (`harness/inference/gateway.py`), with `openai_compatible.py`, `mock_gateway.py`,
  `config.py`, `tokenizer.py`.
- **`tools`** — MCP tool protocol, runtime and built-in tools. `registry.py` is
  "an MCP metadata cache, not a second tool protocol"; execution belongs to the
  runtime (`harness/tools/registry.py:1-5`).
- **`skills`** — filesystem skills with progressive disclosure. The loader puts
  the skill *body* into agent state on demand; the model sees only metadata
  beforehand (`harness/skills/loader.py:1-6`).
- **`subagents`** — context-isolated delegation; each subagent is a directory
  with `AGENT.md` whose front matter declares its tool allowlist
  (`harness/subagents/registry.py:1-14`).
- **`memory`** — long-term semantic memory (Qdrant) and memory formation.
  `MemoryService` is "the only memory interface the rest of MiniAgent knows";
  the context builder depends on the service, never on `QdrantClient`
  (`harness/memory/service.py:1-5`).
- **`context`** — context building, token budget and compaction.
  `ContextManager` owns token counting, message selection and the compaction
  trigger (`harness/context/manager.py:29-45`).
- **`agent`** — state, DTOs, events and the termination guard
  (`harness/__init__.py:12`).
- **`infra`** — config, logging, checkpoint adapters (`harness/__init__.py:13`).
- **`permission`** — the permission layer; deliberately independent of the
  orchestrator, importing only `harness.agent`, `harness.tools.paths` and
  `harness.infra.config` (`harness/permission/__init__.py:1-6`).
- **`cli`** — event-driven front end. "The CLI never executes tools or models
  itself: it turns input into an intent and renders the resulting events"
  (`harness/cli/app.py:1-10`).

### 4.1 Composition root

`harness/core.py` is the **single composition root**: "Everything above the infra
layer is injected here... Nothing else in the codebase constructs a concrete
adapter, which is what keeps the dependency direction clean"
(`harness/core.py:1-7`). `AgentHarness.__init__` builds the workspace, model
config, thread id and service slots (`harness/core.py:62-120`).

## 5. Data flow — one turn end to end

1. **Input** — the CLI turns user input into an intent and invokes the harness;
   it does not execute anything itself (`harness/cli/app.py:1-10`).
2. **`build_context`** — `ContextManager.prepare()` builds a `ModelRequest` and
   counts its tokens, returning a `PreparedContext` with a `should_compact`
   verdict (`harness/context/manager.py:18-50`).
3. **`token_guard`** — routes to `compact` or straight to `llm` based on the
   scratch verdict (`harness/orchestration/routing.py:62-66`).
4. **`compact`** (conditional) — compacts the message history, then loops back to
   `llm` (`harness/orchestration/graph.py:125`).
5. **`llm`** — the gateway runs one completion and returns a normalized response
   (`harness/inference/gateway.py:21-23`).
6. **`permission`** — the gate decides every call before anything executes;
   `after_llm` sends the whole pending set here (`harness/orchestration/routing.py:69-86`).
7. **`act`** — executes the allowed calls (skills, delegation, tools) in the
   model's order (`harness/orchestration/graph.py:109-115`).
8. **Loop or terminate** — `act` returns to `build_context`
   (`harness/orchestration/routing.py:104-107`); `terminate` ends the graph
   (`harness/orchestration/graph.py:143`).

**Memory path** *(static)*: `MemoryService` exposes recall (read) and remember
(write) over a memory store, with a pluggable embedding backend
(`harness/memory/service.py:20-38`). A memory backend that cannot start disables
memory for the session rather than failing the turn
(`harness/memory/service.py:46-50`).

## 6. Entry points

All entry points converge on `harness.cli.app:main`:

| Entry | Path |
| --- | --- |
| Console script `miniagent` | `pyproject.toml:25-26` → `harness.cli.app:main` |
| `python -m harness` | `harness/__main__.py:7-10` |
| `python -m harness.main` | `harness/main.py:7-10` |
| `./miniagent.sh` | `miniagent.sh:13` → `exec "$PYTHON" -m harness.main "$@"` |
| `./run.sh` | wrapper around `miniagent.sh` (`README.md:104`) |

CLI flags (`harness/cli/app.py:49-66`): positional `task` (one-shot,
non-interactive), `-c/--config`, `-m/--model`, `-w/--workspace`, `--mock`,
`--mcp-stdio`, `--plain`, `--no-memory`, `--no-checkpoint`, `--show-events`,
`--no-stream`, `--version`.

`--mock` (or `MINIAGENT_MOCK=1`) swaps the default model for a `provider: mock`
entry, which is the only way to run without an API key
(`harness/cli/app.py:73-80`).

## 7. Configuration

`config.yaml` (142 lines) declares: `models` + `default_model`, `context`,
`runtime`, `tools`, `memory`, `embedding`, `skills`, `subagents`, `permissions`,
`checkpoint`, `logging` (`config.yaml:1-142`).

Notable values:

- Default model `main` → DeepSeek `deepseek-chat` via `openai_compatible`
  (`config.yaml:2-20`).
- Context budget: `max_input_tokens: 128000`, `reserve_output_tokens: 8192`,
  `compact_trigger_ratio: 0.8`, `keep_recent_messages: 8` (`config.yaml:21-25`).
- Runtime: `max_iterations: 40`, `max_repeated_tool_calls: 3`,
  `tool_timeout_seconds: 120` (`config.yaml:26-30`).
- Tools enabled: `list_dir, glob, grep, read_file, write_file, apply_patch,
  shell, git_diff` (`config.yaml:32`).
- Memory: `qdrant_local`, path `.miniagent/memory`, `top_k: 5`
  (`config.yaml:34-40`).
- Embedding: `hashing`, 256 dimensions (`config.yaml:41-46`).
- Permission mode `auto` with `default: ask` (`config.yaml:51-56`).

**Env overrides**: any nested setting can be overridden with `MINIAGENT_` + the
path in double underscores, and the environment wins over the file
(`README.md:71-78`). The API key is read from the environment, never stored in
the config (`README.md:57-62`). A `.env` next to `config.yaml` is loaded
automatically, with existing environment variables winning (`README.md:64-69`).
Relative paths resolve against the config file's directory (`README.md:80-81`).

## 8. Skills & subagents

**Skills** — each skill is a directory with `SKILL.md` carrying YAML front matter
(`name`, `description`, `keywords`) followed by a Markdown body
(`skills/repo_exploration/SKILL.md:1-5`). Four skills ship: `code_review`,
`debugging`, `repo_exploration`, `testing`. Progressive disclosure: the model
sees metadata in every context and calls `load_skill` to pull the body into
agent state (`harness/skills/loader.py:1-6`). Bodies are truncated at 12,000
characters (`harness/skills/loader.py:14,36-38`).

**Subagents** — each is a directory with `AGENT.md`; front matter declares
`name`, `description`, `tools` and `skills`
(`subagents/explorer/AGENT.md:1-6`). Two ship: `explorer` and `planner`. The
declared tool list **is** the isolation boundary: an Explorer simply has no
`write_file` tool, so it cannot modify the repository even if asked
(`harness/subagents/registry.py:11-14`). The harness enforces a read-only ceiling
constant `("list_dir", "glob", "grep", "read_file", "git_diff")`
(`harness/core.py:46`). Subagents inherit the parent's permission policy but may
never ask questions, because their events do not reach the CLI and an unanswered
prompt would hang (`harness/core.py:118-120`).

## 9. Tools

Eight built-in tools plus two harness-native tools (`README.md:109-124`):

| Tool | Purpose | Source |
| --- | --- | --- |
| `list_dir` | List a directory as an indented tree. | MCP tool |
| `glob` | Find files by pattern. | MCP tool |
| `grep` | Search file contents with a regex. | MCP tool |
| `read_file` | Read a file or line range, with line numbers. | MCP tool |
| `write_file` | Create a file or fully overwrite it. | MCP tool |
| `apply_patch` | Edit via patch hunks, unified diff, or SEARCH/REPLACE. | MCP tool |
| `shell` | Run commands from the workspace root. | MCP tool |
| `git_diff` | Show the uncommitted diff. | MCP tool |
| `load_skill` | Read a skill's full instructions on demand. | harness native |
| `delegate` | Hand a task to a read-only subagent. | harness native |

All file paths are confined to the workspace root; escapes (`..`, absolute paths,
symlinks) are rejected (`README.md:126-128`). `apply_patch` matches exactly
first, then tolerates whitespace differences, and leaves a file untouched if any
hunk fails (`README.md:127-128`).

## 10. Permissions

The layer is one gate between the model and execution
(`harness/permission/__init__.py:1-32`):

```text
ToolCall / `!command`
      -> Action.from_tool_call
      -> PermissionEvaluator  (policy + memory + approval provider)
      -> PermissionGate
      -> allow -> execute | deny -> tool error
```

- Three modes: `off` (no permission layer), `ask` (prompt for anything the rules
  do not cover), `auto` (deny anything the rules do not cover)
  (`config.yaml:52-55`).
- Rules are **most-restrictive-wins**: a broad allow cannot hide a narrow ask
  (`config.yaml:57`).
- Reading/inspecting is allowed; `write_file` and `apply_patch` ask; `shell` is
  allowed for `git status`/`git diff`/`git log`/`ls` prefixes and asks otherwise;
  `rm`, `sudo`, `shutdown`, `mkfs`, `dd` are denied outright
  (`config.yaml:59-112`).
- `protected_paths` (`.env`, `*.pem`, `*.key`, `id_rsa*`, `.ssh/`, `.aws/`) are
  denied in every mode and for every approval scope (`config.yaml:125-133`).
- Persistent "always allow" grants live outside the workspace so the agent cannot
  edit its own permission store; `persistent: false` by default
  (`config.yaml:134-137`).
- The gate fails closed: a provider that cannot answer (plain stdin, batch) is a
  real answer (`harness/cli/app.py:108-110`).

## 11. Memory & checkpointing

- **Long-term memory**: embedded Qdrant (`backend: qdrant_local`) at
  `.miniagent/memory`, collection `miniagent_memory`, `top_k: 5`,
  `score_threshold: 0.05` (`config.yaml:34-40`). The default embedding backend is
  `hashing` with 256 dimensions; the config notes hashing embeddings score low
  and the threshold should be raised (0.3+) for real embedding models
  (`config.yaml:40-46`).
- **Checkpointing**: `langgraph-checkpoint-sqlite` persists run state per thread
  at `.miniagent/checkpoints.sqlite` (`config.yaml:138-139`,
  `pyproject.toml:11`). MiniAgent never writes its own save/load
  (`README.md:29`). `--no-checkpoint` disables it for a run
  (`harness/cli/app.py:83-84`).

## 12. How to run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

export DEEPSEEK_API_KEY="sk-..."          # or: echo 'DEEPSEEK_API_KEY=sk-...' > .env

./miniagent.sh                                    # interactive REPL
./miniagent.sh "fix the failing test in calc.py"  # one task, then exit
./miniagent.sh --mock "list the files here"       # offline, no API key needed
./miniagent.sh --mcp-stdio "..."                  # tools served from the MCP stdio server
.venv/bin/python -m harness.main --help           # same entry point, all flags
```

Source: `README.md:44-107`. Requires Python 3.11+ (`pyproject.toml:5`).

## 13. How to test

- pytest config: `testpaths = ["tests"]`, `asyncio_mode = "auto"`,
  `pythonpath = ["."]`, DeprecationWarnings ignored (`pyproject.toml:28-32`).
- Command: `.venv/bin/python -m pytest -q`.
- Test inventory (15 modules): `test_cli_approval.py`, `test_cli_prompt.py`,
  `test_cli_render.py`, `test_cli_ui.py`, `test_infra.py`,
  `test_integration.py`, `test_loop.py`, `test_mcp.py`, `test_memory_service.py`,
  `test_permission.py`, `test_permission_audit.py`, `test_permission_phase3.py`,
  `test_spec_compliance.py`, `test_textarea_edit.py`, `test_tools.py`, plus
  `tests/helpers.py`.
- **Verified**: `.venv/bin/python -m pytest -q` was executed during this analysis
  and passed — `494 passed in 29.60s`.

## 14. Known discrepancies / open questions

- **Doc vs. code drift** *(unverified)*: the design docs
  (`Agent_Harness_Design.md`, `CLI_Design.md`, `Detail.md`, `PERMISSIONS.md`) are
  large and may describe intended rather than actual behaviour. This analysis
  treats code as ground truth; no systematic diff of docs against code was
  performed.
- **Runtime behaviour** *(unverified)*: the loop is described statically. Whether
  it runs end to end depends on a venv and an API key (or `--mock`); no run was
  performed.
- **Peripheral files** *(not analysed)*: `scripts/demo_cli.py`,
  `rename_to_miniagent.py`, `terminal_output_research.md`,
  `CLI_Rendering_Issues.md` exist but were not examined in depth.
- **`harness/cli/app.py` is 1255 lines** and the permission test modules total
  ~80 KB; the CLI is summarised here as a layer, not line by line.
