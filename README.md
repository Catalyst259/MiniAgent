# MiniAgent

A lightweight terminal coding agent: a LangGraph-driven loop with real tools, filesystem
skills, context-isolated subagents, SQLite-checkpointed run state, and Qdrant-backed
long-term memory. It is deliberately a *harness* around a model, not a framework, and it
talks to any OpenAI-compatible chat endpoint.

The design this implements is documented in `Agent_Harness_Design.md`; the required
tools, skills, subagents and slash commands are listed in `Detail.md`. The internal Python
package is still named `harness/` (so imports stay stable) even though the product is
MiniAgent.

## What it is

- A CLI agent you run inside a repository; it reads, edits and tests code through tools.
- Orchestration is explicit: the loop is a LangGraph state machine
  (`build_context -> token_guard -> llm -> tools/skills/delegate -> terminate`).
- Behaviour is configurable from `config.yaml`; capability comes from `skills/` and
  `subagents/`, both plain Markdown on disk.

## Architecture

| Component | Role |
| --- | --- |
| LangGraph orchestration | Runs the loop as a state graph (`build_context -> token_guard -> llm -> act -> terminate`), including checkpointing and conditional routing. |
| `act` node | Executes every pending call of the current turn (skills, delegation, plain tools) in the model's order; a single node keeps one assistant message from silently dropping calls. |
| ModelGateway | Provider-agnostic chat interface; every response is normalized, so providers are swapped by config. |
| MCP tool protocol | Tools are real MCP tools (`harness/tools/mcp_server.py`, server name `miniagent-tools`); the same schemas drive the in-process runtime. |
| SQLite checkpoint | `langgraph-checkpoint-sqlite` persists run state per thread; MiniAgent never writes its own save/load. |
| Qdrant long-term memory | Embedded Qdrant for durable cross-task facts, with a pluggable embedding backend. |
| Filesystem skills | `SKILL.md` files discovered on disk; only metadata is injected until `load_skill` is called. |
| Context builder / compact | Deterministic prompt assembly within a token budget, plus compaction when the budget gets tight. |
| Permission layer | One gate between the model and execution: sandbox, allow/ask/deny rules, approval memory, and a prompt for anything not already decided. See `PERMISSIONS.md`. |
| Termination guard | Stops the loop on final answer, iteration cap, repeated tool calls, fatal errors or tool failures. |
| Subagents | `Planner` and `Explorer` run as separate graphs with their own context and a read-only tool ceiling. |

## Requirements

- Python 3.11 or newer (the interactive prompt uses `prompt_toolkit`, installed from
  `requirements.txt`).
- An API key for an OpenAI-compatible endpoint (DeepSeek, OpenAI, vLLM, ...) unless you
  configure the model provider as `mock`.

## Install

```bash
cd /path/to/Agent
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Configure

`config.yaml` declares models, the context budget, runtime limits, tool settings, memory,
skills, subagents and logging. `default_model` picks the entry used at startup.

The API key is read from the environment, never stored in the config file:

```bash
export DEEPSEEK_API_KEY="sk-..."               # the env var named by api_key_env
export MINIAGENT_MODELS__MAIN__API_KEY="sk-..."  # or override the model entry directly
```

A `.env` file next to `config.yaml` is loaded automatically (existing environment
variables win), so the common setup is simply:

```bash
echo 'DEEPSEEK_API_KEY=sk-...' > .env
```

Any nested setting can be overridden with `MINIAGENT_` + the path in double underscores,
and the environment wins over the file:

```bash
MINIAGENT_MODELS__MAIN__MODEL=deepseek-reasoner
MINIAGENT_CONTEXT__MAX_INPUT_TOKENS=128000
MINIAGENT_RUNTIME__MAX_ITERATIONS=60
```

Relative paths in `config.yaml` (workspace, checkpoint, memory, skills, subagents) resolve
against the config file's directory.

If your shell exports `HTTP_PROXY`/`HTTPS_PROXY` values that httpx cannot parse (a common
WSL/mirror situation, surfacing as `InvalidURL: Invalid port`), set `trust_env: false` on
the model entry to make that client ignore the ambient proxy variables:

```yaml
models:
  main:
    base_url: https://api.deepseek.com
    trust_env: false
```

## Run

```bash
./miniagent.sh
```

Startup takes no arguments and enters an interactive session. Set the model,
workspace, memory, checkpointing and tool transport in `config.yaml`; API keys
belong in `.env`. `./run.sh` remains a compatibility wrapper.

## Tools

The agent gets the eight built-in tools plus two harness-native tools:

| Tool | Purpose | Source |
| --- | --- | --- |
| `list_dir` | List a directory as an indented tree. | MCP tool |
| `glob` | Find files by pattern, e.g. `**/*.py`. | MCP tool |
| `grep` | Search file contents with a regex, with optional context lines. | MCP tool |
| `read_file` | Read a file or a line range, with line numbers and an explicit slice header. | MCP tool |
| `write_file` | Create a file or fully overwrite it. | MCP tool |
| `apply_patch` | Edit files with `*** Begin Patch` hunks, a unified diff, or SEARCH/REPLACE blocks. | MCP tool |
| `shell` | Run commands such as `pytest`, `python` and `git` from the workspace root. | MCP tool |
| `git_diff` | Show the uncommitted diff, optionally for one path. | MCP tool |
| `load_skill` | Read a skill's full instructions on demand. | harness native |
| `delegate` | Hand a task to a read-only subagent. | harness native |

All file paths are confined to the workspace root; escapes (`..`, absolute paths, symlinks)
are rejected. `apply_patch` matches exactly first, then tolerates whitespace differences,
and leaves a file untouched if any hunk fails to apply.

## Permissions

Every tool call passes a permission layer before it runs, whatever the front end:
the model's tools, a subagent's tools, and your own `!command` shell intents.

```text
ToolCall -> Action -> [ sandbox | policy rules | memory ] -> ALLOW / DENY / ASK
                                       ASK -> prompt -> execute or refuse
```

- **`mode`** decides what happens when no rule has an opinion: `off` (no rules),
  `ask` (prompt), or `auto` (deny). The shipped `config.yaml` sets `auto`, so a
  non-interactive run refuses a write instead of blocking on a prompt. Note that
  `off` switches off the *rules*, not the sandbox below.
- **Rules** are `allow` / `ask` / `deny` statements, resolved most-specific-first:
  a blanket `shell: ask` plus `shell prefix="git status": allow` lets `git status`
  run and asks about everything else. Rules can also constrain by `risk`
  (`low`/`medium`/`high`), where an unclassified tool counts as `medium` — a safety
  net for tools nobody wrote a rule for.
- **The sandbox cannot be overridden** — not by a rule, not by an approval, and
  not by `mode: off`. It covers workspace escapes, `protected_paths` (`.env`,
  keys, `.ssh`), `denied_tools`, the read-only ceiling of subagents, and the tool
  ceiling a loaded skill declares.
- **The sandbox inspects side effects, not the argument a rule matched on**: an
  `apply_patch` move destination, a shell command's `--output` path, `grep`'s
  `glob` filter and `glob`'s `pattern` are all checked like file targets.
- **Enforcement is doubled**: the `permission` graph node decides a whole turn
  (so the user is asked once per call and the transcript shows why), and
  `ToolRuntime.run` re-checks anything that did not come through it.
- **Shell prefix rules are parsed, not string-matched**: `npm install x; rm -rf ~`,
  `env npm …` and `git status -c core.pager=evil` do not satisfy a rule for `npm`
  or `git status`.
- **Approvals are remembered** as "allow once" (nothing stored), "allow this
  session" (shared with subagents), or "always allow" (stored in
  `~/.config/miniagent/approvals.json` — deliberately *outside* the workspace, so
  the agent cannot edit its own rules).

`PERMISSIONS.md` documents the full model, the config reference and the security
decisions. `/permissions` shows the current posture and grants; `/permissions clear`
drops the session grants.

## Skills

Skills live in `skills/<name>/SKILL.md` with YAML front matter (`name`, `description`,
`keywords`). Startup loads metadata only; the agent reads a body with `load_skill` and the
instructions then stay in context for the rest of the task.

A skill may also declare `tools:` — the tools it relies on. That list is a *ceiling* for
the rest of the task, never a grant: loading a skill cannot make a tool reachable that the
permission policy would otherwise refuse (see `PERMISSIONS.md`).

| Skill | Use it when |
| --- | --- |
| `repo_exploration` | Orienting in an unfamiliar repository or module. |
| `debugging` | A test, build or runtime path fails and you need the real cause. |
| `testing` | Running, narrowing, adding or trusting tests. |
| `code_review` | Self-reviewing a diff before declaring the task complete. |

## Subagents

`subagents/<name>/AGENT.md` defines a subagent with YAML front matter (`name`,
`description`, `tools`, `skills`). The main agent calls `delegate`; the subagent runs with
its own state and context - the parent transcript is never copied in - and returns a
compact report.
| Subagent | Role | Tools |
| --- | --- | --- |
| `planner` | Breaks a complex task into an ordered, verifiable plan. | `list_dir`, `glob`, `grep`, `read_file`, `git_diff` |
| `explorer` | Finds relevant code and reports files, symbols and call relationships. | `list_dir`, `glob`, `grep`, `read_file` |

Neither can write, and `explorer` has no `shell` at all: the tool list in `AGENT.md` is the
isolation boundary and is clamped to a read-only ceiling by the harness. A subagent also
runs with an auto-deny approver - its events never reach the CLI, so a permission prompt
could never be answered - while still sharing the session's remembered grants.

## The CLI

The front end follows the architecture in `CLI_Design.md`: it consumes events and
renders state, and never executes a tool, a model or a subprocess itself.

```text
KeyEvent -> Composer -> sync popups -> AppState -> Renderer
UserIntent -> Agent Runtime -> AgentEvent -> AppState -> Renderer
```

| Piece | Behaviour |
| --- | --- |
| **Composer** | A real editable buffer (`text` + `cursor`) via prompt_toolkit: insert, Backspace/Delete, Left/Right, Home/End, word-delete (Ctrl+W), multiline (Esc+Enter), paste, history. |
| **Transcript viewport** | The transcript is a scrollable pane with a scrollbar, so the newest line stays visible while the history stays reachable: the **mouse wheel** scrolls three lines per notch, **PageUp/PageDown** page by exactly one screen, **Ctrl+Home/Ctrl+End** jump to the oldest/newest line. At the bottom the view follows the stream; scrolling up pins it (the status line then shows `↑ N line(s) above the latest`) and new output no longer drags the view back down. Enabling the wheel means the terminal reports mouse events — use Shift+drag (or your terminal's selection modifier) to select text. |
| **Expandable cells** | **Ctrl+O** expands/collapses the newest collapsible block: a tool's full output (bounded to 400 lines so a huge dump cannot freeze the renderer), a subagent report, or the chain of thought. Collapsed blocks show their first lines plus `… N more line(s)`. |
| **Chain of thought** | Reasoning returned by the provider (`reasoning_content`/`reasoning`) is rendered as a dim `∴ thinking` block above the answer, collapsed to six lines and expandable with Ctrl+O. It never enters the model context. |
| **Slash popup** | Typing `/` shows a fuzzy-filtered command list above the prompt. The popup owns *navigation* only (↑/↓ move the selection, Tab accepts, Esc dismisses); the composer always owns the text, and the popup re-filters on every keystroke. |
| **Fuzzy match** | Codex-style case-insensitive subsequence matcher: window-size score with a bonus for matching from the start, matched characters highlighted. `/mdl` finds `/model`. |
| **Cell lifecycle** | Events address an *entity*, they never create one: `AssistantStarted` creates the message cell, `AssistantDelta` updates it, `assistant_message`/`AssistantFinished` only mark it complete; `ToolStarted` and `ToolFinished` update the same `ToolCell` by `call_id`. Nothing is appended twice, so an answer is rendered exactly once and one call shows one pending + one finished line. |
| **Assistant streaming** | Raw Markdown is accumulated and split at the last newline into a **stable** region and a **mutable tail**. Finished lines are written once, each on its own row; only the last, still-changing line is redrawn in place (throttled). When the message finishes the preview rows are erased and the answer is rendered once from the full source, so partial Markdown (`**hel` + `lo**`) never renders wrong and never appears twice. The interactive transcript does the same thing per cell: complete lines are Markdown, the still-changing tail is plain text (an unterminated code fence can no longer swallow the rest of the answer). |
| **Final answer** | When a turn ends the live region is erased and the answer is printed under a `── final answer ──` banner (or `── final answer (stopped: <reason>) ──`), so it is unmistakable which text is the result. |
| **Tool trace** | Every call is announced as it starts (`◐ read_file  {"path": …} …` with its arguments), a running tool streams the tail of its output, and the finished cell shows the body preview plus the elapsed time (`(42 ms)`). Skills (`◆ loaded skill testing`), delegations (`⇢ explorer …` plus the subagent's structured report) are traced the same way. Routing goes through one `act` node that executes *all* calls of a turn, so every `tool_call_id` is answered. |
| **Command output** | Slash-command output (`/help`, `/status`, `/tools`, …) is appended to the transcript as cells, so it scrolls, expands and clears like everything else instead of being printed above the application region. |
| **Output sanitising** | Tool output is arbitrary text: ANSI codes, control characters and tabs are stripped/normalised before printing, so a colour code can never show up as `[0m` garbage. |
| **Light Markdown** | Rich's defaults paint code blocks with `bgcolor="black"`; MiniAgent overrides that theme so code gets a colour but **no background fill**, keeping the transcript light instead of a full-width dark bar. |
| **Tool cells** | A tool renders as one cell that flips from `◐ running` to `● done` / `✗ failed` in place (shape, not only colour, tells them apart), with a bounded output preview and timing. |
| **History cells** | History is a list of typed cells (`UserCell`, `AssistantCell`, `ToolCell`, `SubAgentCell`, `SkillCell`, `ErrorCell`, `InfoCell`), not strings, so the renderer has no `elif event.type` chain. |
| **Committed vs active** | The running cell is the *active cell* and is committed to history when it finishes. |
| **Shell intent** | `!command` is not run by the UI: the composer produces a `ShellIntent` and the runtime executes it through the same sandboxed `shell` tool the model uses. |

The interactive prompt requires terminal input (TTY). Piped or redirected input
is rejected with `Error: stdin is not a terminal`.

## Slash commands

| Command | Action |
| --- | --- |
| `/help` | Show command help. |
| `/status` | Current task, iteration count, token budget, loaded skills, memory status. |
| `/model` | Show configured models, or switch with `/model <name>`. |
| `/tools` | List the tools available to the agent. |
| `/skills` | List skills, marking the loaded ones. |
| `/agents` | List the subagents and their tools. |
| `/permissions` | Show the permission posture, rules and remembered grants (`/permissions clear` drops the session grants). |
| `/compact` | Compact the current context now (Summary is separate: it runs at task end). |
| `/clear` | Clear the conversation and start a new thread. |
| `/exit` | Exit the CLI (`/quit` and `/q` are aliases). |

Pressing `/` shows the same list as a popup, filtered as you type.

## Mock / offline mode

Set the active model's `provider` to `mock` in `config.yaml` to exercise the
CLI and tools without an API key. Then launch normally:

```bash
./miniagent.sh
```

The offline model performs a short reconnaissance (`list_dir`, `read_file`, `grep`) and
then says plainly that it did not reason about the task; it never invents an answer. The
mock gateway also streams its answer in chunks, so the streaming path is covered offline.

## Demo

`scripts/demo_cli.py` drives the same `MiniAgentApp` the TUI uses, but with a scripted
input instead of keystrokes, so the CLI can be demonstrated without a terminal:

```bash
.venv/bin/python scripts/demo_cli.py  # offline; workspace comes from config.yaml
```

## Tests

```bash
./miniagent.sh                  # interactive startup
.venv/bin/python -m pytest     # full suite
```

`pyproject.toml` sets `testpaths = ["tests"]`, `asyncio_mode = "auto"` and
`pythonpath = ["."]`, so async tests need no decorators and imports resolve from the repo
root. The suite covers the tool layer (including `apply_patch` dialects), the agent loop
end to end, the termination guard, compaction, memory formation and recall, skills,
subagents and their tool isolation, MCP server/client round trips, SQLite checkpoints,
config overrides, the permission layer (actions, shell parsing, rules, sandbox, grants,
approval keys, skill ceilings), the CLI (fuzzy matcher, popup derivation, composer
editing, cells, stable/tail streaming, event bridge, slash commands, shell intent) and
subprocess-level runs of the real binary.

Note: tests that spawn a pseudo-terminal are intentionally absent - the sandbox cannot
allocate one - so the interactive prompt is covered through prompt_toolkit's own frame and
binding APIs plus a real pipe-input run of the key layer.

## Module layout

The package lives under `harness/` at the repository root:

```
harness/
  agent/          # state, DTOs, events, errors, termination guard
  orchestration/  # LangGraph graph, nodes, routing
  context/        # context builder, token budget, compaction, prompts
  inference/      # ModelGateway protocol, OpenAI-compatible + mock gateways, streaming, tokenizer
  tools/          # 8 tools, patch engine, MCP server, MCP client, registry, runtime
  permission/     # action parsing, sandbox/rules policy, grants, evaluator, gate
  skills/         # skill registry and progressive-disclosure loader
  subagents/      # subagent registry and delegation runtime
  memory/         # embedding backends, memory stores (Qdrant/in-memory), summary service
  infra/          # config, logging, SQLite checkpoint adapters
  cli/
    app.py        # MiniAgentApp: assembly, startup and scripted entry points
    terminal.py   # prompt_toolkit layout, input callbacks and terminal lifecycle
    controller.py # input routing, Agent turns and task cancellation
    presenter.py  # event-driven conversation cells and message lifecycle
    output.py     # console and live transcript output adapters
    events.py     # AgentEvent dataclasses (the runtime/UI boundary)
    state.py      # AppState, TextAreaState, CommandPopupState, StreamState
    events_bridge.py  # runtime events -> UI events
    shell_intent.py   # `!command` as an intent executed by the runtime
    approval.py   # interactive permission approval
    composer/     # composer, slash_commands registry, fuzzy_match, command_popup
    cells/        # base + user/assistant/tool/subagent/error cells
    streaming/    # assistant_stream (stable/tail)
    render/       # renderer, theme, legacy plain/rich renderers
skills/           # 4 skills (repo_exploration, debugging, testing, code_review)
subagents/        # 2 subagents (planner, explorer)
config.yaml       # models, budgets, tools, permissions, memory, skills, subagents
requirements.txt  # runtime + test dependencies
miniagent.sh      # launcher (run.sh is a wrapper)
scripts/          # demo_cli.py: scripted, terminal-free CLI demo
```

Dependency direction is one-way: `orchestration -> agent/context/tools/skills/subagents/
memory/inference -> infra`. The context builder depends on the `MemoryService` interface,
never on `QdrantClient`; the LLM node depends on `ModelGateway`, never on a provider SDK.
