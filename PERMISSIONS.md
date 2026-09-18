# Permissions

MiniAgent runs real commands in a real repository, so every action the model asks
for passes one permission layer before anything is executed. This document
describes what is enforced, how to configure it, and how the implementation maps
onto the design.

```text
LLM ──► ToolCall
          │
          ▼
    Action parser            harness/permission/action.py
          │
          ▼
    Permission gate          harness/permission/gate.py      (a real graph node)
          │
   ┌──────┴───────────────────────────────┐
   │ Sandbox   workspace escape, protected paths, read-only ceiling, skill ceiling
   │ Policy    configured rules: allow / ask / deny
   │ Memory    once · session · persistent
   │ Approval  Allow once | Allow this session | Always allow | Reject
   └──────┬───────────────────────────────┘
          ▼
   ALLOW ──► execute        DENY ──► tool error to the model        ASK ──► prompt
```

Enforcement happens at **two** points, deliberately:

1. the `permission` graph node decides a whole turn before anything runs, so the
   transcript can show the verdict and the user is asked once per call;
2. `ToolRuntime.run` — the method that actually starts a tool — checks again, so a
   call that never reached that node is *decided* rather than assumed safe.

The second is a safety net, not a duplicate prompt: the node marks its batch as
decided, and the runtime only asks about calls it has not seen.

## Decisions

| Verdict | What happens |
| --- | --- |
| `ALLOW` | The tool runs. |
| `DENY` | The tool never runs; the model receives an explicit `PERMISSION DENIED` tool error telling it not to retry, and the transcript shows the refusal. |
| `ASK` | The user is prompted, once per call, before anything runs. |

An `ASK` that nothing can answer is converted to `DENY`: there is no fourth
"maybe" state. Subagents, `--plain` runs with exhausted stdin, and one-shot
`miniagent "task"` runs all fail closed.

## Modes

`permissions.mode` decides what happens when no rule has an opinion:

| Mode | Behaviour | Use it for |
| --- | --- | --- |
| `off` | No permission *rules* at all (the historical behaviour). The sandbox below still applies: workspace boundary, `protected_paths`, `denied_tools`, the read-only ceiling and the skill ceiling. | Debugging, trusted throwaway workspaces. |
| `ask` | Prompt for anything the rules do not cover. | Interactive use where you want to decide. |
| `auto` | Deny anything the rules do not cover. | Non-interactive runs, CI, anything that must not block. **This is the mode the shipped `config.yaml` sets.** |

Note the distinction: `mode: off` switches off the rule set, not the safety
boundary. A config that omits the `permissions:` section entirely gets
`mode: off` (the schema default), so it has no rules — but it still cannot leave
the workspace or touch a protected path.

> **`default: allow` with `mode: auto`** allows every action no rule mentions.
> That combination logs a warning, because it reads like a closed posture and is
> not one.

## Configuration

```yaml
permissions:
  mode: auto            # off | ask | auto
  default: ask          # verdict for an action no rule matches: allow | ask | deny
  rules:
    - tool: read_file
      permission: allow
    - tool: shell
      prefix: git status      # a *simple* command whose leading tokens match
      permission: allow
    - tool: shell
      program: npm            # the program being invoked, path stripped
      permission: ask
    - tool: shell
      permission: ask         # blanket rule for everything else
    - risk: medium            # constrain by risk band instead of tool
      permission: ask
  # shorthand lists, equivalent to rules entries. A bare string names a TOOL:
  allow: [grep]
  ask: [write_file]
  deny: [curl]
  # denied in every mode, for every tool, and no approval can lift these
  protected_paths: [".env", "**/*.pem", "**/.ssh/**"]
  denied_tools: [some_mcp_tool]
  # "Always allow" grants
  persistent: false
  persistent_path: null     # default: ~/.config/miniagent/approvals.json
```

### How a rule is chosen

Resolution is **most specific wins, then most restrictive**:

* `shell: ask` next to `shell prefix='git status': allow` means `git status`
  runs and every other shell command asks — the narrow rule refines the broad one;
* two equally specific rules resolve to the more restrictive verdict, so a deny
  can never be shadowed by an allow of the same breadth.

Specificity counts the fields a rule constrains: `tool` (2), `program` (3),
`prefix` (3), `target` (1).

### Risk levels

Every action carries a risk level — `low`, `medium` or `high` — reported in the
action payload and the transcript. Reading is `low`, changing the workspace is
`medium`, running a shell command is `high`, and **a tool nobody classified is
`medium`, not "harmless"**. That last point is what makes risk useful: an MCP tool
from another server has no rule and no known semantics, and it should not inherit
a permissive default by accident.

```yaml
rules:
  # a safety net for the tools nobody enumerated
  - risk: medium
    permission: ask
  # or a hard floor:
  - risk: high
    permission: deny
```

A risk band is broad by construction, so it is resolved *against* the ordinary
winner instead of competing with it: it applies only where it is strictly more
restrictive (or where no ordinary rule matched at all), and a risk **deny** is a
floor no other rule can talk past. A risk band is never memorised — "always allow
high-risk actions" is not what anyone means by approving one command.

### Rules and memory

Configured rules are consulted first; memory only *widens what is asked*:

| Configured rule | Memory can… |
| --- | --- |
| `deny` | nothing. A deny is final. |
| `allow` | nothing. It is already allowed. |
| `ask` | answer the question — that is what "Allow this session" does. |
| *(no rule)* | allow, or deny. |

A remembered grant therefore never overrides a `deny` rule, and never overrides
the sandbox.

## The sandbox

The sandbox is not a rule set: it is a property of the context, and no rule, no
approval and no `Always allow` can lift it. It is enforced in **every** mode,
including `mode: off`.

| Check | Detail |
| --- | --- |
| Workspace escape | `..`, absolute paths and symlinks out of the workspace are refused, for filesystem tools *and* for shell commands that name an output path. |
| `protected_paths` | Path globs denied for every tool. Applied to *every* path an action can reach — see below. |
| `denied_tools` | Tool names refused outright. |
| Read-only ceiling | Subagents run with this on: their tool list already omits the write tools, and the ceiling holds even when `mode: off`. |
| Skill ceiling | While a skill is active, only the tools that skill declares may be used (see below). |

### Every path an action can reach is inspected

The rule that keeps this honest: **the sandbox inspects the action's side effects,
not the argument a rule matched on.** A tool's real effect frequently lives
somewhere the obvious argument does not mention.

| Tool | Paths inspected |
| --- | --- |
| `read_file`, `write_file`, `list_dir`, `git_diff` | `path` |
| `apply_patch` | every file in the payload — including `*** Move to:` destinations, because a move writes the destination |
| `grep` | `path` **and** the `glob` filter (`glob="*.pem"` opens private keys), and its search is confined to the workspace after resolving symlinks |
| `glob` | `pattern` — an absolute pattern is read as workspace-relative, and `..` is refused; the tool never walks the filesystem root |
| `shell` | the paths the command was told to **write** (`git log --output=…`, `-o`, `--file`, …) and the ones it was told to **read** as operands (`git diff --no-index a /etc/passwd`), plus the raw command line |

The shell row is honest about its limits: the command line is inspected as an
*option-carrying* string, not fully parsed. A program that takes an unbounded path
operand without a recognisable option is not covered — which is why the shipped
rules ask about shell commands rather than allowing them wholesale, and why a
`deny` on a specific program is the reliable way to block it.

`matches_path` gives `` **/ `` its usual "zero or more directories" meaning, so
`**/*.pem` protects a key at the workspace root as well as one under `certs/`.
Plain `fnmatch` does not do this, and a root-level key is exactly where a stray
key ends up.

### Shell commands are parsed, not prefix-matched

A prefix rule is only sound when the whole command *is* that program:

```bash
npm install axios             # prefix rule "npm" applies
npm install axios; rm -rf ~   # NOT matched by any prefix rule - it asks
npm install $(whoami)         # NOT matched - command substitution
env npm install axios         # NOT matched - "env" runs another program
git status -c core.pager=evil # NOT matched - the option executes code
```

Parsing rejects:

* **chaining**: `;`, `&&`, `||`, `|`, `&`, redirection (`>` `<`), backticks,
  `$(`, `${`, newlines, and unbalanced quotes;
* **wrapping**: commands whose job is to run another command — `env`, `nice`,
  `nohup`, `timeout`, `xargs`, `sudo`, `bash -c`, `python -c`, `make`, …
* **self-escalation**: options that make a program run something else
  (`-c`, `--config`, `--upload-pack`, `--pager`, …) or write a file it was told
  to (`--output`, `-o`, `--file`, …).

A command that trips any of these fails safe: it never satisfies an allow rule.

Because such a command can never be matched by a standing rule, the prompt does
not offer "Allow this session" / "Always allow" for it — promising a grant the
layer could not keep would be a lie.

## Approval memory

| Scope | Lifetime | Stored as |
| --- | --- | --- |
| **Allow once** | This one call. | Nothing is stored. |
| **Allow this session** | Until the CLI exits. Shared with subagents, so a child never re-asks what you already answered. | `SessionMemory` |
| **Always allow** | Across restarts. | `~/.config/miniagent/approvals.json` (mode `0600`) |

Two deliberate limits:

* **A standing grant is never target-scoped.** "Always allow `write_file`" grants
  the tool, and that is what the label says. `apply_patch` therefore offers only
  "Allow once": one approved patch must not silently authorise every future patch.
* **The store lives outside the workspace.** The agent can write inside its
  workspace, so a permission file there would let it edit its own rules. A
  configured `persistent_path` inside the workspace is refused and the store is
  disabled with an error in the log rather than silently trusted.

`/permissions` shows the posture, the rules, the session grants and the store
path; `/permissions clear` drops the session grants.

Any setting can also come from the environment, which is the quickest way to try a
posture without editing the file:

```bash
MINIAGENT_PERMISSIONS__MODE=ask ./miniagent.sh          # prompt instead of deny
MINIAGENT_PERMISSIONS__MODE=off ./miniagent.sh          # no rules (sandbox still applies)
MINIAGENT_PERMISSIONS__PERSISTENT=true ./miniagent.sh   # remember "always allow"
```

## Skills

A skill may declare the tools it needs:

```yaml
---
name: code-review
description: Review a diff.
tools: [read_file, grep, git_diff]
---
```

That list is a **ceiling, never a grant** (`Skill Permission <= Tool Permission`):

* a tool outside the list is denied, in every mode — a ceiling is not a question
  to be asked;
* a tool the policy denies is still denied, even if the skill lists it;
* loading a skill in the same turn as the calls it governs already applies the
  ceiling to those calls;
* `load_skill` itself stays available, so the model can switch to another skill
  instead of deadlocking;
* a skill that declares no tools imposes no ceiling (it made no promise).

## Subagents

Each subagent is a separate harness with its own permission stack:

* it is clamped to the read-only tool ceiling by its `AGENT.md`, and the
  permission layer enforces read-only as a second barrier;
* it is built with an auto-deny approver — its events never reach the CLI, so a
  prompt could never be answered and would hang the whole run;
* it **shares the session's remembered grants**, so "allow this session" is not
  asked again by a child, and an inherited grant still cannot beat the ceiling.

## CLI keys

While a question is open the status line shows the action, the reason and the
choices:

| Key | Effect |
| --- | --- |
| `1` / `2` / `3` / `4` | Allow once / this session / always / Reject |
| `←` `→` | Move the highlighted choice |
| `Enter` | Confirm the highlighted choice |
| `Ctrl+R` | Reject |
| `Ctrl+C` | Abort the turn (and the question with it) |

The digit keys only exist while a question is open, so they type normally
otherwise. Anything that cannot be decided in the TUI (`!command` shell intents,
one-shot runs) goes through the same gate.

## Module map

| File | Role |
| --- | --- |
| `harness/permission/action.py` | `Action` + conversion from a `ToolCall` (including the files an `apply_patch` touches) |
| `harness/permission/shell.py` | Shell parsing and the compound-command detector that makes prefix rules sound |
| `harness/permission/rules.py` | `Rule`, `RuleMatcher`, specificity resolution |
| `harness/permission/decision.py` | `Permission`, `Verdict` |
| `harness/permission/policy.py` | Sandbox, rules, mode, skill ceiling |
| `harness/permission/memory.py` | Once / session / persistent stores |
| `harness/permission/evaluator.py` | The decision flow |
| `harness/permission/approval.py` | `ApprovalProvider` protocol + headless providers |
| `harness/permission/gate.py` | Batch evaluation, approval, denial messages |
| `harness/cli/approval.py` | Interactive (future + keys) and plain (stdin) providers |

Integration points:

* **`harness/orchestration/graph.py`** adds a `permission` node between `llm` and
  `act`, and points the gate's emitter at the turn's event stream.
* **`harness/orchestration/nodes.py`** — `act` executes what the gate allowed and
  turns denials into tool errors, so every `tool_call_id` is still answered.
* **`harness/core.py`** builds the stack per harness and wires subagent isolation.

## Events

| Event | Meaning |
| --- | --- |
| `permission_ask` | A question is open (carries `can_session` / `can_persist`) |
| `permission_decision` | A verdict, with its reason and source |
| `tool_denied` | A refused call, rendered as a failed tool cell |

## Tests

| File | Covers |
| --- | --- |
| `tests/test_permission.py` | Actions, shell parsing, rules, policy, sandbox, memory, approval, the gate, and the loop end to end |
| `tests/test_cli_approval.py` | The interactive provider, status line, key dispatch, a real pipe-driven `Application`, and the plain/stdin provider |
| `tests/test_permission_phase3.py` | Skill ceilings and persistent-store location/round-trip |
| `tests/test_permission_audit.py` | Regressions for every finding of the adversarial audit below |

## Audit history

The layer was attacked by an independent reviewer instructed to break it. Five
bypasses were confirmed and fixed; each now has a regression test in
`tests/test_permission_audit.py` named after its finding id.

| Id | Bypass | Fix |
| --- | --- | --- |
| E1 | `mode: off` made the gate return ALLOW before the ceiling was consulted, lifting `denied_tools`, the read-only ceiling and the workspace check — and `off` is the schema default. | The gate always goes through the evaluator. |
| E2 | `*** Move to: secrets/x.pem` wrote a protected path that `targets()` never listed. | Move destinations and unified-diff source paths are targets. |
| E3 | `git log -p --output=/tmp/exfil` matched the shipped `prefix: git log` allow rule and wrote outside the workspace. | Shell write options are targets; write/escalation options disqualify a prefix rule. |
| E4 | `glob("/etc/*.conf")` walked the real filesystem root. | Absolute patterns are workspace-relative; `..` is refused. |
| E5 | `grep(glob="*.pem")` read a protected file because the filter was not a target. | `grep`'s `glob` filter is inspected. |
| B1 | `fnmatch` never gives `**/` a "zero directories" meaning, so the shipped globs did not protect a key at the workspace root. | Segment-aware `matches_path`. |
| E6 | *(found while reproducing the others)* enforcement lived only in the graph node, so calling `ToolRuntime.run` directly executed a call the gate had denied. | `ToolRuntime.run` consults the gate for any call it has not already decided. |
| E7 | `` `./.env` `` did not match the `.env` protection rule — the same file, spelled differently. | Path matching normalises `.` segments and separators first. |
| E8 | `MINIAGENT_PERMISSIONS__MODE=off` was coerced to the boolean `False`, so the documented way of choosing a posture from the environment failed to load. | Word-valued settings are kept verbatim by the env-override coercer. |
| N1 | Every gated call was decided **and prompted** twice: the permission node recorded its batch in a `ContextVar`, which LangGraph never propagates to a sibling node, so the execution boundary re-asked. | The graph hands its decided ids to `run_many` explicitly; the task-scoped set now covers only re-entrant calls within one execution path. |
| N2 | A read-only allow rule was a licence to read the filesystem: `git diff --no-index a.txt /etc/passwd` printed any file the user could read. | Shell *read* operands are targets too, and `--no-index` disqualifies a prefix rule. |
| N3 | Dead duplicate loop after `return None` in `sandbox_check`. | Removed. |

The audit also reported what survived: the shell parser resisted a 40-payload
chaining/redirection/substitution battery with zero bypasses, no way was found to
widen permissions without a real user approval, "allow once" is genuinely spent,
grants cannot widen past a configured `deny`, the persistent store refuses to live
inside the workspace, and the stdin provider fails closed (its `move_to` and
symlink escapes were already refused by `Workspace.resolve`).
