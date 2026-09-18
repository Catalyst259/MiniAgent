"""MiniAgent CLI: event-driven front end.

```text
KeyEvent -> Composer -> (popup sync) -> AppState -> Renderer
UserIntent -> Agent Runtime -> AgentEvent -> AppState -> Renderer
```

The CLI never executes tools or models itself: it turns input into an intent and
renders the resulting events (``CLI_Design.md`` sections 31, 38, 39).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from harness.agent.dto import Message, ToolCall
from harness.agent.state import AgentState, new_state
from harness.cli import events as ui
from harness.cli.approval import InteractiveApprovalProvider, PlainApprovalProvider
from harness.cli.cells import (
    AssistantCell,
    ErrorCell,
    InfoCell,
    SkillCell,
    SubAgentCell,
    ToolCell,
    ToolStatus,
    UserCell,
)
from harness.cli.composer import CommandPopup, CommandRegistry, Composer, build_default_registry
from harness.cli.composer.composer import PromptInterrupt, SlashCompleter, build_key_bindings
from harness.cli.events_bridge import EventBridge
from harness.cli.render.renderer import Renderer
from harness.cli.render.transcript import TranscriptControl, TranscriptPane
from harness.cli.shell_intent import ShellIntent, run_shell_intent
from harness.cli.state import AppState
from harness.infra.checkpoint import AsyncCheckpointStore
from harness.infra.config import HarnessConfig

BANNER_AGENT = "MiniAgent"


# ---------------------------------------------------------------------- config
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="miniagent",
        description="MiniAgent: a lightweight terminal coding agent (LangGraph orchestration, OpenAI-compatible models).",
    )
    parser.add_argument("task", nargs="*", help="run one task non-interactively and exit")
    parser.add_argument("-c", "--config", help="path to config.yaml", default=None)
    parser.add_argument("-m", "--model", help="model name from the config to use", default=None)
    parser.add_argument("-w", "--workspace", help="workspace root the tools are confined to", default=None)
    parser.add_argument("--mock", action="store_true", help="force the offline mock model")
    parser.add_argument("--mcp-stdio", action="store_true", help="run the tools through the MCP stdio server")
    parser.add_argument("--plain", action="store_true", help="line-based mode (no prompt_toolkit)")
    parser.add_argument("--no-memory", action="store_true", help="disable long-term memory for this run")
    parser.add_argument("--no-checkpoint", action="store_true", help="disable LangGraph SQLite checkpointing")
    parser.add_argument("--show-events", action="store_true", help="print every MiniAgent event")
    parser.add_argument("--no-stream", action="store_true", help="disable assistant delta streaming")
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    return parser


def load_config(args: argparse.Namespace) -> HarnessConfig:
    config = HarnessConfig.load(args.config)
    if args.workspace:
        config.runtime.workspace_root = args.workspace
    if args.mock or os.environ.get("MINIAGENT_MOCK", "").lower() in ("1", "true", "yes", "on"):
        from harness.inference.config import ModelConfig

        config.models[config.default_model] = ModelConfig(
            provider="mock", model=f"mock-{config.default_model}"
        )
        if args.model:
            config.models[args.model] = config.models[config.default_model]
    if args.no_memory:
        config.memory.enabled = False
    if args.no_checkpoint:
        config.checkpoint.enabled = False
    if args.mcp_stdio:
        config.tools.transport = "mcp-stdio"
    return config


# --------------------------------------------------------------------- session
class Session:
    """One CLI session: conversation history + harness + checkpointer."""

    def __init__(
        self,
        config: HarnessConfig,
        *,
        model: str | None = None,
        on_event=None,
        approval_provider=None,
    ) -> None:
        self.config = config
        self.model_name = model or config.default_model
        self.history: list[Message] = []
        self.turn = 0
        self.on_event = on_event
        self.harness = None
        #: the permission layer's ``ASK`` branch.  A provider that cannot answer
        #: (plain stdin, batch) is a real answer - it fails closed.
        self.approval_provider = approval_provider
        self._checkpoint_cm: AsyncCheckpointStore | None = None

    async def __aenter__(self) -> "Session":
        from harness.core import AgentHarness

        checkpointer = None
        if self.config.checkpoint.enabled:
            self._checkpoint_cm = AsyncCheckpointStore(str(self.config.checkpoint_path))
            checkpointer = await self._checkpoint_cm.__aenter__()
        harness = AgentHarness(
            self.config,
            model_name=self.model_name,
            on_event=self.on_event,
            approval_provider=self.approval_provider,
        )
        harness.build()
        if checkpointer is not None:
            harness.orchestrator.checkpointer = checkpointer
            harness.orchestrator.build()
        self.harness = harness
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self.harness is not None:
            await self.harness.close()
        if self._checkpoint_cm is not None:
            await self._checkpoint_cm.__aexit__(*exc_info)

    # -------------------------------------------------------------------- turns
    def _state(self, task: str) -> AgentState:
        self.turn += 1
        state = new_state(
            task,
            thread_id=f"{self.harness.thread_id}-t{self.turn}",
            task_id=f"{self.harness.thread_id}-{self.turn}",
        )
        state["messages"] = list(self.history) + [Message(role="user", content=task)]
        return state

    async def ask(self, task: str, *, on_delta=None, write_memory: bool = True) -> AgentState:
        assert self.harness is not None
        state = self._state(task)
        result = await self.harness.run(
            task, state=state, thread_id=state["thread_id"], on_delta=on_delta
        )
        self.history = list(result.get("messages") or [])
        if write_memory and result.get("termination_status") in ("final_answer", None):
            await self.harness.finish(result, repo=str(self.config.workspace_root))
        return result

    async def compact_now(self) -> str:
        assert self.harness is not None
        state = self._state("(manual compact)")
        self.turn -= 1  # a compact is not a conversational turn
        outcome = await self.harness.context_manager.compact(state)
        if not outcome.applied:
            return "Nothing to compact yet."
        self.history = [
            Message(role="user", content=f"[Context compacted]\n{outcome.summary}")
        ] + outcome.tail[len(self.history) :]
        return f"Compacted {outcome.folded} message(s) into {len(outcome.summary)} chars."

    # ------------------------------------------------------------------ commands
    async def command_handlers(self) -> dict[str, Any]:
        return {
            "help": self.cmd_help,
            "status": self.cmd_status,
            "model": self.cmd_model,
            "tools": self.cmd_tools,
            "skills": self.cmd_skills,
            "agents": self.cmd_agents,
            "permissions": self.cmd_permissions,
            "compact": self.cmd_compact,
            "clear": self.cmd_clear,
            "exit": self.cmd_exit,
        }

    async def cmd_help(self, app: "MiniAgentApp", argument: str = "") -> bool:
        app.emit_line("")
        for command in app.registry.all():
            app.emit_line(f"  /{command.name:<9} {command.description}")
        app.emit_line("  !<cmd>    run a shell command in the workspace")
        return False

    async def cmd_status(self, app: "MiniAgentApp", argument: str = "") -> bool:
        harness = self.harness
        info = await harness.status()
        context_status = await harness.context_manager.status(self._state("(status)"))
        self.turn -= 1
        rows: list[tuple[str, Any]] = [
            ("model", info["model"]),
            ("workspace", info["workspace"]),
            ("thread", info["thread_id"]),
            ("turn", self.turn),
            ("messages", len(self.history)),
            ("tools", info["tools"]),
            ("skills", info["skills"]),
            ("loaded skills", ", ".join(_loaded_skills(self.history)) or "(none)"),
            ("subagents", ", ".join(info["subagents"]) or "(none)"),
            ("context", context_status["usage"]),
            ("compactions", context_status["compactions"]),
        ]
        memory = info.get("memory")
        if memory:
            rows.append(
                (
                    "memory",
                    f"{memory['backend']} / {memory['embedding']} / {memory['records']} records"
                    + ("" if memory["enabled"] else " (disabled)"),
                )
            )
        permissions = info.get("permissions")
        if permissions:
            policy = permissions["policy"]
            grants = permissions["memory"]
            rows.append(
                (
                    "permissions",
                    f"{policy['mode']} / {policy['rules']} rule(s) / "
                    f"{policy['sandbox_deny']} protected path(s)",
                )
            )
            rows.append(
                (
                    "grants",
                    f"{grants['session']} session, {grants['temporary']} once"
                    + (
                        f", {grants['persistent']['grants']} saved"
                        if grants.get("persistent")
                        else ", not persisted"
                    ),
                )
            )
        app.emit_line("")
        for key, value in rows:
            app.emit_line(f"  {key:<14} {value}")
        return False

    async def cmd_model(self, app: "MiniAgentApp", argument: str = "") -> bool:
        if not argument:
            app.emit_line("")
            for name, model_config in sorted(self.config.models.items()):
                marker = "❯" if name == self.model_name else " "
                app.emit_line(f"  {marker} {name}  {model_config.label}")
            app.emit_line("  use /model <name> to switch")
            return False
        try:
            self.harness.set_model(argument.strip())
            self.model_name = argument.strip()
            app.notify(f"model switched to {self.harness.model_config.label}")
        except Exception as exc:
            app.notify(f"cannot switch model: {exc}", error=True)
        return False

    async def cmd_tools(self, app: "MiniAgentApp", argument: str = "") -> bool:
        harness = self.harness
        app.emit_line("")
        for name in harness.tool_runtime.visible_tools():
            spec = harness.tool_registry.get(name)
            app.emit_line(f"  {name:<12} {spec.description.splitlines()[0]}")
        for schema in harness.context_builder.tool_schemas:
            function = schema.get("function", {})
            if function.get("name") in ("load_skill", "delegate"):
                description = function["description"].splitlines()[0][:90]
                app.emit_line(f"  {function['name']:<12} {description}")
        return False

    async def cmd_skills(self, app: "MiniAgentApp", argument: str = "") -> bool:
        harness = self.harness
        loaded = set(_loaded_skills(self.history))
        app.emit_line("")
        for name in harness.skill_registry.names():
            metadata = harness.skill_registry.get(name)
            mark = "*" if name in loaded else " "
            app.emit_line(f"  {mark} {name}  {metadata.description}")
        return False

    async def cmd_agents(self, app: "MiniAgentApp", argument: str = "") -> bool:
        harness = self.harness
        app.emit_line("")
        for name in harness.subagent_registry.names():
            spec = harness.subagent_registry.get(name)
            app.emit_line(f"  {name}  {spec.description}")
            app.emit_line(f"      tools: {', '.join(spec.tools) or '(none)'}")
        return False

    async def cmd_permissions(self, app: "MiniAgentApp", argument: str = "") -> bool:
        """Show the permission posture and the grants the user has given.

        ``/permissions clear`` drops the session grants, so "allow this session"
        can be taken back without restarting the agent.
        """

        harness = self.harness
        stack = harness.permission
        if stack is None:  # pragma: no cover - defensive
            app.notify("permissions are not wired into this harness", error=True)
            return False

        action = (argument or "").strip().lower()
        if action in ("clear", "reset"):
            count = len(stack.memory.session.rules())
            stack.memory.clear_session()
            app.notify(f"cleared {count} session permission grant(s)")
            return False

        policy = stack.policy
        app.emit_line("")
        app.emit_line(f"  mode           {policy.mode}")
        app.emit_line(f"  default        {policy.default.value}")
        app.emit_line(f"  rules          {len(policy.rules)}")
        for rule in policy.rules:
            app.emit_line(f"      {rule.describe()}")
        if policy.sandbox_deny:
            app.emit_line(f"  protected      {', '.join(r.target for r in policy.sandbox_deny)}")
        ceiling = policy.skill_ceiling()
        if ceiling["tools"]:
            app.emit_line(
                f"  skill ceiling  {ceiling['name']}: {', '.join(ceiling['tools'])}"
            )
        session = stack.memory.session.rules()
        app.emit_line(f"  session grants {len(session)}")
        for rule in session:
            app.emit_line(f"      {rule.describe()}")
        persistent = stack.memory.persistent
        if persistent is not None:
            state = "on" if persistent.enabled else "disabled"
            app.emit_line(f"  saved grants   {state} ({persistent.path})")
            if persistent.unsafe_reason:
                app.emit_line(f"      refused: {persistent.unsafe_reason}")
            for rule in persistent.rules():
                app.emit_line(f"      {rule.describe()}")
        else:
            app.emit_line("  saved grants   not enabled (permissions.persistent: false)")
        app.emit_line("  /permissions clear   drop the session grants")
        return False

    async def cmd_compact(self, app: "MiniAgentApp", argument: str = "") -> bool:
        app.notify(await self.compact_now())
        return False

    async def cmd_clear(self, app: "MiniAgentApp", argument: str = "") -> bool:
        self.history = []
        self.turn += 1
        harness = self.harness
        harness.thread_id = f"thread-{self.turn}-{os.getpid() % 10000}"
        if harness.orchestrator is not None:
            harness.orchestrator.thread_id = harness.thread_id
        app.state.history_cells.clear()
        app.state.active_cell = None
        app.state.expanded_tool_ids.clear()
        app.renderer.reset_history_tracking()
        app.scroll_to_bottom()
        app.notify(f"conversation cleared (new thread {harness.thread_id})")
        return False

    async def cmd_exit(self, app: "MiniAgentApp", argument: str = "") -> bool:
        return True


def _loaded_skills(messages: list[Message]) -> list[str]:
    found: list[str] = []
    for message in messages:
        if message.role == "assistant":
            for call in message.tool_calls:
                if call.name == "load_skill":
                    name = str(call.arguments.get("name") or "").strip()
                    if name and name not in found:
                        found.append(name)
    return found


# -------------------------------------------------------------------- the app
@dataclass
class MiniAgentApp:
    """Wires composer + cells + renderer + session into one event loop."""

    session: Session
    renderer: Renderer = field(default_factory=Renderer)
    state: AppState = field(default_factory=AppState)
    stream: bool = True
    show_events: bool = False
    registry: CommandRegistry = field(init=False)
    composer: Composer = field(init=False)
    #: the interactive approval provider (permission layer, ``ASK`` branch)
    approval: "InteractiveApprovalProvider" = field(init=False)
    running: bool = False
    _tool_cells: dict[str, ToolCell] = field(default_factory=dict)
    _subagent_cell: SubAgentCell | None = None
    _last_rendered_active: str = ""
    _final_cell: AssistantCell | None = None
    _assistant_cell: AssistantCell | None = None
    _ui_app: Any = field(default=None, init=False)
    _ui_turn_task: asyncio.Task | None = field(default=None, init=False)
    _transcript_pane: TranscriptPane | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.registry = CommandRegistry()
        self.composer = Composer(
            state=self.state.composer,
            popup=CommandPopup(self.registry, self.state.command_popup),
        )
        # The session may already carry a provider chosen for the front end that is
        # actually running (plain stdin, or a fail-closed one-shot run).  Only
        # install the key-driven provider when nothing was chosen - otherwise a
        # headless run would wait forever for a key press.
        supplied = getattr(self.session, "approval_provider", None) if self.session else None
        self.approval = (
            supplied
            if supplied is not None
            else InteractiveApprovalProvider(
                state=self.state,
                on_change=self._invalidate_ui,
                activity=self._set_activity,
            )
        )
        # A key-driven question only works while the application loop is running;
        # this is how the app tells its provider whether that is the case.
        answerable = getattr(self.approval, "_answerable", None)
        if answerable is not None:
            self.approval._answerable = lambda: self._ui_app is not None
        harness = getattr(self.session, "harness", None) if self.session is not None else None
        if harness is not None:
            harness.approval_provider = self.approval
        elif self.session is not None:
            # The harness does not exist until the session is entered, so hand
            # the provider to the session and let it build with it.
            self.session.approval_provider = self.approval

    def _set_activity(self, text: str) -> None:
        self.state.activity = text

    async def setup(self) -> None:
        handlers = await self.session.command_handlers()
        self.registry = build_default_registry(handlers)
        self.composer.popup = CommandPopup(self.registry, self.state.command_popup)
        self.composer.sync_popups()

    # --------------------------------------------------------------- event sink
    def emit(self, event: ui.AgentEvent) -> None:
        """Consume one AgentEvent: mutate state and cells (render happens after)."""

        if self.show_events:
            self.renderer.print(f"[dim]· {type(event).__name__} {_brief(event)}[/dim]")

        if isinstance(event, ui.TurnStarted):
            self._assistant_cell = None
            self._final_cell = None
            self.state.activity = "starting task"
            user_cell = UserCell(text=event.prompt)
            self.state.append_cell(user_cell)
            if self._ui_app is None:
                self.renderer.render_cell(user_cell)

        elif isinstance(event, ui.AssistantStarted):
            # A new assistant message: create the cell once and keep updating it
            # for the rest of the stream.  No other path may append another.
            self._drop_empty_assistant_cell()
            self.state.activity = f"iteration {event.iteration}: waiting for model"
            # a finished message that came before this one (text the model wrote
            # alongside tool calls) belongs in the transcript: render it now
            self._render_completed_assistant()
            if self._ui_app is None:
                self.renderer.begin_stream()
            self._assistant_cell = AssistantCell(message_id=event.message_id)
            self.state.append_cell(self._assistant_cell)

        elif isinstance(event, ui.AssistantDelta):
            self.state.activity = "receiving assistant response"
            if self._ui_app is None:
                self.renderer.push_delta(event.text)
            if self._assistant_cell is None:
                self._assistant_cell = AssistantCell(message_id=event.message_id)
                self.state.append_cell(self._assistant_cell)
            self._assistant_cell.append_delta(event.text)

        elif isinstance(event, ui.AssistantFinished):
            # Completion only marks the streaming cell done; it never appends or
            # renders a second copy of the same answer.
            self._finalize_pending_assistant(event.text, event.reasoning, event.message_id)

        elif isinstance(event, ui.ToolStarted):
            # an assistant message that only produced tool calls leaves an empty
            # cell behind; the turn has clearly moved on, so drop it here
            self._drop_empty_assistant_cell()
            self.state.activity = f"running {event.tool}"
            # identity is the call id: "requested" and "started" are the SAME
            # entity, so the second event updates the existing cell instead of
            # creating another one (that produced duplicated pending lines).
            key = event.call_id or f"anon-{len(self._tool_cells)}"
            cell = self._tool_cells.get(key)
            if cell is None:
                cell = ToolCell(call_id=key, tool=event.tool, arguments=event.arguments)
                self._tool_cells[key] = cell
                self.state.set_active(cell)
                if self._ui_app is None:
                    self.renderer.render_cell(cell)  # announced as soon as it starts
            else:
                if event.arguments and not cell.arguments:
                    cell.arguments = event.arguments
                if event.tool and not cell.tool:
                    cell.tool = event.tool

        elif isinstance(event, ui.ToolOutput):
            cell = self._tool_cells.get(event.call_id)
            if cell is not None:
                cell.append(event.text)

        elif isinstance(event, ui.ToolFinished):
            cell = self._tool_cells.get(event.call_id)
            if cell is None:
                cell = ToolCell(call_id=event.call_id, tool=event.tool or "tool")
            cell.finish(event.ok, text=event.text, duration_ms=event.duration_ms)
            self.state.commit(cell)
            self.state.activity = "waiting for model"
            if self._ui_app is None:
                self._render_tool_result(cell, key=event.call_id)

        elif isinstance(event, ui.ToolFailed):
            cell = self._tool_cells.get(event.call_id)
            if cell is None:
                cell = ToolCell(call_id=event.call_id, tool=event.tool or "tool")
            cell.fail(event.error)
            self.state.commit(cell)
            self.state.activity = "waiting for model"
            if self._ui_app is None:
                self._render_tool_result(cell, key=event.call_id)

        elif isinstance(event, ui.SkillLoaded):
            self.state.append_cell(SkillCell(name=event.name, ok=event.ok))

        elif isinstance(event, ui.SubAgentStarted):
            cell = SubAgentCell(agent=event.agent, task=event.task)
            self._subagent_cell = cell
            self.state.set_active(cell)
            from rich.text import Text

            if self._ui_app is None:
                line = Text()
                line.append("⇢ ", style="blue")
                line.append(event.agent, style="bold blue")
                line.append(f"  {event.task.splitlines()[0][:100] if event.task else ''}", style="dim")
                self.renderer.print(line)
                self.renderer.print("[dim]  … working[/dim]")

        elif isinstance(event, ui.SubAgentFinished):
            cell = self._subagent_cell or SubAgentCell(agent=event.agent)
            cell.ok = event.ok
            cell.iterations = event.iterations
            cell.summary = event.summary or cell.summary
            cell.status = ToolStatus.DONE if event.ok else ToolStatus.FAILED
            self.state.commit(cell)
            self._subagent_cell = None
            # the compact result the subagent returned to the main agent
            if self._ui_app is None:
                self.renderer.render_cell(cell)

        elif isinstance(event, ui.Compacted):
            if event.folded:
                self.state.append_cell(InfoCell(message=f"◆ compacted {event.folded} message(s)"))
        elif isinstance(event, ui.ContextUsage):
            if self.show_events:
                self.state.append_cell(InfoCell(message=f"context {event.tokens} tokens"))

        elif isinstance(event, ui.TurnFinished):
            self.state.activity = ""
            if self._ui_app is not None:
                self._invalidate_ui()
                return
            cell = self._final_cell
            self._final_cell = None
            self._assistant_cell = None
            if cell is not None and not cell.empty:
                # print the messages that came before the answer (text between
                # tool calls is real history), leaving the answer for the banner
                cells = self.state.history_cells
                # Render history before the answer explicitly.  The committed
                # cursor may already have advanced for tool results, so using
                # it alone can skip the user's first message.
                for history_cell in cells[:-1]:
                    if isinstance(history_cell, ToolCell):
                        continue
                    self.renderer.render_cell(history_cell)
                self.renderer.final_answer(cell, status=event.status)
                self.renderer.mark_cells_rendered(len(self.state.history_cells))

        elif isinstance(event, ui.PermissionDecided):
            self._handle_permission_decision(event)

        elif isinstance(event, ui.ErrorEvent):
            self.state.append_cell(ErrorCell(message=event.message, fatal=event.fatal))

        self._invalidate_ui()

    def _handle_permission_decision(self, event: ui.PermissionDecided) -> None:
        """Render one permission verdict.

        Denials become transcript cells so the refusal is visible even in a long
        run; approvals are quiet (the tool cell that follows is the record),
        except when the user answered a prompt - then the choice is echoed so the
        transcript shows *who* allowed it.
        """

        if event.permission == "deny":
            label = f"⛔ permission denied · {event.tool or 'tool'}"
            if event.reason:
                label += f" — {event.reason}"
            self.state.append_cell(InfoCell(message=label))
            return

        if event.approval in ("once", "session", "persistent"):
            scope = {
                "once": "once",
                "session": "for this session",
                "persistent": "always",
            }[event.approval]
            self.state.append_cell(
                InfoCell(message=f"⚠ allowed {scope} · {event.tool or 'tool'}")
            )

    def _finalize_pending_assistant(
        self, text: str | None, reasoning: str | None, message_id: str = ""
    ) -> None:
        """Mark the message that is currently streaming as complete.

        Called for *every* assistant message (including the ones that only
        accompany tool calls) and at the end of a turn, so no cell is left in a
        half-open state.  ``run_prompt`` marks the turn's last message complete,
        which is why callers may pass no text at all.
        """

        cell = self._assistant_cell
        if cell is None:
            return
        if self._ui_app is not None:
            if text:
                cell.complete_with(text)
            else:
                cell.complete = True
            cell.reasoning = reasoning or cell.reasoning
            self._drop_empty_assistant_cell()
            return
        # A message that never streamed text (the model answered with tool calls
        # only) is finalized with the text the caller knows about.
        if text:
            final_source = self.renderer.end_stream(text)
            cell.complete_with(final_source or text)
        else:
            final_source = self.renderer.end_stream()
            if final_source and not cell.source:
                cell.complete_with(final_source)
            else:
                cell.complete = True
        cell.reasoning = reasoning or cell.reasoning
        self._drop_empty_assistant_cell()

    def _render_completed_assistant(self) -> None:
        """Print the previous message once, if it has anything to say."""

        cell = self._assistant_cell
        if cell is None or not cell.complete or cell.empty:
            return
        if self._ui_app is not None:
            return
        self.renderer.render_cell(cell)

    def _drop_empty_assistant_cell(self) -> None:
        """Discard a streamed message that turned out to be empty.

        A turn whose model answer is only tool calls leaves an empty assistant
        cell behind; without this it would sit in the transcript (and be counted)
        for the rest of the session.
        """

        cell = self._assistant_cell
        if cell is None or not cell.empty:
            return
        if self.state.history_cells and self.state.history_cells[-1] is cell:
            self.state.history_cells.pop()
        self._assistant_cell = None

    def render(self) -> None:
        if self._ui_app is not None:
            self._invalidate_ui()
            return
        self.renderer.flush(self.state.history_cells)

    def _invalidate_ui(self) -> None:
        if self._ui_app is not None:
            self._ui_app.invalidate()

    def _toggle_latest_expandable(self) -> None:
        self.state.toggle_latest_expandable()
        self._invalidate_ui()

    #: backwards-compatible name
    def _toggle_latest_tool(self) -> None:
        self._toggle_latest_expandable()

    def _render_tool_result(self, cell: ToolCell, *, key: str = "") -> None:
        """Print the finished cell and drop it from the running set."""

        self._tool_cells.pop(key, None)
        self.renderer.render_cell(cell)
        # the cell is already on screen, so history tracking must skip it
        if self.state.history_cells:
            self.renderer.mark_cells_rendered(len(self.state.history_cells))

    # -------------------------------------------------------------------- turns
    def new_bridge(self) -> EventBridge:
        return EventBridge(self.emit, self.state)

    async def run_prompt(self, text: str) -> AgentState | None:
        bridge = self.new_bridge()
        harness = self.session.harness
        assert harness is not None
        harness.on_event = bridge
        harness.orchestrator.on_event = bridge

        self.emit(ui.TurnStarted(prompt=text))
        self.running = True
        on_delta = None
        if self.stream:

            def on_delta(delta: str) -> None:
                self.emit(ui.AssistantDelta(text=delta))

        try:
            result = await self.session.ask(text, on_delta=on_delta)
        except asyncio.CancelledError:
            self.emit(ui.ErrorEvent(message="interrupted by Ctrl+C"))
            self.render()
            self.running = False
            raise
        except Exception as exc:
            self.emit(ui.ErrorEvent(message=f"{type(exc).__name__}: {exc}", fatal=True))
            self.running = False
            return None
        finally:
            self.running = False

        final = result.get("final_answer") or ""
        # the per-message completion already fired; this only makes sure the last
        # message is closed even if the runtime ended in an unusual way
        self._finalize_pending_assistant(final, None)
        self._final_cell = self._assistant_cell
        self.emit(
            ui.TurnFinished(
                status=str(result.get("termination_status") or ""),
                final_answer=final,
                iterations=int(result.get("iteration") or 0),
            )
        )
        self.render()
        return result

    async def run_shell(self, command: str) -> None:
        """Run ``!command`` through the same permission gate the model's tools use.

        The gate is applied here rather than left to the tool runtime so the
        decision is visible as a trace event and, when the question cannot be
        answered, fails closed instead of blocking the turn.
        """

        harness = self.session.harness
        assert harness is not None
        intent = ShellIntent(command=command, call_id=f"user-shell-{self.session.turn}")
        self.emit(ui.TurnStarted(prompt=f"!{command}"))
        self.emit(ui.ToolStarted(call_id=intent.call_id, tool="shell", arguments={"command": command}))

        call = ToolCall(name="shell", arguments={"command": command}, id=intent.call_id)
        answered = self._approval_is_answerable()
        results = await harness.permission.gate.check_batch(
            [call], allow_when_unavailable=answered
        )
        result = results[0] if results else None
        if result is not None and not result.allowed:
            self.emit(ui.ToolFailed(call_id=intent.call_id, tool="shell", error=result.denial_message))
            self.emit(
                ui.PermissionDecided(
                    call_id=intent.call_id,
                    tool="shell",
                    permission=result.verdict.permission.value,
                    reason=result.verdict.reason,
                    source=result.verdict.source,
                    approval=result.verdict.approval,
                )
            )
            self.render()
            return

        from harness.tools.runtime import calls_already_decided

        with calls_already_decided([intent.call_id]):
            result_event = await run_shell_intent(harness.tool_runtime, intent)
        if result_event.ok:
            self.emit(result_event)
        else:
            self.emit(ui.ToolFailed(call_id=intent.call_id, tool="shell", error=result_event.text))
        self.render()

    def _approval_is_answerable(self) -> bool:
        """Whether a permission question would reach the user right now."""

        provider = self.approval
        checker = getattr(provider, "can_answer", None)
        if isinstance(checker, bool):
            return checker
        if callable(checker):
            return bool(checker())
        return False

    # ------------------------------------------------------------- intent router
    async def handle_input(self, text: str) -> bool:
        """Turn one submitted line into an intent.  Returns ``True`` to exit."""

        text = text.strip()
        if not text:
            return False
        if text.startswith("!"):
            await self.run_shell(text[1:].strip())
            return False
        if text.startswith("/"):
            name, _, argument = text[1:].partition(" ")
            command = self.registry.get(name)
            if command is None or command.handler is None:
                self.notify(f"unknown command `/{name}` — type /help", error=True)
                return False
            return bool(await command.handler(self, argument.strip()))
        await self.run_prompt(text)
        return False

    # --------------------------------------------------------------- interactive
    def create_prompt_session(self):
        """Build the prompt session.

        Kept as its own method so tests can inject a session with pipe input
        (a pseudo-terminal cannot be allocated in every environment).
        """

        from prompt_toolkit import PromptSession

        return PromptSession(
            # The completer feeds prompt_toolkit's async completion path (used by
            # ``complete_while_typing`` and Ctrl+Space); the visible candidate
            # list is the composer popup, so the native menu is given no space.
            completer=SlashCompleter(self.registry),
            key_bindings=build_key_bindings(self.composer, cancel=self._on_cancel),
            complete_while_typing=True,
            reserve_space_for_menu=0,
            bottom_toolbar=self._toolbar,
        )

    def create_application(self):
        """Build the long-lived prompt_toolkit application."""

        from prompt_toolkit.application import Application
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
        from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
        from prompt_toolkit.layout.dimension import Dimension
        from prompt_toolkit.styles import Style

        def sync_buffer(buffer) -> None:  # noqa: ANN001
            self.composer.sync_from_buffer(buffer.text)
            self.composer.sync_popups()
            if self._ui_app is not None:
                self._ui_app.invalidate()

        buffer = Buffer(
            name="miniagent-input",
            multiline=True,
            completer=SlashCompleter(self.registry),
            on_text_changed=sync_buffer,
        )

        def submit(text: str) -> None:
            text = text.strip()
            buffer.reset()
            self.composer.sync_from_buffer("")
            self.composer.sync_popups()
            self.state.command_popup.reset()
            # a new turn belongs at the newest line, even if the user was
            # reading history
            self.scroll_to_bottom()
            if text:
                if self._ui_turn_task is None:
                    self._ui_turn_task = asyncio.create_task(self._run_application_input(text))
                else:
                    # never drop a submitted line silently
                    self.notify("still working on the previous turn — input ignored (Ctrl+C aborts)")

        popup = FormattedTextControl(
            lambda: self.composer.prompt_fragments()[:-1],
            focusable=False,
            show_cursor=False,
        )
        prompt = FormattedTextControl(
            lambda: [("class:prompt", "› ")],
            focusable=False,
            show_cursor=False,
        )
        input_control = BufferControl(buffer=buffer)
        transcript = TranscriptControl(self.state, final_cell=lambda: self._final_cell)
        pane = TranscriptPane(Window(transcript, wrap_lines=True))
        transcript.pane = pane
        self._transcript_pane = pane

        def scroll_history(action: str) -> None:
            if action == "page-up":
                pane.page(-1)
            elif action == "page-down":
                pane.page(1)
            elif action == "wheel-up":
                pane.scroll_lines(-3)  # towards older content
            elif action == "wheel-down":
                pane.scroll_lines(3)  # towards the newest line
            elif action == "top":
                pane.to_top()
            else:
                pane.to_bottom()
            self._invalidate_ui()

        status = FormattedTextControl(self._status_fragments, focusable=False, show_cursor=False)

        key_bindings = build_key_bindings(
            self.composer,
            cancel=self._on_cancel,
            on_submit=submit,
            on_toggle_tool=self._toggle_latest_expandable,
            on_history_scroll=scroll_history,
            approval=self.approval,
        )

        root = HSplit(
            [
                pane,
                Window(popup, height=Dimension(min=0, max=7), dont_extend_height=True),
                VSplit([Window(prompt, width=2), Window(input_control)]),
                Window(status, height=1, dont_extend_height=True),
            ]
        )
        style = Style.from_dict(
            {
                "prompt": "ansigreen bold",
                "final-marker": "ansigreen bold",
                "user": "ansigreen bold",
                "assistant": "",
                "reasoning": "ansibrightblack italic",
                "tool-running": "ansiyellow",
                "tool-done": "ansigreen",
                "tool-failed": "ansired bold",
                "tool-body": "ansibrightblack",
                "subagent": "ansiblue",
                "skill": "ansimagenta",
                "info": "ansibrightblack",
                "error": "ansired",
                "error-fatal": "ansired bold",
                "transcript": "",
                "transcript-dim": "ansibrightblack",
                "activity": "ansibrightblack italic",
                "status": "ansibrightblack",
                "status-hint": "ansibrightblack",
                "scrollbar.background": "bg:#3a3a3a",
                "scrollbar.button": "bg:#8a8a8a",
            }
        )
        return Application(
            layout=Layout(root, focused_element=input_control),
            key_bindings=key_bindings,
            style=style,
            full_screen=False,
            # the wheel scrolls the transcript; Shift+drag still selects text in
            # terminals that support it
            mouse_support=True,
            # no idle repaint: every agent event invalidates the UI itself
            refresh_interval=None,
        )

    def _status_fragments(self):  # noqa: ANN202 - prompt_toolkit accepts a callable
        """One-line status bar: model, run state and the scroll hint."""

        harness = self.session.harness
        model = harness.model_config.model if harness is not None else "?"
        run_state = "running" if self.running else "idle"
        parts = [("class:status", f" {BANNER_AGENT} · {model} · {run_state}")]
        pane = self._transcript_pane
        # A permission question takes over the status line: it is the one thing
        # that blocks the turn, so it must never be scrolled out of sight.
        if self.approval.waiting:
            parts.extend(self.approval.status_fragments())
            parts.append(("class:approval-dim", "  Enter: confirm · ←/→: move · Ctrl+R: reject"))
            return parts
        if pane is not None and pane.scrolled_up_by:
            parts.append(
                (
                    "class:status-hint",
                    f"   ↑ {pane.scrolled_up_by} line(s) above the latest · Ctrl+End jumps back",
                )
            )
        else:
            parts.append(
                ("class:status-hint", "   PageUp/wheel: history · Ctrl+O: expand · Ctrl+C: interrupt")
            )
        return parts

    def scroll_to_bottom(self) -> None:
        """Follow the newest line again (new turn, /clear, Ctrl+End)."""

        if self._transcript_pane is not None:
            self._transcript_pane.to_bottom()
            self._invalidate_ui()

    def notify(self, message: str, *, error: bool = False) -> None:
        """Show a one-line message where the user can actually see it.

        Inside the prompt application that means a transcript cell; in batch or
        ``--plain`` mode it means a line on stdout.
        """

        if self._ui_app is not None:
            self.state.append_cell(ErrorCell(message=message) if error else InfoCell(message=message))
            self._invalidate_ui()
        elif error:
            self.renderer.error(message)
        else:
            self.renderer.info(message)

    def emit_line(self, text: str = "") -> None:
        """One line of command output: a transcript cell, or a printed line.

        Command output used to be written straight to stdout, which put it
        *above* the application region: it could not be scrolled to, searched or
        cleared.  Inside the UI it now becomes part of the transcript.
        """

        if self._ui_app is not None:
            self.state.append_cell(InfoCell(message=text))
            self._invalidate_ui()
        else:
            self.renderer.print(text)

    async def _run_application_input(self, text: str) -> None:
        try:
            if await self.handle_input(text) and self._ui_app is not None:
                self._ui_app.exit()
        except asyncio.CancelledError:
            self.notify("turn aborted")
            raise
        except KeyboardInterrupt:
            self.state.append_cell(InfoCell(message="interrupted"))
        except Exception as exc:  # pragma: no cover - defensive
            self.state.append_cell(ErrorCell(message=f"{type(exc).__name__}: {exc}", fatal=True))
        finally:
            self._ui_turn_task = None
            self._invalidate_ui()

    async def prompt_loop(self, *, prompt_session=None) -> int:
        # Keep the injectable PromptSession path for pipe-driven tests and
        # embedders; real terminals use the long-lived application below.
        if prompt_session is not None or "create_prompt_session" in self.__dict__:
            return await self._legacy_prompt_loop(prompt_session)

        from prompt_toolkit.patch_stdout import patch_stdout

        self.renderer.banner()
        application = self.create_application()
        self._ui_app = application
        try:
            with patch_stdout(raw=True):
                await application.run_async()
        except (PromptInterrupt, EOFError):
            # Ctrl+C on an empty line / Ctrl+D: a clean exit, not a traceback
            return 0
        finally:
            if self._ui_turn_task is not None and not self._ui_turn_task.done():
                self._ui_turn_task.cancel()
            self._ui_app = None
            self._transcript_pane = None
        return 0

    async def _legacy_prompt_loop(self, prompt_session=None) -> int:
        from prompt_toolkit.formatted_text import FormattedText
        from prompt_toolkit.patch_stdout import patch_stdout

        prompt_session = prompt_session or self.create_prompt_session()

        def message():  # noqa: ANN202 - prompt_toolkit accepts a callable
            return FormattedText(self.composer.prompt_fragments())

        with patch_stdout(raw=True):
            while True:
                try:
                    text = await prompt_session.prompt_async(message)
                except (PromptInterrupt, KeyboardInterrupt):
                    return 0
                except EOFError:
                    return 0

                text = (text or "").strip()
                self.composer.sync_from_buffer("")
                self.composer.sync_popups()
                self.state.command_popup.reset()
                if not text:
                    continue
                if await self.handle_input(text):
                    return 0

    def _on_cancel(self) -> None:
        """Ctrl+C while the agent is running aborts the turn; otherwise it is
        just a cleared input line."""

        # A question that belonged to the turn being aborted must not survive it,
        # or the status line would keep asking about work that no longer runs.
        cancel_approval = getattr(self.approval, "cancel", None)
        if callable(cancel_approval):
            cancel_approval()

        task = self._ui_turn_task
        if self.running and task is not None and not task.done():
            task.cancel()
            self.notify("aborting the current turn…")
        else:
            self.notify("input cleared")

    #: the turn task, whichever attribute the caller set
    @property
    def turn_task(self) -> asyncio.Task | None:
        return self._ui_turn_task

    def _toolbar(self):  # noqa: ANN202 - prompt_toolkit accepts a callable
        from prompt_toolkit.formatted_text import FormattedText

        harness = self.session.harness
        model = harness.model_config.model if harness else "?"
        state = "running" if self.running else "idle"
        return FormattedText([("fg:ansibrightblack", f" {BANNER_AGENT} · {model} · {state} ")])

    # --------------------------------------------------------------------- batch
    async def run_batch(self, task: str) -> int:
        result = await self.run_prompt(task)
        if result is None:
            return 1
        status = result.get("termination_status")
        if status and status != "final_answer":
            self.renderer.info(f"(stopped: {status})")
        return 0


# ------------------------------------------------------------------- entrypoint
async def async_main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.version:
        from harness import __version__

        print(f"{BANNER_AGENT} {__version__}")
        return 0

    from harness.infra.logging import configure_logging

    config = load_config(args)
    configure_logging(config.logging.level, logfile=config.logging.file, json=config.logging.json_output)

    # Which front end will run decides how (and whether) a permission question can
    # be answered.  A piped one-shot task has nobody to ask, and guessing would be
    # worse than refusing: the gate fails closed with an explicit reason.
    batch = bool(args.task)
    plain = args.plain or not sys.stdin.isatty()
    approver = None
    if not batch and not plain:
        approver = None  # MiniAgentApp installs the interactive provider
    elif not batch and plain:
        approver = PlainApprovalProvider()
    else:
        from harness.permission import AutoDenyProvider

        approver = AutoDenyProvider(
            note=(
                "this one-shot run cannot ask for approval; re-run it "
                "interactively, or allow the action in config.yaml"
            )
        )

    app = MiniAgentApp(
        session=Session(config, model=args.model, approval_provider=approver),
        renderer=Renderer(),
        stream=not args.no_stream,
        show_events=args.show_events,
    )
    await app.setup()

    async with app.session:
        if args.task:
            return await app.run_batch(" ".join(args.task))
        if plain:
            return await _plain_loop(app)
        return await app.prompt_loop()


async def _plain_loop(app: MiniAgentApp) -> int:
    """Line-based fallback (pipes, ``--plain``, CI): no prompt_toolkit."""

    app.renderer.banner()
    while True:
        try:
            line = await asyncio.to_thread(input, "› ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        try:
            if await app.handle_input(line):
                app.render()
                return 0
        except KeyboardInterrupt:
            app.renderer.info("interrupted")
        except Exception as exc:
            app.renderer.error(f"{type(exc).__name__}: {exc}")
        # A turn that ends in a fatal ErrorEvent appends a cell but never reaches
        # the TurnFinished render path, so flush here or the user sees nothing.
        app.render()


def _brief(event: ui.AgentEvent) -> str:
    for attribute in ("tool", "agent", "name", "status", "message"):
        value = getattr(event, attribute, "")
        if value:
            return str(value)[:60]
    return ""


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(async_main(argv))
    except KeyboardInterrupt:  # pragma: no cover - user interrupt
        return 130


__all__ = [
    "main",
    "async_main",
    "MiniAgentApp",
    "Session",
    "build_arg_parser",
    "load_config",
]
