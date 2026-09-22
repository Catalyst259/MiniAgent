"""Conversation session and slash-command handlers for the CLI."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from harness.agent.dto import Message
from harness.agent.state import AgentState, new_state
from harness.infra.checkpoint import AsyncCheckpointStore
from harness.infra.config import HarnessConfig
from harness.tools.native import NATIVE_TOOL_NAMES

if TYPE_CHECKING:
    from harness.cli.app import MiniAgentApp


class Session:
    """One CLI session: conversation history, harness and commands."""

    def __init__(
        self,
        config: HarnessConfig,
        *,
        model: str | None = None,
        on_event=None,
        approval_provider=None,
        interaction_provider=None,
    ) -> None:
        self.config = config
        self.model_name = model or config.default_model
        self.history: list[Message] = []
        self.turn = 0
        self.on_event = on_event
        self.harness = None
        self.approval_provider = approval_provider
        self.interaction_provider = interaction_provider
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
            interaction_provider=self.interaction_provider,
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
        self.turn -= 1
        outcome = await self.harness.context_manager.compact(state)
        if not outcome.applied:
            return "Nothing to compact yet."
        self.history = [
            Message(role="user", content=f"[Context compacted]\n{outcome.summary}")
        ] + outcome.tail[len(self.history) :]
        return f"Compacted {outcome.folded} message(s) into {len(outcome.summary)} chars."

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
            if function.get("name") in NATIVE_TOOL_NAMES:
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
        harness = self.harness
        stack = harness.permission
        if stack is None:
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
            app.emit_line(f"  skill ceiling  {ceiling['name']}: {', '.join(ceiling['tools'])}")
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


__all__ = ["Session"]
