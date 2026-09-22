"""``AgentHarness``: assembling a runnable terminal agent.

Everything above the infra layer is injected here: the model gateway, the tool
registry/runtime, the skill loader, the subagent runtime, the memory service and
the context manager.  Nothing else in the codebase constructs a concrete
adapter, which is what keeps the dependency direction clean.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Callable

from harness.agent.dto import Message, ModelRequest
from harness.agent.state import AgentState, new_state
from harness.agent.termination import TerminationPolicy
from harness.context.builder import ContextBuilder
from harness.context.compact import CompactService
from harness.context.manager import ContextManager
from harness.context.token_budget import TokenBudgetPolicy
from harness.inference.config import ModelConfig
from harness.inference.openai_compatible import OpenAICompatibleGateway
from harness.inference.tokenizer import build_token_counter
from harness.infra.config import HarnessConfig
from harness.memory.embedding import build_embedding_backend
from harness.memory.service import MemoryService
from harness.memory.store import build_memory_store
from harness.memory.summary import SummaryService
from harness.orchestration.graph import Orchestrator, fresh_thread_id
from harness.permission import build_permission_stack
from harness.skills.loader import SkillLoader
from harness.skills.registry import SkillRegistry
from harness.subagents.registry import SubAgentRegistry
from harness.subagents.runtime import SubAgentRuntime
from harness.tools.fs_tools import ToolContext
from harness.tools.local_backend import LocalToolBackend
from harness.tools.native import native_tool_schemas
from harness.tools.paths import Workspace
from harness.tools.registry import ToolRegistry
from harness.tools.runtime import ToolRuntime

log = logging.getLogger(__name__)

READ_ONLY_CEILING = ("list_dir", "glob", "grep", "read_file", "git_diff")

SUBAGENT_PREAMBLE = """You are the `{name}` subagent of a terminal coding agent.

{body}

Operating constraints:
- You received only the task and the context below; you cannot see the parent conversation.
- Your maximum tool/model loop is {max_iterations} iterations. Use the budget deliberately,
  stop exploring before it is exhausted, and return a concise partial result if needed.
- Stay inside the workspace. Report file paths and line numbers, not large code dumps.
- End with the required section headings of your contract, filled in.
- Be concise: the main agent has a limited context budget.
"""


class AgentHarness:
    """A complete harness instance for one model configuration."""

    def __init__(
        self,
        config: HarnessConfig,
        *,
        model_name: str | None = None,
        workspace_root: str | Path | None = None,
        on_event: Callable | None = None,
        thread_id: str | None = None,
        harness_factory: Callable[..., "AgentHarness"] | None = None,
        approval_provider: Any = None,
        interaction_provider: Any = None,
        permission_memory: Any = None,
    ) -> None:
        self.config = config
        self.model_name = model_name or config.default_model
        self.model_config: ModelConfig = config.resolve_model(self.model_name)
        self.workspace = Workspace(workspace_root or config.workspace_root)
        self.on_event = on_event
        self.thread_id = thread_id or fresh_thread_id()
        self._harness_factory = harness_factory or (lambda **kwargs: AgentHarness(config, **kwargs))

        # services (populated by build())
        self.orchestrator: Orchestrator | None = None
        self.memory: MemoryService | None = None
        self.summary: SummaryService | None = None
        self.skill_registry: SkillRegistry | None = None
        self.skill_loader: SkillLoader | None = None
        self.subagent_registry: SubAgentRegistry | None = None
        self.subagent_runtime: SubAgentRuntime | None = None
        self.tool_registry: ToolRegistry | None = None
        self.tool_runtime: ToolRuntime | None = None
        self.permission: Any = None
        self.context_manager: ContextManager | None = None
        self.context_builder: ContextBuilder | None = None
        self.gateway: Any = None
        self._mcp_backend: Any = None
        self._built = False

        #: set for the duration of a turn so assistant text can stream to the UI
        self.delta_callback: Any = None

        # optional overrides applied by build(); used for subagent isolation
        self.system_prompt_override: str | None = None
        self.tool_allowlist: list[str] | None = None
        self.skill_allowlist: list[str] | None = None
        self.allow_delegation: bool = True
        self.use_memory: bool = True

        #: Permission layer.  The provider decides the ``ASK`` branch; when it is
        #: left unset the gate fails closed (deny) whenever it needs an answer.
        self.approval_provider: Any = approval_provider
        #: Optional front-end adapter for model-requested user choices.  When it
        #: is absent the tool schema is omitted, so an isolated agent cannot ask
        #: a question nobody can answer.
        self.interaction_provider: Any = interaction_provider
        #: Permission memory shared with the session (and its subagents), so an
        #: "allow this session" answer is not asked again by a child agent.
        self.permission_memory: Any = permission_memory
        #: A subagent inherits the parent's policy but may never ask questions:
        #: its events do not reach the CLI, so an unanswered prompt would hang.
        self.permission_isolated: bool = False
        #: Extra permission ceiling applied on top of the config (subagents set
        #: this to read-only, so a mis-configured child still cannot write).
        self.permission_read_only: bool = False

    # -------------------------------------------------------------------- build
    def build(self) -> "AgentHarness":
        if self._built:
            return self
        config = self.config

        # ---- inference -----------------------------------------------------
        self.gateway = build_gateway(self.model_config, name=self.model_name)

        # ---- skills --------------------------------------------------------
        self.skill_registry = SkillRegistry(config.skill_paths())
        self.skill_registry.discover()
        self.skill_loader = SkillLoader(self.skill_registry)
        skill_catalog = self.skill_loader.catalog() if config.skills.enabled else ""

        # ---- tools ---------------------------------------------------------
        if config.tools.transport == "mcp-stdio":
            from harness.tools.mcp_client import build_stdio_backend

            self._mcp_backend = build_stdio_backend(self.workspace.root)
            backend = self._mcp_backend
        else:
            backend = LocalToolBackend(
                ToolContext(
                    workspace=self.workspace,
                    max_output_chars=config.tools.max_output_chars,
                    shell_timeout=config.tools.shell_timeout_seconds,
                    shell_max_output_chars=config.tools.shell_max_output_chars,
                    command_allowlist=tuple(config.tools.command_allowlist),
                    command_denylist=tuple(config.tools.command_denylist),
                )
            )
        self.tool_registry = ToolRegistry()
        self.tool_registry.register_backend(backend)

        # ---- subagents -----------------------------------------------------
        self.subagent_registry = SubAgentRegistry(config.subagent_paths())
        self.subagent_registry.discover()

        # ---- memory --------------------------------------------------------
        # Subagents are context-isolated children.  When memory is disabled for
        # a child, do not even construct the local Qdrant store: construction
        # must not contend for the parent's on-disk memory lock.
        self.memory = build_memory_service(config) if self.use_memory else None

        # ---- tools / skills visible to the main agent ----------------------
        enabled = [name for name in config.tools.enabled if name in set(self.tool_registry.names())]
        if self.tool_allowlist is not None:
            enabled = [name for name in enabled if name in set(self.tool_allowlist)]
        subagent_mode = self.system_prompt_override is not None
        subagent_runtime = SubAgentRuntime(
            self.subagent_registry,
            self._subagent_factory,
            enabled=config.subagents.enabled and self.allow_delegation and not subagent_mode,
        )
        self.tool_runtime = ToolRuntime(
            self.tool_registry,
            allowed=enabled,
            permission_gate=None,  # wired below, once the permission stack exists
            max_output_chars=config.tools.max_output_chars,
            timeout_seconds=config.runtime.tool_timeout_seconds,
        )

        # ---- permissions ---------------------------------------------------
        # One gate per harness.  A subagent gets the parent's memory (so a
        # session grant is not re-asked by a child) but never a prompt.
        from harness.permission import AutoDenyProvider

        approver = self.approval_provider
        if approver is None and self.permission_isolated:
            approver = AutoDenyProvider(
                note="subagents run without an interactive approver"
            )
        if approver is None:
            # No UI was wired up at all: fail closed rather than let an ASK
            # verdict look like something that will eventually be answered.
            approver = AutoDenyProvider(note="no approval UI is attached to this harness")
        self.permission = build_permission_stack(
            config,
            workspace=self.workspace,
            approver=approver,
            persistent_path=str(config.approvals_path()),
        )
        if self.permission_memory is not None:
            self.permission.gate.evaluator.memory = self.permission_memory
            self.permission.memory = self.permission_memory
        if self.permission_read_only:
            self.permission.policy.read_only = True
            self.permission.evaluator.policy.read_only = True
        # The execution boundary enforces the same gate: a tool call that never
        # reached the gate node is decided here instead of running unchecked.
        self.tool_runtime.permission_gate = self.permission.gate

        native = native_tool_schemas(
            delegate_schema=subagent_runtime.tool_schema() if subagent_runtime.enabled else None,
            skills_available=config.skills.enabled and bool(self.skill_registry.names()),
            interaction_available=bool(
                self.interaction_provider is not None
                and getattr(self.interaction_provider, "available", False)
            ),
        )
        extra_schemas = native if config.tools.include_native_tools else []
        tool_schemas = self.tool_runtime.schemas(extra=extra_schemas)
        tool_catalog = self.tool_registry.render_catalog(only=enabled)
        if extra_schemas:
            tool_catalog += "\n" + "\n".join(
                f"- {schema['function']['name']}: {schema['function']['description'].splitlines()[0]}"
                for schema in extra_schemas
            )

        # ---- context -------------------------------------------------------
        counter = build_token_counter("auto", model=self.model_config.model)
        use_memory = self.use_memory and self.memory is not None and self.memory.enabled
        self.context_builder = ContextBuilder(
            model_config=self.model_config,
            workspace_root=str(self.workspace.root),
            tool_catalog=tool_catalog,
            skill_catalog=skill_catalog,
            subagent_catalog=self.subagent_registry.catalog(),
            tool_schemas=tool_schemas,
            extra_rules=config.system_prompt_extra,
            base_prompt=self.system_prompt_override,
            memory_provider=self.memory.as_provider() if use_memory else None,
            memory_top_k=config.memory.top_k,
            max_recent_messages=config.context.max_recent_messages,
            max_iterations=(
                config.subagents.max_iterations if subagent_mode else config.runtime.max_iterations
            ),
            delta_callback=lambda delta: self._forward_delta(delta),
            budget=self.context_manager.usage if self.context_manager else None,
        )
        policy = TokenBudgetPolicy(
            max_input_tokens=config.context.max_input_tokens,
            reserve_output_tokens=config.context.reserve_output_tokens,
            compact_trigger_ratio=config.context.compact_trigger_ratio,
        )
        compact_service = CompactService(
            self.gateway,
            keep_recent_messages=config.context.keep_recent_messages,
            target_summary_tokens=config.context.target_summary_tokens,
        )
        self.context_manager = ContextManager(
            self.context_builder,
            counter,
            policy,
            compact_service,
            keep_recent_messages=config.context.keep_recent_messages,
        )

        # ---- subagent runtime ---------------------------------------------
        self.subagent_runtime = subagent_runtime

        # ---- summary -------------------------------------------------------
        # The service always exists so that Compact and Summary stay separate
        # concepts; writing is gated by the memory config inside the service.
        self.summary = SummaryService(
            self.gateway,
            self.memory,
            max_facts=config.memory.max_facts,
        ) if self.memory else None

        # ---- orchestration -------------------------------------------------
        self.orchestrator = Orchestrator(
            gateway=self.gateway,
            context_manager=self.context_manager,
            tool_runtime=self.tool_runtime,
            skill_loader=self.skill_loader,
            subagent_runtime=self.subagent_runtime,
            interaction_provider=self.interaction_provider,
            permission_gate=self.permission.gate if self.permission else None,
            skill_tool_resolver=self._skill_tool_ceiling,
            termination_policy=TerminationPolicy(
                max_iterations=(
                    config.subagents.max_iterations if subagent_mode else config.runtime.max_iterations
                ),
                max_repeated_tool_calls=config.runtime.max_repeated_tool_calls,
                max_consecutive_tool_errors=config.runtime.max_consecutive_tool_errors,
            ),
            on_event=self.on_event,
            thread_id=self.thread_id,
            recursion_limit=config.runtime.recursion_limit,
        )
        self.orchestrator.build()
        self._built = True
        return self

    # ------------------------------------------------------------- subagent glue
    def _subagent_factory(self, spec, task: str):  # noqa: ANN001 - SubAgentSpec
        """Build a context-isolated, tool-restricted harness for a subagent."""

        allowed = [name for name in spec.tools if name in set(READ_ONLY_CEILING)]
        if not allowed:
            allowed = list(READ_ONLY_CEILING)
        child = self._harness_factory(
            model_name=self.model_name,
            workspace_root=self.workspace.root,
            # A subagent is a context-isolated child tree.  Its internal
            # iterations and tool events must not leak into the parent CLI;
            # the parent delegate node emits only start/end summary events.
            on_event=None,
            thread_id=f"sub-{spec.name}-{uuid.uuid4().hex[:6]}",
            # Permission isolation: a child shares the session's remembered
            # grants (so "allow this session" is not re-asked) but can never
            # raise its own prompt - its events do not reach the user.
            permission_memory=self.permission_memory
            if self.permission_memory is not None
            else (self.permission.memory if self.permission else None),
        )
        child.permission_isolated = True
        child.permission_read_only = True
        child.system_prompt_override = SUBAGENT_PREAMBLE.format(
            name=spec.name,
            body=spec.body,
            max_iterations=self.config.subagents.max_iterations,
        )
        child.tool_allowlist = allowed
        child.skill_allowlist = list(spec.skills)
        child.allow_delegation = False
        child.use_memory = False
        child.build()
        return child

    def _forward_delta(self, delta: str) -> None:
        callback = self.delta_callback
        if callback is not None:
            callback(delta)

    def _skill_tool_ceiling(self, loaded: list[str]) -> tuple[list[str], str]:
        """The tools the most recently loaded skill declares.

        Design document section 12: a skill states which tools it relies on, and
        that statement is a *ceiling*.  The newest loaded skill wins, because it is
        the instructions the model is following right now.
        """

        if self.skill_registry is None:
            return ([], "")
        for name in reversed(list(loaded or [])):
            try:
                metadata = self.skill_registry.get(name)
            except Exception:  # noqa: BLE001 - unknown skill: no ceiling
                continue
            if metadata.tools:
                return (list(metadata.tools), metadata.name)
        return ([], "")

    # -------------------------------------------------------------------- state
    def initial_state(self, task: str, *, thread_id: str | None = None, task_id: str | None = None) -> AgentState:
        if thread_id:
            self.thread_id = thread_id
            if self.orchestrator:
                self.orchestrator.thread_id = thread_id
        return new_state(
            task,
            thread_id=self.thread_id,
            task_id=task_id or uuid.uuid4().hex[:8],
        )

    # ------------------------------------------------------------------ running
    async def run(
        self,
        task: str,
        context: str | None = None,
        *,
        state: AgentState | None = None,
        thread_id: str | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> AgentState:
        """Run one task to completion and return the final state.

        ``context`` is only used when MiniAgent acts as a subagent runner: it
        is appended to the task, never merged with a parent transcript.
        ``on_delta`` receives assistant text fragments while the model streams.
        """

        self.build()
        assert self.orchestrator is not None
        if thread_id:
            self.thread_id = thread_id
            self.orchestrator.thread_id = thread_id
        prompt = task if not context else f"{task}\n\n--- context provided by the main agent ---\n{context}"
        initial = state or self.initial_state(prompt)
        if self.memory is not None and self.memory.enabled:
            await self.memory.ensure_ready()
        previous = self.delta_callback
        self.delta_callback = on_delta
        try:
            return await self.orchestrator.ainvoke(initial)
        finally:
            self.delta_callback = previous

    async def run_as_subagent(self, task: str, context: str | None = None) -> tuple[str, int]:
        """``SubAgentRunner`` protocol implementation."""

        final = await self.run(task, context)
        return (final.get("final_answer") or "", int(final.get("iteration") or 0))

    async def finish(self, state: AgentState, *, repo: str | None = None) -> int:
        """Memory formation after a task ends (Summary, not Compact)."""

        if not (self.summary and self.config.memory.write_on_finish and self.config.memory.enabled):
            return 0
        records = await self.summary.summarize_task(state, repo=repo)
        return len(records)

    # ------------------------------------------------------------- passthroughs
    def set_model(self, name: str) -> None:
        """Switch the model in place (``/model <name>``)."""

        model_config = self.config.resolve_model(name)
        self.model_name = name
        self.model_config = model_config
        self.gateway = build_gateway(model_config, name=name)
        self._rebuild_with_gateway()
        if self.context_builder is not None:
            self.context_builder.model_config = model_config

    def set_gateway(self, gateway: Any, *, name: str | None = None) -> None:
        """Swap the model gateway (tests, mock mode, custom providers)."""

        self.gateway = gateway
        if name:
            self.model_name = name
        self._rebuild_with_gateway()

    def _rebuild_with_gateway(self) -> None:
        """Re-wire everything that holds a reference to the gateway."""

        if not self._built:
            return
        assert self.orchestrator is not None and self.context_manager is not None
        self.orchestrator.gateway = self.gateway
        self.orchestrator.build()
        if self.context_manager.compact_service is not None:
            self.context_manager.compact_service.gateway = self.gateway
        if self.summary is not None:
            self.summary.gateway = self.gateway

    async def status(self) -> dict[str, Any]:
        self.build()
        assert self.context_manager is not None
        info: dict[str, Any] = {
            "model": self.model_config.label,
            "model_name": self.model_name,
            "workspace": str(self.workspace.root),
            "thread_id": self.thread_id,
            "tools": len(self.tool_runtime.visible_tools()) if self.tool_runtime else 0,
            "skills": len(self.skill_registry.names()) if self.skill_registry else 0,
            "subagents": self.subagent_registry.names() if self.subagent_registry else [],
            "permissions": self.permission.describe() if self.permission else None,
        }
        if self.memory is not None:
            info["memory"] = await self.memory.status()
        return info

    async def memory_lines(self, query: str, k: int | None = None) -> list[str]:
        self.build()
        if self.memory is None or not self.memory.enabled:
            return []
        return await self.memory.recall_lines(query, k)

    async def close(self) -> None:
        if self._mcp_backend is not None:
            try:
                self._mcp_backend.close()
            except Exception as exc:  # pragma: no cover - teardown
                log.debug("closing MCP backend failed: %s", exc)
        if self.memory is not None:
            self.memory.close()


# --------------------------------------------------------------------- builders
def build_gateway(model_config: ModelConfig, *, name: str = "main") -> Any:
    provider = (model_config.provider or "openai_compatible").lower()
    if provider in ("mock", "offline"):
        from harness.inference.mock_gateway import MockGateway

        return MockGateway(model_config=model_config, name=name)
    if provider in ("openai_compatible", "openai", "openai-compatible"):
        return OpenAICompatibleGateway(model_config, name=name)
    raise ValueError(f"unknown model provider `{model_config.provider}`")


def build_memory_service(config: HarnessConfig) -> MemoryService:
    embedding = build_embedding_backend(
        config.embedding.backend,
        dimensions=config.embedding.dimensions,
        model=config.embedding.model,
        base_url=config.embedding.base_url,
        api_key_env=config.embedding.api_key_env,
    )
    store = build_memory_store(
        config.memory.backend, path=str(config.memory_path()), collection=config.memory.collection
    )
    return MemoryService(
        store,
        embedding,
        top_k=config.memory.top_k,
        score_threshold=config.memory.score_threshold,
        enabled=config.memory.enabled,
    )


__all__ = ["AgentHarness", "build_gateway", "build_memory_service", "READ_ONLY_CEILING"]
