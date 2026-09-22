"""LangGraph orchestration.

LangGraph owns exactly four things here: state schema, nodes, edges, and
checkpointing.  Every node body comes from :mod:`harness.orchestration.nodes`,
every routing decision from :mod:`harness.orchestration.routing`, and every
termination decision from :mod:`harness.agent.termination`.

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
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from harness.agent.dto import TerminationDecision
from harness.agent.events import Event
from harness.agent.state import AgentState, new_state
from harness.agent.termination import TerminationPolicy
from harness.orchestration import routing
from harness.orchestration.nodes import (
    EventHandler,
    make_act_node,
    make_build_context_node,
    make_compact_node,
    make_llm_node,
    make_permission_node,
    make_token_guard_node,
)
from harness.orchestration.routing import (
    NODE_ACT,
    NODE_COMPACT,
    NODE_CONTEXT,
    NODE_LLM,
    NODE_PERMISSION,
    NODE_TERMINATE,
    NODE_TOKEN_GUARD,
    after_gate,
    after_llm,
    after_side_effect,
    after_token_guard,
)
from harness.subagents.runtime import SubAgentRuntime

log = logging.getLogger(__name__)


@dataclass
class Orchestrator:
    """Wires the services into a runnable agent loop."""

    gateway: Any
    context_manager: Any
    tool_runtime: Any
    skill_loader: Any
    subagent_runtime: SubAgentRuntime
    #: front-end adapter for the model-facing request_user_input native tool
    interaction_provider: Any = None
    #: the permission gate (``harness.permission.PermissionGate``); ``None`` means
    #: no permission node is built and the loop behaves as it did before
    permission_gate: Any = None
    #: resolves the tools a loaded skill declares, for the permission ceiling.
    #: Injected (not imported) so this module stays free of a skill dependency.
    skill_tool_resolver: Callable[[list[str]], tuple[list[str], str]] | None = None
    termination_policy: TerminationPolicy = field(default_factory=TerminationPolicy)
    checkpointer: Any = None
    on_event: EventHandler | None = None
    thread_id: str = "default"
    recursion_limit: int = 200

    graph: Any = None

    # ------------------------------------------------------------------- wiring
    def build(self) -> Any:
        # LangGraph sends synchronous route callbacks through an executor.  That
        # extra thread boundary is unnecessary for these pure, constant-time
        # decisions and can strand a batch run when the host event loop is not
        # woken by executor completion.  Async adapters keep routing on the
        # graph's event loop (the functions themselves remain easy to unit-test).
        async def route_after_token_guard(state: AgentState) -> str:
            return after_token_guard(state)

        async def route_after_llm(state: AgentState) -> str:
            return after_llm(state, self.termination_policy)

        async def route_after_gate(state: AgentState) -> str:
            return after_gate(state)

        async def route_after_side_effect(state: AgentState) -> str:
            return after_side_effect(state)

        builder = StateGraph(AgentState)
        builder.add_node(NODE_CONTEXT, make_build_context_node(self.context_manager, self.on_event))
        builder.add_node(
            NODE_TOKEN_GUARD, make_token_guard_node(self.context_manager, self.on_event)
        )
        builder.add_node(NODE_COMPACT, make_compact_node(self.context_manager, self.on_event))
        builder.add_node(NODE_LLM, make_llm_node(self.gateway, self.on_event))
        # the permission gate decides every call before anything is executed
        if self.permission_gate is not None:
            # The gate was built before the emitter existed; give it one now so
            # permission_ask / permission_decision reach the same event stream as
            # every other node.
            self.permission_gate.emit = self._permission_emit
            builder.add_node(
                NODE_PERMISSION,
                make_permission_node(
                    self.permission_gate, self.on_event, self.skill_tool_resolver
                ),
            )
        # one node executes every pending call of a turn (see make_act_node)
        builder.add_node(
            NODE_ACT,
            make_act_node(
                self.tool_runtime,
                self.skill_loader,
                self.subagent_runtime,
                self.interaction_provider,
                self.on_event,
            ),
        )
        builder.add_node(NODE_TERMINATE, self._terminate_node)

        builder.add_edge(START, NODE_CONTEXT)
        builder.add_edge(NODE_CONTEXT, NODE_TOKEN_GUARD)
        builder.add_conditional_edges(
            NODE_TOKEN_GUARD,
            route_after_token_guard,
            {NODE_COMPACT: NODE_COMPACT, NODE_LLM: NODE_LLM},
        )
        builder.add_edge(NODE_COMPACT, NODE_LLM)

        after_llm_targets = {NODE_ACT: NODE_ACT, NODE_TERMINATE: NODE_TERMINATE}
        if self.permission_gate is not None:
            after_llm_targets[NODE_PERMISSION] = NODE_PERMISSION
        builder.add_conditional_edges(
            NODE_LLM,
            route_after_llm,
            after_llm_targets,
        )
        if self.permission_gate is not None:
            builder.add_conditional_edges(
                NODE_PERMISSION,
                route_after_gate,
                {NODE_ACT: NODE_ACT, NODE_TERMINATE: NODE_TERMINATE},
            )
        builder.add_conditional_edges(
            NODE_ACT, route_after_side_effect, {NODE_CONTEXT: NODE_CONTEXT}
        )
        builder.add_edge(NODE_TERMINATE, END)

        self.graph = builder.compile(checkpointer=self.checkpointer)
        return self.graph

    async def _permission_emit(self, event_type: str, message: str, data: dict[str, Any]) -> None:
        """Bridge the gate's events onto the turn's event stream."""

        await self._emit(Event(type=event_type, message=message, data=data))

    async def _terminate_node(self, state: AgentState) -> dict[str, Any]:
        decision: TerminationDecision = self.termination_policy.check(state)
        await self._emit(
            Event(
                type="terminate",
                message=decision.reason or "terminated",
                data={"reason": decision.reason},
            )
        )
        return {
            "termination_status": decision.reason or "terminated",
            "termination_reason": decision.reason,
            "final_answer": decision.final_answer,
        }

    async def _emit(self, event: Event) -> None:
        if self.on_event is None:
            return
        result = self.on_event(event)
        if hasattr(result, "__await__"):
            await result

    # ------------------------------------------------------------------ running
    def _config(self) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": self.recursion_limit,
        }

    async def ainvoke(self, state: AgentState) -> AgentState:
        """Run one turn with a fresh scratch namespace and the current handler."""

        if self.graph is None:
            self.build()
        from harness.orchestration.nodes import EMITTER, SCRATCH

        scratch_token = SCRATCH.set({})
        emit_token = EMITTER.set(self.on_event)
        try:
            return await self.graph.ainvoke(state, self._config())
        finally:
            EMITTER.reset(emit_token)
            SCRATCH.reset(scratch_token)

    async def astream_steps(self, state: AgentState):
        """Yield ``(node_name, update)`` as the graph advances (debug/CLI)."""

        if self.graph is None:
            self.build()
        from harness.orchestration.nodes import EMITTER, SCRATCH

        scratch_token = SCRATCH.set({})
        emit_token = EMITTER.set(self.on_event)
        try:
            async for chunk in self.graph.astream(state, self._config(), stream_mode="updates"):
                for node_name, update in chunk.items():
                    yield node_name, update
        finally:
            EMITTER.reset(emit_token)
            SCRATCH.reset(scratch_token)

    async def aget_state(self) -> AgentState:
        if self.graph is None:
            self.build()
        snapshot = await self.graph.aget_state(self._config())
        return dict(snapshot.values or {})


def build_orchestrator(
    *,
    gateway: Any,
    context_manager: Any,
    tool_runtime: Any,
    skill_loader: Any,
    subagent_runtime: SubAgentRuntime,
    permission_gate: Any = None,
    skill_tool_resolver: Callable[[list[str]], tuple[list[str], str]] | None = None,
    termination_policy: TerminationPolicy | None = None,
    checkpointer: Any = None,
    on_event: EventHandler | None = None,
    thread_id: str = "default",
) -> Orchestrator:
    return Orchestrator(
        gateway=gateway,
        context_manager=context_manager,
        tool_runtime=tool_runtime,
        skill_loader=skill_loader,
        subagent_runtime=subagent_runtime,
        permission_gate=permission_gate,
        skill_tool_resolver=skill_tool_resolver,
        termination_policy=termination_policy or TerminationPolicy(),
        checkpointer=checkpointer,
        on_event=on_event,
        thread_id=thread_id,
    )


def fresh_thread_id(prefix: str = "thread") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


__all__ = ["Orchestrator", "build_orchestrator", "fresh_thread_id", "new_state"]
