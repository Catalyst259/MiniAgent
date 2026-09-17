"""Graph nodes.

Each node is a small, dependency-injected async function that takes the agent
state and returns a partial state update.  LangGraph only sequences them; the
behaviour (context, budget, tools, skills, delegation, termination) is all here.
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any, Awaitable, Callable, Protocol

from harness.agent.dto import (
    DelegateRequest,
    Message,
    ModelRequest,
    Observation,
    ToolCall,
    TokenUsage,
)
from harness.agent.errors import ModelError
from harness.agent.events import Event
from harness.agent.state import AgentState
from harness.context.manager import PreparedContext
from harness.inference.gateway import ModelGateway
from harness.subagents.runtime import SubAgentRuntime

log = logging.getLogger(__name__)

EventHandler = Callable[[Event], Any]

#: Per-turn working data shared between nodes (prepared request, pending calls,
#: token-guard verdict).  Deliberately *not* part of AgentState: the LangGraph
#: checkpointer serializes state, and a prepared ModelRequest holds callbacks.
SCRATCH: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("miniagent_scratch")

#: Per-turn event handler.  A ContextVar (not a node default) so a caller can
#: subscribe *after* the graph was built; values are inherited by the tasks
#: LangGraph creates for each node.
EMITTER: contextvars.ContextVar[EventHandler | None] = contextvars.ContextVar(
    "miniagent_emitter", default=None
)


class ContextManagerLike(Protocol):
    async def prepare(self, state: AgentState) -> PreparedContext: ...
    async def compact(self, state: AgentState): ...


class ToolRuntimeLike(Protocol):
    async def run_many(self, calls: list[ToolCall]) -> list[Observation]: ...
    async def run(self, call: ToolCall) -> Observation: ...


class SkillLoaderLike(Protocol):
    def load(self, name: str, loaded: list[str] | None = None) -> str: ...
    def names(self) -> list[str]: ...


def scratch() -> dict[str, Any]:
    """The working dict for the current turn (auto-created)."""

    data = SCRATCH.get(None)
    if data is None:
        data = {}
        SCRATCH.set(data)
    return data


async def _emit(handler: EventHandler | None, event: Event) -> None:
    """Send an event to the per-turn emitter, else to the node's own handler."""

    target = EMITTER.get(None) or handler
    if target is None:
        return
    result = target(event)
    if hasattr(result, "__await__"):
        await result  # type: ignore[func-returns-value]


# --------------------------------------------------------------------- context
def make_build_context_node(context_manager: ContextManagerLike, on_event: EventHandler | None = None):
    async def build_context(state: AgentState) -> dict[str, Any]:
        prepared = await context_manager.prepare(state)
        await _emit(
            on_event,
            Event(
                type="context",
                message=prepared.description,
                data={
                    "tokens": prepared.tokens,
                    "messages": len(prepared.request.messages),
                    "should_compact": prepared.should_compact,
                },
            ),
        )
        scratch()["prepared"] = prepared
        return {}

    return build_context


def make_token_guard_node(context_manager: ContextManagerLike, on_event: EventHandler | None = None):
    async def token_guard(state: AgentState) -> dict[str, Any]:
        prepared: PreparedContext = scratch()["prepared"]
        if prepared.should_compact:
            mode = "compact"
        else:
            mode = "normal"
        await _emit(
            on_event,
            Event(
                type="token_guard",
                message=f"{mode} ({prepared.tokens} tokens)",
                data={"mode": mode, "compactable": prepared.compactable_messages},
            ),
        )
        scratch()["token_guard"] = mode
        return {}

    return token_guard


def make_compact_node(context_manager: ContextManagerLike, on_event: EventHandler | None = None):
    async def compact(state: AgentState) -> dict[str, Any]:
        outcome = await context_manager.compact(state)
        if outcome.applied:
            await _emit(
                on_event,
                Event(
                    type="compact",
                    message=f"folded {outcome.folded} message(s) into a summary",
                    data={"folded": outcome.folded, "summary_chars": len(outcome.summary)},
                ),
            )
            scratch()["last_compact"] = outcome.folded
            return outcome.state_patch(state)
        await _emit(
            on_event,
            Event(type="compact", message="nothing to compact; continuing", data={"folded": 0}),
        )
        scratch()["last_compact"] = 0
        return {}

    return compact


# ------------------------------------------------------------------------- llm
def make_llm_node(gateway: ModelGateway, on_event: EventHandler | None = None):
    async def llm(state: AgentState) -> dict[str, Any]:
        prepared: PreparedContext = scratch()["prepared"]
        iteration = int(state.get("iteration") or 0) + 1
        await _emit(
            on_event,
            Event(type="iteration", message=f"iteration {iteration}", data={"iteration": iteration}),
        )
        request: ModelRequest = prepared.request
        try:
            response = await gateway.chat(request)
        except ModelError as exc:
            message = f"Model call failed: {exc}"
            await _emit(on_event, Event(type="error", message=message))
            return {
                "iteration": iteration,
                "termination_status": "model_error",
                "termination_reason": message,
                "final_answer": message,
                "messages": [Message(role="assistant", content=message)],
            }

        message = Message(
            role="assistant",
            content=(response.text or "").strip(),
            tool_calls=response.tool_calls,
            reasoning=response.reasoning,
        )
        pending = {
            "pending_tools": [call for call in response.tool_calls if call.name not in ("load_skill", "delegate")],
            "pending_skills": [call for call in response.tool_calls if call.name == "load_skill"],
            "pending_delegates": [call for call in response.tool_calls if call.name == "delegate"],
        }
        update: dict[str, Any] = {
            "iteration": iteration,
            "messages": [message],
            "tokens_last_request": prepared.tokens,
        }
        scratch().update(pending)
        if response.usage is not None:
            previous = state.get("token_usage") or TokenUsage()
            update["token_usage"] = previous + response.usage
        if message.reasoning:
            await _emit(
                on_event,
                Event(type="assistant_reasoning", message=message.reasoning[:400]),
            )
        if message.content:
            await _emit(on_event, Event(type="assistant_text", message=message.content))
        # Every assistant message completes here - including the ones that only
        # accompany tool calls.  Without this the UI can never mark those cells
        # complete, because the turn-level "finished" event fires once.
        await _emit(
            on_event,
            Event(
                type="assistant_message",
                message=message.content,
                data={"iteration": iteration, "has_tool_calls": bool(message.tool_calls)},
            ),
        )
        for call in response.tool_calls:
            await _emit(
                on_event,
                Event(
                    type="tool_call",
                    message=f"{call.name}({_short_args(call)})",
                    data={"tool": call.name, "id": call.id, "arguments": call.arguments},
                ),
            )
        return update

    return llm


def _short_args(call: ToolCall, limit: int = 140) -> str:
    import json

    try:
        rendered = json.dumps(call.arguments, ensure_ascii=False)
    except (TypeError, ValueError):
        rendered = str(call.arguments)
    return rendered if len(rendered) <= limit else rendered[: limit - 1] + "…"


# ----------------------------------------------------------------------- tools

# ------------------------------------------------------------------------- act
def make_act_node(
    tool_runtime: ToolRuntimeLike,
    skill_loader: SkillLoaderLike,
    subagent_runtime: "SubAgentRuntime",
    on_event: EventHandler | None = None,
):
    """Execute *every* tool call of the current assistant turn, in order.

    A single assistant message may mix skills, delegation and plain tools.  The
    API requires **each** ``tool_call_id`` to be answered by a tool message, so
    dispatching only one category (as separate nodes did) produced conversation
    histories that providers reject with "an assistant message with 'tool_calls'
    must be followed by tool messages".  One node handles them all.
    """

    async def act(state: AgentState) -> dict[str, Any]:
        pending: list[ToolCall] = list(scratch().get("pending_tools") or [])
        skill_calls: list[ToolCall] = list(scratch().get("pending_skills") or [])
        delegate_calls: list[ToolCall] = list(scratch().get("pending_delegates") or [])
        if not (pending or skill_calls or delegate_calls):
            return {}

        # Announce every tool call before executing, so the UI shows timing.
        for call in pending:
            await _emit(
                on_event,
                Event(
                    type="tool_start",
                    message=call.name,
                    data={"tool": call.name, "id": call.id, "arguments": call.arguments},
                ),
            )

        observations: list[Observation] = []
        messages: list[Message] = []
        loaded_names: list[str] = []
        details = dict(state.get("skill_details") or {})
        history: list[dict[str, Any]] = []

        # order of execution follows the model's own ordering of the calls
        order = {call.id: index for index, call in enumerate(
            [*skill_calls, *delegate_calls, *pending]
        )}
        combined = sorted([*skill_calls, *delegate_calls, *pending], key=lambda c: order[c.id])

        for call in combined:
            if call.name == "load_skill":
                observation, name = await _load_one_skill(call, skill_loader, details, on_event)
                observations.append(observation)
                messages.append(observation.to_message())
                if name:
                    loaded_names.append(name)
                continue

            if call.name == "delegate":
                observation, entry = await _run_one_delegate(
                    call, subagent_runtime, on_event
                )
                observations.append(observation)
                messages.append(observation.to_message())
                history.append(entry)
                continue

            # plain tools run concurrently (they are independent)
            tool_calls = [call for call in combined if call.name not in ("load_skill", "delegate")]
            tool_observations = await tool_runtime.run_many(tool_calls)
            for observation in tool_observations:
                await _emit(
                    on_event,
                    Event(
                        type="tool_result",
                        message=_observation_text(observation),
                        data={
                            "duration_ms": observation.duration_ms,
                            "ok": observation.ok,
                            "id": observation.tool_call_id,
                            "tool": observation.tool_name,
                        },
                    ),
                )
            observations.extend(tool_observations)
            messages.extend(observation.to_message() for observation in tool_observations)
            break

        update: dict[str, Any] = {"observations": observations, "messages": messages}
        if loaded_names:
            update["loaded_skills"] = loaded_names
            update["skill_details"] = details
        if history:
            update["delegate_history"] = history
            update["active_subagent"] = None
        return update

    return act


async def _load_one_skill(
    call: ToolCall,
    skill_loader: SkillLoaderLike,
    details: dict[str, str],
    on_event: EventHandler | None,
) -> tuple[Observation, str | None]:
    name = str((call.arguments or {}).get("name") or "").strip()
    try:
        body = skill_loader.load(name, list(details))
    except Exception as exc:
        await _emit(on_event, Event(type="skill_load", message=f"{name}: {exc}", data={"ok": False}))
        return (
            Observation(
                tool_call_id=call.id, tool_name="load_skill", ok=False, content="", error=str(exc)
            ),
            None,
        )
    details[name] = body
    await _emit(
        on_event,
        Event(
            type="skill_load",
            message=f"loaded `{name}` ({len(body)} chars)",
            data={"skill": name, "ok": True, "id": call.id},
        ),
    )
    return (
        Observation(
            tool_call_id=call.id,
            tool_name="load_skill",
            ok=True,
            content=(
                f"Loaded skill `{name}` ({len(body)} chars). "
                "Follow it for the rest of the task."
            ),
        ),
        name,
    )


async def _run_one_delegate(
    call: ToolCall,
    subagent_runtime: "SubAgentRuntime",
    on_event: EventHandler | None,
) -> tuple[Observation, dict[str, Any]]:
    arguments = call.arguments or {}
    request = DelegateRequest(
        agent=str(arguments.get("agent") or ""),
        task=str(arguments.get("task") or ""),
        context=arguments.get("context"),
    )
    await _emit(
        on_event,
        Event(
            type="delegate_start",
            message=f"{request.agent}: {request.task[:120]}",
            data={"agent": request.agent, "task": request.task},
        ),
    )
    run = await subagent_runtime.run(request)
    await _emit(
        on_event,
        Event(
            type="delegate_end",
            message=f"{run.agent} {'ok' if run.ok else 'failed'} ({run.iterations} iterations)",
            data={
                "agent": run.agent,
                "ok": run.ok,
                "iterations": run.iterations,
                "summary": run.summary,
                "artifacts": run.artifacts,
            },
        ),
    )
    header = f"Subagent `{run.agent}` result" + ("" if run.ok else " (FAILED)")
    content = f"{header}:\n\n{run.summary}"
    observation = Observation(
        tool_call_id=call.id,
        tool_name="delegate",
        ok=run.ok,
        content=content,
        error=None if run.ok else run.summary,
    )
    entry = {
        "agent": run.agent,
        "task": run.task,
        "ok": run.ok,
        "iterations": run.iterations,
        "artifacts": run.artifacts,
    }
    return observation, entry


# ----------------------------------------------------------------------- skill

# -------------------------------------------------------------------- delegate


def _observation_text(observation: Observation, limit: int = 8_000) -> str:
    """Full tool output for the UI (the model gets its own capped copy)."""

    text = observation.content if observation.ok else (observation.error or observation.content)
    if len(text) > limit:
        text = text[:limit] + f"\n... [truncated for display: {len(text) - limit} chars]"
    return text


__all__ = [
    "make_act_node",
    "SCRATCH",
    "EMITTER",
    "scratch",
    "make_build_context_node",
    "make_token_guard_node",
    "make_compact_node",
    "make_llm_node",
    "EventHandler",
    "ContextManagerLike",
    "ToolRuntimeLike",
    "SkillLoaderLike",
]
