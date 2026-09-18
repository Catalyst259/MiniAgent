"""Routing: the conditional edges of the agent loop.

Routing is pure decision making - it reads the state and returns the name of the
next node.  The rules are ordered so that termination always wins.
"""

from __future__ import annotations

from harness.agent.dto import Message, ToolCall
from harness.agent.state import AgentState
from harness.agent.termination import TerminationPolicy

# node names (imported by graph.py and the CLI for display)
NODE_CONTEXT = "build_context"
NODE_TOKEN_GUARD = "token_guard"
NODE_COMPACT = "compact"
NODE_LLM = "llm"
NODE_PERMISSION = "permission"
NODE_TOOLS = "tools"
NODE_ACT = "act"
NODE_SKILLS = "skills"
NODE_DELEGATE = "delegate"
NODE_TERMINATE = "terminate"

ROUTE_END = "__end__"


def last_assistant(messages: list[Message]) -> Message | None:
    for message in reversed(messages):
        if message.role == "assistant":
            return message
    return None


def pending_tool_calls(messages: list[Message]) -> list[ToolCall]:
    """Tool calls of the newest assistant turn that are not native tools."""

    message = last_assistant(messages)
    if message is None or not message.tool_calls:
        return []
    return [
        call
        for call in message.tool_calls
        if call.name not in ("load_skill", "delegate")
    ]


def pending_skill_calls(messages: list[Message]) -> list[ToolCall]:
    message = last_assistant(messages)
    if message is None:
        return []
    return [call for call in message.tool_calls if call.name == "load_skill"]


def pending_delegate_calls(messages: list[Message]) -> list[ToolCall]:
    message = last_assistant(messages)
    if message is None:
        return []
    return [call for call in message.tool_calls if call.name == "delegate"]


def after_token_guard(state: AgentState) -> str:
    from harness.orchestration.nodes import scratch

    mode = scratch().get("token_guard")
    return NODE_COMPACT if mode == "compact" else NODE_LLM


def after_llm(state: AgentState, policy: TerminationPolicy) -> str:
    """Termination first, otherwise hand *all* pending calls to the gate.

    Sending the whole set to one node is what keeps the conversation valid: every
    ``tool_call_id`` must be answered, even when one message mixes skills,
    delegation and plain tools.  The permission gate sits between the model and
    execution so no call can be dispatched without a verdict.
    """

    decision = policy.check(state)
    if decision.terminate:
        return NODE_TERMINATE

    messages = list(state.get("messages") or [])
    if pending_skill_calls(messages) or pending_delegate_calls(messages) or pending_tool_calls(messages):
        return NODE_PERMISSION
    # Assistant produced neither text nor tool calls: treat as an implicit stop.
    return NODE_TERMINATE


def after_gate(state: AgentState) -> str:
    """Every call has a verdict by now; the act node executes the allowed ones."""

    from harness.orchestration.nodes import scratch

    data = scratch()
    if (
        data.get("pending_tools")
        or data.get("pending_skills")
        or data.get("pending_delegates")
    ):
        return NODE_ACT
    return NODE_TERMINATE


def after_side_effect(state: AgentState) -> str:
    """The act node returns to the context builder."""

    return NODE_CONTEXT


def pending_calls_by_kind(state: AgentState) -> dict[str, list[ToolCall]]:
    messages = list(state.get("messages") or [])
    return {
        "tools": pending_tool_calls(messages),
        "skills": pending_skill_calls(messages),
        "delegates": pending_delegate_calls(messages),
    }


__all__ = [
    "NODE_CONTEXT",
    "NODE_TOKEN_GUARD",
    "NODE_COMPACT",
    "NODE_LLM",
    "NODE_PERMISSION",
    "NODE_TOOLS",
    "NODE_ACT",
    "NODE_SKILLS",
    "NODE_DELEGATE",
    "NODE_TERMINATE",
    "ROUTE_END",
    "after_token_guard",
    "after_llm",
    "after_gate",
    "after_side_effect",
    "last_assistant",
    "pending_tool_calls",
    "pending_skill_calls",
    "pending_delegate_calls",
    "pending_calls_by_kind",
]
