"""Runtime state of a single task.

``AgentState`` describes *what is happening in the current task*.  It is
deliberately not global memory: it is checkpointed with LangGraph, resumable and
thread-scoped, and it is discarded when the task ends (the interesting parts are
later extracted by :mod:`harness.memory.summary`).
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from harness.agent.dto import Message, Observation, TokenUsage


class ReplaceMessages(list[Message]):
    """Reducer marker for state updates that replace conversation history."""


def append_messages(left: list[Message], right: list[Message]) -> list[Message]:
    if isinstance(right, ReplaceMessages):
        return list(right)
    return list(left or []) + list(right or [])


def extend_unique(left: list[str], right: list[str]) -> list[str]:
    out = list(left or [])
    for item in right or []:
        if item not in out:
            out.append(item)
    return out


class AgentState(TypedDict, total=False):
    """State shared by every node of the orchestrator graph."""

    # conversation + observations
    messages: Annotated[list[Message], append_messages]
    observations: Annotated[list[Observation], operator.add]

    # loop bookkeeping
    iteration: int
    thread_id: str
    task_id: str
    user_input: str

    # progressive disclosure
    loaded_skills: Annotated[list[str], extend_unique]
    skill_details: dict[str, str]

    # context management
    compact_summary: str | None
    compact_count: int
    token_usage: TokenUsage
    tokens_last_request: int

    # delegation
    active_subagent: str | None
    delegate_history: Annotated[list[dict[str, Any]], operator.add]

    # lifecycle
    termination_status: str | None
    termination_reason: str | None
    final_answer: str | None


def new_state(
    user_input: str,
    *,
    thread_id: str,
    task_id: str,
) -> AgentState:
    """Build the initial state for a fresh task."""

    return AgentState(
        messages=[Message(role="user", content=user_input)],
        observations=[],
        iteration=0,
        thread_id=thread_id,
        task_id=task_id,
        user_input=user_input,
        loaded_skills=[],
        skill_details={},
        compact_summary=None,
        compact_count=0,
        token_usage=TokenUsage(),
        tokens_last_request=0,
        active_subagent=None,
        delegate_history=[],
        termination_status=None,
        termination_reason=None,
        final_answer=None,
    )


__all__ = ["AgentState", "new_state", "append_messages", "extend_unique"]
