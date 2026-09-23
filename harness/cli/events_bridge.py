"""Bridge: harness events (``agent.events.Event``) -> CLI ``AgentEvent``s.

The runtime keeps emitting its own small event vocabulary; the CLI translates it
into the UI event model.  Neither side needs to know the other's internals.
"""

from __future__ import annotations

from harness.agent.events import Event
from harness.cli import events as ui

#: harness event type -> UI event type, for the events that map 1:1
_DIRECT = {
    "iteration": ui.AssistantStarted,
    "context": ui.ContextUsage,
    "token_guard": ui.ContextUsage,
    "tool_call": ui.ToolStarted,
    "tool_start": ui.ToolStarted,
    "tool_failed": ui.ToolFailed,
    "tool_denied": ui.ToolFailed,
    "permission_decision": ui.PermissionDecided,
    "permission_ask": None,
    "interaction_request": None,
    "interaction_resolved": ui.InteractionResolved,
    "tool_result": ui.ToolFinished,
    "skill_load": ui.SkillLoaded,
    "delegate_start": ui.SubAgentStarted,
    "delegate_end": ui.SubAgentFinished,
    "compact": ui.Compacted,
    "error": ui.ErrorEvent,
    "memory_write": None,
    "memory_recall": None,
    "assistant_reasoning": None,
    "assistant_text": None,  # streamed through AssistantDelta
    "assistant_message": ui.AssistantFinished,  # rendered from the stream / final answer
    "terminate": None,  # TurnFinished is emitted by the CLI loop exactly once
}


def translate(event: Event) -> ui.AgentEvent | None:
    """Convert one runtime event; ``None`` when the UI has nothing to show."""

    kind = event.type
    assert kind in _DIRECT.keys()
    data = event.data or {}

    if kind == "iteration":
        return ui.AssistantStarted(iteration=int(data.get("iteration") or 0))
    if kind == "context":
        return ui.ContextUsage(
            tokens=int(data.get("tokens") or 0),
            should_compact=bool(data.get("should_compact")),
        )
    if kind in ("tool_call", "tool_start"):
        return ui.ToolStarted(
            call_id=str(data.get("id") or ""),
            tool=str(data.get("tool") or event.message.split("(")[0]),
            arguments=dict(data.get("arguments") or {}),
        )
    if kind == "tool_result":
        return ui.ToolFinished(
            call_id=str(data.get("id") or ""),
            tool=str(data.get("tool") or ""),
            ok=bool(data.get("ok", True)),
            text=event.message,
            duration_ms=int(data.get("duration_ms") or 0),
        )
    if kind == "tool_failed":
        return ui.ToolFailed(
            call_id=str(data.get("id") or ""),
            tool=str(data.get("tool") or ""),
            error=event.message,
        )
    if kind == "tool_denied":
        # The permission layer refused the call; the model sees a tool error, and
        # so does the transcript.
        return ui.ToolFailed(
            call_id=str(data.get("id") or ""),
            tool=str(data.get("tool") or ""),
            error=event.message,
        )
    if kind == "permission_decision":
        return ui.PermissionDecided(
            call_id=str(data.get("id") or ""),
            tool=str(data.get("tool") or ""),
            permission=str(data.get("permission") or ""),
            reason=str(data.get("reason") or event.message),
            source=str(data.get("source") or ""),
            approval=data.get("approval"),
        )
    if kind == "permission_ask":
        # The question is rendered by the approval provider (status line + keys),
        # not as a transcript cell; the decision that follows is the record.
        return None
    if kind == "interaction_request":
        # The injected interaction provider owns the live modal panel.
        return None
    if kind == "interaction_resolved":
        question, _, answer = event.message.partition(": ")
        return ui.InteractionResolved(
            question=question,
            value=str(data.get("value") or answer),
            ok=bool(data.get("ok", True)),
            error=str(data.get("error") or ""),
        )
    if kind == "skill_load":
        return ui.SkillLoaded(name=str(data.get("skill") or event.message), ok=bool(data.get("ok", True)))
    if kind == "delegate_start":
        return ui.SubAgentStarted(
            agent=str(data.get("agent") or ""),
            task=str(data.get("task") or event.message),
        )
    if kind == "delegate_end":
        return ui.SubAgentFinished(
            agent=str(data.get("agent") or ""),
            ok=bool(data.get("ok", True)),
            iterations=int(data.get("iterations") or 0),
            summary=str(data.get("summary") or ""),
        )
    if kind == "assistant_message":
        return ui.AssistantFinished(
            text=event.message,
            reasoning=data.get("reasoning") or None,
            message_id=str(data.get("iteration") or ""),
        )
    if kind == "compact":
        return ui.Compacted(folded=int(data.get("folded") or 0))
    if kind == "error":
        return ui.ErrorEvent(message=event.message)
    if kind == "terminate":
        # The turn's real end is emitted once by the CLI loop, with the final
        # answer attached.  Translating this bookkeeping event as well produced a
        # second TurnFinished that re-rendered the answer.
        return None
    return None


__all__ = ["translate"]
