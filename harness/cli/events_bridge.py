"""Bridge: harness events (``agent.events.Event``) -> CLI ``AgentEvent``s.

The runtime keeps emitting its own small event vocabulary; the CLI translates it
into the UI event model.  Neither side needs to know the other's internals.
"""

from __future__ import annotations

from typing import Any

from harness.agent.events import Event
from harness.cli import events as ui
from harness.cli.state import AppState

#: harness event type -> UI event type, for the events that map 1:1
_DIRECT = {
    "iteration": ui.AssistantStarted,
    "context": ui.ContextUsage,
    "token_guard": ui.ContextUsage,
    "tool_call": ui.ToolStarted,
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


def translate(event: Event, state: AppState | None = None) -> ui.AgentEvent | None:
    """Convert one runtime event; ``None`` when the UI has nothing to show."""

    kind = event.type
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
            reasoning=None,
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


class EventBridge:
    """Collects harness events and forwards the UI-relevant ones to a sink."""

    def __init__(self, sink, state: AppState | None = None) -> None:
        self.sink = sink
        self.state = state
        self.raw: list[Event] = []

    def __call__(self, event: Event) -> None:
        self.raw.append(event)
        translated = translate(event, self.state)
        if translated is not None:
            self.sink(translated)

    def find(self, kind: str) -> list[Event]:
        return [event for event in self.raw if event.type == kind]

    def last(self, kind: str) -> Event | None:
        found = self.find(kind)
        return found[-1] if found else None


__all__ = ["translate", "EventBridge"]
