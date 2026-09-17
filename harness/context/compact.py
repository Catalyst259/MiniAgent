"""Compact: shrink the *current* context so the task can continue.

This is explicitly not :mod:`harness.memory.summary`.  Compact is context
management that happens mid-task; summary is memory formation that happens after
a task ends.  They have different inputs, different prompts and different
lifetimes, so they live in different modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from harness.agent.dto import Message, ModelRequest
from harness.agent.state import ReplaceMessages
from harness.agent.state import AgentState
from harness.context.prompts import COMPACT_PROMPT, COMPACT_USER_TEMPLATE, render_transcript
from harness.context.token_budget import split_for_compaction
from harness.inference.gateway import ModelGateway

log = logging.getLogger(__name__)


class CompactService:
    """Turns old messages into one dense summary message."""

    def __init__(
        self,
        gateway: ModelGateway,
        *,
        keep_recent_messages: int = 8,
        target_summary_tokens: int = 900,
        max_transcript_chars: int = 120_000,
    ) -> None:
        self.gateway = gateway
        self.keep_recent_messages = keep_recent_messages
        self.target_summary_tokens = target_summary_tokens
        self.max_transcript_chars = max_transcript_chars

    def compactable_count(self, messages: list[Message]) -> int:
        head, _tail = split_for_compaction(messages, keep_recent=self.keep_recent_messages)
        return len(head)

    async def compact(self, state: AgentState) -> "CompactionOutcome":
        """Fold the oldest messages into a dense summary.

        Never destroys history on failure: when the model returns nothing useful
        the previous summary and the full history are kept.
        """

        messages = list(state.get("messages") or [])
        head, tail = split_for_compaction(messages, keep_recent=self.keep_recent_messages)
        previous = state.get("compact_summary")
        if not head:
            return CompactionOutcome(summary=previous or "", folded=0, tail=tail, applied=False)

        transcript = render_transcript(
            (_render(message) for message in head), max_chars=self.max_transcript_chars
        )
        if previous:
            transcript = f"Summary of earlier context (already compacted):\n{previous}\n\n{transcript}"

        request = ModelRequest(
            messages=[
                Message(
                    role="system",
                    content=COMPACT_PROMPT.format(target_tokens=self.target_summary_tokens),
                ),
                Message(role="user", content=COMPACT_USER_TEMPLATE.format(transcript=transcript)),
            ],
            temperature=0.0,
            metadata={"kind": "compact", "target_tokens": self.target_summary_tokens},
        )
        response = await self.gateway.chat(request)
        summary = (response.text or "").strip()
        if not summary:
            log.warning("compact produced an empty summary; keeping previous context")
            return CompactionOutcome(summary=previous or "", folded=0, tail=tail, applied=False)
        return CompactionOutcome(summary=summary, folded=len(head), tail=tail, applied=True)


def _render(message: Message) -> str:
    role = message.role
    if role == "assistant" and message.tool_calls:
        calls = ", ".join(
            f"{call.name}({call.raw_arguments or call.arguments})" for call in message.tool_calls
        )
        body = (message.content or "").strip()
        return f"ASSISTANT: {body}\n  [tool calls] {calls}".strip()
    if role == "tool":
        return f"TOOL {message.name or ''}: {(message.content or '').strip()}"
    if role == "system":
        return f"SYSTEM: {(message.content or '').strip()}"
    return f"{role.upper()}: {(message.content or '').strip()}"


@dataclass
class CompactionOutcome:
    """Result of one compact attempt."""

    summary: str
    folded: int
    tail: list[Message]
    applied: bool

    def state_patch(self, state: AgentState) -> dict:
        if not self.applied:
            return {}
        return {
            "messages": ReplaceMessages(self.tail),
            "compact_summary": self.summary,
            "compact_count": int(state.get("compact_count") or 0) + 1,
        }


def apply_compaction(state: AgentState, outcome: CompactionOutcome) -> dict:
    """State patch produced by a compact attempt."""

    return outcome.state_patch(state)


__all__ = ["CompactService", "CompactionOutcome", "apply_compaction"]
