"""Context Manager: counting, budgeting and triggering compaction."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from harness.agent.dto import Message, ModelRequest
from harness.agent.state import AgentState
from harness.context.builder import ContextBuilder
from harness.context.compact import CompactService, CompactionOutcome
from harness.context.token_budget import TokenBudgetPolicy, split_for_compaction
from harness.inference.tokenizer import TokenCounter

log = logging.getLogger(__name__)


@dataclass
class PreparedContext:
    """What the LLM node needs, plus the budget verdict."""

    request: ModelRequest
    tokens: int
    should_compact: bool
    compactable_messages: int
    description: str


class ContextManager:
    """Owns token counting, message selection and the compaction trigger."""

    def __init__(
        self,
        builder: ContextBuilder,
        counter: TokenCounter,
        policy: TokenBudgetPolicy,
        compact_service: CompactService | None = None,
        *,
        keep_recent_messages: int = 8,
    ) -> None:
        self.builder = builder
        self.counter = counter
        self.policy = policy
        self.compact_service = compact_service
        self.keep_recent_messages = keep_recent_messages

    # ------------------------------------------------------------------- build
    async def prepare(self, state: AgentState) -> PreparedContext:
        request = await self.builder.build(state)
        tokens = self.counter.count_request(request)
        messages: list[Message] = list(state.get("messages") or [])
        compactable = len(split_for_compaction(messages, keep_recent=self.keep_recent_messages)[0])
        self.last_tokens = tokens
        return PreparedContext(
            request=request,
            tokens=tokens,
            should_compact=self.policy.should_compact(tokens, compactable_messages=compactable),
            compactable_messages=compactable,
            description=self.policy.describe(tokens),
        )

    def count_request(self, request: ModelRequest) -> int:
        return self.counter.count_request(request)

    def usage(self) -> tuple[int, int]:
        """``(tokens_used_by_the_last_request, usable_budget)`` for the UI."""

        return getattr(self, "last_tokens", 0), self.policy.usable_tokens

    # ------------------------------------------------------------------ compact
    async def compact(self, state: AgentState) -> CompactionOutcome:
        if self.compact_service is None:
            return CompactionOutcome(summary=state.get("compact_summary") or "", folded=0, tail=list(state.get("messages") or []), applied=False)
        return await self.compact_service.compact(state)

    def force_compact_available(self, state: AgentState) -> bool:
        """Whether a manual ``/compact`` has anything to work with."""

        messages = list(state.get("messages") or [])
        head, _tail = split_for_compaction(messages, keep_recent=self.keep_recent_messages)
        return bool(head)

    # ------------------------------------------------------------------- status
    async def status(self, state: AgentState) -> dict:
        prepared = await self.prepare(state)
        messages = list(state.get("messages") or [])
        return {
            "tokens": prepared.tokens,
            "budget": self.policy.usable_tokens,
            "threshold": self.policy.compact_threshold,
            "usage": prepared.description,
            "messages": len(messages),
            "compactable": prepared.compactable_messages,
            "compactions": int(state.get("compact_count") or 0),
            "counter": getattr(self.counter, "name", type(self.counter).__name__),
            "should_compact": prepared.should_compact,
        }


__all__ = ["ContextManager", "PreparedContext"]
