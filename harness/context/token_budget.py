"""Token budget is a *policy*, not a service."""

from __future__ import annotations

from dataclasses import dataclass

from harness.agent.dto import Message, ModelRequest
from harness.inference.tokenizer import TokenCounter


@dataclass
class TokenBudgetPolicy:
    """Decides whether a context fits and when to compact."""

    max_input_tokens: int = 96_000
    reserve_output_tokens: int = 4_096
    compact_trigger_ratio: float = 0.8
    min_recent_messages: int = 4

    @property
    def usable_tokens(self) -> int:
        return max(1, self.max_input_tokens - self.reserve_output_tokens)

    @property
    def compact_threshold(self) -> int:
        return int(self.usable_tokens * self.compact_trigger_ratio)

    def count(self, counter: TokenCounter, request: ModelRequest) -> int:
        return counter.count_request(request)

    def exceeded(self, tokens: int) -> bool:
        return tokens > self.usable_tokens

    def should_compact(self, tokens: int, *, compactable_messages: int) -> bool:
        """Compact only when the budget is tight *and* there is history to drop."""

        return tokens > self.compact_threshold and compactable_messages > 0

    def describe(self, tokens: int) -> str:
        pct = (tokens / self.usable_tokens * 100) if self.usable_tokens else 0.0
        return (
            f"{tokens}/{self.usable_tokens} input tokens ({pct:.0f}% of budget, "
            f"compact at {self.compact_threshold}, reserve {self.reserve_output_tokens})"
        )


def split_for_compaction(
    messages: list[Message],
    *,
    keep_recent: int,
    min_compactable: int = 2,
) -> tuple[list[Message], list[Message]]:
    """Split history into ``(head_to_compact, recent_tail)``.

    The split point never lands inside an assistant/tool exchange: a tool result
    must always stay with the assistant message that requested it.
    """

    if len(messages) <= keep_recent:
        return [], list(messages)

    cut = len(messages) - keep_recent
    while cut > 0 and messages[cut].role == "tool":
        cut -= 1
    while cut > 0 and messages[cut - 1].role == "assistant" and messages[cut - 1].tool_calls:
        cut -= 1
    if cut < min_compactable:
        return [], list(messages)
    return list(messages[:cut]), list(messages[cut:])


__all__ = ["TokenBudgetPolicy", "split_for_compaction"]
