"""Token counting that is not welded to one model.

Priority order:

1. ``tiktoken`` when installed and a matching encoding exists,
2. ``transformers`` tokenizer when explicitly configured and installed,
3. a deterministic character heuristic.

The heuristic is intentionally conservative (it rounds up) because it is used
for a *budget guard*, where over-estimating is safe and under-estimating is not.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from harness.agent.dto import Message, ModelRequest

_ASCII_CHARS_PER_TOKEN = 3.6
_NON_ASCII_CHARS_PER_TOKEN = 1.4
_MESSAGE_OVERHEAD = 4


@runtime_checkable
class TokenCounter(Protocol):
    def count_text(self, text: str) -> int: ...

    def count_messages(self, messages: list[Message]) -> int: ...

    def count_request(self, request: ModelRequest) -> int: ...


@dataclass
class HeuristicTokenCounter:
    """Deterministic, dependency-free estimator."""

    name: str = "heuristic"

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        ascii_chars = sum(1 for ch in text if ord(ch) < 128)
        other_chars = len(text) - ascii_chars
        estimate = ascii_chars / _ASCII_CHARS_PER_TOKEN + other_chars / _NON_ASCII_CHARS_PER_TOKEN
        return max(1, int(estimate + 0.5))

    def count_messages(self, messages: list[Message]) -> int:
        total = 0
        for message in messages:
            total += _MESSAGE_OVERHEAD + self.count_text(message.content or "")
            if message.role == "assistant":
                for call in message.tool_calls:
                    total += self.count_text(call.name) + self.count_text(
                        call.raw_arguments or str(call.arguments)
                    )
        return total

    def count_request(self, request: ModelRequest) -> int:
        total = self.count_messages(request.messages)
        for schema in request.tools or []:
            total += self.count_text(str(schema))
        return total


@dataclass
class TiktokenCounter:
    encoding: Any
    name: str = "tiktoken"

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoding.encode(text, disallowed_special=()))

    def count_messages(self, messages: list[Message]) -> int:
        total = 0
        for message in messages:
            total += _MESSAGE_OVERHEAD + self.count_text(message.content or "")
            for call in message.tool_calls:
                total += self.count_text(call.name) + self.count_text(
                    call.raw_arguments or str(call.arguments)
                )
        return total

    def count_request(self, request: ModelRequest) -> int:
        total = self.count_messages(request.messages)
        for schema in request.tools or []:
            total += self.count_text(str(schema))
        return total


@dataclass
class TransformersCounter:
    tokenizer: Any
    name: str = "transformers"

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_messages(self, messages: list[Message]) -> int:
        total = 0
        for message in messages:
            total += _MESSAGE_OVERHEAD + self.count_text(message.content or "")
            for call in message.tool_calls:
                total += self.count_text(call.name) + self.count_text(
                    call.raw_arguments or str(call.arguments)
                )
        return total

    def count_request(self, request: ModelRequest) -> int:
        total = self.count_messages(request.messages)
        for schema in request.tools or []:
            total += self.count_text(str(schema))
        return total


def build_token_counter(
    backend: str = "auto",
    *,
    model: str | None = None,
    tokenizer_path: str | None = None,
) -> TokenCounter:
    """Return the best available counter.

    ``backend`` is one of ``auto``, ``tiktoken``, ``transformers``, ``heuristic``.
    """

    if backend in ("auto", "tiktoken"):
        counter = _try_tiktoken(model)
        if counter is not None:
            return counter
        if backend == "tiktoken":
            raise RuntimeError("tiktoken is not installed; falling back is not allowed by config")

    if backend in ("auto", "transformers") and tokenizer_path:
        counter = _try_transformers(tokenizer_path)
        if counter is not None:
            return counter

    return HeuristicTokenCounter()


def _try_tiktoken(model: str | None) -> TokenCounter | None:
    if importlib.util.find_spec("tiktoken") is None:
        return None
    try:  # pragma: no cover - depends on optional dependency
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model or "gpt-4o-mini")
        except Exception:
            encoding = tiktoken.get_encoding("o200k_base")
        return TiktokenCounter(encoding=encoding)
    except Exception:
        return None


def _try_transformers(tokenizer_path: str) -> TokenCounter | None:
    if importlib.util.find_spec("transformers") is None:
        return None
    try:  # pragma: no cover - depends on optional dependency
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        return TransformersCounter(tokenizer=tokenizer)
    except Exception:
        return None


__all__ = [
    "TokenCounter",
    "HeuristicTokenCounter",
    "TiktokenCounter",
    "TransformersCounter",
    "build_token_counter",
]
