"""Streaming chat for OpenAI-compatible providers.

Streaming exists so the CLI can show progress: text deltas reach the UI while the
answer is still being generated.  Tool calls are *not* streamed (they are
accumulated and returned in the normalized response), because a half-parsed
``arguments`` JSON string is useless to the harness.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Callable

from harness.agent.dto import ModelRequest, ModelResponse, ToolCall, TokenUsage
from harness.agent.errors import ModelError

log = logging.getLogger(__name__)

DeltaCallback = Callable[[str], None]


async def stream_chat(
    client: Any,
    payload: dict[str, Any],
    *,
    on_delta: DeltaCallback | None = None,
) -> ModelResponse:
    """Consume a streaming completion into one normalized response.

    ``on_delta`` receives only assistant text; the harness keeps the raw source so
    Markdown is rendered once, at the end.  If the provider rejects streaming
    (or the stream breaks), the caller falls back to a non-streaming call.
    """

    stream_payload = dict(payload)
    stream_payload["stream"] = True
    stream_payload["stream_options"] = {"include_usage": True}

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_buffer: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    usage = None
    model = payload.get("model")

    try:
        stream = await client.chat.completions.create(**stream_payload)
        async for chunk in stream:
            chunk_model = getattr(chunk, "model", None)
            if chunk_model:
                model = chunk_model
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = _usage(chunk_usage)
            for choice in getattr(chunk, "choices", None) or []:
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                reasoning = getattr(delta, "reasoning_content", None) or getattr(
                    delta, "reasoning", None
                )
                if reasoning:
                    reasoning_parts.append(reasoning)
                text = getattr(delta, "content", None)
                if text:
                    text_parts.append(text)
                    if on_delta is not None and not tool_buffer:
                        on_delta(text)
                for fragment in getattr(delta, "tool_calls", None) or []:
                    _accumulate_tool_call(tool_buffer, fragment)
    except Exception as exc:
        raise ModelError(f"{type(exc).__name__}: {exc}") from exc

    tool_calls = _finalize_tool_calls(tool_buffer)
    text = "".join(text_parts) or None
    return ModelResponse(
        text=text,
        reasoning="".join(reasoning_parts) or None,
        tool_calls=tool_calls,
        finish_reason=finish_reason or ("tool_calls" if tool_calls else "stop"),
        usage=usage,
        model=model,
        raw={"streamed": True},
    )


def _accumulate_tool_call(buffer: dict[int, dict[str, Any]], fragment: Any) -> None:
    index = getattr(fragment, "index", None) or 0
    entry = buffer.setdefault(index, {"id": None, "name": "", "arguments": ""})
    call_id = getattr(fragment, "id", None)
    if call_id:
        entry["id"] = call_id
    function = getattr(fragment, "function", None)
    if function is not None:
        name = getattr(function, "name", None)
        if name:
            entry["name"] = name
        arguments = getattr(function, "arguments", None)
        if arguments:
            entry["arguments"] += arguments


def _finalize_tool_calls(buffer: dict[int, dict[str, Any]]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for index in sorted(buffer):
        entry = buffer[index]
        if not entry.get("name"):
            continue
        raw = entry.get("arguments") or "{}"
        try:
            arguments = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        calls.append(
            ToolCall(
                id=entry.get("id") or f"call_{index}",
                name=str(entry["name"]).strip(),
                arguments=arguments,
                raw_arguments=raw,
            )
        )
    return calls


def _usage(payload: Any) -> TokenUsage:
    get = payload.get if isinstance(payload, dict) else lambda key, default=None: getattr(payload, key, default)
    details = get("prompt_tokens_details") or {}
    cached = 0
    if details:
        dget = details.get if isinstance(details, dict) else lambda key, default=None: getattr(details, key, default)
        cached = int(dget("cached_tokens") or 0)
    return TokenUsage(
        prompt_tokens=int(get("prompt_tokens") or 0),
        completion_tokens=int(get("completion_tokens") or 0),
        total_tokens=int(get("total_tokens") or 0),
        cached_tokens=cached,
    )


async def iter_text(stream: AsyncIterator[Any]) -> AsyncIterator[str]:
    """Utility for tests: yield the text of an OpenAI-style chunk stream."""

    async for chunk in stream:
        for choice in getattr(chunk, "choices", None) or []:
            delta = getattr(choice, "delta", None)
            text = getattr(delta, "content", None) if delta is not None else None
            if text:
                yield text


__all__ = ["stream_chat", "iter_text", "DeltaCallback"]
