"""OpenAI-compatible adapters.

Streaming lives in :mod:`harness.inference.streaming`; this module owns the
payload shape and the non-streaming fallback.

Two important properties of this module:

* it is an **adapter** - it converts provider payloads into the normalized
  :class:`~harness.agent.dto.ModelResponse` and nothing else,
* it never leaks a provider SDK object upwards; ``raw`` is kept only for
  debugging/telemetry.
"""

from __future__ import annotations

import logging

from typing import Any, AsyncIterator

from harness.agent.dto import (
    Message,
    ModelRequest,
    ModelResponse,
    TokenUsage,
    ToolCall,
)
from harness.agent.errors import ModelError
from harness.inference.config import ModelConfig
from harness.inference.streaming import stream_chat

log = logging.getLogger(__name__)


def normalize_response(payload: Any, *, model: str | None = None) -> ModelResponse:
    """Convert an OpenAI-style chat completion into a ``ModelResponse``.

    Tolerates dicts, pydantic objects and the small dialect differences between
    providers (missing ``usage``, ``reasoning_content`` vs ``reasoning``,
    ``tool_calls`` as dicts vs objects, ``arguments`` as str vs dict).
    """

    get = _getter(payload)
    choices = get("choices") or []
    if not choices:
        raise ModelError("provider returned no choices")
    choice = choices[0]
    cget = _getter(choice)
    message = cget("message") or {}
    mget = _getter(message)

    text = mget("content")
    if isinstance(text, list):  # some providers return content parts
        text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in text)
    reasoning = mget("reasoning_content") or mget("reasoning")

    tool_calls: list[ToolCall] = []
    for raw_call in mget("tool_calls") or []:
        call = ToolCall.from_openai(_getter(raw_call)("function") or raw_call)
        if call.name:
            tool_calls.append(call)

    usage_raw = get("usage")
    usage = None
    if usage_raw is not None:
        uget = _getter(usage_raw)
        details = uget("prompt_tokens_details") or {}
        usage = TokenUsage(
            prompt_tokens=int(uget("prompt_tokens") or 0),
            completion_tokens=int(uget("completion_tokens") or 0),
            total_tokens=int(uget("total_tokens") or 0),
            cached_tokens=int(_getter(details)("cached_tokens") or 0) if details else 0,
        )

    return ModelResponse(
        text=text,
        reasoning=reasoning,
        tool_calls=tool_calls,
        finish_reason=cget("finish_reason"),
        usage=usage,
        model=get("model") or model,
        raw={"id": get("id")},
    )


def _getter(obj: Any):
    if isinstance(obj, dict):
        return lambda key, default=None: obj.get(key, default)
    return lambda key, default=None: getattr(obj, key, default)


class OpenAICompatibleGateway:
    """``ModelGateway`` implementation backed by ``openai.AsyncOpenAI``."""

    def __init__(self, config: ModelConfig, *, client: Any | None = None, name: str = "main") -> None:
        self.config = config
        self.name = name
        self._client = client

    # ------------------------------------------------------------------ client
    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover - dependency is required
                raise ModelError("the `openai` package is required for OpenAI-compatible gateways") from exc
            kwargs: dict[str, Any] = {
                "api_key": self.config.resolved_api_key,
                "timeout": self.config.timeout_seconds,
                "max_retries": self.config.max_retries,
            }
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            if self.config.extra_headers:
                kwargs["default_headers"] = self.config.extra_headers
            if not self.config.trust_env:
                # The SDK honours proxy/TLS env vars through httpx; when the
                # ambient environment is wrong for this endpoint, opt out with
                # our own client instead of failing the request.
                import httpx

                kwargs["http_client"] = httpx.AsyncClient(
                    trust_env=False, timeout=self.config.timeout_seconds
                )
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    # -------------------------------------------------------------------- chat
    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model or self.config.model,
            "messages": [_to_openai(message) for message in request.messages],
        }
        temperature = request.temperature if request.temperature is not None else self.config.temperature
        if temperature is not None:
            payload["temperature"] = temperature
        max_tokens = request.max_tokens if request.max_tokens is not None else self.config.max_tokens
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if request.stop:
            payload["stop"] = request.stop
        if request.tools:
            payload["tools"] = request.tools
            payload["tool_choice"] = "auto"
        if self.config.extra_body:
            payload.update(self.config.extra_body)
        return payload

    async def chat(self, request: ModelRequest) -> ModelResponse:
        """One completion.

        Streams when the caller supplied an ``on_delta`` callback (so the CLI can
        show text as it arrives) and falls back to a non-streaming call when the
        provider does not support streaming.
        """

        payload = self._payload(request)
        on_delta = request.metadata.get("on_delta")
        if on_delta is not None:
            try:
                return await stream_chat(self.client, payload, on_delta=on_delta)
            except ModelError as exc:
                log.warning("streaming failed (%s); falling back to a blocking call", exc)
        try:
            completion = await self.client.chat.completions.create(**payload)
        except Exception as exc:  # provider/SDK errors are normalized here
            raise ModelError(f"{type(exc).__name__}: {exc}") from exc
        return normalize_response(completion, model=payload["model"])

    async def stream(self, request: ModelRequest) -> AsyncIterator[str]:
        """Yield text deltas only (no tool calls)."""

        queue: list[str] = []

        def collect(delta: str) -> None:
            queue.append(delta)

        payload = self._payload(request)
        response = await stream_chat(self.client, payload, on_delta=collect)
        for delta in queue:
            yield delta
        if response.text and not queue:  # pragma: no cover - non-streaming providers
            yield response.text

    # ------------------------------------------------------------- diagnostics
    async def health(self) -> tuple[bool, str]:
        """Cheap reachability probe used by ``/model`` and tests."""

        try:
            await self.client.models.list()
            return True, "ok"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"


def _to_openai(message: Message) -> dict[str, Any]:
    return message.to_openai()


__all__ = ["OpenAICompatibleGateway", "normalize_response"]
