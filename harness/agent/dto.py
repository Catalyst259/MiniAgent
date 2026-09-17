"""Core data-transfer objects shared across the harness.

These are pure value objects: no behaviour that touches the network, the
filesystem or the model.  Everything MiniAgent passes between its layers is
described here so that the layers themselves stay replaceable.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


def _uuid() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ToolCall(BaseModel):
    """A structured request from the model to run one tool."""

    id: str = Field(default_factory=lambda: f"call_{_uuid()}")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str | None = None

    @classmethod
    def from_openai(cls, payload: Any) -> "ToolCall":
        """Normalize an OpenAI-style tool call object/``dict``.

        Accepts both the full shape ``{"id", "type", "function": {...}}`` and the
        already-unwrapped ``{"name", "arguments"}`` form.
        """

        get = payload.get if isinstance(payload, dict) else lambda k, d=None: getattr(payload, k, d)
        inner = get("function")
        if inner is not None:
            get_inner = inner.get if isinstance(inner, dict) else lambda k, d=None: getattr(inner, k, d)
            name = get_inner("name")
            raw = get_inner("arguments")
        else:
            name = get("name")
            raw = get("arguments")
        raw = raw or "{}"
        if not isinstance(raw, str):
            raw = json.dumps(raw, ensure_ascii=False)
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {"value": parsed}
        return cls(
            id=get("id") or f"call_{_uuid()}",
            name=(name or "").strip(),
            arguments=parsed,
            raw_arguments=raw,
        )


class Message(BaseModel):
    """One conversation message.

    ``tool_calls`` is only meaningful for assistant messages and ``tool_call_id``
    only for tool messages, but a single model keeps the state reducers simple.
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    reasoning: str | None = None

    def to_openai(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role}
        if self.role == "assistant" and self.tool_calls:
            payload["content"] = self.content or None
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.raw_arguments or json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in self.tool_calls
            ]
            return payload
        payload["content"] = self.content
        if self.role == "tool":
            payload["tool_call_id"] = self.tool_call_id or "unknown"
            if self.name:
                payload["name"] = self.name
        return payload

    def text_preview(self, limit: int = 160) -> str:
        text = " ".join(self.content.split())
        if len(text) > limit:
            text = text[: limit - 1] + "…"
        return text


class Observation(BaseModel):
    """Normalized result of one tool invocation."""

    tool_call_id: str
    tool_name: str
    ok: bool
    content: str
    error: str | None = None
    duration_ms: int = 0
    truncated: bool = False

    def to_message(self) -> Message:
        body = self.content if self.ok else f"ERROR: {self.error or self.content}"
        return Message(role="tool", content=body, tool_call_id=self.tool_call_id, name=self.tool_name)


class DelegateRequest(BaseModel):
    """Main agent -> subagent hand-off (context-isolated)."""

    agent: str
    task: str
    context: str | None = None
    max_iterations: int | None = None


class DelegateResult(BaseModel):
    """Subagent -> main agent compact return value."""

    agent: str
    summary: str
    ok: bool = True
    artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    iterations: int = 0


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


class ModelRequest(BaseModel):
    """Provider-agnostic model invocation."""

    messages: list[Message]
    tools: list[dict[str, Any]] | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    """Normalized model answer; MiniAgent never sees raw provider payloads."""

    text: str | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    model: str | None = None
    raw: dict[str, Any] | None = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class TerminationDecision(BaseModel):
    terminate: bool
    reason: str | None = None
    final_answer: str | None = None


class FinalAnswer(BaseModel):
    text: str
    reason: str = "final_answer"
    iterations: int = 0


class MemoryRecord(BaseModel):
    id: str = Field(default_factory=_uuid)
    content: str
    memory_type: str
    source: str | None = None
    task_id: str | None = None
    repo: str | None = None
    timestamp: datetime = Field(default_factory=_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillMetadata(BaseModel):
    name: str
    description: str
    keywords: list[str] = Field(default_factory=list)
    path: str
    source: str = "filesystem"


class ToolSpec(BaseModel):
    """MCP tool metadata cache entry."""

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    server: str = "local"

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema or {"type": "object", "properties": {}},
            },
        }


class SubAgentSpec(BaseModel):
    name: str
    description: str
    tools: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    path: str = ""
    body: str = ""


__all__ = [
    "Role",
    "ToolCall",
    "Message",
    "Observation",
    "DelegateRequest",
    "DelegateResult",
    "TokenUsage",
    "ModelRequest",
    "ModelResponse",
    "TerminationDecision",
    "FinalAnswer",
    "MemoryRecord",
    "SkillMetadata",
    "ToolSpec",
    "SubAgentSpec",
    "_uuid",
    "_now",
]
