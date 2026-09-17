"""Tool runtime: find -> validate -> execute -> normalize -> observation.

Intentionally thin.  It owns the *policy* of tool execution (visibility, time
outs, output caps, error normalization) and delegates the actual work to a
:class:`~harness.tools.registry.ToolBackend` (local functions or an MCP server).
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Iterable

from harness.agent.dto import Observation, ToolCall
from harness.agent.errors import (
    HarnessError,
    ToolError,
    ToolNotFound,
    ToolPermissionError,
)
from harness.tools.registry import ToolRegistry

_FATAL_MARKERS = ("FATAL:", "CONTEXT_FAILURE:", "SUBAGENT_FAILURE:")


class ToolRuntime:
    """Executes :class:`ToolCall` objects and returns observations."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        allowed: Iterable[str] | None = None,
        forbidden: Iterable[str] = (),
        max_output_chars: int = 30_000,
        timeout_seconds: float = 180.0,
        max_parallel: int = 4,
    ) -> None:
        self.registry = registry
        self.allowed = set(allowed) if allowed is not None else None
        self.forbidden = set(forbidden)
        self.max_output_chars = max_output_chars
        self.timeout_seconds = timeout_seconds
        self.max_parallel = max(1, max_parallel)

    # ----------------------------------------------------------------- visibility
    def visible_tools(self) -> list[str]:
        names = self.registry.names()
        if self.allowed is not None:
            names = [name for name in names if name in self.allowed]
        return [name for name in names if name not in self.forbidden]

    def schemas(self, *, extra: Iterable[dict[str, Any]] = ()) -> list[dict[str, Any]]:
        return self.registry.openai_schemas(only=self.visible_tools(), extra_tools=extra)

    # ------------------------------------------------------------------- running
    async def run(self, call: ToolCall) -> Observation:
        started = time.perf_counter()

        if call.name in self.forbidden:
            return self._failure(
                call, f"tool `{call.name}` is disabled in this context", started
            )
        if self.allowed is not None and call.name not in self.allowed:
            return self._failure(
                call,
                f"tool `{call.name}` is not available here; available tools: "
                + ", ".join(self.visible_tools()),
                started,
            )

        try:
            spec = self.registry.get(call.name)
        except ToolNotFound as exc:
            return self._failure(call, str(exc), started)

        validation_error = validate_arguments(call, spec.input_schema)
        if validation_error:
            return self._failure(call, validation_error, started)

        try:
            output = await asyncio.wait_for(
                asyncio.to_thread(self.registry.call, call.name, call.arguments),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            return self._failure(
                call, f"tool timed out after {self.timeout_seconds:.0f}s", started
            )
        except ToolPermissionError as exc:
            return self._failure(call, f"permission denied: {exc}", started)
        except ToolError as exc:
            return self._failure(call, str(exc), started)
        except (HarnessError, OSError, ValueError) as exc:
            return self._failure(call, f"{type(exc).__name__}: {exc}", started)
        except Exception as exc:  # pragma: no cover - unexpected tool crash
            return self._failure(call, f"unexpected {type(exc).__name__}: {exc}", started)

        content, truncated = self._cap(output)
        return Observation(
            tool_call_id=call.id,
            tool_name=call.name,
            ok=True,
            content=content,
            truncated=truncated,
            duration_ms=self._ms(started),
        )

    async def run_many(self, calls: list[ToolCall]) -> list[Observation]:
        """Run tool calls concurrently (bounded) preserving input order."""

        if not calls:
            return []
        if len(calls) == 1:
            return [await self.run(calls[0])]
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def guarded(call: ToolCall) -> Observation:
            async with semaphore:
                return await self.run(call)

        return list(await asyncio.gather(*(guarded(call) for call in calls)))

    # ------------------------------------------------------------------- helpers
    def _cap(self, output: str) -> tuple[str, bool]:
        text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False, default=str)
        if len(text) <= self.max_output_chars:
            return text, False
        head = int(self.max_output_chars * 0.75)
        tail = self.max_output_chars - head
        return (
            text[:head]
            + f"\n... [tool output truncated: {len(text) - self.max_output_chars} chars omitted] ...\n"
            + text[-tail:],
            True,
        )

    def _failure(self, call: ToolCall, message: str, started: float) -> Observation:
        return Observation(
            tool_call_id=call.id,
            tool_name=call.name,
            ok=False,
            content="",
            error=message,
            duration_ms=self._ms(started),
        )

    @staticmethod
    def _ms(started: float) -> int:
        return int((time.perf_counter() - started) * 1000)


def validate_arguments(call: ToolCall, schema: dict[str, Any]) -> str | None:
    """Minimal JSON-Schema check: required keys, types, unknown keys.

    Deliberately not a full JSON-Schema implementation - it covers the mistakes
    models actually make and keeps the runtime thin.
    """

    if not schema:
        return None
    properties: dict[str, Any] = schema.get("properties") or {}
    required: list[str] = schema.get("required") or []
    arguments = call.arguments or {}

    missing = [key for key in required if key not in arguments or arguments[key] is None]
    if missing:
        return (
            f"`{call.name}` is missing required argument(s): {', '.join(missing)}. "
            f"Expected schema: {json.dumps(schema, ensure_ascii=False)}"
        )

    if schema.get("additionalProperties") is False:
        unknown = [key for key in arguments if key not in properties]
        if unknown:
            return (
                f"`{call.name}` got unexpected argument(s): {', '.join(unknown)}. "
                f"Allowed: {', '.join(properties) or '(none)'}"
            )

    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    for key, value in arguments.items():
        expected = (properties.get(key) or {}).get("type")
        if not expected or expected not in type_map or value is None:
            continue
        python_type = type_map[expected]
        if isinstance(value, bool) and expected in ("integer", "number"):
            return f"`{call.name}.{key}` must be a {expected}, got boolean"
        if not isinstance(value, python_type):
            return (
                f"`{call.name}.{key}` must be a {expected}, got {type(value).__name__}"
            )
        if expected == "integer" and isinstance(value, bool):
            return f"`{call.name}.{key}` must be an integer, got boolean"
    return None


__all__ = ["ToolRuntime", "validate_arguments"]
