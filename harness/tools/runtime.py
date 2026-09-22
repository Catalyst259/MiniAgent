"""Tool runtime: find -> validate -> execute -> normalize -> observation.

Intentionally thin.  It owns the *policy* of tool execution (visibility, time
outs, output caps, error normalization) and delegates the actual work to a
:class:`~harness.tools.registry.ToolBackend` (local functions or an MCP server).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import json
import logging
import time
from contextlib import contextmanager
from typing import Any, Iterable, Iterator

from harness.agent.dto import Observation, ToolCall
from harness.agent.errors import (
    HarnessError,
    ToolError,
    ToolNotFound,
    ToolPermissionError,
)
from harness.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

_FATAL_MARKERS = ("FATAL:", "CONTEXT_FAILURE:", "SUBAGENT_FAILURE:")

#: Calls the current *task* has already had decided by the permission gate.
#:
#: Deliberately task-scoped, not turn-scoped: a LangGraph node runs in its own
#: task, and a ``ContextVar`` set in one node is **invisible to its siblings**
#: (verified against LangGraph, not assumed).  The graph therefore hands its
#: decisions to :meth:`ToolRuntime.run_many` explicitly; this only covers calls
#: made *within* one execution path - notably ``!command``, where the app decides
#: the intent and then passes it to the runtime.
_DECIDED: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "miniagent_decided_calls", default=frozenset()
)


@contextmanager
def calls_already_decided(call_ids: Iterable[str]) -> Iterator[None]:
    """Mark calls as decided for the duration of one execution path."""

    token = _DECIDED.set(_DECIDED.get() | frozenset(str(item) for item in call_ids))
    try:
        yield
    finally:
        _DECIDED.reset(token)


def is_decided(call_id: str) -> bool:
    return call_id in _DECIDED.get()


class ToolRuntime:
    """Executes :class:`ToolCall` objects and returns observations."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        allowed: Iterable[str] | None = None,
        forbidden: Iterable[str] = (),
        permission_gate: Any = None,
        max_output_chars: int = 30_000,
        timeout_seconds: float = 180.0,
        max_parallel: int = 4,
    ) -> None:
        self.registry = registry
        self.allowed = set(allowed) if allowed is not None else None
        self.forbidden = set(forbidden)
        #: The permission layer, consulted here as well as in the graph.  Two
        #: barriers, because this method *is* the execution boundary: any code
        #: that runs a tool comes through here, so a call that never reached the
        #: gate node (a new entry point, a harness-native tool, a bug) is still
        #: decided instead of silently executed.
        self.permission_gate = permission_gate
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
    async def run(
        self,
        call: ToolCall,
        *,
        decided: Iterable[str] | frozenset[str] = (),
    ) -> Observation:
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

        already = call.id in decided or call.id in _DECIDED.get()
        if not already:
            refusal = await self._permission_check(call)
            if refusal is not None:
                return self._failure(call, refusal, started)

        try:
            spec = self.registry.get(call.name)
        except ToolNotFound as exc:
            return self._failure(call, str(exc), started)

        validation_error = validate_arguments(call, spec.input_schema)
        if validation_error:
            return self._failure(call, validation_error, started)

        try:
            output = await self._call_backend(call)
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

    async def _call_backend(self, call: ToolCall) -> Any:
        """Run a synchronous backend without depending on one thread wake-up.

        Some restricted hosts can lose the selector wake-up generated when an
        executor future completes.  A long ``wait_for(to_thread(...))`` then
        sleeps until its timeout even though the worker is already done.  The
        short event-loop tick below processes that completion while preserving
        non-blocking tool execution and the original overall timeout.
        """

        # Do not use asyncio's *default* executor here.  ``asyncio.Runner`` waits
        # for that executor through another cross-thread wake-up during shutdown,
        # reproducing the same lost-wakeup hang after an otherwise successful
        # batch/test.  A scoped executor is already idle by the time it is joined.
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="miniagent-tool",
        )
        loop = asyncio.get_running_loop()
        task = loop.run_in_executor(
            executor,
            self.registry.call,
            call.name,
            call.arguments,
        )
        completed = False
        try:
            async with asyncio.timeout(self.timeout_seconds):
                while not task.done():
                    await asyncio.sleep(0.02)
                result = task.result()
                completed = True
                return result
        except TimeoutError:
            task.cancel()
            raise asyncio.TimeoutError from None
        finally:
            executor.shutdown(wait=completed, cancel_futures=not completed)

    async def run_many(
        self,
        calls: list[ToolCall],
        *,
        decided: Iterable[str] = (),
    ) -> list[Observation]:
        """Run tool calls concurrently (bounded) preserving input order.

        ``decided`` names the calls the permission gate has already ruled on - the
        graph passes its batch here, so each call is asked about exactly once.
        """

        if not calls:
            return []
        ids = frozenset(str(item) for item in decided)
        if len(calls) == 1:
            return [await self.run(calls[0], decided=ids)]
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def guarded(call: ToolCall) -> Observation:
            async with semaphore:
                return await self.run(call, decided=ids)

        return list(await asyncio.gather(*(guarded(call) for call in calls)))

    # --------------------------------------------------------------- permissions
    async def _permission_check(self, call: ToolCall) -> str | None:
        """Why this call may not run, or ``None`` when it may.

        Returns the model-facing refusal message, or ``None``.  This is the
        execution boundary: a call that arrives without a verdict from the graph's
        permission node (a new entry point, a harness-native tool, a bug) is
        decided here rather than assumed safe.
        """

        if self.permission_gate is None:
            return None

        results = await self.permission_gate.check_batch([call])
        if not results:
            return None
        result = results[0]
        if result.allowed:
            log.debug(
                "permission allowed at the execution boundary: %s (%s)",
                result.action.describe(),
                result.verdict.reason,
            )
            return None
        log.warning(
            "permission DENIED at the execution boundary: %s (%s)",
            result.action.describe(),
            result.verdict.reason,
        )
        return result.denial_message

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
