"""SubAgent runtime: context-isolated delegation.

Design rules implemented here:

* a subagent gets **its own state, its own context and its own graph** - the
  parent's message list is never copied in,
* only a compact :class:`DelegateResult` travels back,
* tool isolation comes from the subagent definition, so a read-only subagent
  physically lacks the write tools.
"""

from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from harness.agent.dto import DelegateRequest, DelegateResult
from harness.agent.errors import HarnessError, SubAgentNotFound
from harness.agent.state import AgentState
from harness.subagents.registry import SubAgentRegistry

log = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 6_000
_PATH_RE = re.compile(r"\b[\w./-]+\.(?:py|md|toml|yaml|yml|json|txt|cfg|ini|sh|js|ts|tsx|jsx|go|rs|java|kt|c|h|cpp|hpp|sql|html|css)\b")


class SubAgentRunner(Protocol):
    """What the runtime needs from a runnable agent."""

    async def run(self, task: str, context: str | None = None) -> Any:
        """Return ``(final_text, iterations)`` or an agent-state mapping."""
        ...


SubAgentFactory = Callable[[Any, str], "SubAgentRunner | Awaitable[SubAgentRunner]"]


def _unpack_run_result(result: Any) -> tuple[str, int]:
    """Accept either ``(text, iterations)`` or an ``AgentState`` mapping."""

    if isinstance(result, tuple) and len(result) == 2:
        return str(result[0] or ""), int(result[1] or 0)
    if isinstance(result, dict):
        return str(result.get("final_answer") or ""), int(result.get("iteration") or 0)
    return str(result or ""), 0


@dataclass
class SubAgentRun:
    """Bookkeeping for one delegation (also used for the event stream)."""

    agent: str
    task: str
    ok: bool
    summary: str
    iterations: int = 0
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None

    def to_result(self) -> DelegateResult:
        return DelegateResult(
            agent=self.agent,
            summary=self.summary,
            ok=self.ok,
            artifacts=self.artifacts,
            iterations=self.iterations,
            metadata={"error": self.error} if self.error else {},
        )


class SubAgentRuntime:
    """Executes subagents declared on the filesystem."""

    def __init__(
        self,
        registry: SubAgentRegistry,
        factory: SubAgentFactory,
        *,
        enabled: bool = True,
    ) -> None:
        self.registry = registry
        self.factory = factory
        self.enabled = enabled
        self.history: list[SubAgentRun] = []

    # ------------------------------------------------------------------- public
    def names(self) -> list[str]:
        return self.registry.names()

    def catalog(self) -> str:
        return self.registry.catalog()

    def tool_schema(self) -> dict:
        return self.registry.tool_schema()

    async def delegate(self, request: DelegateRequest) -> DelegateResult:
        run = await self.run(request)
        return run.to_result()

    async def run(self, request: DelegateRequest) -> SubAgentRun:
        if not self.enabled:
            return SubAgentRun(
                agent=request.agent,
                task=request.task,
                ok=False,
                summary="Delegation is disabled in this context.",
                error="delegation disabled",
            )
        try:
            spec = self.registry.get(request.agent)
        except SubAgentNotFound as exc:
            return SubAgentRun(
                agent=request.agent,
                task=request.task,
                ok=False,
                summary=str(exc),
                error=str(exc),
            )

        task = (request.task or "").strip()
        if not task:
            message = f"`delegate` to `{spec.name}` needs a non-empty `task`."
            return SubAgentRun(agent=spec.name, task=task, ok=False, summary=message, error=message)

        context = request.context
        if context and len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS] + "\n... [context truncated by the harness]"

        log.info("delegating to %s (%d chars of context)", spec.name, len(context or ""))
        try:
            runner = self.factory(spec, task)
            if inspect.isawaitable(runner):
                runner = await runner
            text, iterations = _unpack_run_result(await runner.run(task, context))
        except HarnessError as exc:
            return SubAgentRun(
                agent=spec.name, task=task, ok=False, summary=f"SUBAGENT_FAILURE: {exc}", error=str(exc)
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("subagent %s crashed", spec.name)
            return SubAgentRun(
                agent=spec.name,
                task=task,
                ok=False,
                summary=f"SUBAGENT_FAILURE: {type(exc).__name__}: {exc}",
                error=str(exc),
            )

        summary = (text or "").strip() or "(subagent produced no output)"
        run = SubAgentRun(
            agent=spec.name,
            task=task,
            ok=True,
            summary=summary,
            iterations=iterations,
            artifacts=_extract_artifacts(summary),
        )
        self.history.append(run)
        return run

    async def delegate_tool_calls(self, calls, scratch_state: AgentState) -> list[tuple[str, SubAgentRun]]:
        """Run every ``delegate`` tool call of one assistant turn, in order."""

        results: list[tuple[str, SubAgentRun]] = []
        for call in calls:
            arguments = call.arguments or {}
            request = DelegateRequest(
                agent=str(arguments.get("agent") or ""),
                task=str(arguments.get("task") or ""),
                context=arguments.get("context"),
                max_iterations=arguments.get("max_iterations"),
            )
            run = await self.run(request)
            results.append((call.id, run))
        return results


def _extract_artifacts(text: str, limit: int = 12) -> list[str]:
    seen: list[str] = []
    for match in _PATH_RE.finditer(text or ""):
        path = match.group(0)
        if path not in seen:
            seen.append(path)
        if len(seen) >= limit:
            break
    return seen


__all__ = ["SubAgentRuntime", "SubAgentRun", "SubAgentRunner", "SubAgentFactory"]
