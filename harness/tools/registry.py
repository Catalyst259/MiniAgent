"""Tool registry: an MCP metadata cache, not a second tool protocol.

The registry only stores :class:`~harness.agent.dto.ToolSpec` objects (schema +
description) and applies visibility rules.  Execution belongs to the runtime.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Protocol, runtime_checkable

from harness.agent.dto import ToolSpec
from harness.agent.errors import ToolNotFound, ToolPermissionError

log = logging.getLogger(__name__)


@runtime_checkable
class ToolBackend(Protocol):
    """Anything that can list and execute tools."""

    name: str

    def list_specs(self) -> list[ToolSpec]: ...

    def call(self, name: str, arguments: dict[str, Any]) -> str: ...


class ToolRegistry:
    """In-memory view over one or more backends."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._backends: dict[str, ToolBackend] = {}

    # ------------------------------------------------------------------ wiring
    def register_backend(self, backend: ToolBackend, *, refresh: bool = True) -> None:
        self._backends[backend.name] = backend
        if refresh:
            self.refresh(backend)

    def refresh(self, backend: ToolBackend | None = None) -> int:
        backends = [backend] if backend is not None else list(self._backends.values())
        count = 0
        for item in backends:
            try:
                specs = item.list_specs()
            except Exception as exc:  # a broken backend must not kill MiniAgent
                log.warning("tool backend `%s` failed to list tools: %s", item.name, exc)
                continue
            for spec in specs:
                spec.server = item.name
                self._specs[spec.name] = spec
                count += 1
        return count

    # ------------------------------------------------------------------ lookup
    def names(self) -> list[str]:
        return sorted(self._specs)

    def specs(self, only: Iterable[str] | None = None) -> list[ToolSpec]:
        if only is None:
            return [self._specs[name] for name in self.names()]
        wanted = list(only)
        return [self._specs[name] for name in wanted if name in self._specs]

    def get(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise ToolNotFound(f"tool `{name}` is not registered") from exc

    def openai_schemas(
        self,
        *,
        extra_tools: Iterable[dict[str, Any]] = (),
        only: Iterable[str] | None = None,
        exclude: Iterable[str] = (),
    ) -> list[dict[str, Any]]:
        """Tool schemas in OpenAI function-calling format."""

        blocked = set(exclude)
        schemas = [self.get(name).to_openai_schema() for name in (only or self.names()) if name not in blocked]
        return schemas + list(extra_tools)

    # ------------------------------------------------------------------ calling
    def backend_for(self, name: str) -> ToolBackend:
        spec = self.get(name)
        try:
            return self._backends[spec.server]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ToolNotFound(f"no backend serves tool `{name}`") from exc

    def call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        allowed: Iterable[str] | None = None,
    ) -> str:
        if allowed is not None and name not in set(allowed):
            raise ToolPermissionError(
                f"tool `{name}` is not available in this context (allowed: {', '.join(sorted(allowed))})"
            )
        backend = self.backend_for(name)
        return backend.call(name, arguments)

    def render_catalog(self, *, only: Iterable[str] | None = None) -> str:
        lines = []
        for spec in self.specs(only):
            first_line = (spec.description or "").strip().splitlines()[0]
            lines.append(f"- {spec.name}: {first_line}")
        return "\n".join(lines) or "(no tools registered)"


__all__ = ["ToolRegistry", "ToolBackend"]
