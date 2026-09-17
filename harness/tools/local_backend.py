"""In-process tool backend.

This is the default backend: it calls the same functions the MCP server exposes,
which keeps the CLI fast (no subprocess per call) while remaining protocol
compatible - the schema served here is byte-identical to the MCP schema.
"""

from __future__ import annotations

from typing import Any

from harness.agent.dto import ToolSpec
from harness.tools.fs_tools import TOOL_FUNCS, ToolContext, call_tool


class LocalToolBackend:
    name = "local"

    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx

    def list_specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name=spec.name,
                description=spec.description,
                input_schema=spec.parameters,
                server=self.name,
            )
            for spec in TOOL_FUNCS.values()
        ]

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        return call_tool(name, self.ctx, dict(arguments or {}))


__all__ = ["LocalToolBackend"]
