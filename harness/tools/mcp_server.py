"""MCP server exposing MiniAgent tools.

Run it directly to talk to any MCP client:

```bash
python -m harness.tools.mcp_server --root /path/to/workspace
```

The CLI uses the in-process backend by default and this server only when
``--mcp-stdio`` is requested, but the tool schemas are produced from the same
function signatures, so both paths stay identical.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
from contextlib import suppress
from pathlib import Path
from typing import Any

from harness.tools.fs_tools import TOOL_FUNCS, ToolContext
from harness.tools.paths import Workspace


def build_server(root: Path, *, max_output_chars: int = 30_000):
    from mcp.server.mcpserver import MCPServer

    ctx = ToolContext(workspace=Workspace(root), max_output_chars=max_output_chars)
    server = MCPServer(
        name="miniagent-tools",
        instructions="Filesystem, search, patching and shell tools for a coding agent.",
    )

    for spec in TOOL_FUNCS.values():
        server.tool(
            name=spec.name,
            description=spec.description,
            structured_output=False,
        )(_make_handler(spec, ctx))
    return server


def _make_handler(spec, ctx: ToolContext):
    """Build a handler whose *signature* mirrors the tool's JSON schema.

    The MCP SDK derives the tool's input schema from the Python signature, so a
    single ``**kwargs`` handler would publish ``kwargs`` as the only accepted
    parameter.  Mirroring the schema keeps MCP clients and the in-process
    runtime seeing exactly the same arguments.
    """

    properties: dict[str, Any] = (spec.parameters or {}).get("properties") or {}
    required = set((spec.parameters or {}).get("required") or [])
    annotations = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    def handler(**kwargs: Any) -> str:
        allowed = set(properties)
        unknown = set(kwargs) - allowed
        if unknown:
            # Raise rather than return: MCP then marks the result as an error,
            # which the client adapter normalizes into an ERROR observation.
            raise ValueError(
                f"unexpected argument(s) for `{spec.name}`: {', '.join(sorted(unknown))}; "
                f"allowed: {', '.join(sorted(allowed))}"
            )
        # Optional arguments left unset must fall back to the tool's own
        # defaults, so drop explicit ``None`` values before dispatching.
        cleaned = {
            key: value for key, value in kwargs.items() if not (value is None and key not in required)
        }
        missing = [key for key in required if cleaned.get(key) is None]
        if missing:
            return f"ERROR: missing required argument(s): {', '.join(sorted(missing))}"
        return spec.fn(ctx, **cleaned)

    handler.__name__ = spec.name
    handler.__doc__ = spec.description
    handler.__signature__ = inspect.Signature(
        parameters=[
            inspect.Parameter(
                name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=annotations.get((definition or {}).get("type"), Any),
                default=inspect.Parameter.empty if name in required else None,
            )
            for name, definition in properties.items()
        ],
        return_annotation=str,
    )
    return handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="harness MCP tool server")
    parser.add_argument("--root", default=".", help="workspace root the tools are confined to")
    parser.add_argument("--max-output-chars", type=int, default=30_000)
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"])
    args = parser.parse_args(argv)

    server = build_server(Path(args.root), max_output_chars=args.max_output_chars)
    if args.transport == "stdio":
        asyncio.run(_run_stdio(server))
    else:
        server.run(args.transport)
    return 0


async def _run_stdio(server: Any) -> None:
    """Run stdio while periodically advancing thread-delivered pipe reads."""

    async def tick() -> None:
        while True:
            await asyncio.sleep(0.02)

    ticker = asyncio.create_task(tick())
    try:
        await server.run_stdio_async()
    finally:
        ticker.cancel()
        with suppress(asyncio.CancelledError):
            await ticker


if __name__ == "__main__":  # pragma: no cover - server entry point
    raise SystemExit(main())
