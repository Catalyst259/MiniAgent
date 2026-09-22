"""MCP client adapter.

Mirrors :class:`~harness.tools.local_backend.LocalToolBackend` but talks to a
real MCP server over stdio, so MiniAgent can use *any* MCP server and not only
its built-in tools.

The MCP SDK is asyncio-based while the tool runtime is synchronous (the graph
nodes run tools in a worker thread), so this adapter owns a private event loop
in a daemon thread and submits coroutines to it.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import sys
import threading
from pathlib import Path
from typing import Any

from harness.agent.dto import ToolSpec
from harness.agent.errors import ToolError

log = logging.getLogger(__name__)


class _LoopThread:
    """A single background event loop shared by every MCP session."""

    _instance: "_LoopThread | None" = None

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="mcp-loop", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        # Keep the selector from sleeping indefinitely if this host loses the
        # wake-up byte written by ``run_coroutine_threadsafe``.  MCP requests
        # cross into this loop from the tool worker thread, so the same scoped
        # short tick used by the tool runtime is required here as well.
        self.loop.call_soon(self._tick)
        self.loop.run_forever()

    def _tick(self) -> None:
        if self.loop.is_running():
            self.loop.call_later(0.02, self._tick)

    def run(self, coro, timeout: float | None = None):
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result(timeout=timeout)

    @classmethod
    def shared(cls) -> "_LoopThread":
        if cls._instance is None:
            cls._instance = cls()
            atexit.register(cls._shutdown)
        return cls._instance

    @classmethod
    def _shutdown(cls) -> None:  # pragma: no cover - process teardown
        if cls._instance is not None:
            cls._instance.loop.call_soon_threadsafe(cls._instance.loop.stop)


class MCPClientAdapter:
    """One stdio MCP session, exposed as a synchronous tool backend."""

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        *,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        name: str | None = None,
        call_timeout: float = 300.0,
    ) -> None:
        self.command = command
        self.args = list(args or [])
        self.env = env
        self.cwd = str(cwd) if cwd else None
        self.name = name or f"mcp:{Path(self.command).name}"
        self.call_timeout = call_timeout
        self._loop = _LoopThread.shared()
        self._requests: asyncio.Queue | None = None
        self._runner: asyncio.Future | None = None
        self._closing = False

    # ------------------------------------------------------------------ session
    async def _run_session(self) -> None:
        """Own the whole MCP session inside one task.

        anyio task groups require the context managers to be entered and exited
        by the same task, so every request is funnelled through this single
        coroutine (started lazily on the shared loop).
        """

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self.command, args=self.args, env=self.env, cwd=self.cwd
        )
        queue = self._requests
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    while True:
                        request = await queue.get()
                        if request is None:
                            break
                        kind, payload, future = request
                        if future.cancelled():
                            continue
                        try:
                            if kind == "list":
                                future.set_result(await session.list_tools())
                            else:
                                name, arguments = payload
                                future.set_result(await session.call_tool(name, arguments))
                        except Exception as exc:  # normalized by the caller
                            future.set_exception(exc)
        except Exception as exc:  # pragma: no cover - transport level failure
            log.debug("MCP session for %s ended: %s", self.name, exc)
            if queue is not None:
                while not queue.empty():
                    request = queue.get_nowait()
                    if request is not None and not request[2].done():
                        request[2].set_exception(exc)

    async def _request(self, kind: str, payload: Any):
        if self._requests is None:
            self._requests = asyncio.Queue()
        if self._runner is None or self._runner.done():
            if self._closing:
                raise ToolError(f"MCP backend `{self.name}` is closed")
            self._runner = self._loop.loop.create_task(self._run_session())
        future: asyncio.Future = self._loop.loop.create_future()
        await self._requests.put((kind, payload, future))
        return await future

    def close(self) -> None:  # pragma: no cover - teardown only
        self._closing = True
        if self._requests is None or self._runner is None:
            return
        try:
            self._loop.run(self._requests.put(None), timeout=5)
        except Exception as exc:
            log.debug("closing MCP session failed: %s", exc)

    # -------------------------------------------------------------------- tools
    def list_specs(self) -> list[ToolSpec]:
        try:
            result = self._loop.run(self._request("list", None), timeout=60)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"MCP list_tools failed: {type(exc).__name__}: {exc}") from exc
        specs: list[ToolSpec] = []
        for tool in getattr(result, "tools", []) or []:
            specs.append(
                ToolSpec(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=getattr(tool, "input_schema", None)
                    or getattr(tool, "inputSchema", None)
                    or {"type": "object", "properties": {}},
                    server=self.name,
                )
            )
        return specs

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            result = self._loop.run(
                self._request("call", (name, dict(arguments or {}))), timeout=self.call_timeout
            )
        except Exception as exc:
            raise ToolError(f"MCP call `{name}` failed: {type(exc).__name__}: {exc}") from exc
        return _render_result(result)


def _render_result(result: Any) -> str:
    """Flatten an MCP tool result into the observation string the model reads."""

    parts: list[str] = []
    for item in getattr(result, "content", None) or []:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(text)
            continue
        data = getattr(item, "data", None)
        if data is not None:
            parts.append(f"[binary content: {len(str(data))} bytes]")
            continue
        parts.append(str(item))
    if getattr(result, "structured_content", None):
        parts.append(str(result.structured_content))
    if getattr(result, "is_error", False):
        return "ERROR: " + ("\n".join(parts) or "tool reported an error")
    return "\n".join(parts) or "(no output)"


def build_stdio_backend(
    workspace_root: str | Path,
    *,
    python_executable: str | None = None,
    extra_args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> MCPClientAdapter:
    """Backend that runs MiniAgent' own MCP server in a subprocess."""

    import os

    server_env = dict(os.environ)
    if env:
        server_env.update(env)
    server_env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + server_env.get(
        "PYTHONPATH", ""
    )
    return MCPClientAdapter(
        command=python_executable or sys.executable,
        args=["-m", "harness.tools.mcp_server", "--root", str(workspace_root), *(extra_args or [])],
        env=server_env,
        cwd=str(workspace_root),
        name="mcp:miniagent-tools",
    )


__all__ = ["MCPClientAdapter", "build_stdio_backend"]
