"""LangGraph checkpointer adapter.

Runtime state persistence is delegated to LangGraph's SQLite checkpointer; the
harness never implements ``save_state``/``load_state``/``resume_state`` itself.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class AsyncCheckpointStore:
    """Async SQLite checkpointer (used by the CLI)."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._saver = None
        self._cm = None
        self._loop_tick: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Any:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        # aiosqlite completes operations on a worker thread.  Some restricted
        # event-loop hosts can lose that thread's wake-up byte and then sleep
        # forever despite the result already being ready.  A scoped low-rate
        # timer keeps the selector advancing while this store is open.
        self._loop_tick = asyncio.create_task(self._keep_loop_awake())
        self._cm = AsyncSqliteSaver.from_conn_string(self.path)
        try:
            self._saver = await self._cm.__aenter__()
            await self._saver.setup()
            return self._saver
        except BaseException:
            await self._stop_loop_tick()
            raise

    async def __aexit__(self, *exc_info) -> None:
        try:
            if self._cm is not None:
                await self._cm.__aexit__(*exc_info)
                self._cm = None
                self._saver = None
        finally:
            await self._stop_loop_tick()

    @staticmethod
    async def _keep_loop_awake() -> None:
        while True:
            await asyncio.sleep(0.02)

    async def _stop_loop_tick(self) -> None:
        task = self._loop_tick
        self._loop_tick = None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


class SyncCheckpointStore:
    """Sync SQLite checkpointer (tests, one-shot runs)."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._saver = None
        self._cm = None

    def __enter__(self) -> Any:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        from langgraph.checkpoint.sqlite import SqliteSaver

        # ``from_conn_string`` returns a *context manager* that owns the sqlite
        # connection, so it has to be entered (and kept) rather than unpacked.
        self._cm = SqliteSaver.from_conn_string(self.path)
        self._saver = self._cm.__enter__()
        self._saver.setup()
        return self._saver

    def __exit__(self, *exc_info) -> None:
        if self._cm is not None:
            self._cm.__exit__(*exc_info)
            self._cm = None
            self._saver = None


def memory_checkpointer() -> Any:
    """In-process checkpointer - handy for tests and short-lived runs."""

    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


def sqlite_checkpointer(path: str) -> Any:
    """One-shot sync checkpointer; caller owns the returned context manager."""

    return SyncCheckpointStore(path)


def list_threads(path: str) -> list[str]:
    """Best-effort listing of checkpointed threads (for ``/clear`` and debugging)."""

    import sqlite3

    if not Path(path).exists():
        return []
    try:
        connection = sqlite3.connect(path)
        try:
            rows = connection.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
            return sorted(str(row[0]) for row in rows)
        finally:
            connection.close()
    except sqlite3.Error as exc:  # pragma: no cover - schema differences
        log.debug("cannot list threads: %s", exc)
        return []


__all__ = [
    "AsyncCheckpointStore",
    "SyncCheckpointStore",
    "memory_checkpointer",
    "sqlite_checkpointer",
    "list_threads",
]
