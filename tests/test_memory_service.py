"""Regression tests for MemoryService startup resilience.

A memory backend that cannot start (an embedded Qdrant folder still locked by
another process is the common case) must degrade to "memory off" instead of
taking the whole turn down with it.
"""

from __future__ import annotations

from typing import Any, Sequence

from harness.agent.dto import MemoryRecord
from harness.memory.service import MemoryService


class _BrokenStore:
    """A store whose ``ensure_ready`` always fails, like a locked Qdrant folder."""

    name = "broken"

    async def ensure_ready(self, dimensions: int) -> None:
        raise RuntimeError("Storage folder is already accessed by another instance")

    async def upsert(self, records: Sequence[MemoryRecord], vectors: Sequence[Sequence[float]]) -> int:
        raise AssertionError("upsert must not be reached once memory is disabled")

    async def search(self, vector, *, top_k=5, score_threshold=0.0, filters=None):  # noqa: ANN001
        raise AssertionError("search must not be reached once memory is disabled")

    async def count(self) -> int:
        return 0

    async def clear(self) -> None:
        return None


class _Embedding:
    name = "stub"
    dimensions = 8

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] * self.dimensions for _ in texts]

    async def embed_one(self, text: str) -> list[float]:
        return [0.0] * self.dimensions


async def test_ensure_ready_degrades_instead_of_raising():
    service = MemoryService(_BrokenStore(), _Embedding())

    # Must not raise: startup calls this directly and a raise aborts the turn.
    await service.ensure_ready()

    assert service.enabled is False
    assert service._ready is False
    assert "already accessed" in (service._last_error or "")


async def test_disabled_service_is_inert_after_failure():
    service = MemoryService(_BrokenStore(), _Embedding())
    await service.ensure_ready()

    # Every public path must now be a no-op rather than touching the store.
    assert await service.remember([MemoryRecord(id="1", content="x", memory_type="fact")]) == 0
    assert await service.recall("anything") == []
    assert await service.recall_lines("anything") == []

    status: dict[str, Any] = await service.status()
    assert status["enabled"] is False
    assert status["last_error"]


async def test_healthy_store_still_becomes_ready():
    from harness.memory.store import InMemoryMemoryStore

    service = MemoryService(InMemoryMemoryStore(), _Embedding())
    await service.ensure_ready()

    assert service.enabled is True
    assert service._ready is True
    assert service._last_error is None
