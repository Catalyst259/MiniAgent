"""MemoryService: the only memory interface the rest of MiniAgent knows.

Dependency direction stays correct: the context builder depends on this service,
never on ``QdrantClient``.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from harness.agent.dto import MemoryRecord
from harness.context.prompts import render_transcript
from harness.memory.embedding import EmbeddingBackend
from harness.memory.store import MemoryStore

log = logging.getLogger(__name__)


class MemoryService:
    """Recall (read path) and remember (write path) over a memory store."""

    def __init__(
        self,
        store: MemoryStore,
        embedding: EmbeddingBackend,
        *,
        top_k: int = 5,
        score_threshold: float = 0.05,
        enabled: bool = True,
    ) -> None:
        self.store = store
        self.embedding = embedding
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.enabled = enabled
        self._ready = False
        self._last_error: str | None = None

    # -------------------------------------------------------------------- setup
    async def ensure_ready(self) -> None:
        if self._ready or not self.enabled:
            return
        try:
            await self.store.ensure_ready(self.embedding.dimensions)
        except Exception as exc:
            # A memory backend that cannot start (e.g. an embedded Qdrant folder
            # still locked by another process) must not take the whole turn down:
            # disable memory for this session and carry on without it.
            self._last_error = f"{type(exc).__name__}: {exc}"
            self.enabled = False
            if "Storage folder" in str(exc) and "already accessed" in str(exc):
                log.debug("memory disabled because the local Qdrant folder is busy: %s", self._last_error)
            else:
                log.warning("memory disabled: %s", self._last_error)
            return
        self._ready = True

    # ------------------------------------------------------------------ writing
    async def remember(
        self,
        records: Sequence[MemoryRecord],
        *,
        collection: str | None = None,  # accepted for API symmetry with stores
    ) -> int:
        del collection
        if not self.enabled or not records:
            return 0
        try:
            await self.ensure_ready()
            vectors = await self.embedding.embed([record.content for record in records])
            return await self.store.upsert(records, vectors)
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            log.warning("memory write failed: %s", self._last_error)
            return 0

    # ------------------------------------------------------------------ reading
    async def recall(
        self,
        query: str,
        top_k: int | None = None,
        *,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        if not self.enabled or not (query or "").strip():
            return []
        try:
            await self.ensure_ready()
            vector = await self.embedding.embed_one(query)
            return await self.store.search(
                vector,
                top_k=top_k or self.top_k,
                score_threshold=self.score_threshold,
                filters=filters,
            )
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            log.warning("memory recall failed: %s", self._last_error)
            return []

    async def recall_lines(self, query: str, top_k: int | None = None) -> list[str]:
        """Human/LLM-readable memory lines for the context builder."""

        hits = await self.recall(query, top_k)
        lines: list[str] = []
        for record, score in hits:
            source = f" [{record.memory_type}" + (f", {record.source}" if record.source else "") + "]"
            lines.append(f"({score:.2f}){source} {record.content.strip()}")
        return lines

    # ------------------------------------------------------------------- status
    async def status(self) -> dict[str, Any]:
        count = 0
        try:
            await self.ensure_ready()
            count = await self.store.count()
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
        return {
            "enabled": self.enabled,
            "backend": getattr(self.store, "name", type(self.store).__name__),
            "embedding": getattr(self.embedding, "name", type(self.embedding).__name__),
            "dimensions": self.embedding.dimensions,
            "records": count,
            "last_error": self._last_error,
        }

    def as_provider(self):
        """Adapter for ``ContextBuilder(memory_provider=...)``."""

        async def provider(query: str, k: int) -> list[str]:
            return await self.recall_lines(query, k)

        return provider

    def close(self) -> None:
        """Release backend resources (embedded Qdrant holds a folder lock)."""

        closer = getattr(self.store, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception as exc:  # pragma: no cover - teardown only
                log.debug("closing memory store failed: %s", exc)


def transcript_of(messages, *, limit: int = 60) -> str:
    """Render recent messages for summary/memory formation."""

    entries: list[str] = []
    for message in list(messages)[-limit:]:
        content = message.content or ""
        if message.role == "assistant" and message.tool_calls:
            calls = ", ".join(call.name for call in message.tool_calls)
            entries.append(f"ASSISTANT: {content.strip()}\n  [calls] {calls}")
        elif message.role == "tool":
            entries.append(f"TOOL {message.name or ''}: {content.strip()[:600]}")
        else:
            entries.append(f"{message.role.upper()}: {content.strip()[:2000]}")
    return render_transcript(entries)


__all__ = ["MemoryService", "transcript_of"]
