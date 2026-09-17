"""Long-term memory storage.

``Global Memory != Agent State``: this is cross-task, semantic, and read by the
context builder through the :class:`MemoryService` interface only.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, Sequence, runtime_checkable

from harness.agent.dto import MemoryRecord

log = logging.getLogger(__name__)


@runtime_checkable
class MemoryStore(Protocol):
    async def ensure_ready(self, dimensions: int) -> None: ...

    async def upsert(self, records: Sequence[MemoryRecord], vectors: Sequence[Sequence[float]]) -> int: ...

    async def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[MemoryRecord, float]]: ...

    async def count(self) -> int: ...

    async def clear(self) -> None: ...


class InMemoryMemoryStore:
    """Cosine-similarity store; used for tests and for ``memory.backend: memory``."""

    name = "memory"

    def __init__(self) -> None:
        self._records: list[MemoryRecord] = []
        self._vectors: list[list[float]] = []

    async def ensure_ready(self, dimensions: int) -> None:  # noqa: ARG002 - protocol
        return None

    async def upsert(self, records: Sequence[MemoryRecord], vectors: Sequence[Sequence[float]]) -> int:
        existing = {record.id: index for index, record in enumerate(self._records)}
        written = 0
        for record, vector in zip(records, vectors):
            if record.id in existing:
                index = existing[record.id]
                self._records[index] = record
                self._vectors[index] = list(vector)
            else:
                self._records.append(record)
                self._vectors.append(list(vector))
            written += 1
        return written

    async def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        scored: list[tuple[MemoryRecord, float]] = []
        for record, stored in zip(self._records, self._vectors):
            if filters and not _matches(record, filters):
                continue
            score = _cosine(vector, stored)
            if score >= score_threshold:
                scored.append((record, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    async def count(self) -> int:
        return len(self._records)

    async def clear(self) -> None:
        self._records.clear()
        self._vectors.clear()


class QdrantMemoryStore:
    """Qdrant Local (embedded, on-disk) implementation of the same interface."""

    name = "qdrant_local"

    def __init__(self, path: str, collection: str = "miniagent_memory") -> None:
        self.path = path
        self.collection = collection
        self._client = None
        self._dimensions: int | None = None

    # ------------------------------------------------------------------- client
    def _get_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(path=self.path)
        return self._client

    async def ensure_ready(self, dimensions: int) -> None:
        from qdrant_client.http import models as qmodels

        client = self._get_client()
        self._dimensions = dimensions
        existing = {collection.name for collection in client.get_collections().collections}
        if self.collection not in existing:
            client.create_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(size=dimensions, distance=qmodels.Distance.COSINE),
            )
            return
        info = client.get_collection(self.collection)
        size = None
        try:
            size = info.config.params.vectors.size  # type: ignore[union-attr]
        except AttributeError:  # pragma: no cover - qdrant version differences
            size = None
        if size is not None and int(size) != int(dimensions):
            raise ValueError(
                f"collection `{self.collection}` has dimension {size} but the embedding backend "
                f"produces {dimensions}; delete {self.path} or change the embedding config"
            )

    # -------------------------------------------------------------------- write
    async def upsert(self, records: Sequence[MemoryRecord], vectors: Sequence[Sequence[float]]) -> int:
        from qdrant_client.http import models as qmodels

        if not records:
            return 0
        client = self._get_client()
        points = [
            qmodels.PointStruct(
                id=_point_id(record.id),
                vector=list(vector),
                payload={**record.model_dump(mode="json"), "id": record.id},
            )
            for record, vector in zip(records, vectors)
        ]
        client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(points)

    # --------------------------------------------------------------------- read
    async def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        client = self._get_client()
        query_filter = _qdrant_filter(filters) if filters else None
        try:
            response = client.query_points(
                collection_name=self.collection,
                query=list(vector),
                limit=top_k,
                score_threshold=score_threshold or None,
                query_filter=query_filter,
                with_payload=True,
            )
            hits = response.points
        except AttributeError:  # pragma: no cover - older qdrant-client
            hits = client.search(
                collection_name=self.collection,
                query_vector=list(vector),
                limit=top_k,
                score_threshold=score_threshold or None,
                query_filter=query_filter,
                with_payload=True,
            )
        out: list[tuple[MemoryRecord, float]] = []
        for hit in hits:
            payload = dict(hit.payload or {})
            payload.pop("id", None)
            try:
                record = MemoryRecord(**payload)
            except Exception:  # pragma: no cover - payload drift
                continue
            out.append((record, float(hit.score or 0.0)))
        return out

    async def count(self) -> int:
        client = self._get_client()
        try:
            return int(client.count(self.collection, exact=True).count)
        except Exception:  # pragma: no cover - empty/missing collection
            return 0

    def close(self) -> None:
        """Release the embedded storage lock.

        Qdrant Local allows only one client per storage folder, so the CLI
        releases it on exit and a second harness can then open the same path.
        """

        client = self._client
        self._client = None
        if client is None:
            return
        try:
            client.close()
        except TypeError:  # pragma: no cover - older clients are not context managers
            try:
                client.__exit__(None, None, None)
            except Exception as exc:
                log.debug("closing qdrant client failed: %s", exc)
        except Exception as exc:  # pragma: no cover
            log.debug("closing qdrant client failed: %s", exc)

    async def clear(self) -> None:
        client = self._get_client()
        try:
            client.delete_collection(self.collection)
        except Exception as exc:  # pragma: no cover
            log.debug("clearing collection failed: %s", exc)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _matches(record: MemoryRecord, filters: dict[str, Any]) -> bool:
    for key, value in filters.items():
        actual = getattr(record, key, None)
        if actual is None:
            actual = (record.metadata or {}).get(key)
        if actual != value:
            return False
    return True


def _point_id(record_id: str) -> str:
    import uuid

    try:
        return str(uuid.UUID(record_id))
    except (ValueError, AttributeError, TypeError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"harness-memory:{record_id}"))


def _qdrant_filter(filters: dict[str, Any]):
    from qdrant_client.http import models as qmodels

    conditions = [
        qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value))
        for key, value in filters.items()
    ]
    return qmodels.Filter(must=conditions) if conditions else None


def build_memory_store(backend: str, *, path: str, collection: str) -> MemoryStore:
    if backend == "memory":
        return InMemoryMemoryStore()
    if backend == "qdrant_local":
        return QdrantMemoryStore(path, collection)
    raise ValueError(f"unknown memory backend `{backend}`")


__all__ = [
    "MemoryStore",
    "InMemoryMemoryStore",
    "QdrantMemoryStore",
    "build_memory_store",
]
