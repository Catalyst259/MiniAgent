"""Embedding backend, independent of the main model.

The default ``hashing`` backend needs no model download and works offline, so
MiniAgent is usable and testable without any network access.  Point
``embedding.backend`` at ``openai_compatible`` (or ``sentence_transformers``
locally) when real semantic quality matters.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


@runtime_checkable
class EmbeddingBackend(Protocol):
    dimensions: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_one(self, text: str) -> list[float]: ...


class HashingEmbeddingBackend:
    """Deterministic feature hashing (unigrams + bigrams), L2 normalized."""

    name = "hashing"

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = max(16, int(dimensions))

    def _tokens(self, text: str) -> list[str]:
        lowered = (text or "").lower()
        words = _TOKEN_RE.findall(lowered)
        tokens = list(words)
        tokens.extend(f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1))
        tokens.extend(f"{word[:4]}*" for word in words if len(word) > 4)
        return tokens

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def embed_one(self, text: str) -> list[float]:
        return self._vector(text)


class OpenAICompatibleEmbeddingBackend:
    """Remote ``/embeddings`` endpoint (OpenAI, vLLM, TEI, ...)."""

    name = "openai_compatible"

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str = "EMPTY",
        dimensions: int = 1024,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.dimensions = int(dimensions)
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("the `openai` package is required for remote embeddings") from exc
            kwargs: dict = {"api_key": self.api_key, "timeout": self.timeout}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._get_client().embeddings.create(model=self.model, input=list(texts))
        vectors = [list(item.embedding) for item in response.data]
        if vectors:
            self.dimensions = len(vectors[0])
        return vectors

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]


class SentenceTransformersEmbeddingBackend:
    """Local model (e.g. Qwen3-Embedding) through ``sentence-transformers``."""

    name = "sentence_transformers"

    def __init__(self, model: str, *, dimensions: int = 1024) -> None:
        self.model_name = model
        self.dimensions = int(dimensions)
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "sentence-transformers is not installed; `pip install sentence-transformers`"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
            self.dimensions = int(self._model.get_sentence_embedding_dimension() or self.dimensions)
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import asyncio

        model = await asyncio.to_thread(self._load)
        vectors = await asyncio.to_thread(
            model.encode, list(texts), normalize_embeddings=True
        )
        return [list(map(float, vector)) for vector in vectors]

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]


def build_embedding_backend(
    backend: str = "hashing",
    *,
    dimensions: int = 256,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> EmbeddingBackend:
    import os

    key = api_key or (os.environ.get(api_key_env) if api_key_env else None) or "EMPTY"
    if backend == "hashing":
        return HashingEmbeddingBackend(dimensions=dimensions)
    if backend == "openai_compatible":
        if not model:
            raise ValueError("embedding.backend=openai_compatible requires embedding.model")
        return OpenAICompatibleEmbeddingBackend(
            model=model, base_url=base_url, api_key=key, dimensions=dimensions
        )
    if backend == "sentence_transformers":
        if not model:
            raise ValueError("embedding.backend=sentence_transformers requires embedding.model")
        return SentenceTransformersEmbeddingBackend(model, dimensions=dimensions)
    raise ValueError(f"unknown embedding backend `{backend}`")


__all__ = [
    "EmbeddingBackend",
    "HashingEmbeddingBackend",
    "OpenAICompatibleEmbeddingBackend",
    "SentenceTransformersEmbeddingBackend",
    "build_embedding_backend",
]
