"""Model abstraction.

MiniAgent only ever talks to :class:`ModelGateway`.  Everything provider
specific lives behind that protocol, in
:mod:`harness.inference.openai_compatible`.
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

from harness.agent.dto import ModelRequest, ModelResponse


@runtime_checkable
class ModelGateway(Protocol):
    """Provider-agnostic chat interface."""

    name: str

    async def chat(self, request: ModelRequest) -> ModelResponse:
        """Run one completion and return a *normalized* response."""
        ...

    def stream(self, request: ModelRequest) -> AsyncIterator[str]:
        """Yield text deltas.  Optional for gateways; MiniAgent tolerates a
        single-chunk implementation."""
        ...


__all__ = ["ModelGateway"]
