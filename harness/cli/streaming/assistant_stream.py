"""Assistant streaming: keep the raw source, split it into stable + tail.

Rendering a delta on its own would break Markdown (``**hel`` + ``lo**``), so the
buffer always holds the raw source and only complete lines are considered stable.
See ``CLI_Design.md`` sections 17-22 and 36.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harness.cli.state import StreamState


@dataclass
class AssistantStream:
    """Two-region streaming over a raw Markdown source."""

    state: StreamState = field(default_factory=StreamState)
    #: number of trailing characters kept mutable as a safety margin
    holdback_chars: int = 0

    # ------------------------------------------------------------------- writing
    def append(self, delta: str) -> None:
        if not delta:
            return
        self.state.source += delta
        self._recommit()

    def finish(self, text: str | None = None) -> str:
        """Finalize: everything becomes stable and the canonical source is returned."""

        if text is not None:
            self.state.source = text
        self.state.committed_offset = len(self.state.source)
        self._split_lines()
        return self.state.source

    def reset(self) -> None:
        self.state = StreamState()

    # ------------------------------------------------------------------ boundary
    def _recommit(self) -> None:
        """Commit up to the last newline; the remainder stays mutable."""

        limit = len(self.state.source) - self.holdback_chars
        boundary = self.state.source.rfind("\n", self.state.committed_offset, max(limit, 0))
        if boundary >= 0:
            self.state.committed_offset = boundary + 1
        self._split_lines()

    def _split_lines(self) -> None:
        committed = self.state.committed_source
        pending = self.state.pending_source
        self.state.stable_lines = committed.splitlines()
        self.state.tail_lines = pending.splitlines() or ([] if not pending else [pending])

    # ------------------------------------------------------------------ reading
    @property
    def source(self) -> str:
        return self.state.source

    @property
    def committed_source(self) -> str:
        return self.state.committed_source

    @property
    def pending_source(self) -> str:
        return self.state.pending_source

    @property
    def stable_lines(self) -> list[str]:
        return self.state.stable_lines

    @property
    def tail_lines(self) -> list[str]:
        return self.state.tail_lines

    def preview(self, tail_lines: int = 6) -> str:
        """Stable text plus a bounded live tail (for incremental display)."""

        tail = self.tail_lines[-tail_lines:]
        parts = []
        if self.stable_lines:
            parts.append("\n".join(self.stable_lines))
        if tail:
            parts.append("\n".join(tail))
        return "\n".join(parts)


__all__ = ["AssistantStream"]
