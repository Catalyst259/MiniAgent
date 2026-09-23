"""Interactive adapter for user choice requests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from harness.interaction import Choice, InteractionCancelled


@dataclass
class PendingInteraction:
    title: str
    question: str
    options: list[Choice]
    future: asyncio.Future[Choice]
    detail: str = ""
    selected: int = 0
    on_resolve: Callable[[Choice], None] | None = None
    on_cancel: Callable[[str], None] | None = None
    activity_text: str = "waiting for your choice"

    @property
    def current(self) -> Choice:
        return self.options[self.selected]

    def move(self, delta: int) -> None:
        if self.options:
            self.selected = (self.selected + delta) % len(self.options)


class InteractiveInteractionProvider:
    """One modal choice at a time, rendered and answered by prompt_toolkit."""

    available = True

    def __init__(
        self,
        *,
        state: Any,
        on_change: Callable[[], None] | None = None,
        activity: Callable[[str], None] | None = None,
        answerable: Callable[[], bool] | None = None,
    ) -> None:
        self.state = state
        self.on_change = on_change
        self.activity = activity
        self.pending: PendingInteraction | None = None
        self.history: list[tuple[str, str]] = []
        self._answerable: Callable[[], bool] = answerable if answerable is not None else lambda: True

    def request(
        self,
        question: str,
        options: Sequence[Choice],
        *,
        title: str = "Choose an option",
        detail: str = "",
        on_resolve: Callable[[Choice], None] | None = None,
        on_cancel: Callable[[str], None] | None = None,
        activity_text: str = "waiting for your choice",
    ) -> asyncio.Future[Choice]:
        if self.pending is not None:
            raise RuntimeError("another user interaction is already pending")
        if not self.can_answer:
            raise RuntimeError("no interaction prompt can be shown right now")
        choices = list(options)
        if not choices:
            raise ValueError("an interaction needs at least one option")
        future = asyncio.get_event_loop().create_future()
        self.pending = PendingInteraction(
            title=title,
            question=question,
            options=choices,
            future=future,
            detail=detail,
            on_resolve=on_resolve,
            on_cancel=on_cancel,
            activity_text=activity_text,
        )
        self.state.interaction = self.pending
        self._changed()
        return future

    async def choose(
        self,
        question: str,
        options: Sequence[Choice],
        *,
        title: str = "Choose an option",
        detail: str = "",
    ) -> Choice:
        return await self.request(question, options, title=title, detail=detail)

    def replace_options(self, options: Sequence[Choice]) -> None:
        if self.pending is None:
            return
        choices = list(options)
        if choices:
            self.pending.options = choices
            self.pending.selected = 0
            self._changed()

    def move(self, delta: int) -> None:
        if self.pending is not None:
            self.pending.move(delta)
            self._changed()

    def resolve(self, value: str | None = None) -> bool:
        pending = self.pending
        if pending is None:
            return False
        choice = pending.current if value is None else next(
            (option for option in pending.options if option.value == value), None
        )
        if choice is None or pending.future.done():
            return False
        self.pending = None
        self.state.interaction = None
        self.history.append((pending.question, choice.value))
        if pending.on_resolve is not None:
            pending.on_resolve(choice)
        pending.future.set_result(choice)
        self._changed()
        return True

    def resolve_index(self, index: int) -> bool:
        if self.pending is None or not 0 <= index < len(self.pending.options):
            return False
        return self.resolve(self.pending.options[index].value)

    def accept_selection(self) -> bool:
        return self.resolve()

    def cancel(self, note: str = "the interaction was cancelled") -> bool:
        pending = self.pending
        if pending is None:
            return False
        self.pending = None
        self.state.interaction = None
        if pending.on_cancel is not None:
            pending.on_cancel(note)
        if not pending.future.done():
            pending.future.set_exception(InteractionCancelled(note))
        self.history.append((pending.question, "cancelled"))
        self._changed()
        return True

    @property
    def waiting(self) -> bool:
        return self.pending is not None

    @property
    def can_answer(self) -> bool:
        return self._answerable()

    def fragments(self) -> list[tuple[str, str]]:
        pending = self.pending
        if pending is None:
            return []
        fragments: list[tuple[str, str]] = [
            ("class:interaction-title", f" ? {pending.title}\n"),
            ("class:interaction-question", f" {pending.question}\n"),
        ]
        if pending.detail:
            fragments.append(("class:interaction-detail", f" {pending.detail}\n"))
        for index, option in enumerate(pending.options):
            marker = "▸" if index == pending.selected else " "
            style = (
                "class:interaction-choice"
                if index == pending.selected
                else "class:interaction-dim"
            )
            label = f" {marker}{index + 1}) {option.label}"
            if option.description:
                label += f" — {option.description}"
            fragments.append((style, label + "\n"))
        fragments.append(
            ("class:interaction-dim", " Enter: confirm · ↑/↓ or ←/→: move · Ctrl+R: cancel")
        )
        return fragments

    def _changed(self) -> None:
        if self.activity is not None:
            text = self.pending.activity_text if self.pending is not None else ""
            self.activity(text)
        if self.on_change is not None:
            self.on_change()


__all__ = [
    "InteractiveInteractionProvider",
    "PendingInteraction",
]
