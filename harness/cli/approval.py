"""Permission approval adapters.

The TUI adapter translates permission scopes to the same generic choice modal
used by model-requested questions.  Permission policy stays in the permission
layer; selection state, rendering and keys stay in one UI component.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from harness.cli.interaction import InteractiveInteractionProvider
from harness.interaction import Choice
from harness.permission.action import Action
from harness.permission.approval import Approval, ApprovalRequest

@dataclass
class PendingApproval:
    """Permission metadata while the shared choice modal is open."""

    action: Action
    reason: str
    future: asyncio.Future


def build_options(data: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Which choices to offer, from the ``permission_ask`` event payload.

    "Always allow" and "Allow this session" are hidden when the action cannot be
    expressed as a standing rule (a chained shell command, an ``apply_patch``);
    offering them would promise something the permission layer cannot keep.
    """

    can_session = bool(data.get("can_session"))
    can_persist = bool(data.get("can_persist"))
    options: list[tuple[str, str, str]] = [("once", "Allow once", "1")]
    if can_session:
        options.append(("session", "Allow this session", "2"))
    if can_persist:
        options.append(("persistent", "Always allow", "3"))
    options.append(("reject", "Reject", "4"))
    return options


class InteractiveApprovalProvider:
    """Adapt permission requests to an :class:`InteractiveInteractionProvider`."""

    interactive = True

    def __init__(
        self,
        *,
        state: Any = None,
        interaction: InteractiveInteractionProvider | None = None,
        on_change: Callable[[], None] | None = None,
        activity: Callable[[str], None] | None = None,
    ) -> None:
        if interaction is None:
            if state is None:
                raise ValueError("state or interaction is required")
            interaction = InteractiveInteractionProvider(
                state=state,
                on_change=on_change,
                activity=activity,
            )
        self.interaction = interaction
        self.state = interaction.state
        self.pending: PendingApproval | None = None
        self.history: list[tuple[str, str]] = []
        self._options_factory: Callable[[dict[str, Any]], list[tuple[str, str, str]]] = build_options
        self._note = ""
        self._aborting = False

    # ------------------------------------------------------------------ protocol
    def request(self, action: Action, reason: str) -> ApprovalRequest:
        """Arm the question and show it; the gate awaits the returned request."""

        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self.pending = PendingApproval(action=action, reason=reason, future=future)
        self._note = ""
        self._aborting = False
        try:
            choice_future = self.interaction.request(
                action.describe(),
                self._choices([("once", "Allow once", "1"), ("reject", "Reject", "4")]),
                title="Permission required",
                detail=reason,
                on_resolve=self._selected,
                on_cancel=self._cancelled,
                activity_text="waiting for your approval",
            )
        except Exception:
            self.pending = None
            raise
        # The permission gate awaits ``future`` rather than the modal's Choice
        # future.  Drain a cancellation exception from the latter so aborting a
        # turn cannot create an unhandled-future warning.
        choice_future.add_done_callback(self._drain_choice_future)
        return ApprovalRequest(action=action, future=future)

    def offer(self, data: dict[str, Any]) -> None:
        """Apply the choice list carried by the ``permission_ask`` event."""

        if self.pending is None:
            return
        options = self._options_factory(data)
        if options:
            self.interaction.replace_options(self._choices(options))

    # -------------------------------------------------------------------- answer
    def resolve(self, scope: str | None = None, *, note: str = "") -> bool:
        pending = self.pending
        if pending is None:
            return False
        self._note = note
        return self.interaction.resolve(scope)

    def move(self, delta: int) -> None:
        self.interaction.move(delta)

    def accept_selection(self) -> bool:
        return self.interaction.accept_selection()

    def reject(self) -> bool:
        return self.resolve("reject")

    def cancel(self) -> bool:
        """Drop a pending question without an answer (turn aborted).

        The waiting gate must not be left holding a future nobody will ever
        resolve, and the status line must not keep advertising a question that no
        longer belongs to the running turn.
        """

        pending = self.pending
        if pending is None:
            return False
        self._aborting = True
        return self.interaction.cancel("the turn was cancelled")

    @property
    def waiting(self) -> bool:
        return self.pending is not None

    @property
    def can_answer(self) -> bool:
        return self.interaction.can_answer

    @staticmethod
    def _choices(options: list[tuple[str, str, str]]) -> list[Choice]:
        return [Choice(value=scope, label=label) for scope, label, _key in options]

    def _selected(self, choice: Choice) -> None:
        pending = self.pending
        if pending is None:
            return
        if not pending.future.done():
            pending.future.set_result(Approval(scope=choice.value, note=self._note))
        self.history.append((pending.action.describe(), choice.value))
        self.pending = None
        self._note = ""

    def _cancelled(self, note: str) -> None:
        pending = self.pending
        if pending is None:
            return
        if not pending.future.done():
            pending.future.set_result(Approval(scope="reject", note=note))
        self.history.append(
            (pending.action.describe(), "cancelled" if self._aborting else "reject")
        )
        self.pending = None
        self._note = ""
        self._aborting = False

    @staticmethod
    def _drain_choice_future(future: asyncio.Future) -> None:
        if future.cancelled():
            return
        try:
            future.exception()
        except asyncio.CancelledError:  # pragma: no cover - defensive
            pass


__all__ = [
    "PendingApproval",
    "InteractiveApprovalProvider",
    "build_options",
]
