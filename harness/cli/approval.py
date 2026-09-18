"""Interactive approval: the ``ASK`` branch of the permission layer in the TUI.

Design document sections 7, 8 and 15.  The agent turn already runs as an
``asyncio`` task on the application's event loop, so approval needs no threads,
no ``input()`` and no second prompt: the provider arms a future, the status line
shows the four choices, and a key binding resolves the future.

```text
gate.check_batch ──► provider.request() ──► future returned (armed)
                              │
                              └─► AppState.approval -> status line + keys
                                                    │
                     key "1".."4" / Enter ──────────┘
                              │
                       future.set_result(Approval)
                              │
gate awaits the APProval ◄────┘
```
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from harness.permission.action import Action
from harness.permission.approval import Approval, ApprovalRequest

log = logging.getLogger(__name__)

#: The four choices, in the order the design document lists them.
CHOICES: tuple[tuple[str, str, str], ...] = (
    ("once", "Allow once", "1"),
    ("session", "Allow this session", "2"),
    ("persistent", "Always allow", "3"),
    ("reject", "Reject", "4"),
)

#: Keys that select a choice directly.
CHOICE_KEYS: dict[str, str] = {"1": "once", "2": "session", "3": "persistent", "4": "reject"}


@dataclass
class PendingApproval:
    """One question waiting for the user."""

    action: Action
    reason: str
    future: asyncio.Future
    #: ``(scope, label, key)`` for the choices this action may be given
    options: list[tuple[str, str, str]] = field(default_factory=list)
    selected: int = 0

    def move(self, delta: int) -> None:
        if self.options:
            self.selected = (self.selected + delta) % len(self.options)

    def resolve(self, scope: str | None = None, *, note: str = "") -> bool:
        """Answer the question.  Returns ``False`` when the scope is not offered."""

        chosen = scope or (self.options[self.selected][0] if self.options else "reject")
        if not any(option[0] == chosen for option in self.options):
            log.warning("scope %s is not available for %s", chosen, self.action.describe())
            return False
        if self.future.done():
            return False
        self.future.set_result(Approval(scope=chosen, note=note))
        return True

    @property
    def current(self) -> tuple[str, str, str]:
        if not self.options:
            return ("reject", "Reject", "4")
        return self.options[self.selected]


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
    """An :class:`~harness.permission.approval.ApprovalProvider` backed by the TUI."""

    interactive = True

    def __init__(
        self,
        *,
        state: Any,
        on_change: Callable[[], None] | None = None,
        activity: Callable[[str], None] | None = None,
    ) -> None:
        self.state = state
        self.on_change = on_change
        self.activity = activity
        self.pending: PendingApproval | None = None
        self.history: list[tuple[str, str]] = []
        self._options_factory: Callable[[dict[str, Any]], list[tuple[str, str, str]]] = build_options
        #: how to tell whether the key-dispatching application is live; the app
        #: replaces this once its loop starts.
        self._answerable: Callable[[], bool] = lambda: True

    # ------------------------------------------------------------------ protocol
    def request(self, action: Action, reason: str) -> ApprovalRequest:
        """Arm the question and show it; the gate awaits the returned request."""

        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        # The gate emits the permission_ask event *after* this returns, so the
        # options arrive separately; arm with the safe minimum in the meantime.
        self.pending = PendingApproval(
            action=action,
            reason=reason,
            future=future,
            options=[("once", "Allow once", "1"), ("reject", "Reject", "4")],
        )
        self.state.approval = self.pending
        self._changed()
        return ApprovalRequest(action=action, future=future)

    def offer(self, data: dict[str, Any]) -> None:
        """Apply the choice list carried by the ``permission_ask`` event."""

        if self.pending is None:
            return
        options = self._options_factory(data)
        if options:
            self.pending.options = options
            self.pending.selected = 0
        self._changed()

    # -------------------------------------------------------------------- answer
    def resolve(self, scope: str | None = None, *, note: str = "") -> bool:
        pending = self.pending
        if pending is None:
            return False
        chosen = scope or pending.current[0]
        if not pending.resolve(scope, note=note):
            return False
        self.history.append((pending.action.describe(), chosen))
        self.pending = None
        self.state.approval = None
        self._changed()
        return True

    def move(self, delta: int) -> None:
        if self.pending is not None:
            self.pending.move(delta)
            self._changed()

    def accept_selection(self) -> bool:
        if self.pending is None:
            return False
        return self.resolve(self.pending.current[0])

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
        if not pending.future.done():
            pending.future.set_result(Approval(scope="reject", note="the turn was cancelled"))
        self.history.append((pending.action.describe(), "cancelled"))
        self.pending = None
        self.state.approval = None
        self._changed()
        return True

    @property
    def waiting(self) -> bool:
        return self.pending is not None

    @property
    def can_answer(self) -> bool:
        """Whether a question raised *right now* would reach a human.

        The provider exists as soon as the app is constructed, but the application
        that dispatches the keys only runs inside :meth:`MiniAgentApp.prompt_loop`.
        Outside it (the pipe-driven legacy loop, tests) a question would sit on a
        future nobody can resolve, so callers ask this first.
        """

        return self._answerable()

    # ------------------------------------------------------------------- display
    def status_fragments(self) -> list[tuple[str, str]]:
        """The prompt shown in the status line while a question is open."""

        pending = self.pending
        if pending is None:
            return []
        fragments: list[tuple[str, str]] = [
            ("class:approval", " ⚠ permission "),
            ("class:approval-action", f"{pending.action.describe()} "),
        ]
        for index, (scope, label, key) in enumerate(pending.options):
            marker = "▸" if index == pending.selected else " "
            style = "class:approval-choice" if index == pending.selected else "class:approval-dim"
            fragments.append((style, f"{marker}{key}){label} "))
        return fragments

    # -------------------------------------------------------------------- wiring
    def _changed(self) -> None:
        if self.activity is not None:
            self.activity("waiting for your approval" if self.waiting else "")
        if self.on_change is not None:
            self.on_change()


class PlainApprovalProvider:
    """Line-based approval for ``--plain`` / piped runs.

    Reads one line per question from stdin.  When stdin is exhausted (a piped
    script that does not answer) the question is rejected rather than guessed -
    the same fail-closed rule the headless provider follows.
    """

    interactive = True

    @property
    def can_answer(self) -> bool:
        """Always true: this provider reads stdin directly, needing no live loop."""

        return True

    def cancel(self) -> bool:
        """Nothing is pending: the answer is read synchronously."""

        return False

    def __init__(self, *, stream: Any = None, out: Any = None) -> None:
        self.stream = stream if stream is not None else sys.stdin
        self.out = out if out is not None else sys.stdout
        self.history: list[tuple[str, str]] = []
        #: the choice list for the question currently being asked
        self._options: list[tuple[str, str, str]] = [
            ("once", "Allow once", "1"),
            ("session", "Allow this session", "2"),
            ("persistent", "Always allow", "3"),
            ("reject", "Reject", "4"),
        ]

    def offer(self, data: dict[str, Any]) -> None:
        """Narrow the menu to what this action could actually be given.

        Same rule as the TUI: a chained shell command or an ``apply_patch`` cannot
        be expressed as a standing rule, so offering it here would promise
        something the permission layer cannot keep.
        """

        self._options = build_options(data)

    def request(self, action: Action, reason: str) -> ApprovalRequest:
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        answer = self._ask(action, reason)
        future.set_result(answer)
        self.history.append((action.describe(), answer.scope))
        return ApprovalRequest(action=action, future=future)

    def _ask(self, action: Action, reason: str) -> Approval:
        self._write(f"\nAgent wants to run: {action.describe()}")
        self._write(f"Reason: {reason}")
        self._write("  " + "  ".join(f"{key}) {label}" for _scope, label, key in self._options))
        try:
            line = self.stream.readline()
        except (OSError, ValueError):
            line = ""
        if not line:
            self._write("  (no input available) -> Reject")
            return Approval(scope="reject", note="no input available")

        choice = line.strip().lower()
        by_key = {key: scope for scope, _label, key in self._options}
        by_name = {
            scope: scope for scope, _label, _key in self._options
        } | {
            label.split()[0].lower(): scope for scope, label, _key in self._options
        }
        aliases = {
            "": "reject",
            "y": "once",
            "yes": "once",
            "n": "reject",
            "no": "reject",
            "always": "persistent",
            "once": "once",
        }
        scope = by_key.get(choice) or by_name.get(choice) or aliases.get(choice)
        if scope is None or scope not in {option[0] for option in self._options}:
            self._write(f"  (cannot answer {line.strip()!r} here) -> Reject")
            return Approval(scope="reject", note=f"unavailable answer: {line.strip()}")
        return Approval(scope=scope)

    def _write(self, text: str) -> None:
        try:
            self.out.write(text + "\n")
            self.out.flush()
        except (OSError, ValueError):  # pragma: no cover - closed stream
            pass


__all__ = [
    "CHOICES",
    "CHOICE_KEYS",
    "PendingApproval",
    "InteractiveApprovalProvider",
    "PlainApprovalProvider",
    "build_options",
]
