"""User approval: the ``ASK`` branch of the permission flow.

Design document sections 7 and 8.  The provider interface is deliberately an
*async* call returning a decision, not a blocking ``input()``:

* the graph node that asks can ``await`` it, so a TUI can resolve a future from a
  key binding without a second thread,
* a headless run gets :class:`AutoDenyProvider` and fails closed,
* tests get :class:`ScriptedProvider` and need no terminal at all.

Providers never see the raw tool call - they see the :class:`Action` and the
reason the evaluator produced it, which is exactly what a good approval prompt
shows the user.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from harness.permission.action import Action

log = logging.getLogger(__name__)

#: Scopes a provider may answer with, in the order a prompt usually shows them.
SCOPES = ("once", "session", "persistent", "reject")

#: Human-readable labels for the four choices (design document section 15).
SCOPE_LABELS: dict[str, str] = {
    "once": "Allow once",
    "session": "Allow this session",
    "persistent": "Always allow",
    "reject": "Reject",
}


@dataclass(frozen=True)
class Approval:
    """One user answer."""

    scope: str = "reject"
    #: Optional user-supplied denial reason, passed back to the model.
    note: str = ""

    @property
    def granted(self) -> bool:
        return self.scope in ("once", "session", "persistent")


@dataclass
class ApprovalRequest:
    """A pending question, already armed when it is handed back.

    Splitting "show the question" from "wait for the answer" removes a race: the
    gate emits the ``permission_ask`` event *after* the provider has armed its
    future, so a fast answer can never arrive at a future nobody awaits yet.
    Headless providers return an already-resolved future instead.
    """

    action: Action
    future: asyncio.Future

    def __await__(self):
        return self.future.__await__()

    @classmethod
    def resolved(cls, action: Action, approval: "Approval") -> "ApprovalRequest":
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        future.set_result(approval)
        return cls(action=action, future=future)


@runtime_checkable
class ApprovalProvider(Protocol):
    """What the permission gate needs from a user interface."""

    #: Whether this provider can really reach a human.  A provider that can only
    #: fall back to a default must set this to ``False`` so the evaluator can
    #: deny outright instead of producing a verdict nobody will ever answer.
    interactive: bool = True

    def request(self, action: Action, reason: str) -> ApprovalRequest:
        """Arm a question and return it; the caller awaits the answer."""
        ...


class AutoDenyProvider:
    """Fails closed: every request is rejected.

    Used whenever no human can answer - subagents, batch mode without a terminal,
    ``--plain`` runs with exhausted input.  A denied action never executes, so the
    failure mode of a missing approver is "the agent stops", never "the agent
    proceeded without consent".
    """

    interactive = False

    def __init__(self, note: str = "no approver is available in this context") -> None:
        self.note = note

    def request(self, action: Action, reason: str) -> ApprovalRequest:
        log.info("auto-denying %s (%s)", action.describe(), reason)
        return ApprovalRequest.resolved(action, Approval(scope="reject", note=self.note))


class AutoAllowProvider:
    """Allows everything once.  For tests and explicit ``--yes`` style runs."""

    interactive = False

    def __init__(self, scope: str = "once") -> None:
        self.scope = scope

    def request(self, action: Action, reason: str) -> ApprovalRequest:
        return ApprovalRequest.resolved(action, Approval(scope=self.scope))


class ScriptedProvider:
    """Replays a fixed queue of answers.

    Once the queue is exhausted the last answer is reused, so a test that cares
    about *which* questions were asked does not have to enumerate one answer per
    call.  :attr:`batches` records the requests grouped by ``check_batch`` call,
    which is how a test asserts that one turn produced one round of prompts.
    """

    interactive = True

    def __init__(self, answers: list[str] | None = None, *, reuse_last: bool = False) -> None:
        self._queue: list[Approval] = []
        self._last: Approval | None = None
        self.reuse_last = reuse_last
        self.requests: list[tuple[Action, str]] = []
        self.batches: list[list[str]] = []
        self._current_batch: list[str] = []
        self._in_batch = False
        for answer in answers or []:
            self.queue(answer)

    # ------------------------------------------------------------------ scripting
    def queue(self, answer: str) -> "ScriptedProvider":
        self._queue.append(_parse_answer(answer))
        return self

    def queue_chunks(self, answer: str, count: int) -> "ScriptedProvider":
        """Queue ``count`` copies of one answer."""

        for _ in range(max(1, count)):
            self.queue(answer)
        return self

    def begin_batch(self) -> None:
        """Called by :meth:`~harness.permission.gate.PermissionGate.check_batch`."""

        self._current_batch = []
        self._in_batch = True

    def end_batch(self) -> None:
        if self._in_batch:
            self.batches.append(self._current_batch)
        self._in_batch = False

    # ------------------------------------------------------------------- protocol
    def request(self, action: Action, reason: str) -> ApprovalRequest:
        self.requests.append((action, reason))
        if self._in_batch:
            self._current_batch.append(action.describe())
        if self._queue:
            self._last = self._queue.pop(0)
            answer = self._last
        elif self._last is not None and self.reuse_last:
            answer = self._last
        else:
            log.warning("scripted approver has no answer for %s; rejecting", action.describe())
            answer = Approval(scope="reject", note="no scripted answer left")
        return ApprovalRequest.resolved(action, answer)

    # -------------------------------------------------------------------- reading
    @property
    def questioned(self) -> list[str]:
        return [action.describe() for action, _ in self.requests]


def _parse_answer(answer: str) -> Approval:
    text = (answer or "").strip().lower()
    if text.startswith("grant"):
        _, _, tail = text.partition(":")
        scope = tail.strip() or "session"
        if scope not in SCOPES or scope == "reject":
            scope = "session"
        return Approval(scope=scope)
    if text in ("once", "allow", "yes", "y", "1"):
        return Approval(scope="once")
    if text in ("session", "2"):
        return Approval(scope="session")
    if text in ("persistent", "always", "3"):
        return Approval(scope="persistent")
    if text in ("reject", "deny", "no", "n", "4"):
        return Approval(scope="reject")
    return Approval(scope="reject", note=answer)


__all__ = [
    "Approval",
    "ApprovalRequest",
    "ApprovalProvider",
    "AutoDenyProvider",
    "AutoAllowProvider",
    "ScriptedProvider",
    "SCOPES",
    "SCOPE_LABELS",
]
