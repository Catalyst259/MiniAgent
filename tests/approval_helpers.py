"""Permission provider doubles for tests."""
from __future__ import annotations

import logging

from harness.permission.action import Action
from harness.permission.approval import Approval, ApprovalRequest, SCOPES

log = logging.getLogger(__name__)


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
    can_answer = True

    def __init__(self, answers: list[str] | None = None, *, reuse_last: bool = False) -> None:
        self._queue: list[Approval] = []
        self._last: Approval | None = None
        self.reuse_last = reuse_last
        self.requests: list[tuple[Action, str]] = []
        self.answers: list[str] = []
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
        self.answers.append(answer.scope)
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


