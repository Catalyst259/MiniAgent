"""User approval: the ``ASK`` branch of the permission flow.

Design document sections 7 and 8.  The provider interface is deliberately an
*async* call returning a decision, not a blocking ``input()``:

* the graph node that asks can ``await`` it, so a TUI can resolve a future from a
  key binding without a second thread,
* a headless run gets :class:`AutoDenyProvider` and fails closed,

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

    Used whenever no human can answer, including isolated subagents.  A denied action never executes, so the
    failure mode of a missing approver is "the agent stops", never "the agent
    proceeded without consent".
    """

    interactive = False

    def __init__(self, note: str = "no approver is available in this context") -> None:
        self.note = note

    def request(self, action: Action, reason: str) -> ApprovalRequest:
        log.info("auto-denying %s (%s)", action.describe(), reason)
        return ApprovalRequest.resolved(action, Approval(scope="reject", note=self.note))


__all__ = [
    "Approval",
    "ApprovalRequest",
    "ApprovalProvider",
    "AutoDenyProvider",
    "SCOPES",
    "SCOPE_LABELS",
]
