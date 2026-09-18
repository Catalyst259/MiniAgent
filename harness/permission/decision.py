"""Permission decisions.

The design document defines exactly three verdicts.  Keeping the enum this small
is deliberate: adding a fourth state ("maybe") is how permission systems become
unauditable.  Every :class:`Verdict` therefore carries the *reason* it was
produced, so the CLI and the log can always answer "which rule decided this?".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Permission(str, Enum):
    """The three verdicts of the permission layer."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.value


#: Order used to combine rules from several sources: the most restrictive wins.
#: DENY beats ASK beats ALLOW, so a session grant can never punch through a
#: configured deny rule.
SEVERITY: dict[Permission, int] = {
    Permission.ALLOW: 0,
    Permission.ASK: 1,
    Permission.DENY: 2,
}


def most_restrictive(*permissions: Permission) -> Permission:
    return max(permissions, key=lambda item: SEVERITY[item])


@dataclass(frozen=True)
class Verdict:
    """One decision about one :class:`~harness.permission.action.Action`."""

    permission: Permission
    reason: str
    source: str = "policy"
    rule: Any = None
    #: set when the verdict came from user approval rather than from a rule
    approval: str | None = None

    @property
    def allowed(self) -> bool:
        return self.permission is Permission.ALLOW

    @property
    def denied(self) -> bool:
        return self.permission is Permission.DENY

    @property
    def needs_approval(self) -> bool:
        return self.permission is Permission.ASK

    def to_dict(self) -> dict[str, Any]:
        return {
            "permission": self.permission.value,
            "reason": self.reason,
            "source": self.source,
            "rule": getattr(self.rule, "id", None),
            "approval": self.approval,
        }


__all__ = ["Permission", "Verdict", "SEVERITY", "most_restrictive"]
