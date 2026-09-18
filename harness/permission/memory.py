"""Permission memory: temporary, session and persistent grants.

Design document section 9.  The three layers exist so that "allow once" costs one
keystroke and "always allow" costs one file:

* :class:`TemporaryMemory` - consumed by a single action ("Allow once"),
* :class:`SessionMemory`   - lives as long as the agent session ("Allow this
  session"), shared with every subagent of that session,
* :class:`PersistentMemory` - survives restarts ("Always allow"), stored
  **outside the workspace** (see the security note below).

Security note
-------------
The agent can write everywhere inside its workspace, and MiniAgent's
``config.yaml`` lives in the workspace root.  A permission store inside the
workspace would therefore let the agent edit its own rules, which is textbook
self-authorisation.  The default persistent path is ``~/.config/miniagent/``.

Combining layers
----------------
Grants never *override* rules: :meth:`PermissionMemory.lookup` returns the most
restrictive verdict across all layers, and the evaluator only consults memory
when the configured rules have nothing to say.  A remembered "always allow npm"
can thus be revoked by adding a deny rule for npm.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from harness.permission.action import Action
from harness.permission.decision import Permission
from harness.permission.rules import Rule

log = logging.getLogger(__name__)

#: Where "Always allow" is stored unless configured otherwise.
DEFAULT_PERSISTENT_PATH = "~/.config/miniagent/approvals.json"

#: How many times an identical grant may be recorded before it is treated as
#: noise; keeps a pathological loop from growing the file without bound.
_MAX_GRANTS = 500


class MemoryLayer:
    """A store of granted rules."""

    #: Short name used in audit reasons and the CLI.
    name = "memory"

    def add(self, rule: Rule) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def rules(self) -> list[Rule]:  # pragma: no cover - interface
        return []

    def clear(self) -> None:  # pragma: no cover - interface
        pass

    def to_list(self) -> list[dict[str, Any]]:
        return [rule.to_dict() for rule in self.rules()]

    @classmethod
    def from_list(cls, entries: Iterable[Any]) -> "MemoryLayer":
        layer = cls()
        for entry in entries or ():
            try:
                layer.add(Rule.parse(entry))
            except ValueError as exc:
                log.warning("ignoring invalid stored grant %r: %s", entry, exc)
        return layer


@dataclass
class TemporaryMemory(MemoryLayer):
    """Grants that apply to exactly one action ("Allow once")."""

    name = "once"
    _rules: list[Rule] = field(default_factory=list)

    def add(self, rule: Rule) -> None:
        self._rules.append(rule)

    def rules(self) -> list[Rule]:
        return list(self._rules)

    def clear(self) -> None:
        self._rules.clear()

    def consume(self, action: Action) -> bool:
        """True when a one-shot grant covers ``action`` (and spends it)."""

        for index, rule in enumerate(self._rules):
            if rule.permission is Permission.ALLOW and rule.matches(action):
                self._rules.pop(index)
                return True
        return False


@dataclass
class SessionMemory(MemoryLayer):
    """Grants that live as long as the agent session."""

    name = "session"
    _rules: list[Rule] = field(default_factory=list)

    def add(self, rule: Rule) -> None:
        if len(self._rules) >= _MAX_GRANTS:
            log.warning("session permission memory is full; ignoring %s", rule.describe())
            return
        if any(existing.key() == rule.key() for existing in self._rules):
            return
        self._rules.append(rule)

    def rules(self) -> list[Rule]:
        return list(self._rules)

    def clear(self) -> None:
        self._rules.clear()


class PersistentMemory(MemoryLayer):
    """Grants stored on disk, outside the workspace."""

    name = "persistent"

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        enabled: bool = True,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.enabled = enabled
        self.path = Path(path).expanduser() if path else Path(DEFAULT_PERSISTENT_PATH).expanduser()
        self._rules: list[Rule] = []
        self._loaded = False
        # Refuse a store the agent itself could rewrite: a permission file inside
        # the workspace is self-authorisation waiting to happen.
        self.unsafe_reason = self._check_location(workspace_root)
        if self.unsafe_reason:
            log.error(
                "disabling the persistent permission store at %s: %s",
                self.path,
                self.unsafe_reason,
            )
            self.enabled = False

    def _check_location(self, workspace_root: str | Path | None) -> str:
        if workspace_root is None:
            return ""
        try:
            root = Path(workspace_root).expanduser().resolve()
            target = self.path.resolve()
        except OSError:  # pragma: no cover - unreadable path
            return "the path could not be resolved"
        if target == root or root in target.parents:
            return (
                f"`{target}` is inside the workspace `{root}`; the agent can write "
                "there and would be able to edit its own permission rules"
            )
        return ""

    # ------------------------------------------------------------------ storage
    def load(self) -> "PersistentMemory":
        self._loaded = True
        if not self.enabled or not self.path.exists():
            return self
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("cannot read permission store %s: %s", self.path, exc)
            return self
        entries = payload.get("grants") if isinstance(payload, dict) else payload
        self._rules = []
        for entry in entries or ():
            try:
                self._rules.append(Rule.parse(entry))
            except ValueError as exc:
                log.warning("ignoring invalid stored grant %r: %s", entry, exc)
        return self

    def save(self) -> bool:
        """Write the store atomically.  Returns ``False`` when it could not."""

        if not self.enabled:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"version": 1, "grants": [rule.to_dict() for rule in self._rules]}
            handle, temp_name = tempfile.mkstemp(
                dir=str(self.path.parent), prefix=".approvals-", suffix=".tmp"
            )
            try:
                with os.fdopen(handle, "w", encoding="utf-8") as stream:
                    json.dump(payload, stream, indent=2, sort_keys=True)
                    stream.write("\n")
                os.replace(temp_name, self.path)
            except BaseException:
                Path(temp_name).unlink(missing_ok=True)
                raise
            # The rules decide what the agent may do, so keep them owner-only.
            os.chmod(self.path, 0o600)
            return True
        except OSError as exc:
            log.warning("cannot write permission store %s: %s", self.path, exc)
            return False

    # ------------------------------------------------------------------- grants
    def add(self, rule: Rule) -> None:
        if any(existing.key() == rule.key() for existing in self._rules):
            return
        if len(self._rules) >= _MAX_GRANTS:
            log.warning("persistent permission store is full; ignoring %s", rule.describe())
            return
        self._rules.append(rule)
        self.save()

    def rules(self) -> list[Rule]:
        if not self._loaded:
            self.load()
        return list(self._rules)

    def clear(self) -> None:
        self._rules.clear()
        self.save()

    def forget(self, rule: Rule) -> bool:
        before = len(self._rules)
        self._rules = [item for item in self._rules if item.key() != rule.key()]
        if len(self._rules) != before:
            self.save()
            return True
        return False

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "path": str(self.path),
            "grants": len(self.rules()),
            "unsafe_reason": self.unsafe_reason or None,
        }


@dataclass
class PermissionMemory:
    """The three layers, queried as one."""

    session: SessionMemory = field(default_factory=SessionMemory)
    temporary: TemporaryMemory = field(default_factory=TemporaryMemory)
    persistent: PersistentMemory | None = None

    def layers(self) -> list[MemoryLayer]:
        layers: list[MemoryLayer] = [self.temporary, self.session]
        if self.persistent is not None:
            layers.append(self.persistent)
        return layers

    def lookup(self, action: Action) -> tuple[Permission, Rule | None, str] | None:
        """The most restrictive memory verdict for ``action``, or ``None``.

        Layer order sets the *reason*; severity sets the *verdict*.
        """

        verdicts: list[tuple[int, str, Rule]] = []
        for layer in self.layers():
            for rule in layer.rules():
                if rule.matches(action):
                    verdicts.append((_rank(rule.permission), layer.name, rule))
        if not verdicts:
            return None
        rank, layer_name, rule = max(verdicts, key=lambda item: item[0])
        return Permission(rule.permission), rule, layer_name

    def remember(self, rule: Rule, scope: str) -> bool:
        """Store ``rule`` in the layer named by ``scope``.

        Returns ``False`` when the scope cannot store it, which the caller must
        surface as "not remembered" rather than pretend it worked.
        """

        if not rule.memorizable:
            log.info("refusing to memorise target-scoped rule %s", rule.describe())
            return False
        if scope == "session":
            self.session.add(rule)
            return True
        if scope == "persistent":
            if self.persistent is None or not self.persistent.enabled:
                return False
            self.persistent.add(rule)
            return True
        return False

    def spend(self, action: Action) -> bool:
        """Consume a one-shot grant for ``action``."""

        return self.temporary.consume(action)

    def clear_session(self) -> None:
        self.session.clear()
        self.temporary.clear()

    def status(self) -> dict[str, Any]:
        return {
            "session": len(self.session.rules()),
            "temporary": len(self.temporary.rules()),
            "persistent": self.persistent.status() if self.persistent else None,
        }


def _rank(permission: Permission) -> int:
    return {Permission.ALLOW: 0, Permission.ASK: 1, Permission.DENY: 2}[permission]


__all__ = [
    "MemoryLayer",
    "TemporaryMemory",
    "SessionMemory",
    "PersistentMemory",
    "PermissionMemory",
    "DEFAULT_PERSISTENT_PATH",
]
