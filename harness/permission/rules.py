"""Permission rules and their matching semantics.

A rule is a small, declarative statement about one kind of action:

```yaml
permissions:
  rules:
    - "git status: allow"                        # shorthand
    - tool: shell
      program: npm
      permission: ask
    - tool: read_file
      target: "secrets/**"
      permission: deny
```

An empty field is a wildcard, so ``{"tool": "shell", "permission": "ask"}``
matches every shell command.  Two rules are special:

* rules that match **only** on tool name+specificity are *memorizable* - that is
  what "Always allow npm" stores,
* rules that pin a **target glob** can never be memorised, because approving one
  file must never silently approve every file.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Any, Iterable

from harness.permission.action import RISK_LEVELS, Action
from harness.permission.decision import Permission
from harness.permission import shell as shell_mod

#: Shorthand strings accepted in the ``allow`` / ``ask`` / ``deny`` lists.
_SHORTHAND = {
    "always": Permission.ALLOW,
    "allow": Permission.ALLOW,
    "ask": Permission.ASK,
    "prompt": Permission.ASK,
    "deny": Permission.DENY,
    "never": Permission.DENY,
    "block": Permission.DENY,
}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def _normalize(raw: str) -> str:
    """Canonical form for glob matching.

    ``./.env`` and ``.env`` are the same file, so a rule protecting ``.env`` must
    see both.  Trailing slashes and repeated separators are collapsed too; ``..``
    is left alone because it is the workspace guard's business, not the matcher's.
    """

    text = str(raw or "").replace("\\", "/")
    parts = [part for part in text.split("/") if part not in ("", ".")]
    return "/".join(parts)


def matches_path(candidate: str, pattern: str) -> bool:
    """Glob matching with the ``**/`` semantics people actually expect.

    ``fnmatch`` translates ``*`` to "anything, including ``/``", so it never gives
    ``**/`` its usual "zero or more directories" meaning: with plain ``fnmatch``,
    ``**/*.pem`` fails to match a private key sitting in the workspace root, which
    is exactly where a stray key ends up.

    Matching is segment-aware instead: a single ``*`` stays inside one path
    segment, and only ``**`` or a leading ``**/`` may cross a separator.
    """

    if not pattern:
        return False
    text = _normalize(candidate)
    glob = _normalize(pattern)
    if not glob:
        return False

    # "**/x" means "x, at the root or at any depth".
    if glob.startswith("**/"):
        tail = glob[3:]
        return _match_glob(text, tail) or _match_glob(text, "**/" + tail)

    segments = text.split("/") if text else []
    return _segment_match(segments, glob.split("/"))


def _match_glob(text: str, pattern: str) -> bool:
    """Segment-aware glob where only ``**`` crosses a separator."""

    return _segment_match(text.split("/") if text else [], pattern.split("/"))


def _segment_match(segments: list[str], parts: list[str]) -> bool:
    if not parts:
        return not segments
    head, rest = parts[0], parts[1:]

    if head == "**":
        # "**" matches zero or more whole segments.
        if _segment_match(segments, rest):
            return True
        return bool(segments) and _segment_match(segments[1:], parts)
    if not segments:
        return False
    if not fnmatch.fnmatchcase(segments[0], head):
        return False
    return _segment_match(segments[1:], rest)


@dataclass(frozen=True)
class Rule:
    """One permission rule."""

    permission: Permission
    tool: str = ""
    target: str = ""
    program: str = ""
    prefix: str = ""
    #: constrain by the action's risk level (``low`` / ``medium`` / ``high``)
    risk: str = ""
    id: str = ""

    # ------------------------------------------------------------------ matching
    def matches(self, action: Action) -> bool:
        if self.tool and self.tool != action.tool:
            return False
        if self.risk:
            if self.risk == "high":
                wanted = {"high"}
            elif self.risk == "medium":
                wanted = {"medium", "high"}
            else:
                wanted = set(RISK_LEVELS)
            if action.risk not in wanted:
                return False
        if self.program or self.prefix:
            parsed = action.shell_command() if action.type == "shell" else _EMPTY
            if self.program and not shell_mod.matches_program(parsed, self.program):
                return False
            if self.prefix and not shell_mod.matches_prefix(parsed, self.prefix):
                return False
        if self.target:
            targets = action.targets() or [action.target]
            if not any(matches_path(candidate, self.target) for candidate in targets):
                return False
        return True

    # -------------------------------------------------------------------- traits
    @property
    def pins_target(self) -> bool:
        """True when the rule names a specific object (a path glob)."""

        return bool(self.target)

    @property
    def specificity(self) -> int:
        """How narrow the rule is; the most specific match wins.

        The score counts the fields the rule *constrains*, so ``shell + program
        rm`` (5) beats a blanket ``shell`` rule (2) for ``rm -rf /`` while a bare
        ``shell: ask`` still governs everything it does not name.
        """

        score = 0
        if self.tool:
            score += 2
        if self.program:
            score += 3
        if self.prefix:
            score += 3
        if self.target:
            score += 1
        if self.risk:
            # a risk band is broad by construction: "anything high risk" must not
            # outrank a rule that names the exact tool
            score += 1
        return score

    @property
    def memorizable(self) -> bool:
        """Whether "Always allow this" may persist this rule.

        A rule that pins a target path would turn one approval into a permanent
        blanket grant for every file matching the glob, so it is never stored.  A
        risk band is not memorised either: "always allow high-risk actions" is never
        what a user means by approving one command.
        """

        return bool(self.tool) and not self.pins_target and not self.risk

    def key(self) -> tuple[str, str, str, str, str]:
        return (self.tool, self.target, self.program, self.prefix, self.risk)

    def describe(self) -> str:
        parts = [f"tool={self.tool or '*'}"]
        if self.risk:
            parts.append(f"risk={self.risk}")
        if self.target:
            parts.append(f"target={self.target}")
        if self.program:
            parts.append(f"program={self.program}")
        if self.prefix:
            parts.append(f"prefix={self.prefix!r}")
        return f"{self.permission.value} if " + " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"permission": self.permission.value}
        for name in ("tool", "target", "program", "prefix", "risk"):
            value = getattr(self, name)
            if value:
                data[name] = value
        if self.id:
            data["id"] = self.id
        return data

    @classmethod
    def parse(cls, raw: Any, *, permission: Permission | None = None) -> "Rule":
        """Build a rule from ``"tool: allow"`` or a mapping."""

        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                raise ValueError("empty permission rule")
            if permission is not None:  # came from a allow/ask/deny list
                return cls(permission=permission, tool=text)
            head, _, tail = text.rpartition(":")
            if not head or tail.strip().lower() not in _SHORTHAND:
                raise ValueError(
                    f"rule {text!r} must look like `tool: allow|ask|deny`"
                )
            return cls(permission=_SHORTHAND[tail.strip().lower()], tool=head.strip())

        if not isinstance(raw, dict):
            raise ValueError(f"permission rule must be a string or mapping, got {type(raw).__name__}")

        data = dict(raw)
        if permission is not None:
            data.setdefault("permission", permission.value)
        value = str(data.pop("permission", data.pop("decision", "ask"))).strip().lower()
        if value not in _SHORTHAND:
            raise ValueError(f"unknown permission `{value}`; expected allow, ask or deny")
        known = {"tool", "target", "program", "prefix", "risk", "id"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown rule field(s): {', '.join(sorted(unknown))}")
        risk = str(data.get("risk") or "").strip().lower()
        if risk and risk not in RISK_LEVELS:
            raise ValueError(
                f"unknown risk `{risk}`; expected one of {', '.join(RISK_LEVELS)}"
            )
        return cls(
            permission=_SHORTHAND[value],
            tool=str(data.get("tool") or "").strip(),
            target=str(data.get("target") or "").strip(),
            program=str(data.get("program") or "").strip(),
            prefix=str(data.get("prefix") or "").strip(),
            risk=risk,
            id=str(data.get("id") or "").strip(),
        )


_EMPTY = shell_mod.ShellCommand(raw="", program="", tokens=())


def parse_rules(
    entries: Iterable[Any] = (),
    *,
    permission: Permission | None = None,
) -> list[Rule]:
    """Parse a list of rule sources, skipping blanks and duplicates."""

    rules: list[Rule] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for entry in entries or ():
        if entry is None or (isinstance(entry, str) and not entry.strip()):
            continue
        rule = Rule.parse(entry, permission=permission)
        key = (rule.permission.value, *rule.key())
        if key in seen:
            continue
        seen.add(key)
        rules.append(rule)
    return rules


def _effective_specificity(rule: Rule, action: Action) -> int:
    """Specificity *as applied to this action*.

    Only the fields that actually match count, so a rule cannot borrow weight
    from a constraint it did not satisfy.
    """

    score = 0
    if rule.tool and rule.tool == action.tool:
        score += 2
    if rule.program and action.program and rule.program == action.program:
        score += 3
    if rule.prefix:
        score += 3
    if rule.target:
        score += 1
    if rule.risk:
        # broad by construction: "anything high risk" must not outrank a rule that
        # names the exact tool it is talking about
        score += 1
    return score


class RuleMatcher:
    """Applies an ordered rule set to an action.

    Resolution is **most specific wins, then most restrictive**:

    * ``shell: ask`` next to ``shell prefix='git status': allow`` means git
      status runs and every other command asks - the narrow rule refines the
      broad one,
    * two equally specific rules resolve to the more restrictive verdict, so a
      deny can never be shadowed by an allow that is just as narrow.
    """

    def __init__(self, rules: Iterable[Rule] = ()) -> None:
        self.rules: list[Rule] = list(rules)

    def __bool__(self) -> bool:
        return bool(self.rules)

    def with_rules(self, rules: Iterable[Rule]) -> "RuleMatcher":
        return RuleMatcher([*self.rules, *rules])

    def match(self, action: Action) -> Rule | None:
        """The rule that decides ``action``, or ``None``.

        Risk-band rules are resolved *against* the ordinary winner rather than
        competing with it, because a band is broad by construction:

        * a risk **deny** is a safety floor - it is what an operator writes for the
          tools they did not enumerate, so no ordinary rule may talk past it;
        * otherwise a matching band applies only when it is *strictly* more
          restrictive than the ordinary winner, or when no ordinary rule matched at
          all.  A blanket ``risk: medium, ask`` therefore does not undo a specific
          ``prefix: git status, allow``.
        """

        risks = [rule for rule in self.rules if rule.risk and rule.matches(action)]
        floor = next((rule for rule in risks if rule.permission is Permission.DENY), None)
        if floor is not None:
            return floor

        best: Rule | None = None
        best_key: tuple[int, int, int] = (-1, -1, -1)
        for rule in self.rules:
            if rule.risk or not rule.matches(action):
                continue
            key = (
                _effective_specificity(rule, action),
                _severity_rank(rule.permission),
                rule.specificity,
            )
            if key > best_key:
                best, best_key = rule, key

        if not risks:
            return best
        strictest = max(risks, key=lambda rule: _severity_rank(rule.permission))
        if best is None:
            return strictest
        return strictest if _severity_rank(strictest.permission) > _severity_rank(best.permission) else best


def _severity_rank(permission: Permission) -> int:
    return {Permission.ALLOW: 0, Permission.ASK: 1, Permission.DENY: 2}[permission]


__all__ = ["Rule", "RuleMatcher", "parse_rules"]
