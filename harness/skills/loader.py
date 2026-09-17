"""Progressive disclosure for skills.

The model sees metadata in every context and calls ``load_skill`` when it needs
the procedure.  The loader then puts the *body* into the agent state, where the
context builder injects it from that point on.
"""

from __future__ import annotations

from harness.agent.dto import SkillMetadata
from harness.agent.errors import SkillNotFound
from harness.skills.registry import SkillRegistry

MAX_SKILL_BODY_CHARS = 12_000


class SkillLoader:
    """Reads skill bodies on demand and keeps track of what is loaded."""

    def __init__(self, registry: SkillRegistry, *, max_body_chars: int = MAX_SKILL_BODY_CHARS) -> None:
        self.registry = registry
        self.max_body_chars = max_body_chars
        self._cache: dict[str, str] = {}

    def metadata(self, name: str) -> SkillMetadata:
        return self.registry.get(name)

    def load(self, name: str, loaded: list[str] | None = None) -> str:
        """Return the body of ``name``, truncated to the budget."""

        key = (name or "").strip()
        if not key:
            raise SkillNotFound("`load_skill` needs a skill name")
        if key not in self._cache:
            body = self.registry.body(key)
            if len(body) > self.max_body_chars:
                body = body[: self.max_body_chars] + "\n... [skill body truncated]"
            self._cache[key] = body
        body = self._cache[key]
        already = key in (loaded or [])
        prefix = (
            f"(skill `{key}` was already loaded; showing it again)\n\n" if already else ""
        )
        return (
            f"{prefix}--- BEGIN SKILL: {key} ---\n{body}\n--- END SKILL: {key} ---"
        )

    def catalog(self) -> str:
        return self.registry.catalog()

    def names(self) -> list[str]:
        return self.registry.names()


__all__ = ["SkillLoader", "MAX_SKILL_BODY_CHARS"]
