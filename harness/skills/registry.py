"""Skill registry: metadata only, loaded lazily.

At startup MiniAgent reads just the YAML front matter of every ``SKILL.md``.
The body is read only when the model asks for it, which is the whole point of
progressive disclosure.
"""

from __future__ import annotations

import logging
from pathlib import Path

from harness.agent.dto import SkillMetadata
from harness.agent.errors import SkillNotFound

log = logging.getLogger(__name__)


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Split a Markdown document into ``(front_matter, body)``.

    Uses PyYAML when available and falls back to a tiny key/value parser so the
    skill system keeps working without the dependency.
    """

    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    end = None
    for index in range(1, len(lines)):
        if lines[index].strip() in ("---", "..."):
            end = index
            break
    if end is None:
        return {}, text
    raw = "\n".join(lines[1:end])
    body = "\n".join(lines[end + 1 :]).lstrip("\n")
    try:
        import yaml

        data = yaml.safe_load(raw) or {}
        if not isinstance(data, dict):
            return {}, body
        return data, body
    except ImportError:  # pragma: no cover - PyYAML is a dependency
        data = {}
        for line in raw.splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            data[key.strip()] = value.strip().strip("[]").replace(",", " ").split()
        return data, body
    except Exception as exc:
        log.warning("invalid YAML front matter: %s", exc)
        return {}, body


class SkillRegistry:
    """Discovers ``<skill_dir>/SKILL.md`` files under the configured paths."""

    def __init__(self, paths: list[str | Path] | None = None) -> None:
        self.paths = [Path(p) for p in (paths or [])]
        self._metadata: dict[str, SkillMetadata] = {}

    # ---------------------------------------------------------------- discovery
    def discover(self) -> dict[str, SkillMetadata]:
        self._metadata = {}
        for base in self.paths:
            if not base.exists():
                log.debug("skill path does not exist: %s", base)
                continue
            for skill_file in sorted(base.glob("*/SKILL.md")):
                metadata = self._read_metadata(skill_file)
                if metadata is None:
                    continue
                self._metadata[metadata.name] = metadata
        return dict(self._metadata)

    def _read_metadata(self, skill_file: Path) -> SkillMetadata | None:
        try:
            text = skill_file.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("cannot read %s: %s", skill_file, exc)
            return None
        front, _body = parse_front_matter(text)
        name = str(front.get("name") or skill_file.parent.name)
        description = str(front.get("description") or "").strip()
        if not description:
            description = "(no description)"
        keywords = front.get("keywords") or []
        if isinstance(keywords, str):
            keywords = [word.strip() for word in keywords.replace(",", " ").split() if word.strip()]
        return SkillMetadata(
            name=name,
            description=description,
            keywords=[str(word).lower() for word in keywords],
            path=str(skill_file),
        )

    # ------------------------------------------------------------------- access
    def names(self) -> list[str]:
        return sorted(self._metadata)

    def get(self, name: str) -> SkillMetadata:
        try:
            return self._metadata[name]
        except KeyError as exc:
            raise SkillNotFound(
                f"unknown skill `{name}`; available: {', '.join(self.names()) or '(none)'}"
            ) from exc

    def body(self, name: str) -> str:
        metadata = self.get(name)
        text = Path(metadata.path).read_text(encoding="utf-8")
        _front, body = parse_front_matter(text)
        return body.strip()

    def catalog(self, *, max_description: int = 160) -> str:
        """Render the *metadata only* block injected into every context."""

        if not self._metadata:
            return ""
        lines = ["Available skills (call `load_skill` to read one before using it):"]
        for name in self.names():
            metadata = self._metadata[name]
            description = metadata.description
            if len(description) > max_description:
                description = description[: max_description - 1] + "…"
            lines.append(f"- {name}: {description}")
            if metadata.keywords:
                lines.append(f"  keywords: {', '.join(metadata.keywords)}")
        return "\n".join(lines)


__all__ = ["SkillRegistry", "parse_front_matter"]
