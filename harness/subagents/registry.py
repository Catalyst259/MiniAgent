"""SubAgent registry: filesystem definitions with tool isolation.

Each subagent is a directory containing ``AGENT.md``:

```text
subagents/
├── planner/AGENT.md
└── explorer/AGENT.md
```

The front matter declares which tools the subagent may use.  That list is the
isolation boundary: an ``Explorer`` simply has no ``write_file`` tool, so it
cannot modify the repository even if the model asks it to.
"""

from __future__ import annotations

import logging
from pathlib import Path

from harness.agent.dto import SubAgentSpec
from harness.agent.errors import SubAgentNotFound
from harness.skills.registry import parse_front_matter

log = logging.getLogger(__name__)

DEFAULT_AGENT_FILE = "AGENT.md"


class SubAgentRegistry:
    def __init__(self, paths: list[str | Path] | None = None) -> None:
        self.paths = [Path(p) for p in (paths or [])]
        self._specs: dict[str, SubAgentSpec] = {}

    # ---------------------------------------------------------------- discovery
    def discover(self) -> dict[str, SubAgentSpec]:
        self._specs = {}
        for base in self.paths:
            if not base.exists():
                log.debug("subagent path does not exist: %s", base)
                continue
            for agent_file in sorted(base.glob(f"*/{DEFAULT_AGENT_FILE}")):
                spec = self._read(agent_file)
                if spec is not None:
                    self._specs[spec.name] = spec
        return dict(self._specs)

    def _read(self, agent_file: Path) -> SubAgentSpec | None:
        try:
            text = agent_file.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("cannot read %s: %s", agent_file, exc)
            return None
        front, body = parse_front_matter(text)
        name = str(front.get("name") or agent_file.parent.name)
        tools = front.get("tools") or []
        if isinstance(tools, str):
            tools = [item.strip().strip("[]") for item in tools.split(",") if item.strip()]
        skills = front.get("skills") or []
        if isinstance(skills, str):
            skills = [item.strip().strip("[]") for item in skills.split(",") if item.strip()]
        description = str(front.get("description") or "").strip() or "(no description)"
        return SubAgentSpec(
            name=name,
            description=description,
            tools=[str(item) for item in tools],
            skills=[str(item) for item in skills],
            path=str(agent_file),
            body=body.strip(),
        )

    # ------------------------------------------------------------------- access
    def names(self) -> list[str]:
        return sorted(self._specs)

    def get(self, name: str) -> SubAgentSpec:
        key = (name or "").strip().lower()
        if key not in self._specs:
            matches = [n for n in self._specs if n.lower() == key]
            if not matches:
                raise SubAgentNotFound(
                    f"unknown subagent `{name}`; available: {', '.join(self.names()) or '(none)'}"
                )
            key = matches[0]
        return self._specs[key]

    def catalog(self) -> str:
        if not self._specs:
            return "(no subagents configured)"
        return "\n".join(
            f"- {name}: {self._specs[name].description}" for name in self.names()
        )

    def tool_schema(self) -> dict:
        """OpenAI-compatible schema for the built-in ``delegate`` tool."""

        names = self.names()
        return {
            "type": "function",
            "function": {
                "name": "delegate",
                "description": (
                    "Hand a self-contained task to a read-only subagent and get a compact, "
                    "structured answer back. Use it to plan a complex change or to map an "
                    f"unfamiliar part of the repository. Available subagents: {', '.join(names)}."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "description": "Subagent name.",
                            "enum": names or ["planner"],
                        },
                        "task": {"type": "string", "description": "The task, stated self-containedly."},
                        "context": {
                            "type": "string",
                            "description": "Only the context the subagent needs (it does not see this conversation).",
                        },
                    },
                    "required": ["agent", "task"],
                    "additionalProperties": False,
                },
            },
        }


__all__ = ["SubAgentRegistry", "DEFAULT_AGENT_FILE"]
