"""Native, non-MCP tools: ``load_skill`` and ``delegate``.

These are part of MiniAgent itself rather than filesystem/shell capabilities,
so they are declared here and handled by dedicated graph nodes.  Their schemas
are still OpenAI function schemas, keeping the model-facing protocol uniform.
"""

from __future__ import annotations

LOAD_SKILL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "load_skill",
        "description": (
            "Read the full instructions of a skill by name and keep them for the rest of "
            "the task. Call this before doing work that matches a skill."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name, exactly as listed."}
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
}

NATIVE_TOOL_NAMES = ("load_skill", "delegate")


def native_tool_schemas(*, delegate_schema: dict | None = None, skills_available: bool = True) -> list[dict]:
    schemas: list[dict] = []
    if skills_available:
        schemas.append(LOAD_SKILL_SCHEMA)
    if delegate_schema is not None:
        schemas.append(delegate_schema)
    return schemas


__all__ = ["LOAD_SKILL_SCHEMA", "NATIVE_TOOL_NAMES", "native_tool_schemas"]
