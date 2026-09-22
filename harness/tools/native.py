"""Native, non-MCP tools: skills, delegation and user interaction.

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

REQUEST_USER_INPUT_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "request_user_input",
        "description": (
            "Ask the user one necessary multiple-choice question. Use only when the "
            "answer materially changes the work and cannot be discovered from context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "A concise question."},
                "title": {"type": "string", "description": "Short panel title."},
                "options": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string", "description": "Stable result value."},
                            "label": {"type": "string", "description": "Short user-facing label."},
                            "description": {
                                "type": "string",
                                "description": "One sentence explaining the tradeoff.",
                            },
                        },
                        "required": ["label"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["question", "options"],
            "additionalProperties": False,
        },
    },
}

NATIVE_TOOL_NAMES = ("load_skill", "delegate", "request_user_input")


def native_tool_schemas(
    *,
    delegate_schema: dict | None = None,
    skills_available: bool = True,
    interaction_available: bool = False,
) -> list[dict]:
    schemas: list[dict] = []
    if skills_available:
        schemas.append(LOAD_SKILL_SCHEMA)
    if delegate_schema is not None:
        schemas.append(delegate_schema)
    if interaction_available:
        schemas.append(REQUEST_USER_INPUT_SCHEMA)
    return schemas


__all__ = [
    "LOAD_SKILL_SCHEMA",
    "NATIVE_TOOL_NAMES",
    "REQUEST_USER_INPUT_SCHEMA",
    "native_tool_schemas",
]
