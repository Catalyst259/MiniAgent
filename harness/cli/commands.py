"""Compatibility shim for the original slash-command table.

The interactive CLI derives its commands from
:mod:`harness.cli.composer.slash_commands` (a real registry with handlers and fuzzy
matching).  ``COMMANDS`` is kept as the plain ``name -> description`` view of that
registry so older checks - and ``Detail.md`` - still resolve to a single source of
truth.
"""

from __future__ import annotations

from harness.cli.composer.slash_commands import build_default_registry

REGISTRY = build_default_registry({})

#: ``/name`` -> description, in declaration order.
COMMANDS: dict[str, str] = {
    command.display: command.description for command in REGISTRY.all()
}

#: Alias kept for backwards compatibility with earlier code and tests.
KNOWN_COMMANDS: tuple[str, ...] = tuple(COMMANDS)

HELP_TEXT = "\n".join(f"  {name:<10} {description}" for name, description in COMMANDS.items())


def command_names() -> list[str]:
    """Command names without the leading slash, in declaration order."""

    return REGISTRY.names()


__all__ = ["COMMANDS", "KNOWN_COMMANDS", "HELP_TEXT", "REGISTRY", "command_names"]
