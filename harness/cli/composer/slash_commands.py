"""Slash command registry.

Commands are data (name + description + handler), not branches inside the popup or
the renderer.  The popup only displays/filters/selects; the dispatcher executes.
See ``CLI_Design.md`` sections 13 and 34.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable, Iterable, Sequence

from harness.cli.composer.fuzzy_match import FuzzyResult, fuzzy_filter, fuzzy_match

Handler = Callable[..., Awaitable[bool] | bool]


@dataclass
class SlashCommand:
    name: str
    description: str
    handler: Handler | None = None
    supports_args: bool = False
    order: int = 0
    aliases: tuple[str, ...] = ()

    @property
    def display(self) -> str:
        return f"/{self.name}"

    @property
    def usage(self) -> str:
        return f"/{self.name} <{self.name}>" if self.supports_args else f"/{self.name}"


@dataclass
class CommandMatch:
    command: SlashCommand
    matched_indices: list[int] = field(default_factory=list)
    score: int = 0
    order: int = 0


class CommandRegistry:
    def __init__(self, commands: Iterable[SlashCommand] = ()) -> None:
        self._commands: list[SlashCommand] = []
        for position, command in enumerate(commands):
            command.order = position
            self._commands.append(command)

    # ------------------------------------------------------------------ access
    def all(self) -> list[SlashCommand]:
        return list(self._commands)

    def names(self) -> list[str]:
        return [command.name for command in self._commands]

    def get(self, name: str) -> SlashCommand | None:
        key = name.lstrip("/").lower()
        for command in self._commands:
            if command.name == key or key in command.aliases:
                return command
        return None

    # ------------------------------------------------------------------ filter
    def filter(self, query: str) -> list[CommandMatch]:
        """Fuzzy subsequence match over command names, best score first."""

        candidates: list[tuple[str, object]] = []
        for command in self._commands:
            candidates.append((command.name, command))
            for alias in command.aliases:
                candidates.append((alias, command))
        ranked = fuzzy_filter(candidates, query)
        matches: list[CommandMatch] = []
        seen: set[str] = set()
        for command, result in ranked:
            assert isinstance(command, SlashCommand)
            if command.name in seen:
                continue
            seen.add(command.name)
            matches.append(
                CommandMatch(
                    command=command,
                    matched_indices=list(result.matched_indices),
                    score=result.score,
                    order=command.order,
                )
            )
        return matches

    def highlighter(self, query: str):
        """Return a callable that marks matched characters in a command name."""

        def highlight(name: str) -> list[tuple[str, bool]]:
            result = fuzzy_match(name, query)
            if result is None or not result.matched_indices:
                return [(name, False)]
            marked: list[tuple[str, bool]] = []
            for index, char in enumerate(name):
                marked.append((char, index in result.matched_indices))
            return marked

        return highlight


def parse_slash(text: str) -> tuple[str, str] | None:
    """Split ``/name args`` into ``("name", "args")``; ``None`` for plain text."""

    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    body = stripped[1:]
    if not body:
        return "", ""
    name, _, argument = body.partition(" ")
    return name, argument.strip()


def build_default_registry(handlers: dict[str, Handler]) -> CommandRegistry:
    """Wire the MiniAgent command set to a dispatcher's handlers.

    The names and ordering mirror ``Detail.md``; ``/quit``/``/q`` are aliases of
    ``/exit``.
    """

    descriptions: Sequence[tuple[str, str, bool, tuple[str, ...]]] = (
        ("help", "show command help", False, ()),
        ("status", "current task, iterations, tokens, loaded skills", False, ()),
        ("model", "show or switch the model (/model <name>)", True, ()),
        ("tools", "list the tools available to the agent", False, ()),
        ("skills", "list skills, marking the loaded ones", False, ()),
        ("agents", "list the available subagents", False, ()),
        ("compact", "compact the current context now", False, ()),
        ("clear", "start a new thread (clear the conversation)", False, ()),
        ("exit", "leave the CLI", False, ("quit", "q")),
    )
    commands = [
        SlashCommand(
            name=name,
            description=description,
            handler=handlers.get(name),
            supports_args=supports_args,
            aliases=aliases,
            order=index,
        )
        for index, (name, description, supports_args, aliases) in enumerate(descriptions)
    ]
    return CommandRegistry(commands)


__all__ = [
    "SlashCommand",
    "CommandMatch",
    "CommandRegistry",
    "parse_slash",
    "build_default_registry",
    "FuzzyResult",
]
