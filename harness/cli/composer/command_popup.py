"""Command popup: derived state + rendering + selection.

The popup owns no input; it reads ``composer.text`` every time the text changes
(``sync_popups()``), filters the registry, clamps the selection and renders the
candidate list above the prompt.  See ``CLI_Design.md`` sections 8, 9 and 14.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.cli.composer.slash_commands import CommandMatch, CommandRegistry, parse_slash
from harness.cli.state import CommandPopupState


@dataclass
class CommandPopup:
    registry: CommandRegistry
    state: CommandPopupState

    # --------------------------------------------------------------------- sync
    def sync(self, text: str) -> None:
        """Recompute matches from the composer text (never from its own buffer)."""

        parsed = parse_slash(text)
        if parsed is None:
            self.state.reset()
            return
        name, argument = parsed
        if argument:
            # an argument means the command is already chosen
            self.state.update(name, [], dismissed=True)
            return
        matches = self.registry.filter(name)
        self.state.update(name, matches)

    # ---------------------------------------------------------------- selection
    def move(self, delta: int) -> None:
        self.state.move(delta)

    def select_first(self) -> None:
        self.state.selected = 0
        self.state.scroll = 0

    def dismiss(self) -> None:
        self.state.dismiss()

    def completion(self) -> str | None:
        """The text to insert when Tab/Enter accepts the highlighted candidate."""

        match = self.state.selected_match
        if match is None:
            return None
        return f"/{match.command.name}"

    # ------------------------------------------------------------------ reading
    @property
    def visible(self) -> bool:
        return self.state.visible

    @property
    def matches(self) -> list[CommandMatch]:
        return self.state.matches

    def highlight(self, match: CommandMatch) -> list[tuple[str, bool]]:
        """``[(char, matched)]`` for rendering the fuzzy highlights."""

        name = match.command.name
        indices = set(match.matched_indices)
        return [(char, index in indices) for index, char in enumerate(name)]

    # ---------------------------------------------------------------- rendering
    def render(self, console, *, max_rows: int | None = None) -> None:
        """Print the candidate list with the selection marker and highlights."""

        from rich.text import Text

        if not self.state.visible:
            return
        rows = max_rows or self.state.window
        matches = self.state.matches
        start = self.state.scroll
        window = matches[start : start + rows]
        if not window:
            return
        console.print(f"[dim]commands matching[/dim] [bold]/{self.state.query}[/bold]")
        for offset, match in enumerate(window):
            index = start + offset
            selected = index == self.state.selected
            line = Text()
            marker = "❯ " if selected else "  "
            spans = self.highlight(match)
            body = Text()
            for char, matched in spans:
                body.append(
                    char,
                    style="bold yellow" if matched and selected else ("bold cyan" if matched else ""),
                )
            line.append(marker, style="cyan" if selected else "dim")
            line.append("/", style="dim")
            line.append_text(body)
            padding = max(1, 22 - len(match.command.name))
            line.append(" " * padding)
            line.append(match.command.description, style="dim")
            console.print(line, highlight=False)
        if len(matches) > rows:
            hidden = len(matches) - rows
            console.print(f"[dim]  … {hidden} more (↑/↓ to scroll)[/dim]")


__all__ = ["CommandPopup"]
