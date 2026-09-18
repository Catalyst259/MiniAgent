"""ChatComposer: the editable prompt plus its key routing.

Responsibilities (``CLI_Design.md`` sections 7, 8, 15):

* own the :class:`~harness.cli.state.TextAreaState` (the real input data),
* route keys: popup first when it is visible, otherwise the text area,
* call ``sync_popups()`` after every change so the candidate list follows the
  buffer and cursor.

The prompt_toolkit buffer is only a *mirror* of the text area state used for
rendering; every key handler mutates the composer state and then writes it back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterator

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.completion.base import CompleteEvent
from prompt_toolkit.document import Document

from harness.cli.composer.command_popup import CommandPopup
from harness.cli.composer.slash_commands import CommandRegistry
from harness.cli.state import CommandPopupState, TextAreaState

log = logging.getLogger(__name__)


class PromptInterrupt(Exception):
    """Ctrl+C on an empty prompt line.

    Deliberately *not* ``KeyboardInterrupt``: that is a ``BaseException`` and any
    escape into the asyncio event loop aborts the whole application (which is how
    a stuck prompt turned into "Ctrl+C kills the CLI").  This is a normal
    exception the CLI loop can count and act on.
    """

PROMPT_MARK = "› "


@dataclass
class Composer:
    """Mirror of the prompt_toolkit buffer, plus popup synchronisation."""

    state: TextAreaState = field(default_factory=TextAreaState)
    popup: CommandPopup | None = None
    on_submit: Callable[[str], None] | None = None

    # ------------------------------------------------------------------- editing
    def sync_from_buffer(self, text: str) -> None:
        self.state.set_text(text)

    def insert(self, chunk: str) -> None:
        self.state.insert(chunk)

    def backspace(self) -> None:
        self.state.backspace()

    def submit_text(self) -> str | None:
        """Return the submitted line and clear the buffer, or ``None`` if empty."""

        text = self.state.text.strip()
        if not text:
            self.state.clear()
            return None
        self.state.clear()
        if self.popup is not None:
            self.popup.sync("")
        return text

    # -------------------------------------------------------------------- popups
    def sync_popups(self, text: str | None = None) -> None:
        """Recompute derived popup state from the composer text."""

        if self.popup is None:
            return
        self.popup.sync(self.state.text if text is None else text)

    # ------------------------------------------------------------------ prompt UI
    def prompt_fragments(self) -> list[tuple[str, str]]:
        """Formatted-text fragments for the prompt frame (popup rows + input mark).

        prompt_toolkit turns this into ``StyleAndTextTuples``; the popup rows are
        separated by literal newlines, so the whole frame is one flat sequence.
        """

        lines: list[tuple[str, str]] = []
        popup = self.popup
        if popup is not None and popup.visible:
            state = popup.state
            matches = state.matches[state.scroll : state.scroll + state.window]
            for offset, match in enumerate(matches):
                index = state.scroll + offset
                selected = index == state.selected
                marker = "❯ " if selected else "  "
                row: list[tuple[str, str]] = [
                    (("fg:ansicyan bold" if selected else "fg:ansibrightblack"), marker),
                    ("fg:ansibrightblack", "/"),
                ]
                for char, matched in popup.highlight(match):
                    if matched:
                        style = "fg:ansiyellow bold" if selected else "fg:ansicyan bold"
                    else:
                        style = "fg:ansicyan" if selected else ""
                    row.append((style, char))
                padding = max(1, 20 - len(match.command.name))
                row.append(("", " " * padding))
                row.append(("fg:ansibrightblack", match.command.description))
                lines.extend(row)
                lines.append(("", "\n"))
            if len(state.matches) > len(matches):
                lines.append(
                    ("fg:ansibrightblack", f"  … {len(state.matches) - len(matches)} more\n")
                )
        lines.append(("fg:ansigreen bold", PROMPT_MARK))
        return lines


class SlashCompleter(Completer):
    """Registry-backed completer for prompt_toolkit.

    It *must* subclass :class:`prompt_toolkit.completion.Completer`: with
    ``complete_while_typing`` the buffer drives completions through
    ``get_completions_async``, which only exists on the base class.  Duck-typing
    the protocol raises ``AttributeError`` inside the event loop and wedges the
    prompt (the terminal stops accepting input entirely).
    """

    def __init__(self, registry: CommandRegistry) -> None:
        self.registry = registry

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterator[Completion]:
        try:
            yield from self._completions(document)
        except Exception:  # pragma: no cover - defensive
            # A completer exception is raised inside the prompt_toolkit event
            # loop, which used to wedge the whole prompt ("Press ENTER to
            # continue").  Degrade to "no suggestions" instead; the composer
            # popup keeps working.
            log.exception("slash completion failed")
            return

    def _completions(self, document: Document) -> Iterator[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return
        query = text[1:]
        for match in self.registry.filter(query):
            yield Completion(
                f"/{match.command.name}",
                start_position=-len(text),
                display=f"/{match.command.name}",
                display_meta=match.command.description,
            )


def build_key_bindings(
    composer: Composer,
    *,
    cancel=None,
    on_submit=None,
    on_toggle_tool=None,
    on_history_scroll=None,
    approval=None,
):
    """Key routing: popup owns navigation, the text area owns editing.

    Enter must *end the prompt* with the submitted text: a prompt_toolkit
    application only returns when the buffer is accepted or ``app.exit()`` is
    called.  A handler that merely clears the buffer leaves the loop waiting
    forever, so no input ever reaches the agent.

    Ctrl+C clears the current line; on an already empty line it raises
    ``KeyboardInterrupt`` out of the prompt so the caller can count consecutive
    interrupts and quit.

    ``approval`` is a :class:`~harness.cli.approval.InteractiveApprovalProvider`.
    While a permission question is open its keys take priority over the buffer,
    which is why they carry an ``approval_pending`` filter: without it, typing a
    "2" in the middle of a sentence would answer a question the user never saw.
    """

    from prompt_toolkit.application.current import get_app
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.key_binding import KeyBindings

    popup_visible = Condition(lambda: bool(composer.popup and composer.popup.visible))
    approval_pending = Condition(lambda: bool(approval is not None and approval.waiting))
    kb = KeyBindings()

    def _refill(buffer, text: str) -> None:
        buffer.text = text
        buffer.cursor_position = len(text)

    def _accept_selection() -> bool:
        if not (composer.popup and composer.popup.visible):
            return False
        completion = composer.popup.completion()
        if completion is None:
            return False
        _refill(get_app().current_buffer, completion)
        composer.sync_from_buffer(get_app().current_buffer.text)
        composer.sync_popups()
        return True

    @kb.add("up", filter=popup_visible)
    def _popup_up(event) -> None:  # noqa: ANN001
        _select(event, -1)

    @kb.add("down", filter=popup_visible)
    def _popup_down(event) -> None:  # noqa: ANN001
        _select(event, 1)

    @kb.add("tab")
    def _tab(event) -> None:  # noqa: ANN001
        composer.sync_from_buffer(event.current_buffer.text)
        composer.sync_popups()
        if not _accept_selection():
            # Up/Down already wrote the candidate into the buffer, so Tab here
            # means "indent" (and typing a space confirms the command, which
            # hides the popup).
            event.current_buffer.insert_text("    ")

    @kb.add("escape")
    def _escape(event) -> None:  # noqa: ANN001
        if composer.popup and composer.popup.visible:
            composer.popup.dismiss()

    # ------------------------------------------------------- permission prompt
    # Carries its own filter, so these bindings only exist while a question is
    # open.  Digits are safe to capture then: the question is modal, and the
    # composer keeps whatever the user had typed.
    if approval is not None:
        from harness.cli.approval import CHOICE_KEYS

        def _choice_for(scope: str):
            def _choice(event) -> None:  # noqa: ANN001
                approval.resolve(scope)

            return _choice

        for key, scope in CHOICE_KEYS.items():
            kb.add(key, filter=approval_pending)(_choice_for(scope))

        @kb.add("left", filter=approval_pending)
        def _approval_left(event) -> None:  # noqa: ANN001
            approval.move(-1)

        @kb.add("right", filter=approval_pending)
        def _approval_right(event) -> None:  # noqa: ANN001
            approval.move(1)

        @kb.add("enter", filter=approval_pending)
        def _approval_accept(event) -> None:  # noqa: ANN001
            approval.accept_selection()

        @kb.add("c-r", filter=approval_pending)
        def _approval_reject(event) -> None:  # noqa: ANN001
            approval.reject()

    @kb.add("c-c")
    def _interrupt(event) -> None:  # noqa: ANN001
        buffer = event.current_buffer
        if buffer.text:
            # first press clears the line and stays in the prompt
            buffer.text = ""
            composer.sync_from_buffer("")
            composer.sync_popups()
            if cancel is not None:
                cancel()
            return
        # already empty: let the caller count interrupts (the second one quits)
        event.app.exit(exception=PromptInterrupt())

    @kb.add("c-d")
    def _eof(event) -> None:  # noqa: ANN001
        if not event.current_buffer.text:
            event.app.exit(exception=EOFError)

    @kb.add("c-o")
    def _toggle_tool(event) -> None:  # noqa: ANN001
        if on_toggle_tool is not None:
            on_toggle_tool()

    @kb.add("pageup")
    def _history_up(event) -> None:  # noqa: ANN001
        if on_history_scroll is not None:
            on_history_scroll("page-up")

    @kb.add("pagedown")
    def _history_down(event) -> None:  # noqa: ANN001
        if on_history_scroll is not None:
            on_history_scroll("page-down")

    # Ctrl+Home / Ctrl+End, not plain Home/End: the plain keys belong to the
    # input line (README documents them as composer keys), and binding them
    # globally stole "end of line" from the buffer.
    @kb.add("c-home")
    def _history_top(event) -> None:  # noqa: ANN001
        if on_history_scroll is not None:
            on_history_scroll("top")

    @kb.add("c-end")
    def _history_bottom(event) -> None:  # noqa: ANN001
        if on_history_scroll is not None:
            on_history_scroll("bottom")

    if on_history_scroll is not None:
        from prompt_toolkit.key_binding.bindings.mouse import load_mouse_bindings
        from prompt_toolkit.keys import Keys as _Keys

        from harness.cli.mouse import wheel_direction

        # prompt_toolkit's own wheel dispatch needs a known terminal height
        # (CPR); intercepting the raw event here makes the wheel work on every
        # terminal.  Non-wheel mouse events are handed back to prompt_toolkit.
        default_mouse_bindings = load_mouse_bindings()

        @kb.add(_Keys.Vt100MouseEvent)
        def _wheel(event) -> None:  # noqa: ANN001
            direction = wheel_direction(event.data)
            if direction is not None:
                on_history_scroll("wheel-up" if direction < 0 else "wheel-down")
                return
            for binding in default_mouse_bindings.get_bindings_for_keys((_Keys.Vt100MouseEvent,)):
                if binding.filter():
                    binding.call(event)
                    return

    # Enter and Escape+Enter step aside while a permission question is open:
    # the approval bindings above already own Enter, and registration order would
    # otherwise let this one win (prompt_toolkit calls the *last* enabled match).
    @kb.add("enter", filter=~approval_pending)
    def _submit(event) -> None:  # noqa: ANN001
        # Enter submits *the buffer*; it must never just accept a completion,
        # or the prompt would never return and no input would reach the agent.
        # Tab (and Up/Down, which write the candidate into the buffer) handle
        # completion.
        text = event.current_buffer.text
        event.current_buffer.text = ""
        composer.sync_from_buffer("")
        composer.sync_popups()
        if on_submit is not None:
            on_submit(text)
            return
        # This is what actually returns control to the CLI loop.
        event.app.exit(result=text)

    def _select(event, delta: int) -> None:
        """Move the popup selection and mirror the candidate into the buffer.

        Re-syncing the popup with the buffer keeps the invariants: the buffer is
        the source of truth, the popup is derived from it.
        """

        composer.popup.move(delta)
        match = composer.popup.state.selected_match
        if match is None:
            return
        _refill(event.current_buffer, f"/{match.command.name}")
        composer.sync_from_buffer(event.current_buffer.text)
        composer.sync_popups()

    @kb.add("escape", "enter", filter=~approval_pending)
    def _newline(event) -> None:  # noqa: ANN001
        event.current_buffer.insert_text("\n")

    return kb


__all__ = [
    "Composer",
    "SlashCompleter",
    "build_key_bindings",
    "PromptInterrupt",
    "PROMPT_MARK",
]
