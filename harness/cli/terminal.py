"""prompt_toolkit layout and terminal lifecycle; execution arrives via callbacks."""
from __future__ import annotations

from typing import Callable

from prompt_toolkit.application import Application
from prompt_toolkit.patch_stdout import patch_stdout

from harness.cli.cells import AssistantCell
from harness.cli.composer import Composer
from harness.cli.composer.composer import PromptInterrupt, SlashCompleter, build_key_bindings
from harness.cli.interaction import InteractiveInteractionProvider
from harness.cli.render.transcript import TranscriptControl, TranscriptPane
from harness.cli.state import AppState

BANNER_AGENT = "MiniAgent"


class TerminalUI:
    def __init__(
        self,
        state: AppState,
        composer: Composer,
        interaction: InteractiveInteractionProvider,
        *,
        on_submit: Callable[[str], None],
        on_cancel: Callable[[], None],
        final_cell: Callable[[], AssistantCell | None],
        status: Callable[[], tuple[str, bool]],
    ) -> None:
        self.state = state
        self.composer = composer
        self.interaction = interaction
        self.on_submit = on_submit
        self.on_cancel = on_cancel
        self.final_cell = final_cell
        self.status = status
        self.application: Application | None = None
        self.pane: TranscriptPane | None = None

    @property
    def active(self) -> bool:
        return self.application is not None

    def invalidate(self) -> None:
        if self.application is not None:
            self.application.invalidate()

    def exit(self) -> None:
        if self.application is not None and self.application.is_running:
            self.application.exit()

    async def run(self) -> int:
        self.application = self.build()
        try:
            with patch_stdout(raw=True):
                await self.application.run_async()
        except (PromptInterrupt, EOFError):
            return 0
        finally:
            self.application = None
            self.pane = None
        return 0

    def _toggle_latest_expandable(self) -> None:
        self.state.toggle_latest_expandable()
        self.invalidate()

    def build(self) -> Application:
        """Build the long-lived prompt_toolkit application."""
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, VSplit, Window
        from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
        from prompt_toolkit.layout.dimension import Dimension
        from prompt_toolkit.styles import Style

        def sync_buffer(buffer) -> None:  # noqa: ANN001
            self.composer.sync_from_buffer(buffer.text)
            self.composer.sync_popups()
            self.invalidate()

        buffer = Buffer(
            name="miniagent-input",
            multiline=True,
            completer=SlashCompleter(self.composer.popup.registry),
            on_text_changed=sync_buffer,
        )

        def submit(text: str) -> None:
            text = text.strip()
            buffer.reset()
            self.composer.sync_from_buffer("")
            self.composer.sync_popups()
            self.state.command_popup.reset()
            # a new turn belongs at the newest line, even if the user was
            # reading history
            self.scroll_to_bottom()
            if text:
                self.on_submit(text)

        popup = FormattedTextControl(
            lambda: self.composer.prompt_fragments()[:-1],
            focusable=False,
            show_cursor=False,
        )
        popup_container = ConditionalContainer(
            Window(popup, height=Dimension(min=1, max=7), dont_extend_height=True),
            filter=Condition(lambda: self.state.command_popup.visible),
        )
        interaction = FormattedTextControl(
            self.interaction.fragments,
            focusable=False,
            show_cursor=False,
        )
        interaction_container = ConditionalContainer(
            Window(interaction, wrap_lines=True, dont_extend_height=True),
            filter=Condition(lambda: self.interaction.waiting),
        )
        prompt = FormattedTextControl(
            lambda: [("class:prompt", "› ")],
            focusable=False,
            show_cursor=False,
        )
        input_control = BufferControl(buffer=buffer)
        transcript = TranscriptControl(self.state, final_cell=self.final_cell)
        pane = TranscriptPane(Window(transcript, wrap_lines=True))
        transcript.pane = pane
        self.pane = pane

        def scroll_history(action: str) -> None:
            if action == "page-up":
                pane.page(-1)
            elif action == "page-down":
                pane.page(1)
            elif action == "wheel-up":
                pane.scroll_lines(-3)  # towards older content
            elif action == "wheel-down":
                pane.scroll_lines(3)  # towards the newest line
            elif action == "top":
                pane.to_top()
            else:
                pane.to_bottom()
            self.invalidate()

        status = FormattedTextControl(self._status_fragments, focusable=False, show_cursor=False)

        key_bindings = build_key_bindings(
            self.composer,
            cancel=self.on_cancel,
            on_submit=submit,
            on_toggle_tool=self._toggle_latest_expandable,
            on_history_scroll=scroll_history,
            interaction=self.interaction,
        )

        root = HSplit(
            [
                pane,
                interaction_container,
                popup_container,
                VSplit([Window(prompt, width=2), Window(input_control)]),
                Window(status, height=1, dont_extend_height=True),
            ]
        )
        style = Style.from_dict(
            {
                "prompt": "ansigreen bold",
                "final-marker": "ansigreen bold",
                "user": "ansigreen bold",
                "assistant": "",
                "reasoning": "ansibrightblack italic",
                "tool-running": "ansiyellow",
                "tool-done": "ansigreen",
                "tool-failed": "ansired bold",
                "tool-body": "ansibrightblack",
                "subagent": "ansiblue",
                "skill": "ansimagenta",
                "info": "ansibrightblack",
                "error": "ansired",
                "error-fatal": "ansired bold",
                "transcript": "",
                "transcript-dim": "ansibrightblack",
                "activity": "ansibrightblack italic",
                "interaction-title": "ansicyan bold",
                "interaction-question": "ansiwhite bold",
                "interaction-detail": "ansibrightblack",
                "interaction-choice": "ansigreen bold",
                "interaction-dim": "ansibrightblack",
                "status": "ansibrightblack",
                "status-hint": "ansibrightblack",
                "scrollbar.background": "bg:#3a3a3a",
                "scrollbar.button": "bg:#8a8a8a",
            }
        )
        return Application(
            layout=Layout(root, focused_element=input_control),
            key_bindings=key_bindings,
            style=style,
            full_screen=False,
            # the wheel scrolls the transcript; Shift+drag still selects text in
            # terminals that support it
            mouse_support=True,
            # no idle repaint: every agent event invalidates the UI itself
            refresh_interval=None,
        )

    def _status_fragments(self):  # noqa: ANN202 - prompt_toolkit accepts a callable
        """One-line status bar: model, run state and the scroll hint."""

        model, running = self.status()
        run_state = "running" if running else "idle"
        parts = [("class:status", f" {BANNER_AGENT} · {model} · {run_state}")]
        pane = self.pane
        if pane is not None and pane.scrolled_up_by:
            parts.append(
                (
                    "class:status-hint",
                    f"   ↑ {pane.scrolled_up_by} line(s) above the latest · Ctrl+End jumps back",
                )
            )
        else:
            parts.append(
                ("class:status-hint", "   PageUp/wheel: history · Ctrl+O: expand · Ctrl+C: interrupt")
            )
        return parts

    def scroll_to_bottom(self) -> None:
        """Follow the newest line again (new turn, /clear, Ctrl+End)."""

        if self.pane is not None:
            self.pane.to_bottom()
            self.invalidate()
