"""Compose the CLI's terminal, input controller, presenter and conversation session.

Input: TerminalUI -> InputController -> Session -> AgentHarness.
Output: runtime events -> ConversationPresenter -> console or live transcript.
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field

from prompt_toolkit.application import Application

from harness.agent.events import Event
from harness.agent.state import AgentState
from harness.cli import events as ui
from harness.cli.approval import InteractiveApprovalProvider
from harness.cli.composer import CommandPopup, CommandRegistry, Composer, build_default_registry
from harness.cli.controller import InputController
from harness.cli.interaction import InteractiveInteractionProvider
from harness.cli.output import ConsoleOutput, TranscriptOutput
from harness.cli.presenter import ConversationPresenter
from harness.cli.render.renderer import Renderer
from harness.cli.session import Session
from harness.cli.state import AppState
from harness.cli.terminal import TerminalUI
from harness.infra.config import HarnessConfig


def load_config() -> HarnessConfig:
    return HarnessConfig.load()


@dataclass
class MiniAgentApp:
    """Application assembly and the entry points used by scripts and tests."""

    session: Session
    renderer: Renderer = field(default_factory=Renderer)
    state: AppState = field(default_factory=AppState)
    composer: Composer = field(init=False)
    approval: InteractiveApprovalProvider = field(init=False)
    interaction: InteractiveInteractionProvider = field(init=False)
    presenter: ConversationPresenter = field(init=False)
    controller: InputController = field(init=False)
    terminal: TerminalUI = field(init=False)

    def __post_init__(self) -> None:
        registry = CommandRegistry()
        self.composer = Composer(
            state=self.state.composer,
            popup=CommandPopup(registry, self.state.command_popup),
        )
        self.presenter = ConversationPresenter(self.state, ConsoleOutput(self.renderer))
        supplied_interaction = getattr(self.session, "interaction_provider", None)
        self.interaction = (
            supplied_interaction
            if supplied_interaction is not None
            else InteractiveInteractionProvider(
                state=self.state,
                on_change=lambda: self.terminal.invalidate(),
                activity=self._set_activity,
                answerable=lambda: self.terminal.active,
            )
        )
        supplied_approval = getattr(self.session, "approval_provider", None)
        self.approval = (
            supplied_approval
            if supplied_approval is not None
            else InteractiveApprovalProvider(interaction=self.interaction)
        )
        self.controller = InputController(
            self.session, self.presenter, registry,
            approval=self.approval, interaction=self.interaction,
            on_exit=lambda: self.terminal.exit(),
            on_clear=lambda: self.terminal.scroll_to_bottom(),
        )
        self.terminal = TerminalUI(
            self.state, self.composer, self.interaction,
            on_submit=self.controller.submit, on_cancel=self.controller.cancel,
            final_cell=lambda: self.presenter.final_cell, status=self._status,
        )
        if self.session is not None:
            self.session.on_event = self.on_runtime_event
            self.session.approval_provider = self.approval
            self.session.interaction_provider = self.interaction
            harness = getattr(self.session, "harness", None)
            if harness is not None:
                harness.approval_provider = self.approval
                harness.interaction_provider = self.interaction

    @property
    def registry(self) -> CommandRegistry:
        return self.controller.registry

    @registry.setter
    def registry(self, registry: CommandRegistry) -> None:
        self.controller.registry = registry
        self.composer.popup = CommandPopup(registry, self.state.command_popup)

    @property
    def running(self) -> bool:
        return self.controller.running

    def _set_activity(self, text: str) -> None:
        self.state.activity = text

    def _status(self) -> tuple[str, bool]:
        harness = getattr(self.session, "harness", None)
        return (harness.model_config.model if harness is not None else "?", self.running)

    async def setup(self) -> None:
        self.registry = build_default_registry(await self.session.command_handlers())
        self.composer.sync_popups()

    def emit(self, event: ui.AgentEvent) -> None:
        self.presenter.emit(event)

    def on_runtime_event(self, event: Event) -> None:
        self.controller.on_runtime_event(event)

    async def handle_input(self, text: str) -> bool:
        return await self.controller.handle_input(text)

    async def run_prompt(self, text: str) -> AgentState | None:
        return await self.controller.run_prompt(text)

    async def run_shell(self, command: str) -> None:
        await self.controller.run_shell(command)

    def render(self) -> None:
        self.presenter.render()

    def notify(self, message: str, *, error: bool = False) -> None:
        self.presenter.notify(message, error=error)

    def emit_line(self, text: str = "") -> None:
        self.presenter.emit_line(text)

    def clear_transcript(self) -> None:
        self.controller.clear_transcript()

    def scroll_to_bottom(self) -> None:
        self.terminal.scroll_to_bottom()

    def create_application(self) -> Application:
        return self.terminal.build()

    def _on_cancel(self) -> None:
        self.controller.cancel()

    async def prompt_loop(self) -> int:
        self.renderer.banner()
        self.presenter.output = TranscriptOutput(self.terminal.invalidate)
        try:
            return await self.terminal.run()
        finally:
            try:
                await self.controller.close()
            finally:
                self.presenter.output = ConsoleOutput(self.renderer)


# ------------------------------------------------------------------- entrypoint
async def async_main() -> int:
    from harness.infra.logging import configure_logging

    config = load_config()
    configure_logging(config.logging.level, logfile=config.logging.file, json=config.logging.json_output)

    app = MiniAgentApp(session=Session(config), renderer=Renderer())
    await app.setup()

    async with app.session:
        return await app.prompt_loop()


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv:
        print("Error: start MiniAgent without arguments", file=sys.stderr)
        return 2
    if not sys.stdin.isatty():
        print("Error: stdin is not a terminal", file=sys.stderr)
        return 1
    try:
        # 等待内部 async_main 返回
        return asyncio.run(async_main())
    except KeyboardInterrupt:  # pragma: no cover - user interrupt
        return 130


__all__ = [
    "main",
    "async_main",
    "MiniAgentApp",
    "Session",
    "load_config",
]
