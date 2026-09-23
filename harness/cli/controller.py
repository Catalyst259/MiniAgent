"""Input routing and ownership of the single active CLI task."""
from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Callable

from harness.agent.events import Event
from harness.agent.state import AgentState
from harness.cli import events as ui
from harness.cli.composer import CommandRegistry
from harness.cli.events_bridge import translate
from harness.cli.presenter import ConversationPresenter
from harness.cli.session import Session
from harness.cli.shell_intent import ShellIntent, execute_shell_intent


class InputController:
    def __init__(
        self, session: Session, presenter: ConversationPresenter, registry: CommandRegistry,
        *, approval, interaction, on_exit: Callable[[], None],
        on_clear: Callable[[], None],
    ) -> None:
        self.session = session
        self.presenter = presenter
        self.registry = registry
        self.approval = approval
        self.interaction = interaction
        self.on_exit = on_exit
        self.on_clear = on_clear
        self._task: asyncio.Task | None = None
        self._executing = False

    @property
    def running(self) -> bool:
        return self._executing or (self._task is not None and not self._task.done())

    def on_runtime_event(self, event: Event) -> None:
        translated = translate(event)
        if translated is not None:
            self.presenter.emit(translated)

    def submit(self, text: str) -> None:
        if not text.strip():
            return
        if self.running:
            self.notify("still working on the previous turn — input ignored (Ctrl+C aborts)")
            return
        self._task = asyncio.create_task(self._run_input(text))
        self.presenter.render()

    async def _run_input(self, text: str) -> None:
        try:
            if await self.handle_input(text):
                self.on_exit()
        except asyncio.CancelledError:
            self.notify("turn aborted")
            raise
        except KeyboardInterrupt:
            self.notify("interrupted")
        except Exception as exc:
            self.presenter.emit(ui.ErrorEvent(message=f"{type(exc).__name__}: {exc}", fatal=True))
        finally:
            self._task = None
            self.presenter.render()

    async def handle_input(self, text: str) -> bool:
        """Execute one input; return True when a command requests exit."""
        text = text.strip()
        if not text:
            return False
        self._executing = True
        try:
            if text.startswith("!"):
                await self.run_shell(text[1:].strip())
            elif text.startswith("/"):
                name, _, argument = text[1:].partition(" ")
                command = self.registry.get(name)
                if command is None or command.handler is None:
                    self.notify(f"unknown command `/{name}` — type /help", error=True)
                else:
                    return bool(await command.handler(self, argument.strip()))
            else:
                await self.run_prompt(text)
            return False
        finally:
            self._executing = False

    async def run_prompt(self, text: str) -> AgentState | None:
        assert self.session.harness is not None
        self.presenter.emit(ui.TurnStarted(prompt=text))
        was_executing = self._executing
        self._executing = True
        try:
            result = await self.session.ask(
                text, on_delta=lambda delta: self.presenter.emit(ui.AssistantDelta(text=delta))
            )
            self.presenter.emit(ui.TurnFinished(
                status=str(result.get("termination_status") or ""),
                final_answer=result.get("final_answer") or "",
                iterations=int(result.get("iteration") or 0),
            ))
            return result
        except asyncio.CancelledError:
            self.presenter.stop_turn("interrupted by Ctrl+C")
            self.presenter.emit(ui.ErrorEvent(message="interrupted by Ctrl+C"))
            raise
        except Exception as exc:
            self.presenter.stop_turn(str(exc))
            self.presenter.emit(ui.ErrorEvent(message=f"{type(exc).__name__}: {exc}", fatal=True))
            return None
        finally:
            self._executing = was_executing
            self.presenter.render()

    async def run_shell(self, command: str) -> None:
        harness = self.session.harness
        assert harness is not None
        checker = getattr(self.approval, "can_answer", False)
        can_answer = bool(checker() if callable(checker) else checker)
        intent = ShellIntent(command=command, call_id=f"user-shell-{self.session.turn}")
        try:
            await execute_shell_intent(harness, intent, can_answer=can_answer, emit=self.presenter.emit)
        except asyncio.CancelledError:
            self.presenter.stop_turn("interrupted by Ctrl+C")
            raise
        except Exception as exc:
            self.presenter.stop_turn(str(exc))
            raise
        else:
            self.presenter.emit(ui.TurnFinished())
        finally:
            self.presenter.render()

    def cancel(self) -> None:
        self._cancel_interactions()
        if self._task is not None and not self._task.done():
            if not self._task.cancelling():
                self._task.cancel()
            self.notify("aborting the current turn…")
        else:
            self.notify("input cleared")

    def _cancel_interactions(self) -> None:
        cancel = getattr(self.approval, "cancel", None)
        if callable(cancel):
            cancel()
        cancel = getattr(self.interaction, "cancel", None)
        if callable(cancel):
            cancel("the turn was cancelled")

    async def close(self) -> None:
        """Finish cancellation before the caller releases the session."""
        self._cancel_interactions()
        task = self._task
        if task is not None:
            if not task.done() and not task.cancelling():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            self._task = None

    def notify(self, message: str, *, error: bool = False) -> None:
        self.presenter.notify(message, error=error)

    def emit_line(self, text: str = "") -> None:
        self.presenter.emit_line(text)

    def clear_transcript(self) -> None:
        self.presenter.clear()
        self.on_clear()
