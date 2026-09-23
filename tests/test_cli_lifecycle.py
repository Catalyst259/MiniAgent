"""Behavior at the presenter, task controller and terminal seams."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from prompt_toolkit.application.current import create_app_session
from prompt_toolkit.input import create_pipe_input

from harness.cli import events as ui
from harness.cli.app import MiniAgentApp
from harness.cli.cells import AssistantCell, ToolCell, ToolStatus
from harness.cli.composer import CommandRegistry, SlashCommand
from harness.cli.controller import InputController
from harness.cli.output import ConsoleOutput, TranscriptOutput
from harness.cli.presenter import ConversationPresenter
from harness.cli.render.renderer import Renderer
from harness.cli.state import AppState
from tests.helpers import RecordingConsole, SizedDummyOutput


@pytest.fixture(params=["console", "transcript"])
def presenter(request):
    renderer = Renderer(console=RecordingConsole(), use_live_tail=False)
    output = ConsoleOutput(renderer) if request.param == "console" else TranscriptOutput(lambda: None)
    return ConversationPresenter(AppState(), output)


def test_completed_tool_ignores_late_output_in_both_modes(presenter):
    presenter.emit(ui.ToolStarted(call_id="tool-1", tool="read_file"))
    presenter.emit(ui.ToolFinished(call_id="tool-1", text="complete"))
    cell = presenter.state.history_cells[-1]
    presenter.emit(ui.ToolOutput(call_id="tool-1", text="late output"))
    assert cell.output == "complete"
    assert presenter.state.active_cell is None


def test_clear_discards_pending_and_final_message_state(presenter):
    presenter.emit(ui.AssistantStarted(message_id="old"))
    presenter.emit(ui.AssistantDelta(text="old answer"))
    presenter.emit(ui.TurnFinished(final_answer="old answer"))
    presenter.emit(ui.ToolStarted(call_id="old-tool", tool="read_file"))
    presenter.clear()
    assert presenter.final_cell is None
    assert presenter.state.active_cell is None
    assert presenter.state.activity == ""
    assert not presenter.state.history_cells
    presenter.emit(ui.ToolStarted(call_id="old-tool", tool="read_file"))
    presenter.emit(ui.ToolFinished(call_id="old-tool", text="new result"))
    assert len(presenter.state.history_cells) == 1
    assert presenter.state.history_cells[0].output == "new result"


def test_turn_completion_closes_stream_without_message_finished(presenter):
    presenter.emit(ui.TurnStarted(prompt="hello"))
    presenter.emit(ui.AssistantStarted(message_id="answer"))
    presenter.emit(ui.AssistantDelta(text="partial"))
    presenter.emit(ui.TurnFinished(final_answer="complete answer"))
    cell = presenter.final_cell
    assert cell.complete and cell.source == "complete answer"
    assert sum(isinstance(cell, AssistantCell) for cell in presenter.state.history_cells) == 1
    assert presenter.state.activity == ""


async def test_shell_submission_is_busy_and_cancellation_waits_for_cleanup(presenter):
    started, cleaning, release, cleaned = (asyncio.Event() for _ in range(4))
    calls = []

    async def check_batch(calls, **kwargs):
        return [SimpleNamespace(allowed=True)]

    async def run(call):
        calls.append(call)
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            cleaned.set()

    session = SimpleNamespace(turn=0, harness=SimpleNamespace(
        permission=SimpleNamespace(gate=SimpleNamespace(check_batch=check_batch)),
        tool_runtime=SimpleNamespace(run=run),
    ))
    controller = InputController(
        session, presenter, CommandRegistry(), approval=None, interaction=None,
        on_exit=lambda: None, on_clear=lambda: None,
    )
    controller.submit("!sleep 10")
    await asyncio.wait_for(started.wait(), 2)
    assert controller.running
    controller.submit("!another command")
    assert len(calls) == 1
    controller.cancel()
    await asyncio.wait_for(cleaning.wait(), 2)
    # Closing an already-cancelling task must not interrupt its cleanup again.
    closing = asyncio.create_task(controller.close())
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    await asyncio.wait_for(closing, 2)
    assert cleaned.is_set()
    assert not controller.running
    assert presenter.state.active_cell is None
    assert presenter.state.activity == ""


def test_stopped_turn_completes_partial_messages_and_fails_pending_tools(presenter):
    presenter.emit(ui.AssistantStarted(message_id="partial"))
    presenter.emit(ui.AssistantDelta(text="Working on it"))
    presenter.emit(ui.ToolStarted(call_id="pending", tool="shell"))
    presenter.stop_turn("cancelled")
    assistant, tool = presenter.state.history_cells
    assert isinstance(assistant, AssistantCell) and assistant.complete
    assert assistant.source == "Working on it"
    assert isinstance(tool, ToolCell) and tool.status == ToolStatus.FAILED
    assert tool.error == "cancelled"
    assert presenter.final_cell is None
    assert presenter.state.active_cell is None
    assert presenter.state.activity == ""


async def test_terminal_enter_reaches_agent_and_exit_closes_controller():
    answered = asyncio.Event()
    prompts = []

    async def ask(text, *, on_delta):
        prompts.append(text)
        on_delta("answer")
        answered.set()
        return {"final_answer": "answer", "termination_status": "final_answer"}

    async def exit_command(context, argument):
        return True

    session = SimpleNamespace(
        harness=SimpleNamespace(model_config=SimpleNamespace(model="test")), ask=ask,
    )
    app = MiniAgentApp(session, renderer=Renderer(console=RecordingConsole(), use_live_tail=False))
    app.registry = CommandRegistry([SlashCommand(name="exit", description="Exit", handler=exit_command)])
    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=SizedDummyOutput()):
            loop = asyncio.create_task(app.prompt_loop())
            try:
                pipe.send_text("hello\r")
                await asyncio.wait_for(answered.wait(), 3)
                assert prompts == ["hello"]
                pipe.send_text("/exit\r")
                assert await asyncio.wait_for(loop, 3) == 0
            finally:
                if not loop.done():
                    loop.cancel()
                    await asyncio.gather(loop, return_exceptions=True)
    assert not app.running
    assert not app.terminal.active
    assert isinstance(app.presenter.output, ConsoleOutput)
    assert app.presenter.final_cell.source == "answer"
