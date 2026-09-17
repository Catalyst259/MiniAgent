"""Tests for the real prompt_toolkit integration (frame, completer, bindings).

A pseudo-terminal cannot be allocated inside this sandbox, so the interactive
frame is exercised through the composer/prompt-frame API that prompt_toolkit
itself calls, plus a real application run with pipe input for the key layer.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from harness.cli.app import MiniAgentApp, Session
from harness.cli.composer import CommandPopup, CommandRegistry, build_default_registry
from harness.cli.composer.composer import PromptInterrupt, SlashCompleter, build_key_bindings
from harness.cli.render import Renderer
from harness.cli.state import AppState, CommandPopupState
from harness.infra.config import HarnessConfig
from tests.helpers import RecordingConsole

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def app(tmp_path) -> MiniAgentApp:
    config = HarnessConfig.load(None)
    config.base_dir = str(ROOT)
    config.runtime.workspace_root = str(tmp_path)
    config.memory.enabled = False
    config.checkpoint.enabled = False
    from harness.inference.config import ModelConfig

    config.models = {"main": ModelConfig(provider="mock", model="mock-main")}
    config.default_model = "main"
    return MiniAgentApp(
        session=Session(config),
        renderer=Renderer(console=RecordingConsole(), use_live_tail=False),
        stream=False,
    )


async def test_popup_rows_render_above_the_prompt_mark(app):
    await app.setup()
    app.composer.sync_from_buffer("/mo")
    app.composer.sync_popups()
    fragments = app.composer.prompt_fragments()
    plain = "".join(text for _, text in fragments)
    assert plain.rstrip().endswith("›")
    assert "model" in plain and "show or switch the model" in plain
    assert app.composer.popup.visible


async def test_prompt_mark_only_when_popup_is_hidden(app):
    await app.setup()
    app.composer.sync_from_buffer("plain question")
    app.composer.sync_popups()
    fragments = app.composer.prompt_fragments()
    assert fragments == [("fg:ansigreen bold", "› ")]


async def test_completer_offers_only_slash_commands(app):
    await app.setup()
    from prompt_toolkit.document import Document

    completer = SlashCompleter(app.registry)
    assert [c.text for c in completer.get_completions(Document("/mod", 4), None)] == ["/model"]
    assert list(completer.get_completions(Document("plain text", 10), None)) == []


async def test_key_bindings_cover_the_required_keys(app):
    await app.setup()
    bindings = build_key_bindings(app.composer)
    keys = {tuple(str(key) for key in binding.keys) for binding in bindings.bindings}
    # prompt_toolkit normalises Tab to ControlI and Enter to ControlM
    for expected in ("Keys.Up", "Keys.Down", "Keys.ControlI", "Keys.Escape", "Keys.ControlC", "Keys.ControlM"):
        assert any(expected in key for key in keys), (expected, keys)


async def test_key_bindings_expose_named_handlers(app):
    """Each key route is a named handler, so the wiring is auditable."""

    await app.setup()
    bindings = build_key_bindings(app.composer)
    names = {binding.handler.__name__ for binding in bindings.bindings}
    for expected in ("_submit", "_popup_up", "_popup_down", "_tab", "_escape", "_interrupt", "_newline"):
        assert expected in names, (expected, names)


async def test_popup_navigation_moves_the_selection(app):
    """Driving the composer the way the key handlers do."""

    await app.setup()
    app.composer.sync_from_buffer("/")
    app.composer.sync_popups()
    popup = app.composer.popup
    assert popup.visible
    first = popup.state.selected_match.command.name
    popup.move(1)
    second = popup.state.selected_match.command.name
    assert first != second
    popup.move(-1)
    assert popup.state.selected_match.command.name == first


# ------------------------------------------------------------------ regression
async def test_slash_completer_is_a_real_prompt_toolkit_completer():
    """Regression: duck-typing the Completer protocol wedged the prompt.

    ``complete_while_typing`` drives completions through
    ``get_completions_async``, which only exists on the base class, so a plain
    class raised ``AttributeError`` inside the prompt_toolkit event loop and the
    terminal stopped accepting input entirely.
    """

    from prompt_toolkit.completion import CompleteEvent, Completer
    from prompt_toolkit.document import Document

    completer = SlashCompleter(build_default_registry({}))
    assert isinstance(completer, Completer)
    assert hasattr(completer, "get_completions_async")

    completions = [
        completion
        async for completion in completer.get_completions_async(
            Document("/mo", 3), CompleteEvent()
        )
    ]
    assert [completion.text for completion in completions] == ["/model"]


async def test_slash_completer_swallows_registry_errors():
    """A failing completer must degrade to "no suggestions", never crash."""

    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    class Exploding:
        def filter(self, query):  # noqa: ANN001, ARG002
            raise RuntimeError("boom")

    completer = SlashCompleter(Exploding())
    completions = [
        completion
        async for completion in completer.get_completions_async(
            Document("/mo", 3), CompleteEvent()
        )
    ]
    assert completions == []


async def test_prompt_session_runs_with_complete_while_typing(app):
    """The real PromptSession configuration must survive typing a slash."""

    import asyncio

    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    await app.setup()
    with create_pipe_input() as pipe:
        session = PromptSession(
            input=pipe,
            output=DummyOutput(),
            completer=SlashCompleter(app.registry),
            key_bindings=build_key_bindings(app.composer),
            complete_while_typing=True,
            reserve_space_for_menu=0,
        )

        def message():
            return FormattedText(app.composer.prompt_fragments())

        task = asyncio.ensure_future(session.prompt_async(message))
        await asyncio.sleep(0.2)
        pipe.send_bytes(b"/mo")  # this is what used to raise inside the loop
        await asyncio.sleep(0.4)
        state = session.default_buffer.complete_state
        assert state is not None and [c.text for c in state.completions] == ["/model"]
        task.cancel()
        try:
            await task
        except BaseException:  # noqa: BLE001 - cancellation is expected
            pass


# -------------------------------------------------- regression: loop wiring
#
# These cover a class of bug that shipped once: the key bindings cleared the
# buffer without ever ending the prompt, so Enter did nothing, no request ever
# reached the model and Ctrl+C could not quit.  Everything here drives a real
# PromptSession through pipe input.


def _pipe_session(app, pipe):
    """A PromptSession wired exactly like MiniAgentApp.create_prompt_session()."""

    from prompt_toolkit import PromptSession

    return PromptSession(
        input=pipe,
        output=DummyOutput(),
        completer=SlashCompleter(app.registry),
        key_bindings=build_key_bindings(app.composer, cancel=app._on_cancel),
        complete_while_typing=True,
        reserve_space_for_menu=0,
    )


async def _send(pipe, session, keys: bytes, settle: float = 0.35):
    pipe.send_bytes(keys)
    await asyncio.sleep(settle)
    return session


async def test_enter_returns_the_submitted_text(app):
    """Enter must END the prompt; otherwise no input ever reaches the agent."""

    await app.setup()
    with create_pipe_input() as pipe:
        session = _pipe_session(app, pipe)
        task = asyncio.ensure_future(session.prompt_async("› "))
        await asyncio.sleep(0.2)
        await _send(pipe, session, b"/help")
        assert session.default_buffer.text == "/help"
        await _send(pipe, session, b"\r")
        assert task.done(), "Enter did not return from the prompt"
        assert task.result() == "/help"


async def test_enter_does_not_accept_a_completion_forever(app):
    """With the popup visible, Enter still submits the buffer."""

    await app.setup()
    app.registry = build_default_registry({})
    with create_pipe_input() as pipe:
        session = _pipe_session(app, pipe)
        task = asyncio.ensure_future(session.prompt_async("› "))
        await asyncio.sleep(0.2)
        await _send(pipe, session, b"/")
        await _send(pipe, session, b"\r")
        assert task.done(), "Enter kept accepting the popup instead of submitting"
        assert task.result() == "/"


async def test_arrow_and_tab_complete_the_command(app):
    await app.setup()
    with create_pipe_input() as pipe:
        session = _pipe_session(app, pipe)
        task = asyncio.ensure_future(session.prompt_async("› "))
        await asyncio.sleep(0.2)
        await _send(pipe, session, b"/")
        await _send(pipe, session, b"\x1b[B")  # ↓ moves the selection
        await _send(pipe, session, b"\t")  # Tab writes it into the buffer
        assert session.default_buffer.text in {"/help", "/status", "/model", "/tools"}
        await _send(pipe, session, b"\r")
        assert task.done() and task.result().startswith("/")


async def test_ctrl_c_clears_text_then_interrupts(app):
    await app.setup()
    with create_pipe_input() as pipe:
        session = _pipe_session(app, pipe)
        task = asyncio.ensure_future(session.prompt_async("› "))
        await asyncio.sleep(0.2)
        await _send(pipe, session, b"draft text")
        await _send(pipe, session, b"\x03")  # first press: clear the line
        assert not task.done() and session.default_buffer.text == ""
        await _send(pipe, session, b"\x03")  # second press: interrupt the prompt
        assert task.done()
        with pytest.raises(PromptInterrupt):
            task.result()


async def test_ctrl_d_exits_with_eof(app):
    await app.setup()
    with create_pipe_input() as pipe:
        session = _pipe_session(app, pipe)
        task = asyncio.ensure_future(session.prompt_async("› "))
        await asyncio.sleep(0.2)
        await _send(pipe, session, b"\x04")
        assert task.done()
        with pytest.raises(EOFError):
            task.result()


async def test_prompt_loop_runs_commands_and_exits(app):
    """The whole loop, not just the bindings: slash command -> request -> exit."""

    await app.setup()
    console = app.renderer.console
    with create_pipe_input() as pipe:
        app.create_prompt_session = lambda: _pipe_session(app, pipe)
        async with app.session:
            loop = asyncio.ensure_future(app.prompt_loop())
            await asyncio.sleep(0.3)
            for text in ("/help", "!echo loop-check", "list the files", "/exit"):
                pipe.send_bytes(text.encode())
                await asyncio.sleep(0.25)
                pipe.send_bytes(b"\r")
                await asyncio.sleep(1.6 if text.startswith("list") else 0.6)
                if loop.done():
                    break
            assert loop.done(), "the loop never exited"
            assert await asyncio.wait_for(loop, timeout=5) == 0

    printed = console.rich_text() if hasattr(console, "rich_text") else console.plain()
    assert "/status" in printed, "the /help handler did not run"
    assert "loop-check" in printed, "the !shell intent did not run"
    kinds = [type(cell).__name__ for cell in app.state.history_cells]
    assert "AssistantCell" in kinds, "no request reached the agent"


async def test_plain_loop_renders_a_fatal_error_turn(app, monkeypatch):
    """Regression: a turn ending in a fatal ErrorEvent rendered nothing.

    ``run_prompt`` catches the exception, emits a fatal ``ErrorEvent`` and
    returns ``None`` without ever reaching the ``TurnFinished`` render path.
    ``_plain_loop`` therefore has to flush the renderer itself, or the user
    sees a blank screen and no explanation.
    """

    from harness.cli import app as app_module
    from harness.cli import events as ui

    await app.setup()
    console = app.renderer.console

    async def fake_handle_input(line: str) -> bool:
        # Mimic the memory-lock failure: a fatal error cell, no TurnFinished.
        app.emit(ui.ErrorEvent(message="RuntimeError: memory store is locked", fatal=True))
        return True

    monkeypatch.setattr(app, "handle_input", fake_handle_input)
    monkeypatch.setattr(app_module.asyncio, "to_thread", lambda fn, *a, **k: _ready(fn(*a, **k)))

    lines = iter(["hello"])

    def fake_input(prompt: str = "") -> str:
        try:
            return next(lines)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)

    assert await app_module._plain_loop(app) == 0
    printed = console.rich_text() if hasattr(console, "rich_text") else console.plain()
    assert "memory store is locked" in printed, "the fatal error was never rendered"


async def _ready(value):
    return value
