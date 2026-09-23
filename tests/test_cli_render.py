"""Regression tests for rendering: streaming layout, tool trace, sanitising."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from harness.agent.events import Event
from harness.cli import events as ui
from harness.cli.cells import AssistantCell, SubAgentCell, ToolCell
from harness.cli.events_bridge import translate
from harness.cli.render.renderer import Renderer
from harness.cli.render.transcript import TranscriptControl
from harness.cli.sanitize import clip_to_width, display_width, safe_text, strip_ansi


class _Stream(Renderer):
    """Renderer whose terminal writes are captured instead of printed."""

    def __init__(self, **kwargs):
        self.written = io.StringIO()
        super().__init__(console=Console(file=io.StringIO(), width=80, highlight=False), **kwargs)

    def _write_raw(self, text: str) -> None:  # pragma: no cover - helper
        self.written.write(text)


@pytest.fixture()
def live(monkeypatch):
    """A renderer with the live tail on, capturing sys.stdout writes.

    The rich console keeps its own buffer, so the console file is patched too;
    both then land in the same place and the assertion can read one stream.
    """

    captured = io.StringIO()
    renderer = Renderer(
        console=Console(file=captured, width=80, highlight=False),
        output=captured,
        use_live_tail=True,
    )
    return renderer, captured


# ------------------------------------------------------------------- sanitising
def test_strip_ansi_removes_colour_codes():
    assert strip_ansi("\x1b[32mgreen\x1b[0m plain") == "green plain"
    assert strip_ansi("a\x1b[2mb\x1b[0mc") == "abc"


def test_safe_text_removes_control_chars_and_expands_tabs():
    dirty = "col1\tcol2\r\n\x1b[31mred\x1b[0m\x07"
    assert safe_text(dirty) == "col1    col2\nred"


def test_safe_text_never_emits_a_half_escape_sequence():
    """The "[0m" garbage in terminals is a colour code whose ESC byte vanished."""

    tool_output = "ok \x1b[32mPASSED\x1b[0m done"
    cleaned = safe_text(tool_output)
    assert "\x1b" not in cleaned and "[0m" not in cleaned and "[32m" not in cleaned


def test_display_width_and_clipping():
    assert display_width("abc") == 3
    assert display_width("中文") == 4
    assert clip_to_width("中文abc", 5) == "中文a"
    assert clip_to_width("abc", 0) == ""


# ------------------------------------------------------------------ tool trace
def test_running_tool_announces_and_finished_tool_shows_body():
    console = io.StringIO()
    renderer = Renderer(console=Console(file=console, width=100, highlight=False), use_live_tail=False)

    running = ToolCell(call_id="1", tool="read_file", arguments={"path": "a.py"})
    renderer.render_cell(running)
    first = console.getvalue()
    assert "read_file" in first and "…" in first
    assert "│" not in first, "a running call must not print a body"

    running.finish(True, text="a.py (2 lines)\n1\tx = 1", duration_ms=7)
    renderer.render_cell(running)
    second = console.getvalue()
    assert "a.py (2 lines)" in second, "the tool result body was never rendered"
    assert second.count("read_file") == 2, "one announcement + one result header"


def test_duplicate_tool_render_is_suppressed():
    console = io.StringIO()
    renderer = Renderer(console=Console(file=console, width=100, highlight=False), use_live_tail=False)
    cell = ToolCell(call_id="1", tool="grep", arguments={"pattern": "x"})
    cell.finish(True, text="one hit")
    renderer.render_cell(cell)
    renderer.render_cell(cell)  # same state again (duplicate event)
    assert console.getvalue().count("grep") == 1


def test_same_tool_with_same_arguments_twice_still_renders_twice():
    console = io.StringIO()
    renderer = Renderer(console=Console(file=console, width=100, highlight=False), use_live_tail=False)
    for call_id in ("1", "2"):
        cell = ToolCell(call_id=call_id, tool="list_dir", arguments={"path": "."})
        cell.finish(True, text="f.py")
        renderer.render_cell(cell)
    assert console.getvalue().count("list_dir") == 2


def test_tool_output_with_ansi_is_cleaned_before_printing():
    console = io.StringIO()
    renderer = Renderer(console=Console(file=console, width=100, highlight=False), use_live_tail=False)
    cell = ToolCell(call_id="1", tool="shell")
    cell.finish(True, text="\x1b[32m48 passed\x1b[0m in 1.8s")
    renderer.render_cell(cell)
    rendered = console.getvalue()
    assert "48 passed" in rendered
    assert "[32m" not in rendered and "[0m" not in rendered


def test_tool_preview_is_shared_and_truncated():
    cell = ToolCell(call_id="1", tool="grep", output="\n".join(f"line {i}" for i in range(12)))
    lines, hidden = cell.preview_body()
    assert len(lines) == 8
    assert hidden == 4

    console = io.StringIO()
    renderer = Renderer(console=Console(file=console, width=100, highlight=False), use_live_tail=False)
    cell.finish(True)
    renderer.render_cell(cell)
    rendered = console.getvalue()
    assert "line 7" in rendered and "line 8" not in rendered
    assert "4 more line(s)" in rendered


# ------------------------------------------------------------------- streaming
def test_streamed_lines_land_one_per_row(live):
    renderer, captured = live
    renderer.begin_stream()
    for delta in ("Hello", ", I will", " fix this.\nThe", " problem is", " the regex.\n", "Done."):
        renderer.push_delta(delta)
    renderer.end_stream()

    out = captured.getvalue()
    # finished lines are newline-terminated...
    assert "  │ Hello, I will fix this.\n" in out
    assert "  │ The problem is the regex.\n" in out
    # ...while the mutable tail is redrawn in place on a single row
    assert "\r\x1b[2K  │ Done." in out
    # and the whole raw preview (2 committed rows) is erased at the end, so the
    # canonical Markdown rendering can replace it instead of repeating it
    assert out.rstrip().endswith("\x1b[2A\x1b[J")


def test_finished_answer_is_not_printed_twice_on_a_terminal():
    """Streaming preview + final Markdown must not duplicate the answer."""

    import re

    class _TTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    captured = _TTY()
    renderer = Renderer(
        console=Console(file=captured, width=100, highlight=False),
        output=captured,
        use_live_tail=True,
    )
    renderer.begin_stream()
    for delta in ("hello world\n", "second line\n"):
        renderer.push_delta(delta)
    source = renderer.end_stream()
    renderer.final_answer(AssistantCell(source=source or "hello world\nsecond line", complete=True))

    out = captured.getvalue()
    assert "  │ hello world\n" in out, "the live preview should still be streamed"
    assert "\x1b[2A\x1b[J" in out, "the raw preview rows must be erased"

    # Replay the escape sequences: after erasing, the answer appears once.
    rows = [""]
    cursor = 0
    for chunk in re.split(r"(\x1b\[\d*A|\x1b\[J|\r\x1b\[2K|\n)", out):
        if not chunk:
            continue
        if chunk == "\n":
            rows.append("")
            cursor += 1
        elif chunk == "\r\x1b[2K":
            rows[cursor] = ""
        elif chunk == "\x1b[J":
            del rows[cursor + 1 :]
            rows[cursor] = ""
        elif re.fullmatch(r"\x1b\[\d+A", chunk):
            cursor = max(0, cursor - int(chunk[2:-1]))
        else:
            rows[cursor] += chunk
    visible = "\n".join(rows)
    assert visible.count("hello world") == 1, visible
    assert "final answer" in visible


def test_tool_print_erases_the_live_row_first(live):
    renderer, captured = live
    renderer.begin_stream()
    renderer.push_delta("thinking about it")
    renderer.render_cell(ToolCell(call_id="1", tool="read_file", arguments={"path": "a.py"}))
    out = captured.getvalue()
    # the row holding the streamed preview must be cleared before the tool line
    assert "\r\x1b[2K◐ read_file" in out, "the live row was not erased before the tool header"


def test_final_answer_has_its_own_marker_and_block():
    console = io.StringIO()
    renderer = Renderer(console=Console(file=console, width=100, highlight=False), use_live_tail=False)
    renderer.final_answer(AssistantCell(source="**done** and verified"), status="final_answer")
    rendered = console.getvalue()
    assert "── final answer ──" in rendered
    assert "done" in rendered
    assert rendered.count("──") >= 2


def _state(cells, **kwargs):
    """A minimal AppState stand-in for transcript tests."""

    from types import SimpleNamespace

    base = {"history_cells": list(cells), "expanded_tool_ids": set(), "activity": ""}
    base.update(kwargs)
    return SimpleNamespace(**base)


def _rendered(control) -> str:
    return "".join(text for _style, text in control._get_fragments())


def test_transcript_pane_pages_by_a_full_screen():
    """PageUp moves exactly one window height, not eight logical lines."""

    from prompt_toolkit.layout import Window

    from harness.cli.render.transcript import TranscriptPane

    pane = TranscriptPane(Window(TranscriptControl(_state([]))))
    pane.window_height = 10
    pane.content_height = 100
    pane.to_bottom()
    assert pane.vertical_scroll == 90
    assert pane.at_bottom

    pane.page(-1)
    assert pane.vertical_scroll == 81  # one screen up, immediately visible
    assert pane.scrolled_up_by == 9

    pane.page(1)
    assert pane.vertical_scroll == 90
    assert pane.at_bottom


def test_transcript_pane_keeps_following_only_while_at_the_bottom():
    from prompt_toolkit.layout import Window

    from harness.cli.render.transcript import TranscriptPane

    pane = TranscriptPane(Window(TranscriptControl(_state([]))))
    pane.window_height = 10
    pane.content_height = 100
    pane.to_bottom()

    # new content arrives while the user reads the newest line: follow it
    pane.content_height = 120
    pane._apply_scroll()
    assert pane.vertical_scroll == 110

    # the user scrolls up: new content must not move the viewport
    pane.page(-1)
    assert pane.vertical_scroll == 101
    pane.content_height = 140
    pane._apply_scroll()
    assert pane.vertical_scroll == 101

    # Ctrl+End goes back to the tail and resumes following
    pane.to_bottom()
    assert pane.vertical_scroll == 130
    pane.content_height = 150
    pane._apply_scroll()
    assert pane.vertical_scroll == 140


def test_transcript_hides_nothing_about_a_running_tool():
    """A running tool cell is not in history yet and used to be invisible."""

    from harness.cli.cells import ToolStatus

    cell = ToolCell(call_id="c1", tool="shell", arguments={"command": "pytest -q"})
    state = _state([], active_cell=cell, activity="running shell")
    rendered = _rendered(TranscriptControl(state))
    assert "shell" in rendered
    assert "pytest -q" in rendered  # arguments are shown
    assert "running shell" in rendered


def test_transcript_tool_header_shows_arguments_and_duration():
    from harness.cli.cells import ToolStatus

    cell = ToolCell(
        call_id="c2",
        tool="read_file",
        arguments={"path": "harness/cli/app.py"},
        output="line",
        status=ToolStatus.DONE,
        duration_ms=42,
    )
    rendered = _rendered(TranscriptControl(_state([cell])))
    assert 'read_file  {"path": "harness/cli/app.py"}' in rendered
    assert "(42 ms)" in rendered


def test_transcript_uses_tool_preview_limit():
    from harness.cli.cells import ToolStatus

    cell = ToolCell(
        call_id="1",
        tool="read_file",
        output="\n".join(f"line {i}" for i in range(12)),
        status=ToolStatus.DONE,
    )
    fragments = TranscriptControl(_state([cell]))._get_fragments()
    rendered = "".join(text for _style, text in fragments)
    assert "line 2" in rendered
    assert "line 3" not in rendered
    assert "9 more line(s)" in rendered


def test_transcript_tool_can_expand_without_changing_tool_output():
    """Ctrl+O expands the body to every line, not to a bigger cap."""

    from harness.cli.cells import ToolStatus

    cell = ToolCell(
        call_id="expand-me",
        tool="grep",
        output="\n".join(f"line {i}" for i in range(12)),
        status=ToolStatus.DONE,
    )
    state = _state([cell], expanded_tool_ids={"expand-me"})
    fragments = TranscriptControl(state)._get_fragments()
    rendered = "".join(text for _style, text in fragments)
    assert "line 7" in rendered
    assert "line 11" in rendered
    assert "more line(s)" not in rendered
    assert cell.output.count("line") == 12


def test_transcript_clips_a_long_tool_line_to_one_row():
    """A 5000-column line must not push the whole transcript off screen."""

    from harness.cli.cells import ToolStatus

    cell = ToolCell(call_id="wide", tool="shell", output="A" * 5000, status=ToolStatus.DONE)
    control = TranscriptControl(_state([cell]))
    control.width = 40
    rendered = _rendered(control)
    body = [line for line in rendered.splitlines() if line.startswith("  │ ")]
    assert len(body) == 1
    assert len(body[0]) <= 40
    assert body[0].endswith("…")


def test_transcript_shows_reasoning_next_to_the_answer():
    from harness.cli.cells import AssistantCell

    cell = AssistantCell(source="done", reasoning="step 1\nstep 2", complete=True)
    rendered = _rendered(TranscriptControl(_state([cell])))
    assert "thinking" in rendered
    assert "step 1" in rendered
    assert "done" in rendered


def test_transcript_collapses_long_reasoning_until_expanded():
    from harness.cli.cells import AssistantCell

    cell = AssistantCell(
        source="done",
        reasoning="\n".join(f"thought {i}" for i in range(20)),
        complete=True,
    )
    collapsed = _rendered(TranscriptControl(_state([cell])))
    assert "thought 0" in collapsed
    assert "thought 19" not in collapsed
    assert "more line(s)" in collapsed

    from harness.cli.cells.base import reasoning_key

    expanded = _rendered(
        TranscriptControl(_state([cell], expanded_tool_ids={reasoning_key(cell)}))
    )
    assert "thought 19" in expanded


def test_transcript_state_toggle_expands_the_latest_expandable_cell():
    from harness.cli.cells import AssistantCell, ToolStatus
    from harness.cli.state import AppState

    tool = ToolCell(call_id="t1", tool="grep", output="a\nb", status=ToolStatus.DONE)
    assistant = AssistantCell(source="answer", reasoning="why", complete=True)
    state = AppState(history_cells=[tool, assistant])
    state.toggle_latest_expandable()
    assert state.expanded_tool_ids  # the assistant reasoning block
    state.toggle_latest_expandable()
    assert not state.expanded_tool_ids


def test_transcript_shows_non_persistent_activity():
    rendered = _rendered(TranscriptControl(_state([], activity="waiting for model")))
    assert "waiting for model" in rendered


def test_transcript_renders_assistant_markdown():
    state = _state([AssistantCell(source="## Title\n\n- item", complete=True)])
    rendered = _rendered(TranscriptControl(state))
    assert "Title" in rendered and "• item" in rendered
    assert "## Title" not in rendered
    styles = [style for style, _text in TranscriptControl(state)._get_fragments()]
    assert any("underline" in style or "bold" in style for style in styles)


def test_transcript_streaming_tail_is_not_rendered_as_markdown():
    """Half a code fence in the live tail must not restyle the answer."""

    cell = AssistantCell(source="intro\n```python\nx = ", complete=False)
    rendered = _rendered(TranscriptControl(_state([cell])))
    assert "intro" in rendered
    assert "x = " in rendered


def test_markdown_keeps_links_and_angle_brackets():
    """Regression: OSC-8 payload leaked as text; <value> was swallowed."""

    state = _state(
        [
            AssistantCell(
                source="see [docs](https://example.com) and use --flag <value> plus Vec<T>",
                complete=True,
            )
        ]
    )
    rendered = _rendered(TranscriptControl(state))
    assert "8;id=" not in rendered
    assert "8;;" not in rendered
    assert "docs" in rendered
    assert "<value>" in rendered
    assert "Vec<T>" in rendered


def test_markdown_escapes_only_outside_code_spans():
    from harness.cli.render.transcript import _escape_angle_brackets

    assert _escape_angle_brackets("a <b> c") == "a \\<b\\> c"
    assert _escape_angle_brackets("`a <b> c`") == "`a <b> c`"
    assert _escape_angle_brackets("```\n<b>\n```") == "```\n<b>\n```"
    # a blockquote marker is syntax, not text
    assert _escape_angle_brackets("> quoted <b>") == "> quoted \\<b\\>"
    assert _escape_angle_brackets(">> nested <b>") == ">> nested \\<b\\>"


def test_wheel_direction_understands_both_encodings():
    from harness.cli.mouse import wheel_direction

    assert wheel_direction("\x1b[<64;12;6M") == -1  # SGR wheel up
    assert wheel_direction("\x1b[<65;12;6M") == 1  # SGR wheel down
    assert wheel_direction("\x1b[<68;12;6M") == -1  # shift+wheel up
    assert wheel_direction("\x1b[<64;12;6m") is None  # release, not a wheel
    assert wheel_direction("\x1b[<0;12;6M") is None  # left button
    assert wheel_direction("\x1b[M" + chr(32 + 64) + "ab") == -1  # X10 wheel up
    assert wheel_direction("\x1b[M" + chr(32 + 65) + "ab") == 1  # X10 wheel down
    assert wheel_direction("") is None
    assert wheel_direction("\x1b[5~") is None  # PageUp is not a mouse event


def test_reasoning_reaches_the_transcript_control():
    """The bridge used to hardcode reasoning=None, so CoT never rendered."""

    from types import SimpleNamespace

    from harness.agent.events import Event

    translated = translate(
        Event(
            type="assistant_message",
            message="answer",
            data={"iteration": 1, "reasoning": "first I looked at the code"},
        )
    )
    assert "reasoning" in dir(translated)
    assert translated.reasoning == "first I looked at the code"

    cell = AssistantCell(source="answer", reasoning=translated.reasoning, complete=True)
    rendered = "".join(
        text for _style, text in TranscriptControl(_state([cell]))._get_fragments()
    )
    assert "first I looked at the code" in rendered


def test_runtime_assistant_message_event_carries_reasoning():
    """nodes.py must put the reasoning into the event payload."""

    import inspect

    from harness.orchestration import nodes

    source = inspect.getsource(nodes)
    assert '"reasoning": message.reasoning' in source


def test_command_output_becomes_transcript_cells():
    """Slash-command output must live in the transcript, not above the UI."""

    import asyncio

    from harness.cli.app import MiniAgentApp, Session
    from harness.cli.output import TranscriptOutput

    async def run() -> None:
        app = MiniAgentApp(session=None)
        app.presenter.output = TranscriptOutput(lambda: None)
        await Session.cmd_help(object(), app)
        assert app.state.history_cells, "command output did not enter the transcript"
        text = "".join(getattr(cell, "message", "") for cell in app.state.history_cells)
        assert "shell command" in text

    asyncio.run(run())


def test_markdown_renders_at_the_requested_width(monkeypatch):
    """The layout width comes from the window, not a hardcoded 120.

    ``TERM`` must be a real terminal name: rich falls back to 80x25 and ignores
    ``width`` when the terminal looks "dumb" (which is the case under pytest).
    """

    monkeypatch.setenv("TERM", "xterm-256color")
    from harness.cli.render.transcript import _render_markdown
    from harness.cli.sanitize import display_width

    source = "word " * 60
    narrow = "".join(text for _style, text in _render_markdown(source, width=60))
    wide = "".join(text for _style, text in _render_markdown(source, width=120))
    narrow_width = max(display_width(line) for line in narrow.splitlines() if line.strip())
    wide_width = max(display_width(line) for line in wide.splitlines() if line.strip())
    assert narrow_width <= 60
    assert wide_width > narrow_width


def test_stopped_turn_says_why():
    console = io.StringIO()
    renderer = Renderer(console=Console(file=console, width=100, highlight=False), use_live_tail=False)
    renderer.final_answer(AssistantCell(source="partial"), status="max_iterations")
    assert "stopped: max_iterations" in console.getvalue()


def test_subagent_cell_shows_task_and_result():
    console = io.StringIO()
    renderer = Renderer(console=Console(file=console, width=100, highlight=False), use_live_tail=False)
    renderer.render_cell(
        SubAgentCell(agent="explorer", task="find the auth code", summary="## Findings\n- auth.py", iterations=3)
    )
    rendered = console.getvalue()
    assert "explorer" in rendered and "find the auth code" in rendered and "auth.py" in rendered


# ------------------------------------------------------------------ event wiring
def test_delegate_end_event_carries_the_summary():
    event = Event(
        type="delegate_end",
        message="explorer ok (3 iterations)",
        data={"agent": "explorer", "ok": True, "iterations": 3, "summary": "## Findings\n- auth.py"},
    )
    translated = translate(event)
    assert isinstance(translated, ui.SubAgentFinished)
    assert translated.summary.startswith("## Findings")


def test_delegate_start_event_carries_the_task():
    translated = translate(
        Event(type="delegate_start", message="explorer: find auth", data={"agent": "explorer", "task": "find auth"})
    )
    assert isinstance(translated, ui.SubAgentStarted) and translated.task == "find auth"


def test_skill_and_tool_events_carry_what_the_ui_needs():
    skill = translate(Event(type="skill_load", message="loaded `debugging` (1200 chars)", data={"skill": "debugging", "ok": True}))
    assert isinstance(skill, ui.SkillLoaded) and skill.name == "debugging"

    started = translate(
        Event(type="tool_start", data={"id": "c1", "tool": "apply_patch", "arguments": {"patch": "..."}})
    )
    assert isinstance(started, ui.ToolStarted) and started.arguments == {"patch": "..."}
