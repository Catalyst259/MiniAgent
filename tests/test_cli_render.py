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
    # and the live row is erased at the end
    assert out.rstrip().endswith("\r\x1b[2K")


def test_tool_print_erases_the_live_row_first(live):
    renderer, captured = live
    renderer.begin_stream()
    renderer.push_delta("thinking about it")
    renderer.render_cell(ToolCell(call_id="1", tool="read_file", arguments={"path": "a.py"}))
    out = captured.getvalue()
    # the row holding the streamed preview must be cleared before the tool line
    assert "\r\x1b[2K● read_file" in out, "the live row was not erased before the tool header"


def test_final_answer_has_its_own_marker_and_block():
    console = io.StringIO()
    renderer = Renderer(console=Console(file=console, width=100, highlight=False), use_live_tail=False)
    renderer.final_answer(AssistantCell(source="**done** and verified"), status="final_answer")
    rendered = console.getvalue()
    assert "── final answer ──" in rendered
    assert "done" in rendered
    assert rendered.count("──") >= 2


def test_transcript_cursor_tracks_last_line():
    from types import SimpleNamespace
    from harness.cli.cells import AssistantCell, UserCell

    state = SimpleNamespace(
        history_cells=[UserCell(text="question"), AssistantCell(source="answer")]
    )
    control = TranscriptControl(state)
    assert control.get_cursor_position().y == 1


def test_transcript_uses_tool_preview_limit():
    from types import SimpleNamespace
    from harness.cli.cells import ToolStatus

    cell = ToolCell(
        call_id="1",
        tool="read_file",
        output="\n".join(f"line {i}" for i in range(12)),
        status=ToolStatus.DONE,
    )
    fragments = TranscriptControl(SimpleNamespace(history_cells=[cell]))._get_fragments()
    rendered = "".join(text for _style, text in fragments)
    assert "line 2" in rendered
    assert "line 3" not in rendered
    assert "9 more line(s)" in rendered


def test_transcript_tool_can_expand_without_changing_tool_output():
    from types import SimpleNamespace
    from harness.cli.cells import ToolStatus

    cell = ToolCell(
        call_id="expand-me",
        tool="grep",
        output="\n".join(f"line {i}" for i in range(12)),
        status=ToolStatus.DONE,
    )
    state = SimpleNamespace(history_cells=[cell], expanded_tool_ids={"expand-me"})
    fragments = TranscriptControl(state)._get_fragments()
    rendered = "".join(text for _style, text in fragments)
    assert "line 7" in rendered
    assert "line 8" not in rendered
    assert "4 more line(s)" in rendered
    assert cell.output.count("line") == 12


def test_transcript_shows_non_persistent_activity():
    from types import SimpleNamespace

    state = SimpleNamespace(history_cells=[], expanded_tool_ids=set(), activity="waiting for model")
    rendered = "".join(text for _style, text in TranscriptControl(state)._get_fragments())
    assert "waiting for model" in rendered


def test_transcript_renders_assistant_markdown():
    from types import SimpleNamespace

    state = SimpleNamespace(
        history_cells=[AssistantCell(source="## Title\n\n- item")],
        expanded_tool_ids=set(),
    )
    rendered = "".join(text for _style, text in TranscriptControl(state)._get_fragments())
    assert "Title" in rendered and "• item" in rendered
    assert "## Title" not in rendered


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
