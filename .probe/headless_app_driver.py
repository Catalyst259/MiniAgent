#!/usr/bin/env python3
"""Headless renderer-capture driver for MiniAgentApp.create_application().

WHY THIS EXISTS
---------------
The audit asked for a real PTY (`pty.openpty`).  In this sandbox, allocating a
PTY is impossible: opening /dev/ptmx for writing is denied with EACCES by the
DSH file sandbox ("landlock-run: partial enforcement"), and the escalation
request was refused.

So this driver drives the *real* ``MiniAgentApp.create_application()``
prompt_toolkit ``Application`` (same layout, same key bindings, same
``refresh_interval``, same ``mouse_support=False``, same ``full_screen=False``)
inside ``patch_stdout`` exactly like ``MiniAgentApp.prompt_loop`` does, with two
fake terminal endpoints:

  * input  = prompt_toolkit's ``PosixPipeInput`` (a plain pipe; keystrokes are
    injected as the exact bytes a terminal would send),
  * output = ``Vt100_Output`` -- the *real* vt100 renderer, hence byte-for-byte
    the escape sequences a real terminal would receive -- writing into a StringIO
    whose ``isatty()`` is True and whose size comes from our own
    ``get_size()`` callback (so terminal width/rows are what we choose).

Every byte the renderer produces is captured in order, so the same ANSI screen
emulator used for PTY logs can replay it.

Usage:  python3 .probe/headless_app_driver.py <plan.json>
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import signal
import sys
import time

REPO = "/home/catalyst259/Personal-Files/Agent"
PROBE = os.path.join(REPO, ".probe")
sys.path.insert(0, REPO)

from prompt_toolkit.application.current import create_app_session  # noqa: E402
from prompt_toolkit.data_structures import Size  # noqa: E402
from prompt_toolkit.input.defaults import create_pipe_input  # noqa: E402
from prompt_toolkit.output.base import Output  # noqa: E402


class FakeStdout(io.StringIO):
    """A StringIO that claims to be a TTY (so Vt100_Output is used)."""

    encoding = "utf-8"

    def isatty(self) -> bool:  # noqa: D102
        return True

    def fileno(self) -> int:  # noqa: D102
        raise io.UnsupportedOperation("fileno")


def make_capture_output(rows: int, cols: int, log: list[str], live_chars: list[int] | None = None):
    """Build the real Vt100 renderer with its output redirected into `log`.

    ``write_raw``/``write`` are pure VT100 generation, so capturing them yields
    exactly the bytes the process would have written to the terminal.
    """

    from prompt_toolkit.output.vt100 import Vt100_Output

    state = {"rows": rows, "cols": cols}

    class CaptureOutput(Vt100_Output):
        def __init__(self) -> None:
            import contextlib

            sink = io.StringIO()
            with contextlib.redirect_stderr(sink):
                super().__init__(
                    FakeStdout(),
                    get_size=lambda: Size(rows=state["rows"], columns=state["cols"]),
                    term="xterm-256color",
                    default_color_depth=None,
                    enable_bell=False,
                )

        def write_raw(self, data: str) -> None:
            log.append(data)
            if live_chars is not None:
                live_chars[0] += len(data)

        def write(self, data: str) -> None:
            log.append(data.replace("\x1b", "?"))
            if live_chars is not None:
                live_chars[0] += len(data)

        def flush(self) -> None:
            pass

        @property
        def responds_to_cpr(self) -> bool:
            # The pipe-input harness cannot deliver a CPR reply reliably
            # (prompt_toolkit's own docs say CPR is unsupported when input is
            # piped), so we declare it unsupported: this only suppresses the
            # "terminal doesn't support CPR" warning and skips the 2s wait.
            return False

    return CaptureOutput(), state


def build_app(rows: int, cols: int):
    """Build the real MiniAgentApp with a mock model, offline."""

    from harness.cli.app import MiniAgentApp, Session, load_config
    from harness.cli.render.renderer import Renderer

    argv = ["--mock", "--no-memory", "--no-checkpoint"]
    from harness.cli.app import build_arg_parser

    args = build_arg_parser().parse_args(argv)
    config = load_config(args)
    app = MiniAgentApp(
        session=Session(config, model=args.model),
        renderer=Renderer(),
        stream=True,
        show_events=False,
    )
    return app


async def main() -> int:
    plan = json.load(open(sys.argv[1], encoding="utf-8"))
    name = plan["name"]
    rows = int(plan.get("rows", 24))
    cols = int(plan.get("cols", 100))
    idle_window = float(plan.get("idle_window", 3.0))
    hard_timeout = float(plan.get("hard_timeout", 25.0))
    refresh_interval = plan.get("refresh_interval")

    os.chdir(REPO)
    os.environ.pop("DEEPSEEK_API_KEY", None)
    os.environ.pop("EMBEDDING_API_KEY", None)

    log: list[str] = []
    marks: list[list] = []
    SNAPSHOTS: list[dict] = []
    SIZE_WATCH: dict[str, object] = {}
    started = time.monotonic()
    live_chars = [0]

    def stamp(label: str) -> None:
        marks.append([label, round(time.monotonic() - started, 4), live_chars[0]])
    app = build_app(rows, cols)
    await app.setup()

    output, size_state = make_capture_output(rows, cols, log, live_chars)
    write_times: list[float] = []

    # A live screen emulator plays the role of the terminal: it advances on
    # every byte the renderer emits and answers CPR (cursor position) queries.
    from ansi_emulator import Screen

    screen = Screen(rows, cols)
    cpr_replies = [0]
    cpr_pipe_r, cpr_pipe_w = os.pipe()
    os.set_blocking(cpr_pipe_w, False)

    def _record_write(data: str) -> None:
        write_times.append(time.monotonic() - started)
        log.append(data)
        live_chars[0] += len(data)
        screen.feed(data.encode("utf-8", "replace"))
        # Answer the renderer's CPR query the way a real terminal emulator
        # would, so prompt_toolkit does not emit its "terminal doesn't support
        # cursor position requests" warning.
        if "\x1b[6n" in data:
            cpr_replies[0] += 1
            reply = f"\x1b[{screen.row + 1};{screen.col + 1}R".encode()
            try:
                os.write(cpr_pipe_w, reply)
            except OSError:
                pass

    output.write_raw = lambda data: _record_write(data)
    output.write = lambda data: _record_write(data.replace("\x1b", "?"))

    async with app.session:
        with create_pipe_input() as pipe_input:
            with create_app_session(input=pipe_input, output=output):
                from prompt_toolkit.patch_stdout import patch_stdout

                # The Application binds input/output from the *active* app
                # session at construction time, so it must be built in here
                # (exactly like MiniAgentApp.prompt_loop does).
                application = app.create_application()
                if refresh_interval is not None:
                    application.refresh_interval = float(refresh_interval)
                app._ui_app = application

                app.renderer.banner()
                stamp("banner")
                with patch_stdout(raw=True):
                    run_task = asyncio.ensure_future(application.run_async())

                    def log_text() -> str:
                        return "".join(log)

                    async def wait_idle(seconds: float, timeout: float = 20.0) -> None:
                        """Wait for a quiet gap; return as soon as one is seen.

                        If the renderer never goes quiet (which is itself the
                        finding), give up after `timeout` and let the caller
                        measure real elapsed time instead of the nominal window.
                        """

                        deadline = time.monotonic() + timeout
                        last_count = len(write_times)
                        last_change = time.monotonic()
                        while time.monotonic() < deadline:
                            await asyncio.sleep(0.05)
                            if len(write_times) != last_count:
                                last_count = len(write_times)
                                last_change = time.monotonic()
                            elif time.monotonic() - last_change >= seconds:
                                return
                        return

                    async def wait_for(pattern: str, timeout: float = 20.0) -> bool:
                        deadline = time.monotonic() + timeout
                        while time.monotonic() < deadline:
                            if pattern in log_text():
                                return True
                            await asyncio.sleep(0.05)
                        return False

                    async def scripted() -> None:
                        for action in plan.get("actions", []):
                            await asyncio.sleep(float(action.get("delay", 0.0)))
                            kind = action["type"]
                            if kind == "send":
                                ok = await wait_for(action["after"]) if action.get("after") else True
                                if not ok:
                                    stamp(f"timeout_waiting:{action['after']!r}")
                                pipe_input.send_text(action["data"])
                                stamp(f"send:{action.get('label', action['data'])!r}")
                            elif kind == "wait_for":
                                ok = await wait_for(action["pattern"], float(action.get("timeout", 20.0)))
                                stamp(f"wait_for:{action['pattern']!r}:{'ok' if ok else 'TIMEOUT'}")
                            elif kind == "resize":
                                SIZE_WATCH["before"] = screen.visible()
                                SIZE_WATCH["scrollback_before"] = list(screen.scrollback)
                                SIZE_WATCH["before_at"] = round(
                                    time.monotonic() - started, 3
                                )
                                size_state["rows"] = int(action["rows"])
                                size_state["cols"] = int(action["cols"])
                                screen.rows = int(action["rows"])
                                screen.cols = int(action["cols"])
                                os.kill(os.getpid(), signal.SIGWINCH)
                                stamp(f"resize:{action['cols']}x{action['rows']}")
                            elif kind == "snapshot":
                                SNAPSHOTS.append(
                                    {
                                        "label": action["label"],
                                        "at": round(time.monotonic() - started, 3),
                                        "visible": screen.visible(),
                                        "scrollback_len": len(screen.scrollback),
                                        "write_calls": len(write_times),
                                    }
                                )
                                stamp(f"snapshot:{action['label']}")
                            elif kind == "run_turn":
                                stamp(f"turn_start:{action['text']!r}")
                                task = asyncio.ensure_future(
                                    app._run_application_input(action["text"])
                                )
                                app._ui_turn_task = task
                                await task
                                stamp(f"turn_end:{action['text']!r}")
                            elif kind == "idle":
                                await wait_idle(idle_window)
                                stamp("idle_measured")
                            elif kind == "sleep":
                                await asyncio.sleep(float(action["seconds"]))
                            elif kind == "quit":
                                pipe_input.send_text("\x03")
                                await asyncio.sleep(0.3)
                                pipe_input.send_text("\x03")
                                stamp("quit")

                    script = asyncio.ensure_future(scripted())
                    done, pending = await asyncio.wait(
                        {run_task, script}, timeout=hard_timeout,
                        return_when=asyncio.ALL_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                    if run_task in pending:
                        run_task.cancel()
                    await asyncio.sleep(0.2)

    raw = "".join(log)
    with open(os.path.join(PROBE, f"raw_{name}.bin"), "wb") as handle:
        handle.write(raw.encode("utf-8"))

    # final reconstructed terminal state
    final_screen = {
        "visible": screen.visible(),
        "scrollback_len": len(screen.scrollback),
        "scrollback": screen.scrollback,
        "modes": screen.modes,
        "mode_events": screen.mode_events,
        "unknown_sequences": screen.unknown[:40],
        "other_events": screen.other_events[:40],
    }
    with open(os.path.join(PROBE, f"screen_{name}.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {"final": final_screen, "snapshots": SNAPSHOTS, "resize": SIZE_WATCH},
            handle,
            indent=2,
            ensure_ascii=False,
        )
    with open(os.path.join(PROBE, f"visible_{name}.txt"), "w", encoding="utf-8") as handle:
        for index, line in enumerate(screen.visible()):
            handle.write(f"{index:3d}|{line}\n")
        handle.write(f"--- scrollback: {len(screen.scrollback)} lines ---\n")
        for index, line in enumerate(screen.scrollback):
            handle.write(f"{index:5d}|{line}\n")

    # split the stream by phase boundaries from the mark timestamps
    boundaries = []
    total = len(raw)
    for label, when, offset in marks:
        boundaries.append({"label": label, "at": when, "offset": offset})
    boundaries.append({"label": "END", "at": round(time.monotonic() - started, 4), "offset": total})

    # idle volume: bytes emitted between the last two marks of the idle window
    idle_stats = None
    for index, item in enumerate(boundaries):
        if item["label"] == "idle_measured" and index > 0:
            start_item = boundaries[index - 1]
            idle_stats = {
                "window_label": f"{start_item['label']} -> idle_measured",
                "seconds": round(item["at"] - start_item["at"], 3),
                "bytes": item["offset"] - start_item["offset"],
            }
    if idle_stats and idle_stats["seconds"]:
        idle_stats["bytes_per_second"] = round(idle_stats["bytes"] / idle_stats["seconds"], 2)

    # write-level stats for the final 4 seconds of the run (pure idle tail)
    tail_start = boundaries[-1]["at"] - 4.0
    tail = [when for when in write_times if when >= tail_start]
    gaps = [round(tail[i + 1] - tail[i], 4) for i in range(len(tail) - 1)]
    all_gaps = [round(write_times[i + 1] - write_times[i], 4) for i in range(len(write_times) - 1)]
    write_stats = {
        "total_write_calls": len(write_times),
        "tail_4s_write_calls": len(tail),
        "tail_4s_writes_per_second": round(len(tail) / 4.0, 2),
        "tail_4s_max_gap_seconds": max(gaps) if gaps else None,
        "run_max_gap_seconds": max(all_gaps) if all_gaps else None,
        "run_write_calls_per_second": round(len(write_times) / (boundaries[-1]["at"] or 1), 2),
    }

    meta = {
        "name": name,
        "mode": "headless-vt100-capture",
        "rows_rows_note": "terminal size used for Vt100_Output.get_size()",
        "rows": rows,
        "cols": cols,
        "total_bytes": total,
        "probe_wall_seconds": round(time.monotonic() - started, 3),
        "marks": boundaries,
        "idle": idle_stats,
        "writes": write_stats,
        "cpr_queries": cpr_replies[0],
        "final_scrollback_len": len(screen.scrollback),
        "modes": screen.modes,
        "hard_timeout": hard_timeout,
    }
    with open(os.path.join(PROBE, f"meta_{name}.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(asyncio.wait_for(main(), timeout=float(json.load(
        open(sys.argv[1], encoding="utf-8")).get("hard_timeout", 25.0)) + 10)))
