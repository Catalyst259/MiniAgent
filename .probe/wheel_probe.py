#!/usr/bin/env python3
"""Probe: is the wheel event delivered to TranscriptControl.mouse_handler?"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, "/home/catalyst259/Personal-Files/Agent/.probe")
os.chdir("/home/catalyst259/Personal-Files/Agent")
os.environ.pop("DEEPSEEK_API_KEY", None)
os.environ.pop("EMBEDDING_API_KEY", None)
os.environ["TERM"] = "xterm-256color"

from ansi_emulator import Screen  # noqa: E402
from headless_app_driver import build_app, make_capture_output  # noqa: E402
from prompt_toolkit.application.current import create_app_session  # noqa: E402
from prompt_toolkit.input.defaults import create_pipe_input  # noqa: E402
from prompt_toolkit.patch_stdout import patch_stdout  # noqa: E402

results: list[str] = []


def say(*parts) -> None:
    results.append(" ".join(str(p) for p in parts))


async def main() -> int:
    from harness.cli.render.transcript import TranscriptControl

    calls: list[str] = []
    original = TranscriptControl.mouse_handler

    def spy(self, mouse_event):  # noqa: ANN001
        calls.append(str(mouse_event.event_type))
        return original(self, mouse_event)

    TranscriptControl.mouse_handler = spy  # type: ignore[method-assign]

    app = build_app(24, 100)
    await app.setup()
    log: list[str] = []
    output, _ = make_capture_output(24, 100, log, [0])
    screen = Screen(24, 100)
    cpr_r, cpr_w = os.pipe()
    os.set_blocking(cpr_w, False)

    def record(data: str) -> None:
        log.append(data)
        screen.feed(data.encode("utf-8", "replace"))
        if "\x1b[6n" in data:
            try:
                os.write(cpr_w, f"\x1b[{screen.row + 1};{screen.col + 1}R".encode())
            except OSError:
                pass

    output.write_raw = record
    output.write = lambda data: record(data.replace("\x1b", "?"))

    async with app.session:
        with create_pipe_input() as pipe_input:
            with create_app_session(input=pipe_input, output=output):
                application = app.create_application()
                app._ui_app = application
                app.renderer.banner()
                with patch_stdout(raw=True):
                    run_task = asyncio.ensure_future(application.run_async())
                    await asyncio.sleep(0.8)
                    for text in ("hi", "hi again", "one more", "last one"):
                        await app._run_application_input(text)
                    await asyncio.sleep(0.6)

                    pane = app._transcript_pane
                    say("pane:", type(pane).__name__,
                        "window_height:", pane.window_height,
                        "content_height:", pane.content_height,
                        "scroll:", pane.vertical_scroll,
                        "max_scroll:", pane.max_scroll)

                    pipe_input.send_text("\x1b[5~")
                    await asyncio.sleep(0.5)
                    say("[PageUp] scroll =", pane.vertical_scroll, "calls:", calls)
                    calls.clear()

                    for sequence in ("\x1b[<64;12;6M", "\x1b[<64;12;6m"):
                        pipe_input.send_text(sequence)
                        await asyncio.sleep(0.5)
                        say(f"[wheel-up {sequence!r}] scroll =", pane.vertical_scroll,
                            "control.mouse_handler calls:", calls)
                        calls.clear()

                    pipe_input.send_text("\x1b[<65;12;6M")
                    await asyncio.sleep(0.5)
                    say("[wheel-down] scroll =", pane.vertical_scroll, "calls:", calls)

                    pipe_input.send_text("\x1b[1;5F")           # Ctrl+End -> bottom
                    await asyncio.sleep(0.5)
                    say("[Ctrl+End] scroll =", pane.vertical_scroll,
                        "at_bottom =", pane.at_bottom)
                    pipe_input.send_text("\x1b[<64;12;6M")        # wheel up 3
                    await asyncio.sleep(0.5)
                    say("[wheel-up from bottom] scroll =", pane.vertical_scroll,
                        "scrolled_up_by =", pane.scrolled_up_by)

                    app._ui_app.exit()
                    try:
                        await asyncio.wait_for(run_task, timeout=5)
                    except Exception:
                        run_task.cancel()
    print("\n".join(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
