#!/usr/bin/env python3
"""Probe: does the in-app transcript viewport react to scroll state at all?

Reuses the headless driver's helpers (fake terminal + ANSI screen emulator).
Answers two questions separately:
  A. Does sending PageUp change ``app.state.transcript_scroll``?  (key delivery)
  B. Does setting ``transcript_scroll`` by hand change the rendered screen?
     (viewport mechanism)
"""

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


async def main() -> int:
    app = build_app(24, 100)
    await app.setup()

    log: list[str] = []
    live_chars = [0]
    output, _size_state = make_capture_output(24, 100, log, live_chars)
    screen = Screen(24, 100)
    cpr_pipe_r, cpr_pipe_w = os.pipe()
    os.set_blocking(cpr_pipe_w, False)

    def record(data: str) -> None:
        log.append(data)
        screen.feed(data.encode("utf-8", "replace"))
        if "\x1b[6n" in data:
            try:
                os.write(cpr_pipe_w, f"\x1b[{screen.row + 1};{screen.col + 1}R".encode())
            except OSError:
                pass

    output.write_raw = record
    output.write = lambda data: record(data.replace("\x1b", "?"))

    from harness.cli.render.transcript import TranscriptControl

    results: list[str] = []

    def say(*parts) -> None:
        results.append(" ".join(str(p) for p in parts))

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

                    control = TranscriptControl(app.state, final_cell=lambda: app._final_cell)
                    say("transcript 逻辑行数 =", control.line_count)
                    say("初始 transcript_scroll =", app.state.transcript_scroll)

                    before_screen = screen.visible()
                    top = lambda vis: next((l for l in vis if l.strip()), "")
                    say("")
                    for press in range(1, 5):
                        prev = screen.visible()
                        pipe_input.send_text("\x1b[5~")          # PageUp
                        await asyncio.sleep(0.6)
                        say(f"[A{press}] PageUp#{press}: scroll={app.state.transcript_scroll}"
                            f" 屏幕是否变化={screen.visible() != prev}"
                            f" 首行={top(screen.visible())[:44]!r}")

                    pipe_input.send_text("\x1b[F")               # End
                    await asyncio.sleep(0.5)
                    say(f"[A5] End: scroll={app.state.transcript_scroll}"
                        f" 首行={top(screen.visible())[:44]!r}")

                    say("")
                    app.state.transcript_scroll = 5              # 手动设滚动位置作为对照
                    app._invalidate_ui()
                    await asyncio.sleep(0.7)
                    say(f"[B] 手动 scroll=5: 屏幕首行={top(screen.visible())[:44]!r}")

                    app._ui_app.exit() if app._ui_app else None
                    try:
                        await asyncio.wait_for(run_task, timeout=5)
                    except Exception:
                        run_task.cancel()
    print("\n".join(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
