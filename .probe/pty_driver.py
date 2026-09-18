#!/usr/bin/env python3
"""PTY probe driver for the MiniAgent interactive TUI.

Runs a command inside a real pseudo-terminal, feeds scripted keystrokes,
records every byte the process writes to the master fd with timestamps,
and supports a hard timeout so nothing can hang.

Usage:  python3 .probe/pty_driver.py <plan.json>

Plan schema:
{
  "name": "run1",
  "argv": ["./.venv/bin/python", "-m", "harness", "--mock", ...],
  "env": {"DEEPSEEK_API_KEY": null, ...},   # null/"" => unset
  "rows": 24, "cols": 100,
  "actions": [
      {"at": 2.0, "type": "send", "data": "hi"},
      {"at": 2.4, "type": "send", "data": "\r"},
      {"at": 9.0, "type": "resize", "rows": 24, "cols": 100, "sigwinch": true},
      {"at": 10.0, "type": "signal", "sig": "SIGINT"}
  ],
  "quiesce": 1.0,          # seconds of output silence = "turn finished"
  "idle_window": 3.0,      # seconds of idle to measure after quiesce
  "hard_timeout": 25.0,
  "cwd": "/path/to/repo"
}

Outputs (under .probe/):
  raw_<name>.bin          exact bytes written by the process
  timeline_<name>.jsonl   {"t": seconds, "hex": "...", "repr": "..."} per read chunk
  meta_<name>.json        plan + summary counters
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import pty
import select
import signal
import struct
import sys
import termios
import time

REPO = "/home/catalyst259/Personal-Files/Agent"
PROBE = os.path.join(REPO, ".probe")


def set_winsize(fd: int, rows: int, cols: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


def jdump(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def main() -> int:
    plan = json.load(open(sys.argv[1], encoding="utf-8"))
    name = plan["name"]
    rows = int(plan.get("rows", 24))
    cols = int(plan.get("cols", 100))
    hard_timeout = float(plan.get("hard_timeout", 25.0))
    quiesce = float(plan.get("quiesce", 1.0))
    idle_window = float(plan.get("idle_window", 3.0))
    cwd = plan.get("cwd", REPO)

    env = dict(os.environ)
    for key, value in (plan.get("env") or {}).items():
        if value is None or value == "":
            env.pop(key, None)
        else:
            env[key] = value
    env.setdefault("TERM", "xterm-256color")
    env["PYTHONUNBUFFERED"] = "1"

    argv = plan["argv"]
    actions = sorted(plan.get("actions", []), key=lambda item: item["at"])
    start = time.monotonic()

    pid, master = pty.fork()
    if pid == 0:  # child
        try:
            os.chdir(cwd)
            os.execvpe(argv[0], argv, env)
        except BaseException as exc:  # pragma: no cover
            os.write(2, f"exec failed: {exc}\n".encode())
            os._exit(127)

    set_winsize(master, rows, cols)
    os.set_blocking(master, False)

    os.environ["COLUMNS"] = str(cols)
    os.environ["LINES"] = str(rows)

    chunks: list[tuple[float, bytes]] = []
    buf = bytearray()
    action_index = 0
    last_output_at = time.monotonic()
    idle_measured: float | None = None
    idle_bytes: int | None = None
    idle_started: float | None = None
    idle_taken_at: float | None = None
    saw_quiesce_at: float | None = None
    killed_for_timeout = False
    exit_status = None

    def now() -> float:
        return time.monotonic() - start

    def reap() -> None:
        nonlocal exit_status
        try:
            done, status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return
        if done:
            exit_status = status

    def fire(action: dict) -> None:
        kind = action.get("type")
        if kind == "send":
            os.write(master, action["data"].encode("utf-8"))
        elif kind == "resize":
            new_rows = int(action.get("rows", rows))
            new_cols = int(action.get("cols", cols))
            set_winsize(master, new_rows, new_cols)
            if action.get("sigwinch"):
                os.kill(pid, signal.SIGWINCH)
        elif kind == "signal":
            os.kill(pid, getattr(signal, action.get("sig", "SIGINT")))

    while True:
        elapsed = now()

        # ---- fire scheduled actions
        while action_index < len(actions) and actions[action_index]["at"] <= elapsed:
            fire(actions[action_index])
            action_index += 1

        # ---- fire actions whose "after" watch-pattern has appeared in the stream
        while action_index < len(actions):
            action = actions[action_index]
            watch = action.get("after")
            if not watch or watch.encode() not in bytes(buf):
                break
            fire(action)
            action_index += 1

        # ---- idle measurement window (starts once the stream has gone quiet
        #      for `quiesce` seconds *after* the last scheduled action)
        if saw_quiesce_at is not None and idle_measured is None:
            if idle_started is None:
                idle_started = elapsed
                idle_bytes = 0
            if elapsed - idle_started >= idle_window:
                idle_measured = elapsed - idle_started
                idle_taken_at = elapsed

        # ---- read whatever the process wrote
        readable, _, _ = select.select([master], [], [], 0.02)
        if readable:
            try:
                data = os.read(master, 65536)
            except OSError as exc:
                if exc.errno in (errno.EIO, errno.EBADF):
                    data = b""
                else:
                    raise
            if data:
                chunks.append((now(), data))
                buf.extend(data)
                last_output_at = time.monotonic()
                if idle_started is not None and idle_measured is None:
                    idle_bytes += len(data)

        # ---- quiesce detection
        if (
            saw_quiesce_at is None
            and action_index >= len(actions)
            and actions
            and (time.monotonic() - last_output_at) >= quiesce
            and elapsed > 3.0
        ):
            saw_quiesce_at = now()

        # ---- stop conditions
        reap()
        if idle_measured is not None and exit_status is not None:
            break
        if idle_measured is not None:
            # drain a little more, then finish
            if elapsed - idle_taken_at > 0.5:
                break
        if elapsed > hard_timeout:
            killed_for_timeout = True
            break
        if exit_status is not None and not actions:
            break

    # ---- shutdown
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass
    try:
        os.close(master)
    except OSError:
        pass

    raw = bytes(buf)
    with open(os.path.join(PROBE, f"raw_{name}.bin"), "wb") as handle:
        handle.write(raw)

    with open(os.path.join(PROBE, f"timeline_{name}.jsonl"), "w", encoding="utf-8") as handle:
        for when, data in chunks:
            handle.write(
                json.dumps(
                    {"t": round(when, 4), "n": len(data), "repr": repr(data)[:4000]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    gaps = [
        (round(chunks[i + 1][0] - chunks[i][0], 4), round(chunks[i][0], 4))
        for i in range(len(chunks) - 1)
    ]
    biggest_gap = max(gaps, default=(0.0, 0.0))

    jdump(
        {
            "name": name,
            "argv": argv,
            "cwd": cwd,
            "rows": rows,
            "cols": cols,
            "plan": plan,
            "total_bytes": len(raw),
            "read_chunks": len(chunks),
            "first_output_at": round(chunks[0][0], 4) if chunks else None,
            "last_output_at": round(chunks[-1][0], 4) if chunks else None,
            "quiesce_at": round(saw_quiesce_at, 4) if saw_quiesce_at else None,
            "idle_window_seconds": idle_measured,
            "idle_bytes_in_window": idle_bytes,
            "idle_bytes_per_second": (round(idle_bytes / idle_measured, 2) if idle_measured else None),
            "biggest_gap_seconds": biggest_gap[0],
            "biggest_gap_starts_at": biggest_gap[1],
            "hard_timeout_hit": killed_for_timeout,
            "exit_status": exit_status,
        },
        os.path.join(PROBE, f"meta_{name}.json"),
    )
    print(json.dumps(json.load(open(os.path.join(PROBE, f"meta_{name}.json"))), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
