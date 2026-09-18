"""Mouse-wheel decoding for the transcript viewport.

prompt_toolkit only dispatches mouse events to a control when the renderer knows
the terminal height (``Renderer.height_is_known``), which on a non-full-screen
application requires the terminal to answer a cursor-position request (CPR).
Terminals that do not answer CPR would therefore never scroll — the exact
symptom this module exists to remove.

The wheel is decoded here and handled at application level (see
``composer.build_key_bindings``); every other mouse event is handed back to
prompt_toolkit's own handler.
"""

from __future__ import annotations

from typing import Final

#: X10/SGR button bit that marks a wheel event
_WHEEL_BIT: Final = 64
#: SGR prefix
_SGR_PREFIX: Final = "\x1b[<"


def wheel_direction(data: str) -> int | None:
    """Return ``-1`` for wheel-up, ``+1`` for wheel-down, ``None`` otherwise.

    Understands the two encodings terminals actually send: SGR
    (``\\x1b[<64;12;6M``) and the original X10 form (``\\x1b[M`` + 3 bytes).
    Modifier bits (shift/alt/ctrl) are ignored.
    """

    if not data:
        return None
    if data.startswith(_SGR_PREFIX):
        # SGR: "\x1b[<64;12;6M".  A trailing "m" is a button *release*; the
        # wheel only ever presses.
        if not data.endswith("M"):
            return None
        code = data[len(_SGR_PREFIX) :].split(";", 1)[0]
        try:
            button = int(code)
        except ValueError:
            return None
    elif data.startswith("\x1b[M") and len(data) >= 4:
        button = ord(data[3]) - 32
    else:
        return None

    if not button & _WHEEL_BIT:
        return None
    return -1 if (button & 0b11) == 0 else 1


__all__ = ["wheel_direction"]
