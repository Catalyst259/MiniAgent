"""Focused regression tests for the text-area editing arithmetic."""

from harness.cli.state import TextAreaState


def test_insert_in_the_middle_of_a_word():
    """Place a cursor inside the word instead of using hand-computed offsets."""

    state = TextAreaState()
    state.set_text("hello world")
    state.cursor = state.text.index("o") + 1  # just after the first "o"
    state.insert("X")
    assert state.text == "helloX world"
    assert state.cursor == state.text.index("X") + 1


def test_backspace_then_retype_roundtrip():
    state = TextAreaState()
    state.set_text("hello world")
    for _ in range(5):
        state.backspace()
    assert state.text == "hello "
    state.insert("there")
    assert state.text == "hello there"
    assert state.cursor == len("hello there")
