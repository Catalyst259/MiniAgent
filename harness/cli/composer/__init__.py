"""Composer package: editable buffer, slash registry, fuzzy match, popup."""

from harness.cli.composer.command_popup import CommandPopup
from harness.cli.composer.composer import PROMPT_MARK, Composer, SlashCompleter, build_key_bindings
from harness.cli.composer.fuzzy_match import FuzzyResult, fuzzy_filter, fuzzy_match
from harness.cli.composer.slash_commands import (
    CommandMatch,
    CommandRegistry,
    SlashCommand,
    build_default_registry,
    parse_slash,
)

from harness.cli.state import CommandPopupState, TextAreaState

__all__ = [
    "CommandPopupState",
    "TextAreaState",
    "CommandPopup",
    "Composer",
    "SlashCompleter",
    "build_key_bindings",
    "PROMPT_MARK",
    "FuzzyResult",
    "fuzzy_filter",
    "fuzzy_match",
    "CommandMatch",
    "CommandRegistry",
    "SlashCommand",
    "build_default_registry",
    "parse_slash",
]
