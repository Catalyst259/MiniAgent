"""Shell command parsing for permission rules.

A permission rule such as ``{"tool": "shell", "prefix": "npm"}`` is only sound if
the *whole* command is that program.  Three things break that assumption, and all
three are handled here:

1. **chaining** - ``npm install x; rm -rf ~`` also starts with ``npm``, so the
   line is rejected for prefix matching when it contains ``;``, ``&&``, ``|``,
   redirection, backticks, ``$(`` or a newline;
2. **wrapping** - ``env npm ...``, ``xargs rm ...`` and ``bash -c '...'`` mean the
   visible program is not the one doing the work;
3. **self-escalation** - ``git status -c core.pager=evil`` is a status command
   that executes arbitrary code, so options that make a program run something
   else disqualify a prefix rule too.

Parsing is best effort and always fails safe: anything that cannot be split, or
that trips one of the checks above, is reported as :attr:`ShellCommand.compound`
and therefore never satisfies a prefix rule.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass

#: Characters that let one command line run another command.  A prefix rule must
#: never match a command containing any of them.
_SEPARATORS = (";", "&&", "||", "|", "&", "\n", "\r")

#: Redirections and substitution: the target of a prefix rule stops being the
#: program the user approved.
_REDIRECTIONS = (">", "<", "`")

#: ``$(`` / ``${`` introduce command or variable substitution.
_SUBSTITUTION = ("$(", "${", "<(")

#: Programs that run whatever they are given, so "the command runs X" says
#: nothing about what actually happens.
_WRAPPERS = frozenset(
    {
        "env", "nice", "nohup", "setsid", "stdbuf", "timeout", "time", "watch",
        "xargs", "sudo", "doas", "su", "eval", "exec", "command", "script",
        "parallel", "bash", "sh", "zsh", "dash", "fish", "python", "python3",
        "perl", "ruby", "node", "make", "just", "poetry", "uv", "pipx",
    }
)

#: Options that make a program execute something else, or take a path operand
#: beyond the command's own arguments.  ``git status -c core.pager=evil`` is a
#: *status* command that runs arbitrary code, and ``git diff --no-index`` takes two
#: arbitrary paths, so a prefix rule for those programs must not cover them.
_ESCALATING_OPTIONS = frozenset(
    {
        "-c", "--config", "-C", "--git-dir", "--work-tree", "--exec-path",
        "--upload-pack", "--receive-pack", "--pager", "-O", "--eval",
        "--command", "--exec", "-e", "--module",
        # writing somewhere the workspace guard would never see
        "--output", "-o", "--outfile", "--out", "--file", "-f", "--to",
        # taking a path operand the command's own semantics do not imply
        "--no-index", "--no-prefix", "--src-prefix", "--dst-prefix",
    }
)

#: Options that make a program *write* the file they name, so a prefix allow-rule
#: must not cover them even though the shell never sees a redirect.
_WRITING_OPTIONS = frozenset(
    {"--output", "-o", "--outfile", "--out", "--file", "-f", "--to", "--dest", "--target"}
)

#: Options after which **every** following path-like operand is a file the program
#: will read, not a mode or a revision.  ``git diff --no-index a.txt /etc/passwd``
#: prints any file the user can read.
_READ_OPERAND_OPTIONS = frozenset({"--no-index"})


def _option_name(token: str) -> str:
    return token.split("=", 1)[0] if token.startswith("--") else token


def _option_values(tokens: tuple[str, ...], options: frozenset[str]) -> list[str]:
    """Values given to ``options``, in every spelling the shell allows."""

    found: list[str] = []
    for index, token in enumerate(tokens[1:], start=1):
        if token.startswith("--") and "=" in token:
            name, _, value = token.partition("=")
            if name in options and value:
                found.append(value)
            continue
        if token in options and index + 1 < len(tokens):
            found.append(tokens[index + 1])
    for token in tokens[1:]:
        if len(token) > 2 and token[:2] in options:
            # attached short form: -o/tmp/x, -f/etc/passwd
            found.append(token[2:].lstrip("="))
    return [value for value in found if value]


def writing_targets(tokens: tuple[str, ...]) -> list[str]:
    """Paths a command will write because of its own options.

    ``git log -p --output=/tmp/x`` writes a file while looking like a read-only
    command.  A prefix rule approves a *command*, so anything the command was told
    to write must be inspected like a filesystem target.
    """

    return _option_values(tokens, _WRITING_OPTIONS)


def reading_targets(tokens: tuple[str, ...]) -> list[str]:
    """Paths a command will read because of its own options.

    The read-side twin of :func:`writing_targets`: ``git diff --no-index a /etc/x``
    reads outside the workspace although every argument it was matched on looks
    harmless.  When an operand option is present, *every* later path-like operand
    is reported - ``--no-index`` takes two of them, and a bare word after it is as
    likely to be a revision as a file.
    """

    found: list[str] = []
    for index, token in enumerate(tokens):
        if token.split("=", 1)[0] in _READ_OPERAND_OPTIONS:
            found.extend(tokens[index + 1 :])
    return [value for value in found if _looks_like_path(value)]


def _escalates(tokens: tuple[str, ...]) -> str:
    """Why a syntactically simple command may still run something else."""

    if not tokens:
        return ""
    program = tokens[0].rsplit("/", 1)[-1]
    if program.endswith(".exe"):
        program = program[:-4]
    if program in _WRAPPERS:
        return f"`{program}` runs another command"

    for token in tokens[1:]:
        name = token.split("=", 1)[0] if token.startswith("--") else token
        if name in _ESCALATING_OPTIONS:
            return f"`{program}` option `{name}` can execute a different command"
    return ""


@dataclass(frozen=True)
class ShellCommand:
    """A parsed shell command line."""

    raw: str
    program: str
    tokens: tuple[str, ...]
    #: True when the line chains, redirects or substitutes - prefix rules are
    #: ignored for compound commands.
    compound: bool = False
    #: Why the command was classified as compound (for the audit reason).
    reason: str = ""

    @property
    def name(self) -> str:
        """The program without any path, e.g. ``./node_modules/.bin/pytest``."""

        last = self.program.rsplit("/", 1)[-1]
        return last[:-4] if last.endswith(".exe") else last


def _scan_raw(raw: str) -> tuple[bool, str]:
    """Classify a raw command line without trusting any parser."""

    for marker in _SEPARATORS:
        if marker in raw:
            return True, f"contains the shell operator `{marker}`"
    for marker in _REDIRECTIONS:
        if marker in raw:
            return True, f"contains the shell operator `{marker}`"
    for marker in _SUBSTITUTION:
        if marker in raw:
            return True, f"contains the substitution `{marker}`"
    if "$" in raw:
        return True, "contains `$` (variable expansion)"
    return False, ""


def _looks_like_path(value: str) -> bool:
    """Whether an option value names a path rather than a mode or a ref."""

    return value.startswith(("/", "./", "../", "~")) or "/" in value


def parse(command: str) -> ShellCommand:
    """Split ``command`` into a program and its arguments."""

    raw = (command or "").strip()
    if not raw:
        return ShellCommand(raw="", program="", tokens=(), compound=True, reason="empty command")

    compound, reason = _scan_raw(raw)
    try:
        tokens = tuple(shlex.split(raw))
    except ValueError as exc:
        # Unbalanced quotes: force a prompt rather than guess.
        return ShellCommand(
            raw=raw,
            program="",
            tokens=(),
            compound=True,
            reason=f"unparseable command ({exc})",
        )

    if not compound:
        escalation = _escalates(tokens)
        if escalation:
            compound, reason = True, escalation

    return ShellCommand(
        raw=raw,
        program=tokens[0] if tokens else "",
        tokens=tokens,
        compound=compound,
        reason=reason,
    )


def matches_program(command: ShellCommand, wanted: str) -> bool:
    """True when ``command`` invokes exactly the program ``wanted``."""

    if not wanted:
        return False
    target = wanted[:-4] if wanted.endswith(".exe") else wanted
    return command.name == target


def matches_prefix(command: ShellCommand, prefix: str) -> bool:
    """True for a *simple* command whose leading tokens equal ``prefix``.

    Compound commands never match, and neither does a command that only shares a
    string prefix (``npm-debug`` does not match ``npm``).
    """

    if command.compound or not prefix:
        return False
    wanted = prefix.split()
    if not wanted or len(wanted) > len(command.tokens):
        return False
    return list(command.tokens[: len(wanted)]) == wanted


__all__ = ["ShellCommand", "parse", "matches_program", "matches_prefix"]
