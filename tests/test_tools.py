"""Tests for the tool layer: path guard, read_file, grep, apply_patch, shell."""

from __future__ import annotations

import pytest

from harness.agent.errors import PatchError, ToolError, ToolPermissionError
from harness.tools.fs_tools import (
    ALL_TOOL_NAMES,
    ToolContext,
    apply_patch,
    call_tool,
    glob,
    grep,
    list_dir,
    read_file,
    shell,
    write_file,
)
from harness.tools.paths import Workspace
from harness.tools.patch import apply_hunks, parse_patch


@pytest.fixture()
def ctx(tmp_path):
    return ToolContext(workspace=Workspace(tmp_path))


@pytest.fixture()
def project(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text(
        "import os\n\n\ndef main():\n    value = 1\n    return value\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text("# demo\n", encoding="utf-8")
    (tmp_path / "src" / "util.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    return tmp_path


# ------------------------------------------------------------------ path guard
def test_path_escape_is_blocked(ctx):
    with pytest.raises(ToolPermissionError):
        read_file(ctx, "../../etc/passwd")
    with pytest.raises(ToolPermissionError):
        write_file(ctx, "/etc/hosts", "nope")


def test_absolute_path_outside_root_blocked(ctx):
    with pytest.raises(ToolPermissionError):
        write_file(ctx, "/tmp/definitely-outside.txt", "x")


# ------------------------------------------------------------------- read_file
def test_read_file_reports_range_and_numbers(ctx, project):
    out = read_file(ctx, "src/app.py", offset=4, limit=2)
    assert "src/app.py" in out
    assert "showing 4-5" in out
    assert "4\tdef main():" in out
    assert "5\t    value = 1" in out
    assert "more lines" in out


def test_read_file_offset_past_end(ctx, project):
    assert "past the end of the file" in read_file(ctx, "src/app.py", offset=500)


def test_read_file_missing(ctx):
    with pytest.raises(FileNotFoundError):
        read_file(ctx, "nope.py")


def test_read_file_directory(ctx, project):
    with pytest.raises(ToolError):
        read_file(ctx, "src")


# ------------------------------------------------------------------ list / glob
def test_list_dir_tree_and_depth(ctx, project):
    out = list_dir(ctx, ".", depth=2)
    assert "src/" in out
    assert "app.py" in out
    assert "README.md" in out


def test_glob_finds_nested(ctx, project):
    out = glob(ctx, "**/*.py")
    assert "src/app.py" in out and "src/util.py" in out
    assert "README.md" not in out


def test_glob_no_match(ctx, project):
    assert "no files match" in glob(ctx, "**/*.rs")


# ------------------------------------------------------------------------ grep
def test_grep_with_context_and_glob(ctx, project):
    out = grep(ctx, "return", glob="*.py", context_lines=1)
    assert "src/app.py" in out and "src/util.py" in out


def test_grep_ignore_case(ctx, project):
    assert "no matches" in grep(ctx, "IMPORT")
    assert "src/app.py" in grep(ctx, "IMPORT", ignore_case=True)


def test_grep_invalid_regex(ctx):
    with pytest.raises(ToolError):
        grep(ctx, "([unclosed")


# ------------------------------------------------------------------ write_file
def test_write_file_new_and_overwrite(ctx):
    assert "created" in write_file(ctx, "new/thing.txt", "hello\n")
    assert "overwrote" in write_file(ctx, "new/thing.txt", "hello\nworld\n")
    assert (ctx.workspace.root / "new" / "thing.txt").read_text() == "hello\nworld\n"


# ----------------------------------------------------------------- apply_patch
def test_apply_patch_codex_envelope(ctx, project):
    out = apply_patch(
        ctx,
        """*** Begin Patch
*** Update File: src/app.py
@@ def main():
     value = 1
-    return value
+    return value + 1
*** End Patch""",
    )
    assert "1/1 files ok" in out
    assert "+1/-1" in out
    assert (project / "src" / "app.py").read_text().endswith("    return value + 1\n")


def test_apply_patch_add_and_delete(ctx, project):
    out = apply_patch(
        ctx,
        """*** Begin Patch
*** Add File: src/brand_new.py
+print("hi")
*** Delete File: src/util.py
*** End Patch""",
    )
    assert "2/2 files ok" in out
    assert (project / "src" / "brand_new.py").read_text() == 'print("hi")\n'
    assert not (project / "src" / "util.py").exists()


def test_apply_patch_unified_diff(ctx, project):
    out = apply_patch(
        ctx,
        """--- a/src/app.py
+++ b/src/app.py
@@ -3,4 +3,4 @@
 
 def main():
-    value = 1
+    value = 2
     return value
""",
    )
    assert "1/1 files ok" in out
    assert "value = 2" in (project / "src" / "app.py").read_text()


def test_apply_patch_search_replace(ctx, project):
    out = apply_patch(
        ctx,
        """*** Update File: src/util.py
<<<<<<< SEARCH
def helper():
    return 1
=======
def helper():
    return 2
>>>>>>> REPLACE""",
    )
    assert "1/1 files ok" in out
    assert "return 2" in (project / "src" / "util.py").read_text()


def test_apply_patch_whitespace_tolerant(ctx, project):
    out = apply_patch(
        ctx,
        """*** Begin Patch
*** Update File: src/util.py
@@
 def   helper():
-    return 1
+    return 3
*** End Patch""",
    )
    assert "1/1 files ok" in out
    assert "fuzzy" in out
    assert "return 3" in (project / "src" / "util.py").read_text()


def test_apply_patch_mismatch_leaves_file_untouched(ctx, project):
    before = (project / "src" / "app.py").read_text()
    out = apply_patch(
        ctx,
        """*** Begin Patch
*** Update File: src/app.py
@@
-nothing like this exists
+replacement
*** End Patch""",
    )
    assert "0/1 files ok" in out
    assert "did not match" in out
    assert (project / "src" / "app.py").read_text() == before


def test_apply_patch_dry_run(ctx, project):
    before = (project / "src" / "app.py").read_text()
    out = apply_patch(
        ctx,
        """*** Begin Patch
*** Update File: src/app.py
@@
-    value = 1
+    value = 9
*** End Patch""",
        dry_run=True,
    )
    assert "dry run" in out
    assert (project / "src" / "app.py").read_text() == before


def test_apply_patch_bad_syntax(ctx, project):
    with pytest.raises(PatchError):
        apply_patch(ctx, "just some text without any patch structure")


def test_apply_patch_add_existing_file_is_an_error(ctx, project):
    out = apply_patch(ctx, "*** Begin Patch\n*** Add File: README.md\n+x\n*** End Patch")
    assert "already exists" in out


def test_parse_patch_split_hunk_keeps_shared_context():
    patches = parse_patch("*** Begin Patch\n*** Update File: a\n@@\n keep\n-drop\n+add\n*** End Patch")
    hunk = patches[0].hunks[0]
    assert hunk.old_lines == ["keep", "drop"]
    assert hunk.new_lines == ["keep", "add"]


def test_apply_hunks_multiple_in_order():
    lines = ["a", "b", "c", "d", "e"]
    patches = parse_patch(
        """*** Begin Patch
*** Update File: x
@@
-a
+A
@@
-e
+E
*** End Patch"""
    )
    new_lines, added, removed, _fuzzy = apply_hunks(lines, patches[0].hunks)
    assert new_lines == ["A", "b", "c", "d", "E"]
    assert (added, removed) == (2, 2)


# ----------------------------------------------------------------------- shell
def test_shell_captures_output_and_exit_code(ctx, tmp_path):
    out = shell(ctx, "echo hello && exit 3")
    assert "exit_code: 3" in out
    assert "hello" in out


def test_shell_cwd_is_workspace(ctx, tmp_path):
    (tmp_path / "marker.txt").write_text("x")
    assert "marker.txt" in shell(ctx, "ls")


def test_shell_timeout(ctx):
    out = shell(ctx, "sleep 5", timeout=1)
    assert "timeout after 1s" in out


def test_shell_blocks_destructive(ctx):
    with pytest.raises(ToolError):
        shell(ctx, "rm -rf /")


def test_shell_allowlist_enforced(tmp_path):
    guarded = ToolContext(workspace=Workspace(tmp_path), command_allowlist=("echo",))
    assert "ok" in shell(guarded, "echo ok")
    with pytest.raises(ToolError):
        shell(guarded, "python -c 'print(1)'")


# ------------------------------------------------------------------ dispatching
def test_call_tool_rejects_unknown_arguments(ctx):
    with pytest.raises(ToolError):
        call_tool("read_file", ctx, {"path": "x", "nonsense": 1})


def test_call_tool_unknown_name(ctx):
    with pytest.raises(ToolError):
        call_tool("nope", ctx, {})


def test_all_required_tools_exist():
    assert set(ALL_TOOL_NAMES) == {
        "list_dir",
        "glob",
        "grep",
        "read_file",
        "write_file",
        "apply_patch",
        "shell",
        "git_diff",
    }
