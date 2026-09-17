# Coding Agent Terminal Output Research

## Findings

主流 coding agent 的默认终端视图通常展示：工具名称、关键参数摘要、运行状态、成功/失败、耗时、短结果摘要，以及用户需要确认的权限信息。完整参数、大段原始日志、内部事件 ID、模型协议字段和内部 reasoning 通常不放在默认 transcript 中。

工具输出通常采用两层视图：默认历史视图只显示摘要或有限行数；用户通过 transcript viewer、快捷键或展开动作查看详细输出。Claude Code 官方文档明确提供 transcript viewer、`Ctrl+O` 展开详细执行信息，并对 Bash 输出设置模型读取上限和截断策略。Aider 官方文档和源码将工具输出、错误、警告和 assistant 输出分成不同显示路径，但没有声明固定默认折叠行数。Codex CLI 官方资料确认命令、diff、搜索和审查结果是可观察工作流，但没有公开稳定的固定截断阈值。

错误应保留类型、命令、退出码和首段诊断，并明确提示输出已截断；完整 stderr/traceback 通过展开或文件查看。diff 更适合显示文件级摘要和增删统计，完整 diff 使用独立查看器。

## Primary Sources

- Claude Code permissions: https://code.claude.com/docs/en/permissions
- Claude Code interactive controls and transcript viewer: https://code.claude.com/docs/en/interactive-mode#general-controls
- Claude Code output limits: https://code.claude.com/docs/en/tools-reference#output-limits
- Claude Code diff panel: https://code.claude.com/docs/en/interactive-mode#diff-panel
- Claude Code output styles: https://code.claude.com/docs/en/output-styles#built-in-output-styles
- Aider configuration options: https://aider.chat/docs/config/options.html
- Aider usage and diffs: https://aider.chat/docs/usage.html#making-changes
- Aider output implementation: https://raw.githubusercontent.com/Aider-AI/aider/main/aider/io.py
- Aider command execution: https://raw.githubusercontent.com/Aider-AI/aider/main/aider/run_cmd.py
- Codex CLI: https://learn.chatgpt.com/docs/codex/cli
- prompt-toolkit `FormattedTextControl`: https://python-prompt-toolkit.readthedocs.io/en/master/pages/reference.html#prompt_toolkit.layout.FormattedTextControl
- prompt-toolkit `Window`: https://python-prompt-toolkit.readthedocs.io/en/master/pages/reference.html#prompt_toolkit.layout.Window
- prompt-toolkit `ScrollablePane`: https://python-prompt-toolkit.readthedocs.io/en/master/pages/reference.html#prompt_toolkit.layout.ScrollablePane
- prompt-toolkit `patch_stdout`: https://python-prompt-toolkit.readthedocs.io/en/master/pages/reference.html#prompt_toolkit.patch_stdout.patch_stdout

## Repository Mapping

- `ToolCell` already has `RUNNING`, `DONE`, `FAILED` states and a Rich-side `max_preview_lines=12` preview.
- `TranscriptControl` currently renders completed tool output line by line and does not reuse `ToolCell.preview()`, so the interactive transcript can grow without a display limit.
- Tool output and model context are already separate: runtime/tool limits control data sent through the agent, while terminal rendering should apply an independent preview limit.
- `safe_text()` is the existing centralized ANSI/control-character sanitizer and should remain the display boundary.

## Proposed Plan

1. Define display policy separately from model/runtime limits: preview line count, head/tail behavior, expanded output cap, and argument preview length.
2. Make `TranscriptControl` reuse one preview formatter shared with `ToolCell`, preserving complete output in state while truncating only display.
3. Add explicit tool display state: collapsed by default, expanded tool IDs, selected tool, and an indicator showing hidden line count.
4. Add interaction: a simple expand/collapse key for the selected tool, navigation between tool cells, and a detailed transcript mode for full output.
5. Keep errors more visible than successful output: show status, exit code, first diagnostic lines, and an explicit truncation marker.
6. Add focused tests for short output, long output, multiline Unicode output, errors, expansion, and ensuring model messages remain complete even when the terminal is truncated.
7. Validate with CLI screenshots/PTY smoke tests and the full test suite.
