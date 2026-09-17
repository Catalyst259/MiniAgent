# Tools
| Tool          | 作用                         |    必要性 |
| ------------- | -------------------------- | -----: |
| `list_dir`    | 查看目录内容                     |     必须 |
| `glob`        | 按文件模式搜索，如 `**/*.py`        |     必须 |
| `grep`        | 搜索代码内容                     |     必须 |
| `read_file`   | 读取文件/指定行范围                 |     必须 |
| `write_file`  | 新建或整体写文件                   |     必须 |
| `apply_patch` | 局部修改文件                     | **必须** |
| `shell`       | 执行 `pytest`、Python、git 等命令 |     必须 |
| `git_diff`    | 查看当前修改                     |     推荐 |
其中真正需要特别做好的是 read_file 和 apply_patch

# Skill
| Skill              | 内容                     |
| ------------------ | ---------------------- |
| `repo_exploration` | 如何快速理解陌生代码仓库           |
| `debugging`        | 如何从错误 → 定位 → 修改 → 验证   |
| `testing`          | 如何运行测试、缩小失败范围、回归验证     |
| `code_review`      | 修改完成后如何检查正确性、最小改动、潜在问题 |

例如 debugging/SKILL.md 不需要特别长：
## Debugging

When debugging:

1. Reproduce the failure first.
2. Read the complete error and relevant stack trace.
3. Locate the smallest relevant code region.
4. Form a concrete hypothesis before editing.
5. Make the smallest reasonable change.
6. Re-run the narrowest relevant test.
7. Run the full test suite before completion.

# SubAgent
| SubAgent   | 职责                    |
| ---------- | --------------------- |
| `Planner`  | 复杂任务拆解、制定修改计划，不改代码    |
| `Explorer` | 独立阅读仓库、查找相关代码、返回结构化结论 |

Planner

输入：

Task
+
必要的 Repository Context

输出：

Goal
Steps
Affected Areas
Risks
Verification Plan

不需要 Tool 或只给只读 Tool。

Explorer

给：

glob
grep
read_file

不给：

write
apply_patch
shell destructive command

负责类似：

找出用户认证相关代码在哪里，并说明调用链。
输出：

Relevant files
Important symbols
Call relationships
Findings

然后把简短结果还给 Main Agent。

# Slash Command
| Command    | 作用                        |
| ---------- | ------------------------- |
| `/help`    | 显示命令帮助                    |
| `/status`  | 当前任务、轮数、token、已加载 skill 等 |
| `/model`   | 查看/切换模型                   |
| `/tools`   | 查看当前可用 Tools              |
| `/skills`  | 查看当前可用 / 已加载 Skill        |
| `/agents`  | 查看可用 SubAgent             |
| `/compact` | 手动触发 Context Compact      |
| `/clear`   | 清空当前会话/新建 thread          |
| `/exit`    | 退出 CLI                    |
