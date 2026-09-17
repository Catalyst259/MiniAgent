# Codex CLI 前端交互实现调研与 Mini Coding Agent 设计方案

## 1. 调研目标

本文调研 OpenAI Codex CLI 当前 TUI 前端的主要实现模式，重点关注以下能力：

1. 可持续编辑的命令行输入框
2. Backspace、光标移动、文本追加等编辑行为
3. `/` 斜杠命令列表与匹配机制
4. Popup 与输入框之间的键盘事件路由
5. Agent / LLM 的流式输出
6. Tool / Shell 命令的流式执行输出
7. Terminal 历史区域与动态区域的协调
8. 对 Mini Coding Agent CLI 的可复用架构

本文不试图完整复刻 Codex TUI，而是提取适合 Mini Coding Agent 的最小实现模式。

---

# 2. Codex CLI 总体技术模式

Codex CLI 当前前端实现基于：

* Rust
* Ratatui
* Crossterm
* Tokio
* 自定义 TextArea
* 自定义 Streaming Controller
* 自定义 Command Popup
* 事件驱动 UI

整体可以抽象为：

```text
                    Keyboard / Paste / Resize
                              │
                              ▼
                         Crossterm
                              │
                              ▼
                         BottomPane
                              │
                   ┌──────────┴──────────┐
                   │                     │
             Active Popup          ChatComposer
                                         │
                                         ▼
                                      TextArea
                                  text + cursor
                                         │
                                      changed
                                         │
                                         ▼
                                    sync_popups()
                                         │
                              ┌──────────┴──────────┐
                              │                     │
                       Slash Command          Mention / File
                           Popup                  Popup


Agent / Core
     │
     │ semantic events
     ▼
Streaming Controller / History Cells
     │
     ├── stable content
     │
     └── mutable tail
             │
             ▼
          Renderer
             │
             ▼
          Terminal
```

核心原则是：

> UI 不直接执行 Agent、Tool 或 LLM，而是消费状态和事件。

因此整个系统实际上可以理解成：

```text
Event
  ↓
State Mutation
  ↓
Render
```

---

# 3. 可编辑输入框：TextArea

Codex 并没有使用 shell 自带的 `readline` 输入，也不是简单调用：

```python
input("> ")
```

而是自己维护一个真正的文本编辑器状态。

核心结构类似：

```rust
struct TextArea {
    text: String,
    cursor_pos: usize,
    wrap_cache: ...,
    preferred_col: ...,
    elements: ...,
}
```

其中：

```text
text
```

保存用户当前输入内容；

```text
cursor_pos
```

保存逻辑光标位置；

```text
wrap_cache
```

负责终端宽度变化和软换行；

```text
elements
```

用于表示不可随意拆分的特殊元素，例如某些附件、mention 或 placeholder。Codex 当前 TextArea 还实现了 Vim 编辑模式、kill buffer、Unicode 边界、自动换行等更复杂的功能。

因此界面上的：

```text
> please fix thsi bug
                  ^
```

实际上对应：

```text
text = "please fix thsi bug"
cursor = 16
```

终端上的文字只是这个状态的渲染结果。

---

# 4. 字符输入实现

用户输入：

```text
a
```

并不是直接向 stdout 写入 `a`。

事件流程为：

```text
Keyboard
   ↓
KeyEvent(Char('a'))
   ↓
ChatComposer
   ↓
TextArea
   ↓
insert_str("a")
   ↓
修改 text
   ↓
修改 cursor
   ↓
invalidate wrap cache
   ↓
request redraw
```

逻辑可以抽象成：

```python
def insert(text):
    buffer = buffer[:cursor] + text + buffer[cursor:]
    cursor += len(text)
    invalidate_layout()
```

例如：

```text
hello wrld
       ^
```

输入：

```text
o
```

得到：

```text
hello world
        ^
```

并不是修改已经输出的 terminal 字符，而是修改内存中的字符串。

---

# 5. Backspace 实现

Backspace 同样不是：

```python
print("\b \b")
```

这种传统终端技巧。

Codex 会把 Backspace 转换成编辑操作：

```text
KeyCode::Backspace
      ↓
delete_backward
      ↓
修改 TextArea.text
      ↓
修改 cursor_pos
      ↓
重新渲染
```

TextArea 中存在明确的：

```rust
delete_backward(...)
delete_backward_word(...)
```

等操作。

例如：

```text
text   = "hello worlld"
cursor = 12
```

按一次 Backspace：

```text
text   = "hello world"
cursor = 11
```

随后下一帧重新绘制：

```text
> hello world
             ^
```

因此：

> Backspace 的本质是 editable buffer mutation，而不是 terminal character deletion。

---

# 6. Unicode 与终端显示宽度

不能简单使用：

```python
cursor -= 1
```

作为 Backspace。

原因包括：

```text
ASCII
中文
emoji
combining characters
ZWJ emoji
```

例如：

```text
你
```

UTF-8 占多个 byte，但终端通常显示两个 column。

再例如：

```text
👨‍👩‍👧‍👦
```

从 Unicode codepoint 看由多个字符组成，但视觉上应该作为一个整体处理。

Codex 的 TextArea 因此会：

1. 维护合法 UTF-8 边界
2. 处理 Unicode grapheme
3. 独立计算 display width
4. 根据 terminal width 计算 soft wrap

其单元测试明确覆盖中文、泰文组合字符、分解拉丁字符以及 ZWJ family emoji 的删除行为。

因此终端输入框实际存在两种位置：

```text
logical cursor
```

与：

```text
visual cursor
```

转换关系：

```text
UTF-8 text
    ↓
grapheme / atomic boundary
    ↓
soft wrapping
    ↓
display width
    ↓
terminal x/y
```

---

# 7. ChatComposer：输入框外层状态机

TextArea 只负责：

```text
文本
光标
删除
插入
移动
换行
```

真正负责用户输入逻辑的是：

```text
ChatComposer
```

Codex 源码将其定义为 bottom-pane text input state machine。

它处理：

```text
Enter
Tab
Esc
Up / Down
Paste
/history
/slash commands
@ mentions
$ skills
attachments
shell mode
popup
```

因此结构应该理解为：

```text
ChatComposer
│
├── TextArea
│
├── CommandPopup
│
├── FileSearchPopup
│
├── MentionPopup
│
├── History
│
└── Paste handling
```

而不是：

```text
TextArea 自己负责所有功能
```

---

# 8. 键盘事件路由

Codex 的 ChatComposer 大致执行：

```text
handle_key_event(event)
        │
        ├── popup visible
        │       ↓
        │   popup handler
        │
        └── no popup
                ↓
      handle_key_event_without_popup
                ↓
            TextArea

最后：

sync_popups()
```

Codex 源码中特别指出，在每次键盘处理之后都会同步 popup 状态，使候选列表始终跟随最新 buffer 和 cursor。

因此输入：

```text
/m
```

流程不是：

```text
CommandPopup 接收到 m
```

而是：

```text
KeyEvent('m')
      ↓
TextArea.insert("m")
      ↓
text = "/m"
      ↓
sync_popups()
      ↓
CommandPopup 重新读取 query
      ↓
更新候选
```

这是一个很重要的设计。

---

# 9. Slash Command Popup

例如：

```text
/model
/review
/status
/mcp
/help
```

输入：

```text
/
```

显示：

```text
/model      choose model
/review     review changes
/status     show status
/mcp        list MCP servers
/help       show help

> /
```

继续输入：

```text
/mo
```

则变成：

```text
/model      choose model

> /mo
```

这里 popup 并没有自己的 editable buffer。

真正的数据源始终是：

```text
ChatComposer.TextArea.text
```

CommandPopup 只是一个派生状态：

```text
query = parse(TextArea.text)

matches = filter(commands, query)
```

因此：

```text
/
 ↓
Popup visible

/m
 ↓
filter("m")

Backspace
 ↓
text = "/"
 ↓
filter("")

Esc
 ↓
Popup dismissed

TextArea 内容仍然存在
```

---

# 10. Codex 当前 Slash 匹配机制

需要特别注意：

> Codex 仓库里虽然存在 fuzzy matcher，但当前 Slash Command Popup 本身主要采用 exact + prefix 匹配。

`CommandPopup::filtered()` 当前逻辑明确区分：

```text
exact
prefix
```

然后按照：

```text
exact results
+
prefix results
```

返回。

例如：

```text
/model
/memories
/mention
/mcp
```

输入：

```text
/m
```

得到：

```text
/model
/memories
/mention
/mcp
```

输入：

```text
/mo
```

主要得到：

```text
/model
```

但：

```text
/md
```

不会单纯因为 `m...d` 构成子序列就匹配 `/model`。

最简实现：

```python
def match_commands(query, commands):
    q = query.lower()

    exact = [
        command
        for command in commands
        if command.name.lower() == q
    ]

    prefix = [
        command
        for command in commands
        if command.name.lower().startswith(q)
        and command.name.lower() != q
    ]

    return exact + prefix
```

---

# 11. Codex 的 Fuzzy Matcher

Codex 项目同时存在独立：

```text
codex-rs/utils/fuzzy-match
```

其算法是一个：

```text
case-insensitive subsequence matcher
```

例如：

```text
haystack: hello
needle:   hl
```

匹配：

```text
hello
^ ^
h l
```

返回：

```text
match_indices
score
```

评分思想：

1. 匹配字符越集中越好
2. 从字符串开头匹配有额外奖励
3. score 越小越优

源码中的核心公式近似：

```text
window =
    last_match
    - first_match
    + 1
    - needle_length

score = max(window, 0)

if first_match == 0:
    score -= 100
```

例如：

```text
abc
abc

score = -100
```

而：

```text
a-b-c
abc

score = -98
```

因此前者优先。

Codex Slash 相关辅助函数也会使用 fuzzy matcher 判断某个输入是否仍可能匹配到命令。

---

# 12. Mini Coding Agent 建议直接采用 Fuzzy Slash Command

虽然 Codex 当前 popup 主要使用 prefix，但 Mini Coding Agent 可以直接升级到：

```text
subsequence fuzzy matching
```

例如命令：

```text
/read
/review
/resume
```

输入：

```text
/rv
```

可以匹配：

```text
/review
 ^   ^
```

建议匹配结果：

```python
@dataclass
class CommandMatch:
    command: SlashCommand
    matched_indices: list[int]
    score: int
    original_order: int
```

排序：

```python
matches.sort(
    key=lambda item: (
        item.score,
        item.original_order,
    )
)
```

渲染时利用：

```text
matched_indices
```

高亮：

```text
/review
 ^   ^
```

第一版完全可以实现 Codex 同款简单算法，不需要引入第三方 fuzzy-search 框架。

---

# 13. Slash Command Registry

不要把命令逻辑直接写进 Popup。

应该存在独立 registry：

```python
@dataclass
class SlashCommand:
    name: str
    description: str
    handler: Callable
    supports_args: bool = False
```

例如：

```python
COMMANDS = [
    SlashCommand(
        name="model",
        description="Switch model",
        handler=handle_model,
    ),
    SlashCommand(
        name="tools",
        description="List available tools",
        handler=handle_tools,
    ),
    SlashCommand(
        name="skills",
        description="List available skills",
        handler=handle_skills,
    ),
    SlashCommand(
        name="agents",
        description="List available subagents",
        handler=handle_agents,
    ),
]
```

Popup 只负责：

```text
显示
过滤
选择
滚动
```

Dispatcher 负责：

```text
执行
```

---

# 14. Popup 状态

建议：

```python
@dataclass
class CommandPopupState:
    visible: bool = False

    query: str = ""

    matches: list[CommandMatch] = field(
        default_factory=list
    )

    selected_index: int = 0

    scroll_offset: int = 0
```

当 query 改变：

```text
重新过滤
↓
selected_index reset / clamp
↓
确保 selected item 可见
```

Codex 当前实现同样会在 filter 改变后重置或约束 selection 与 scroll state。

---

# 15. Popup 键盘处理

当 Slash Popup 可见：

```text
Up
    ↓
selected_index - 1

Down
    ↓
selected_index + 1

Tab
    ↓
apply completion

Enter
    ↓
执行 selected command

Esc
    ↓
dismiss popup

普通字符
Backspace
Left / Right
    ↓
仍然交给 TextArea
    ↓
重新 sync popup
```

因此 popup 并不是 modal input。

它只拥有：

```text
selection/navigation
```

TextArea 始终拥有：

```text
文本编辑
```

---

# 16. Shell Command 模式

Codex 支持：

```text
!git status
```

这种输入。

提交时如果：

```text
text.startswith("!")
```

则不会送给模型，而是转换成：

```text
run_user_shell_command
```

进入 core。Codex 当前 ChatWidget 中明确存在这一特殊分支。

因此：

```text
!pytest
```

流程：

```text
Composer
   ↓
Submit
   ↓
ShellCommand intent
   ↓
Core
   ↓
subprocess
   ↓
Exec events
   ↓
UI
```

UI 本身不应该：

```python
subprocess.run(...)
```

UI 只产生 intent。

---

# 17. Agent Streaming

Codex 的 Agent 输出不是：

```python
async for token in model:
    print(token, end="")
```

而是：

```text
Streaming Controller
```

当前实现明确采用：

```text
two-region streaming
```

即：

```text
stable region
+
mutable tail
```

源码中将已经稳定的内容提交到 scrollback，而将仍可能变化的内容保留在 active-cell tail 中。

可以表示为：

```text
Agent Response

Hello, this function does three things:

1. parses input
2. validates parameters
──────────────────────────────────
          stable region

3. execu...
──────────────────────────────────
          mutable tail
```

---

# 18. 为什么不能简单 token render

模型可能流式返回：

```text
**
hel
lo
**
```

完整内容才是：

```markdown
**hello**
```

如果：

```python
render_markdown(delta)
```

会导致每个 delta 的 Markdown 语义错误。

代码块更加明显：

````text
```py
print(
"x")
````

````

在 fence 没闭合之前，当前 Markdown AST 始终是不完整的。

表格：

```markdown
| name | value |
|---|---|
| abc
````

同样存在结构不确定性。

因此必须保存：

```text
raw source
```

而不是保存：

```text
rendered fragments
```

---

# 19. Streaming Collector

Codex 流程可以抽象为：

```text
AssistantDelta
     ↓
Raw Markdown Buffer
     ↓
Streaming Collector
     │
     ├── committed source
     │
     └── pending source
              │
              ▼
          Markdown Render
              │
         ┌────┴────┐
         │         │
      stable      tail
         │         │
         ▼         ▼
    scrollback   active cell
```

当前实现只在遇到完整 source boundary 时将内容加入稳定区域，其中 newline 是非常重要的 commit 边界。

例如收到：

```text
delta 1:
"Hello, I will"

delta 2:
" fix this.\nThe"

delta 3:
" problem is"

delta 4:
" ..."
```

此时：

```text
raw:

Hello, I will fix this.
The problem is ...
```

可以分成：

```text
stable:

Hello, I will fix this.
```

与：

```text
tail:

The problem is ...
```

---

# 20. Stable Region 与 Mutable Tail

这种设计的最大作用是减少闪烁和重复重绘。

已经确定的：

```text
stable
```

可以直接进入 terminal scrollback。

仍在变化的：

```text
tail
```

则由 TUI 在 active viewport 中持续重画。

因此：

```text
terminal history
```

不需要每来一个 token 就重新绘制。

最终形成：

```text
┌──────────────────────────────┐
│ committed previous messages  │
│ committed tool results       │
│ stable assistant lines       │
│                              │
├──────────────────────────────┤
│ current mutable assistant    │
│ tail...                      │
│                              │
│ > composer                   │
└──────────────────────────────┘
```

---

# 21. Markdown Table Holdback

Codex 当前甚至专门处理 streaming Markdown table。

原因：

```markdown
| name | value |
|---|---|
| foo | 1 |
```

后续新增：

```markdown
| extremely-long-value | 2 |
```

可能改变整个 table 的 column width。

因此：

```text
table header 开始之后
```

不能轻易认为内容稳定。

Codex 会：

```text
普通 prose
    ↓
允许进入 stable

检测到 table
    ↓
table header 之后进入 holdback

table streaming
    ↓
保持 mutable

stream finalize
    ↓
重新完整 render
```

该机制当前由 `table_holdback_state` 等逻辑维护。

Mini Coding Agent 第一版不需要完整复刻，但需要保留：

```text
raw source
+
final canonical render
```

这个设计。

---

# 22. Streaming Finalize

流结束时，Codex 不简单执行：

```text
stable + tail
```

作为最终 transcript。

而是：

```text
完整 raw Markdown
        ↓
render_source(...)
        ↓
final canonical output
```

源码中的 `finalize_remaining()` 明确重新基于完整 source 渲染最终结果。

因此：

```text
Streaming render
```

只是临时 UI。

真正历史记录应该来自：

```text
final render
```

这样可以正确处理：

```text
Markdown
table
fence
resize
wrap
```

---

# 23. Tool Streaming

Tool 输出和 Assistant Markdown Streaming 应该是两条不同管线。

例如：

```text
● Running tests
  └ python -m pytest

    test_parser PASSED
    test_agent PASSED
    test_tool ...
```

其事件应该是：

```text
ToolStarted
     ↓
ToolCell(status=RUNNING)

ToolOutput(delta)
     ↓
ToolCell.append(delta)

ToolOutput(delta)
     ↓
ToolCell.append(delta)

ToolFinished
     ↓
ToolCell(status=DONE)

commit
```

因此不要设计：

```python
stream(text)
```

一个 API 同时处理 Agent 和 Tool。

建议分别：

```text
Assistant Stream
    → Markdown semantics

Tool Stream
    → Raw / structured log semantics
```

---

# 24. Agent 与 UI 的事件边界

Mini Coding Agent 推荐定义：

```python
class AgentEvent:
    pass
```

具体事件：

```text
TurnStarted

AssistantStarted
AssistantDelta
AssistantFinished

ToolStarted
ToolOutput
ToolFinished
ToolFailed

SkillLoaded

SubAgentStarted
SubAgentOutput
SubAgentFinished

ApprovalRequested

Error

TurnFinished
```

Agent Runtime 只产生：

```python
await event_bus.emit(
    ToolStarted(
        call_id=call_id,
        tool="grep",
        arguments={
            "pattern": "AgentLoop"
        },
    )
)
```

UI 不知道：

```text
MCP
LangGraph
OpenAI API
DeepSeek
Qwen
Subprocess
```

内部怎么实现。

反过来：

```text
Renderer
```

也永远不能直接调用 Tool。

---

# 25. 推荐 UI State

第一版不需要复制 Codex 的复杂度。

可以直接：

```python
@dataclass
class TextAreaState:
    text: str = ""
    cursor: int = 0


@dataclass
class CommandPopupState:
    visible: bool = False
    query: str = ""
    matches: list[CommandMatch] = field(
        default_factory=list
    )
    selected: int = 0
    scroll: int = 0


@dataclass
class StreamState:
    source: str = ""
    committed_offset: int = 0
    stable_lines: list[str] = field(
        default_factory=list
    )
    tail_lines: list[str] = field(
        default_factory=list
    )


@dataclass
class AppState:
    composer: TextAreaState
    command_popup: CommandPopupState

    history_cells: list["HistoryCell"]

    active_cell: "HistoryCell | None"

    assistant_stream: StreamState | None
```

---

# 26. History Cell

不要把 UI 历史保存成：

```python
history: list[str]
```

应该采用：

```text
HistoryCell
```

例如：

```text
UserCell
AssistantCell
ToolCell
SubAgentCell
ErrorCell
InfoCell
```

接口：

```python
class HistoryCell(Protocol):

    def render(
        self,
        width: int,
    ) -> Renderable:
        ...
```

这样：

```text
Agent semantic event
        ↓
Cell State
        ↓
Renderer
```

不会变成：

```python
if event.type == ...
elif event.type == ...
elif event.type == ...
```

全部堆积在 Renderer 中。

---

# 27. Committed Cell + Active Cell

Mini Coding Agent 最值得借鉴 Codex 的 UI 抽象是：

```text
committed_history
+
active_cell
```

例如：

```text
history

UserCell
AssistantCell
ToolCell
AssistantCell
```

当前：

```text
active_cell
    =
RunningToolCell
```

工具结束：

```text
active_cell.status = DONE

history.append(active_cell)

active_cell = None
```

这样运行中的东西可以：

```text
原地更新
```

而不是：

```text
Running grep...
grep output...
grep done
```

刷三行。

---

# 28. Python 技术选型

Mini Coding Agent 主体已经是 Python，因此没必要为了复刻 Codex 而引入 Rust。

推荐：

```text
prompt_toolkit
    │
    ├── Editable Buffer
    ├── Cursor
    ├── Backspace
    ├── Delete
    ├── Multiline
    ├── Key Binding
    ├── Completion
    ├── History
    └── Unicode

Rich
    │
    ├── Markdown
    ├── Syntax
    ├── Styled Text
    ├── Spinner
    └── Tool Output

asyncio
    │
    ├── Agent Loop
    ├── UI events
    └── Queue
```

推荐：

```text
prompt_toolkit + Rich + asyncio
```

而不是第一版直接上 Textual。

---

# 29. 为什么优先 prompt_toolkit

Codex 自己实现 TextArea 是因为：

```text
Rust TUI
+
高度定制
```

而 Python 已经有成熟的 terminal editor。

如果自己手搓：

```text
Backspace
cursor
Unicode
wrapping
multiline
paste
history
Ctrl shortcuts
terminal resize
```

实际上是在重复造 readline / prompt_toolkit。

因此建议：

```text
输入编辑能力
    → prompt_toolkit

应用状态和 Agent UI
    → 自己写
```

而不是：

```text
整个 TUI 都交给框架
```

---

# 30. Mini Coding Agent CLI 模块设计

推荐：

```text
cli/
├── app.py
│
├── events.py
│
├── state.py
│
├── composer/
│   ├── composer.py
│   ├── slash_commands.py
│   ├── fuzzy_match.py
│   └── command_popup.py
│
├── cells/
│   ├── base.py
│   ├── user.py
│   ├── assistant.py
│   ├── tool.py
│   ├── subagent.py
│   └── error.py
│
├── streaming/
│   ├── assistant_stream.py
│   └── tool_stream.py
│
└── render/
    ├── renderer.py
    └── theme.py
```

---

# 31. app.py

职责：

```text
启动 TUI
启动 Agent Loop
监听键盘
监听 AgentEvent
维护 AppState
触发 render
```

不负责：

```text
MCP 实现
Tool 实现
LLM API
SubAgent 实现
```

---

# 32. events.py

定义：

```python
@dataclass
class AssistantDelta:
    text: str


@dataclass
class ToolStarted:
    call_id: str
    tool: str
    arguments: dict


@dataclass
class ToolOutput:
    call_id: str
    text: str


@dataclass
class ToolFinished:
    call_id: str
    result: object
```

这是：

```text
Agent Runtime
```

与：

```text
CLI
```

之间最重要的接口。

---

# 33. composer/

负责：

```text
文本输入
Slash Command
Fuzzy Match
历史输入
Tab completion
```

结构：

```text
Composer
├── editable buffer
├── command registry
├── popup state
└── completion
```

---

# 34. slash_commands.py

建议第一版至少：

```text
/help
/tools
/skills
/agents
/model
/status
/clear
/exit
```

其中：

```text
/tools
```

展示 MCP Tools；

```text
/skills
```

展示 Skill；

```text
/agents
```

展示 SubAgent；

```text
/model
```

查看 / 修改当前 OpenAI-compatible model 配置。

模型层仍保持 vendor-neutral，不绑定具体 Qwen、DeepSeek 或 vLLM。

---

# 35. fuzzy_match.py

直接实现简单 subsequence fuzzy：

```python
def fuzzy_match(
    haystack: str,
    needle: str,
):
    haystack = haystack.lower()
    needle = needle.lower()

    positions = []

    cursor = 0

    for char in needle:
        found = haystack.find(
            char,
            cursor,
        )

        if found == -1:
            return None

        positions.append(found)

        cursor = found + 1

    window = (
        positions[-1]
        - positions[0]
        + 1
        - len(needle)
    )

    score = max(window, 0)

    if positions[0] == 0:
        score -= 100

    return positions, score
```

然后：

```python
matches.sort(
    key=lambda x: (
        x.score,
        x.command.order,
    )
)
```

第一版已经足够。

---

# 36. assistant_stream.py

状态：

```python
@dataclass
class AssistantStream:

    source: str = ""

    committed_source: str = ""

    pending_source: str = ""

    stable_lines: list[str] = field(
        default_factory=list
    )

    tail_lines: list[str] = field(
        default_factory=list
    )
```

收到：

```text
AssistantDelta
```

执行：

```text
source += delta
    ↓
查找稳定边界
    ↓
stable markdown
    ↓
render
    ↓
tail preview
```

第一版可以简单采用：

```text
最后一个 newline
```

作为 stable boundary。

即：

```python
split = source.rfind("\n")

if split >= 0:
    committed = source[:split + 1]
    pending = source[split + 1:]
```

---

# 37. Tool Stream

Tool Stream 不需要 Markdown。

状态：

```python
@dataclass
class ToolCell:

    call_id: str

    tool: str

    arguments: dict

    output: str = ""

    status: ToolStatus = RUNNING
```

收到：

```text
ToolOutput
```

直接：

```python
cell.output += delta
```

然后重新 render active cell。

---

# 38. Frame / Redraw

不要：

```text
一个 token
↓
一次 terminal print
```

应该：

```text
Event
↓
Mutation
↓
request_render()
```

由 UI 控制刷新节奏。

第一版可以不实现 Codex 那种复杂 frame limiter，但至少保持：

```text
Event → State → Render
```

而不是：

```text
Event → print()
```

---

# 39. 第一版完整事件链

## 用户输入

```text
KeyEvent
   ↓
Composer
   ↓
Editable Buffer
   ↓
sync command popup
   ↓
AppState changed
   ↓
Render
```

## 用户提交

```text
Enter
   ↓
Composer.submit()
   ↓
判断：
  ├── /command
  ├── !shell
  └── normal prompt
   ↓
产生 Intent
   ↓
Agent Runtime
```

## LLM

```text
AssistantDelta
   ↓
AssistantStream
   ↓
stable / tail
   ↓
Active AssistantCell
   ↓
Render
```

## Tool

```text
ToolStarted
   ↓
Active ToolCell

ToolOutput
   ↓
append output

ToolFinished
   ↓
commit ToolCell
```

---

# 40. 第一版不要做的东西

Codex 当前已经包含大量高级 TUI 功能，例如：

```text
Vim mode
kill buffer
Ctrl+Y
复杂 paste detection
attachment placeholder
remote images
history search
mention popup
table holdback
Mermaid streaming
terminal scroll-region optimization
animation
alternate screen
多个 overlay
```

Mini Coding Agent 第一版都没有必要。

第一版只实现：

```text
Editable Prompt
Slash Command
Fuzzy Match
Assistant Streaming
Tool Streaming
History Cell
```

已经足够形成完整 CLI Harness。

---

# 41. MVP 功能范围

建议 Phase 1：

```text
[ ] prompt_toolkit 输入框

[ ] Backspace / Delete

[ ] Left / Right

[ ] multiline

[ ] Enter submit

[ ] Ctrl+C interrupt

[ ] /

[ ] Slash command popup

[ ] fuzzy filter

[ ] Up / Down selector

[ ] Tab completion

[ ] AssistantDelta streaming

[ ] ToolStarted

[ ] ToolOutput

[ ] ToolFinished

[ ] committed history

[ ] active cell
```

完成这批之后，CLI 已经具备 Codex / Claude Code 类产品最核心的交互结构。

---

# 42. 最终架构

```text
                       Mini Coding Agent

┌────────────────────────────────────────────────────────┐
│                         CLI                            │
│                                                        │
│   ┌─────────────────┐        ┌─────────────────────┐   │
│   │    Composer     │        │   History Cells     │   │
│   │                 │        │                     │   │
│   │ Editable Buffer │        │ User                │   │
│   │ Slash Commands  │        │ Assistant           │   │
│   │ Fuzzy Matcher   │        │ Tool                │   │
│   │ Popup           │        │ SubAgent            │   │
│   └────────┬────────┘        └──────────┬──────────┘   │
│            │                            │              │
│            └────────────┬───────────────┘              │
│                         ▼                              │
│                     AppState                           │
│                         │                              │
│                         ▼                              │
│                      Renderer                          │
└─────────────────────────┬──────────────────────────────┘
                          │
                          │ UserIntent
                          ▼
┌────────────────────────────────────────────────────────┐
│                    Agent Runtime                       │
│                                                        │
│   Agent Loop                                           │
│      │                                                 │
│      ├── LLM                                           │
│      ├── Tool                                          │
│      ├── Skill                                         │
│      └── SubAgent                                      │
│                                                        │
└─────────────────────────┬──────────────────────────────┘
                          │
                          │ AgentEvent
                          ▼
                       CLI State
```

---

# 43. 核心设计原则

整个 CLI 前端可以归结为五条原则。

### 1. 输入框是状态，不是 stdout

```text
text + cursor
```

才是真正的数据。

Terminal 只是 render target。

### 2. Popup 是输入框的派生状态

Slash Command Popup 不拥有第二套 input。

它读取：

```text
composer.text
```

生成：

```text
matches
```

### 3. Agent 与 UI 只通过事件通信

```text
AgentEvent
```

是唯一边界。

UI 不知道 Tool 如何执行。

Agent 不知道 Terminal 如何渲染。

### 4. Streaming 保存原始 source

Assistant：

```text
raw Markdown
```

Tool：

```text
raw log
```

而不是保存 token 渲染碎片。

### 5. 历史内容与动态内容分离

使用：

```text
committed_history
+
active_cell
```

运行中的内容原地更新；

结束后才进入正式历史。

---

# 44. 最终建议

Mini Coding Agent 第一版建议采用：

```text
Python

prompt_toolkit
    → 输入框 / 光标 / Backspace / Completion

Rich
    → Markdown / Syntax / Tool UI

asyncio.Queue
    → AgentEvent Bus

自定义：
    SlashCommand Registry
    Fuzzy Matcher
    HistoryCell
    ActiveCell
    StreamingController
```

不要第一版就复刻 Codex 的完整 Ratatui 架构。

真正应该复用的是它的几个核心抽象：

```text
Editable Buffer

Command Popup

Event-driven UI

Committed History + Active Cell

Stable Stream + Mutable Tail
```

这五个抽象建立之后，后续增加：

```text
MCP Tools
Skills
SubAgents
Approval
Model Selection
File Search
Mention
Diff View
```

都只是继续扩展 UI state 和 AgentEvent，而不需要重新修改整个 CLI 架构。
