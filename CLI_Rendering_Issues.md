# MiniAgent 终端渲染问题清单（v1 审计）

> **修复状态（本轮完成后）**
>
> | 编号 | 状态 | 修复位置 / 说明 |
> |---|---|---|
> | P0-1 滚轮无效 | ✅ 已修 | `harness/cli/mouse.py` 解码 SGR/X10 滚轮；`composer.build_key_bindings` 在应用层拦截 `Vt100MouseEvent`（绕过 prompt_toolkit 对 CPR 的依赖），非滚轮事件原样转交默认处理；`app.py` 打开 `mouse_support=True` |
> | P0-2 伪光标视口 | ✅ 已修 | 新增 `transcript.TranscriptPane`（包装 `ScrollablePane`）：滚轮 3 行/格、PageUp/PageDown 按窗口高度整页、Ctrl+Home/Ctrl+End 到顶/到底、到底跟随新内容、上翻后固定位置；`state.transcript_scroll` 已删除 |
> | P0-3 运行中 cell 不可见 | ✅ 已修 | transcript 现在渲染 `active_cell`；工具头显示参数与耗时；运行中显示输出尾部预览 |
> | P0-4 展开被卡在 8 行 | ✅ 已修 | `ToolCell.preview_body(limit)` 语义修正（显式上限=真展开，上限 400 行保护渲染器）；行按显示宽度裁剪，一行只占一行 |
> | P0-5 固定 120 列 / 每帧全量重渲染 | ✅ 已修 | Markdown 按 pane 实际宽度渲染；每个 cell 的片段按 `(内容, 宽度)` 缓存；`refresh_interval=None`（空闲不再 10 Hz 重绘） |
> | P0-6 思维链不显示 | ✅ 已修 | `nodes.py` 事件携带 `reasoning`，`events_bridge` 透传，transcript 渲染 `∴ thinking` 折叠块（Ctrl+O 展开） |
> | P0-7 链接变乱码 / 尖括号被吞 | ✅ 已修 | `_render_markdown` 剥离 OSC 序列；`_escape_angle_brackets` 在代码段之外转义 `<`/`>` |
> | P1-1 两套渲染器不一致 + markup 崩溃 | ✅ 已修 | cells 的 rich 渲染全部改用 `Text`（不再把文本插值进 markup），`AssistantCell` 也过 `safe_text` |
> | P1-2 stable/tail 只在非交互实现 | ✅ 已修 | 交互路径：完整行渲染 Markdown、变化中的尾行按纯文本；批处理：预览行在最终渲染前被擦除 |
> | P1-3 命令输出在 UI 之上 | ✅ 已修 | 新增 `MiniAgentApp.emit_line/notify`，所有斜杠命令输出进入 transcript |
> | P1-4 无状态栏/无按键提示 | ✅ 已修 | 底部状态行（模型 · 状态 · 上翻行数/按键提示）；README 记录全部快捷键 |
> | P1-5 End 键冲突 | ✅ 已修 | transcript 改用 Ctrl+Home/Ctrl+End，`End` 归还输入框 |
> | P1-6 批处理答案渲染两次 | ✅ 已修 | `Renderer._erase_streamed()`：预览行在渲染最终答案前被擦除 |
> | P2-1/3/4/5/6/7/8/10/11/12/13 | ✅ 已修 | 行数统计、提交回到底部、运行中提交有提示、Ctrl+C 不再 AttributeError、ToolOutput 不再空转、预览裁剪、fatal/skill 文案、空闲重绘、Markdown 主题、多行输入续行缩进、运行中用 `◐`（形状而非仅颜色区分） |
> | P2-9 搜索/导出/独立详情视图 | ⏳ 未做 | 属于新功能（交互式搜索、专用 diff/pager 视图、导出），本轮只把"完整输出"通过 Ctrl+O 展开补齐 |
>
> 验证：`pytest -q` → **248 passed**；无头抓包复验见 §6 附录（滚轮生效、首次 PageUp 翻整页、出现鼠标模式序列、空闲 0 字节、CoT/运行中工具/命令输出均在 transcript 内）。

审计对象：`harness/cli/`（交互式 TUI 路径 = prompt_toolkit `Application`）
审计方式：源码走查 + 可复现的量化实验（下称 E1–E5）+ PTY 抓包实测（见 §6）
环境：Python 3.12 / prompt_toolkit 3.0.53 / rich 15.0.0 / 终端 80×24 与 60/100 列

---

## 0. 结论摘要

用户最直观的两个投诉（**滚轮无法上翻回看**、**看不到历史思维链**）不是孤立 bug，而是同一个架构选择的后果：

1. **transcript 是"应用内的一个 Window"，不是终端 scrollback。** 历史内容只存在于内存的 `history_cells` + `TranscriptControl` 里，视口靠"伪造光标位置"（`get_cursor_position`）骗 prompt_toolkit 滚动。于是：滚轮没有接线、翻页步长与可视高度无关、换行后行数算不准、`/clear` 后越界、没有"已上翻"提示。
2. **交互路径与非交互路径是两套渲染器**（rich markup 字符串 vs prompt_toolkit fragments），能力与净化口径都不对齐。README 里描述的工具参数、耗时、运行中宣告、失败区别，在交互界面里大多**不存在**；反过来 rich 路径会把方括号文本当 markup 吃掉甚至抛异常。

按严重程度排序，**P0 共 7 条**（其中 4 条直接决定"能不能顺畅回看历史"），P1 共 6 条，P2 若干净化细节若干。

**症状索引（先看这里）**

| 你看到的现象 | 对应问题 |
|---|---|
| 滚轮上滚没反应 / 滚上去又被弹回底部 | P0-1 |
| PageUp 跳得莫名其妙、不知道自己在哪、`/clear` 后视图空白 | P0-2 |
| 长工具只有底部一行 `· running shell`，看不到参数/耗时 | P0-3 |
| `Ctrl+O` 展开后还是"… N more line(s)"；一条长输出占满整屏 | P0-4 |
| 代码块/表格在窄终端里错位、宽终端右侧大片空白、越用越卡 | P0-5 |
| 从来没有思维链（reasoning）显示 | P0-6 |
| Markdown 链接显示成 `8;id=…;url…8;;`；`<value>`/`Vec<T>` 凭空消失 | P0-7 |
| 同一份内容在 `--plain`/批处理/交互下长得不一样；`[main]`、`[dim]` 被吃掉、偶尔报 MarkupError | P1-1 |
| 流式时半截 Markdown 闪烁、代码块突然吞掉后文 | P1-2 |
| `/help`、`/status` 的输出滚不动、`/clear` 也清不掉 | P1-3 |
| 界面上没有状态栏和快捷键提示 | P1-4 |
| 输入框里按 End 不跳到行尾 | P1-5 |
| `./miniagent.sh "任务"` 的答案被打印两遍 | P1-6 |

---

## 1. P0：直接破坏核心体验

### P0-1 鼠标滚轮完全没有接线（且单开 `mouse_support` 也修不好）

**现象**：滚轮上滚看不到先前内容；任何按键/新输出都会把视图弹回底部。

**实测（无头抓包，见 §6 附录）**：整场会话里鼠标模式序列（`?1000h/1002h/1003h/1006h/1015h`）出现 **0 次**；注入真实 SGR 滚轮事件 `\x1b[<64;12;6M` 后屏幕**毫无反应**。此外会话结束终端 scrollback 长度为 **0** —— 历史根本不进终端 scrollback，所以"靠终端自己滚"也没有内容可滚。

**证据**
- `harness/cli/app.py:759-766`：`Application(..., full_screen=False, mouse_support=False, refresh_interval=0.1)`。`mouse_support=False` 时应用**从不向终端请求鼠标事件**（不会发 `\x1b[?1000h/1002h/1006h`），滚轮事件根本到不了应用，只能由终端自己滚 scrollback。
- `harness/cli/composer/composer.py:161-300`：`build_key_bindings()` 只绑了 `pageup` / `pagedown` / `end` / `c-o`，**没有 `scroll-up` / `scroll-down`**。
- 即使把 `mouse_support=True` 打开也不够：prompt_toolkit 的 SGR 滚轮事件会派发到指针所在 `Window` 的 `Window._mouse_handler()`（`prompt_toolkit/layout/containers.py:2543-2571`），它只改 `window.vertical_scroll`；而 `containers.py:2429-2436` 在**每一帧**都会按 `UIContent.cursor_position` 重新夹紧 `vertical_scroll`，加上 `refresh_interval=0.1`（10 Hz 重绘）与每次 delta 都 `invalidate()`，滚动位置会立刻被拉回。
- 另注：`Keys.ScrollUp/ScrollDown` 的默认处理器只是喂一个 `Up`/`Down` 键（`prompt_toolkit/key_binding/bindings/mouse.py:286-300`），而本应用只在弹窗可见时绑定 `up`/`down`，所以"靠默认绑定"也不会滚动 transcript。

**影响**：唯一可靠的历史回看手段是 `PageUp`/`End` 两个隐藏快捷键（README 未记载），核心诉求（回看 CoT / 工具输出）无法用最自然的交互完成。

**修复方向**：`mouse_support=True` + 在 `TranscriptControl` 上实现 `mouse_handler`（滚轮事件由"指针所在 Window"派发，光绑 `Keys.ScrollUp` 收不到 xterm 的 SGR 滚轮），把偏移写回同一套 `transcript_scroll` 状态；滚动必须由应用状态驱动，不能依赖 `Window.vertical_scroll`。开启鼠标后要提供"Shift+拖拽=原生选择复制"的说明。

```python
# app.py: Application(..., mouse_support=True)
# transcript.py: TranscriptControl 内
def mouse_handler(self, mouse_event):
    from prompt_toolkit.mouse_events import MouseEventType
    if mouse_event.event_type == MouseEventType.SCROLL_UP:
        self.on_scroll(-3); return None          # None = 已处理，需要重绘
    if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
        self.on_scroll(+3); return None
    return NotImplemented                        # 其它事件交回 Window
```

---

### P0-2 滚动模型是"伪光标"，没有真正的视口概念

**证据**
- `harness/cli/render/transcript.py:42-47`：`_cursor_position()` 返回 `Point(x=0, y=scroll)`，即**把光标当成滚动条**。
- `transcript.py:49-51`：`line_count` 只统计片段里的 `"\n"` 数 → 是**逻辑行**；而窗口是 `Window(transcript, wrap_lines=True)`（`app.py:734`），**视觉行数 ≠ 逻辑行数**。一个 200 列宽的段落算 1 行，实际占 3 屏行。
- `app.py:714-722`：`scroll_history()` 每次 `±8` **逻辑行**（不是一页、也不是一行），初始位置 `line_count - 1`；只有下界 `max(0, ...)`，**没有上界夹紧**，也不感知窗口高度。
- 实测 **E5**（`TranscriptControl.create_content(width=40)`）：一段长文本 `line_count=2`（逻辑行），40 列下 `get_height_for_line` 合计 **4 个可视行**（比例 2×）⇒ "每次 ±8 行"实际跳动 **≈16 个可视行**，在 24 行终端里相当于 2/3 屏。滚动步长与视口高度完全脱钩。
- 无"当前已上翻 / 按 End 回到底部"提示；流式输出期间停留在旧位置，用户不知道有新内容。
- 实测（无头抓包，见 §6 附录）：**第一次按 `PageUp` 屏幕完全不变**（`transcript_scroll` 从 `None` 变 21，但该行仍在可视窗口内），第 2~4 次才逐次移动；也就是说"翻页"手感是"按了没反应 → 突然跳一大段"。
- `app.py:267-278`（`/clear`）只清 `history_cells` / `active_cell` / `renderer` 计数，**不清 `state.transcript_scroll` 与 `state.expanded_tool_ids`**（字段定义见 `state.py:198-200`）→ 清空后滚动索引可能越界。
- 没有 Home/顶部跳转、没有 transcript 内搜索。

**修复方向**：把 transcript 改成"按视觉行计数的滚动视图"：要么换 `ScrollablePane`，要么自己维护 `offset_from_bottom`（相对底部而非绝对行号，天然抗新增内容），并在标题/状态行给出 `↑ 已上翻 N 行 · End 回到底部`；`/clear`、提交新输入时重置滚动。

---

### P0-3 运行中的工具 / 子代理在交互界面完全不显示

**现象**：长工具（几十秒的 shell/pytest）在界面上只有底部一行小字 `· running shell`，看不到"哪个工具、什么参数、什么时候开始"。

**证据**
- `harness/cli/state.py:220-231`：`set_active()` 只写 `self.active_cell`，**不 append 到 `history_cells`**；`commit()`（`state.py:209-218`）才在完成时入列。
- `transcript.py:53-62`：`_get_fragments()` **只遍历 `state.history_cells`**。
- 全仓 grep：`active_cell` 在 CLI 里**只被写入、从不被读取**（`state.py:201/212-230`，`app.py:275`）→ 运行中的 cell 没有任何渲染路径。
- 推论：`transcript.py:82-89` 里 `ToolStatus.RUNNING` 的分支是**死代码**（未完成的 cell 永远不在 `history_cells` 里）。
- 非交互路径反而是对的：`app.py:387` 在 `ToolStarted` 时立刻 `render_cell(cell)`。

**连带问题（交互路径信息量比 `--plain` 还少）**
- `transcript.py:82-83`：工具行只有 `● tool …`，**参数不显示**（`ToolCell.arguments` 没被用），**耗时也不显示**（`duration_ms` 没被用）；而 rich 路径两者都有（`cells/base.py:185-188`）。
- `README.md`（CLI 章 **Tool trace** 行）写着 "Every call is announced as it starts (`● read_file  {"path": …} …`)"——交互模式下该承诺不成立。
- 实测（无头抓包，`visible_keys_probe.txt`）：交互屏上的工具行就是光秃秃的 `● list_dir`，**没有参数、没有耗时**（同一份输出在 rich 路径里两者都有）。

**修复方向**：把 `active_cell` 也纳入片段生成（未完成 cell 渲染头部 + `…`），并把参数摘要、耗时、起止状态补齐到同一套 formatter。

---

### P0-4 `Ctrl+O`「展开」实际上只从 3 行扩到 8 行；预览限制按"行"而不是按"屏"

**证据**
- `transcript.py:85`：`cell.preview_body(None if expanded else 3)`；`cells/base.py:140-147`：`max_lines=None` 时回落到 `self.max_preview_lines`，默认 **8**（`base.py:124`）。所以"展开"只是 3 → 8 行，仍会显示 `… N more line(s)`。
- 实测 **E2**（`ToolCell.output` = 一条 5000 字符长行 + 40 行）：
  - 折叠：`3 行 / 隐藏 38 行 / 最长行宽 5000`
  - "展开"：`8 行 / 仍隐藏 33 行`
  → 一条长行算 1 行，"3 行预览"可能直接占满整屏；隐藏计数也是错的（按逻辑行统计）。
- 只能切换**最新的一个**工具（`state.py:241-253` 从后往前找第一个完成的 `ToolCell`），无法选中具体 cell；也没有可见的展开提示/折叠标记。
- 现有测试把缺陷固化了：`tests/test_cli_render.py:203-219` 断言展开后 `line 8` 不出现、`4 more line(s)` 存在——即把"只展开到 8 行"当成正确行为。

**修复方向**：`expanded` 时用真正的"全量"上限（或 `max_preview_lines=None` 表示不限）；预览上限改为**显示行/字符双阈值**（先按显示宽度折行再截断）；给展开态一个可见标记（如 `▾ 全量 / ▸ 折叠`）与 cell 级选择。

---

### P0-5 Markdown 固定按 120 列排版，且每帧全量重渲染

**证据**
- `transcript.py:106-119`：每次都新建 `Console(file=io.StringIO(), width=120, force_terminal=True, color_system="standard")`。
  - **宽度硬编码 120**，与真实终端宽度无关（实测 **E1b**：`TERM=xterm-256color` 时长段落按 120 列排版、最长行 117 列；同一段落在 60/80 列终端里也是 120 列）。
  - 附带的环境陷阱：当 `TERM` 未设置或为 `dumb` 时，rich 的 `Console.size` 走 `is_dumb_terminal` 分支，**直接忽略 `width=120` 返回 80×25** —— 也就是"渲染宽度"取决于 `TERM` 与硬编码常量的组合，而不是窗口的真实宽度（本仓库的沙箱里 `TERM=dumb`，因此本地测到的是 80 列）。
  - 端到端抓包（`.probe/`，窗口 **60 列**，`TERM=dumb` ⇒ 排版宽度 80）：答案段落被"词中硬折行"——`...so it inspect` / `ed the workspace but`、`./m` / `iniagent.sh ""`，可见行宽正好顶到 60。这是排版宽度 ≠ 窗口宽度的直接后果；100→60 resize 也不会按 60 重新排版（只会被窗口再折一次）。
  - 窄终端：rich 先按 120 列排好 → prompt_toolkit 再按窗口宽度折一次 → 代码块/表格**二次折行、边框错位**；宽终端：右侧大片留白；由窄变宽时（80→200）不会重新按新宽度排版。
  - `color_system="standard"` 把 Markdown 高亮降级为 16 色（即使终端支持 truecolor）。
- 性能：`TranscriptControl` 每帧为**每个** assistant cell 重跑 rich Markdown（`_fragment_cache` 只按 `render_counter` 缓存一帧）。而 `refresh_interval=0.1`（10 Hz）+ 每个 delta 都 `_invalidate_ui()`（`app.py:359-366`、`app.py:481`）⇒ 每来一个 token 就把**整段历史**重新排版一次，成本随会话长度线性增长。

**修复方向**：把宽度接到 `get_app().output.get_size()`（或让 rich 自己探测真实终端宽度，去掉 `force_terminal=True`/`width=120` 的组合）；对**已完成**的 cell 缓存渲染结果（只失效正在流式的那一个）；`refresh_interval` 只在真需要动画时开。

---

### P0-6 交互模式下思维链根本没有被渲染（所以"回看 CoT"目前无内容可看）

**证据**
- `events_bridge.py:83-88`：`assistant_message` → `ui.AssistantFinished(..., reasoning=None)`，**硬编码丢弃**；即使事件 `data` 里带了 reasoning 也丢（实测 **E4**：`translate()` 返回的 `reasoning=None`）。
- `events_bridge.py:29`：`assistant_reasoning` 映射为 `None`，即 `nodes.py:192-196` 发出的 `assistant_reasoning` 事件在 UI 侧被整条丢弃。
- `streaming.py:62-66`：reasoning 被单独收集，但 `on_delta` 只回调 `text`（同文件 67-69）→ **CoT 从不流式显示**。
- 于是 `transcript.py:71-72` 的 `cell.reasoning` 分支在交互路径是**死代码**（`AssistantCell.reasoning` 永远是 `None`）。
- 唯一会打印 CoT 的代码是 `render/legacy.py` 的 `PlainRenderer.format()`/`RichRenderer`（`(thinking) …`），但全仓 grep 显示这两个类**只被 `render/__init__.py` 重新导出、从未被实例化**（无人调用）⇒ 当前产品里 CoT 在任何模式下都不显示。

**修复方向**：bridge 透传 `reasoning`（并考虑 `assistant_reasoning` 的流式增量事件），transcript 给 CoT 一个可折叠的独立块（默认折叠/默认展开可配置），否则"滚动回看思维链"这个需求即使修好滚动也看不到东西。

---

### P0-7 Markdown 过 rich→ANSI→fragments 桥接时内容被污染/丢失（链接变乱码、尖括号文本被吞）

**现象**：模型答案里只要出现 Markdown 链接，交互 transcript 里就会显示成 `see 8;id=9249094;https://example.comdocs8;; for details` 这种乱码；写成 `<value>`、`<stdout>`、`Vec<T>` 的文本会**整段消失**。

**证据**（`transcript.py:106-119`，实测 **E6**）
- rich 在 `force_terminal=True` 下为链接输出 OSC 8 超链接：`\x1b]8;id=…;URL\x1b\\` + 文本 + `\x1b]8;;\x1b\\`。
- `to_formatted_text(ANSI(rendered))` 只解析 SGR（`prompt_toolkit/formatted_text/ansi.py`），OSC 序列的 `ESC ]` 与终止符被丢掉、**载荷被当普通文字保留** ⇒ 链接文字被 URL 与 `8;id=` 夹在中间。
- 实测样例：
  - `see [docs](https://example.com) for details` → `see 8;id=9249094;https://example.comdocs8;; for details`
  - `Use tool --flag <value> to continue.` → `Use tool --flag  to continue.`（`<value>` 丢失）
  - `Write to <stdout> and read <stdin>.` → `Write to  and read .`
  - `The generic Vec<T> is used here.` → `The generic Vec is used here.`
  - 对照：`[docs]`/`<value>` 放在反引号代码段里就正常 ⇒ 是 rich Markdown 的 HTML/autolink 解析所致，不是 `safe_text`。
- 受影响的是**主路径**（交互 transcript 全部走这个函数），而 `--plain`/批处理路径直接把 rich 输出写到终端，超链接与文本本身是正常的。
- 实测（无头抓包）：交互屏的最终答案里出现 `export DEEPSEEK_API_KEY=... # or edit config.yaml ./miniagent.sh ""` —— 模型原文中的 `<your task>` 在主路径被吞掉，且两行被并成一段。

**修复方向**：渲染时关掉超链接（`Console(..., hyperlinks=False)`）或改用自己的 ANSI→fragment 转换并剥离 OSC；尖括号文本需要在渲染前对非代码段做转义/替换（或换掉用 rich Markdown 做终端渲染这条路）。

---

## 2. P1：一致性、正确性与可发现性

### P1-1 两套渲染器：同一份内容在 `--plain` / 批处理 / 交互下长得不一样，且 rich 路径会吃字甚至崩

**证据**
- 只存在于 rich 路径、交互路径缺失：工具参数与耗时、子代理摘要/迭代数/失败图标（`cells/base.py:214-229` vs `transcript.py:91-92`）、`ErrorCell.fatal` 的红色加粗区分（`base.py:237-239` vs `transcript.py:96-97`）、`SkillCell` 失败文案（rich: "skill failed"，交互: 永远 "◆ skill loaded: …" —— `transcript.py:93-95`）。
- **rich markup 注入**：rich 路径把工具/错误文本直接插进 markup 字符串（`base.py:200/202/225/227/239/248/258`）。实测 **E3**：
  - `[main] init done` → 渲染成 ` init done`（`[main]` 被当标签吃掉）；`[dim]`、`result: [ok]` 同理 → **工具输出静默丢字**；`[INFO]`、`[Errno 2]` 因不合标签语法而幸存（只有小写开头的方括号词中招）。
  - `error: closing [/] tag`、`bad [/] tag` → **抛 `rich.errors.MarkupError`**（未捕获）。触发场景很现实：`grep` 到 Rich 源码/markdown 文本、pytest 输出、模型给出的代码片段。
  - 修正方式只有 `markup=False` 或 `rich.markup.escape()`；`safe_text()` 只清 ANSI/控制字符，**不清 markup**（`harness/cli/sanitize.py:35-53`）。
- 净化口径不一致：`transcript.py:117` 对 source 走 `safe_text`，而 `base.py:108` 的 `console.print(Markdown(self.source))` **不过** `safe_text`。

**修复方向**：统一为"先 sanitize + escape，再交给 rich"或统一走 fragments；两条路径共用同一个 cell→行 的 formatter（这正是 `terminal_output_research.md` 里"reuse one preview formatter"的意图）。

---

### P1-2 设计文档的 stable/tail 双区流式，只在非交互路径实现了

**证据**
- `renderer.py:128-215` 实现了 `begin_stream / push_delta / _write_committed / _write_tail`（`CLI_Design.md` §19–20 描述的 two-region streaming），但 UI 模式下这些调用全被 `if self._ui_app is None:` 跳过（`app.py:343/354/361/387/407/417/429/445`）。
- 交互路径改为每帧把**半截 Markdown** 整段重渲染（`transcript.py:73-74` → `_render_markdown`）⇒ 未闭合的代码围栏会让后文整段变代码块、未完成表格/加粗会跳动；文档承诺的"减少闪烁和重复重绘"在真实终端下失效。
- 反过来，交互路径**没有** stable/tail 拆分，非交互路径**没有** transcript 历史 → 两套机制各修一半。

**修复方向**：交互路径也按"已提交前缀渲染 Markdown + 尾部一行按纯文本渲染"来做，或把 rich 渲染结果按 stable/tail 缓存。

---

### P1-3 斜杠命令的输出不进 transcript，而是落在 UI 区域上方

**证据**：`/help`、`/status`、`/model`、`/tools`、`/skills`、`/agents` 全部走 `app.renderer.print(...)`（`app.py:170-261`），`/_cancel`、未知命令走 `renderer.info/error`（`app.py:642/833`）。这些写的是 rich `Console`（经 `patch_stdout` 的 `StdoutProxy` 输出到应用区域**上方**），**不进入 `history_cells`**。

**后果**：命令输出无法用 transcript 的滚动/展开看到，`/clear` 也清不掉；界面上存在"两套历史"，回看体验被切成两半。

**修复方向**：命令输出改为生成 `InfoCell`/`ErrorCell` 并 `append_cell`（交回 transcript），只在非 UI 模式保留直接打印。

---

### P1-4 交互模式没有状态栏，也没有任何按键提示

**证据**：`create_application()`（`app.py:669-766`）没有为 `Application` 提供底部工具栏——`_toolbar()`（`app.py:837-843`，显示 `MiniAgent · model · running/idle`）只挂在**旧的 `PromptSession` 路径**上（`app.py:666`）。README 也没有记录 `PageUp` / `End` / `Ctrl+O`。

**后果**：用户不知道能滚动、不知道滚到哪、不知道模型/运行状态；这是"滚轮不好用"体感的一部分。

**修复方向**：在 layout 底部加一行状态/提示条（模型 · 状态 · 已上翻行数 · `Ctrl+O 展开 / End 回到底部`），并把快捷键写进 README。

---

### P1-5 `End` 键与输入框编辑冲突（README 明确承诺 Composer 支持 End）

**证据**
- `composer.py:260-263`：`@kb.add("end")` 全局把 End 绑成"transcript 回到底部"。
- prompt_toolkit 的合并顺序保证应用级绑定优先：`application.py:1497`（`key_bindings.append(self.app._default_bindings)` 后整体 `[::-1]`，注释写明"current control's key bindings … need priority"）+ `key_processor.py:189`（取 `matches[-1]`）。
- 默认 emacs 绑定里 `end` = 移到行尾 → **被覆盖**；`pageup/pagedown` 同理被抢走（弹窗可见时只绑了 `up/down`，翻页仍作用于 transcript）。
- README 的 Composer 行明确写着 "Home/End, word-delete (Ctrl+W) …" → 文档与实现矛盾。

**修复方向**：把 transcript 滚动键换成不冲突的组合（如 `Ctrl+U/Ctrl+D`、`Alt+↑/↓`、`Shift+PageUp`）或加 filter（仅在输入框为空/焦点在 transcript 时生效）。

---

### P1-6 批处理 / TTY 模式下答案被渲染两次（流式原文一次 + 最终 Markdown 一次）

**证据**（实测 **E7**，端到端跑 `async_main(["--mock", ..., "list the files here"])` 并把 `sys.stdout` 换成 `isatty()=True` 的缓冲）
- 非 UI 路径下 `push_delta()` 走 `renderer.py:136-183`：把**已稳定的 Markdown 原文**逐行写成 `  │ …`；
- 回合结束时 `app.py:461-475` 又调用 `final_answer(cell)`，把同一份 source 渲染成 Markdown 打到 `── final answer ──` 之下；
- 结果：答案文本出现 **2 次**（一次是带 `  │ ` 前缀的原始 Markdown，一次是排版后的最终答案），实测计数：`'This run used the offline deterministic model' → 2`、`'[mock model - no API key configured]' → 2`、`'── final answer ──' → 1`。
- 触发条件：stdout 是 TTY（`Renderer.use_live_tail` 自动探测，`renderer.py:52-53`）——也就是 `./miniagent.sh "task"` 这种最常用的批处理用法。

**修复方向**：二选一——要么流式阶段只写"尾部预览"、最终答案写一次（并在 `final_answer` 前擦除流式区），要么 `final_answer` 只在流式未启用时才完整重打。

---

## 3. P2：细节与健壮性

| # | 问题 | 证据 |
|---|---|---|
| P2-1 | `line_count` 少算一行（只数 `"\n"`，末片段无换行时会偏 1），伪光标定位随之偏一行 | `transcript.py:45,50-51` |
| P2-2 | `transcript_scroll` 只有下界，无上界夹紧；越界后视口行为依赖 prompt_toolkit 的夹紧逻辑 | `app.py:721` |
| P2-3 | 提交输入时不重置滚动位置：上翻状态下发起新回合，新内容在视口下方，用户看不到 | `app.py:692-699` |
| P2-4 | 回合运行中提交的输入被**静默丢弃**（无排队、无提示） | `app.py:698` |
| P2-5 | `Ctrl+C` 中断运行中的回合会 **AttributeError 崩溃**：`app.py:831-832` 引用了从不存在的 `self.turn_task`（实际字段是 `_ui_turn_task`）。实测：`MiniAgentApp(session=None); app.running=True; app._on_cancel()` → `AttributeError: 'MiniAgentApp' object has no attribute 'turn_task'`。空输入行时走 `event.app.exit(exception=PromptInterrupt())`（`composer.py:238`），而 `prompt_loop`（`app.py:780-798`）未捕获该异常 | 实测 |
| P2-6 | 每个 `ToolOutput` 事件都触发整帧 invalidate，但运行中的工具 body 又不渲染 ⇒ 纯浪费 | `app.py:395-398` + `transcript.py:86` |
| P2-7 | 工具输出预览不按显示宽度折行/裁剪，超长单行不 `clip_to_width`（交互路径），依赖窗口 wrap；CJK/emoji 宽度未参与分行计算 | `transcript.py:85-89` |
| P2-8 | `ErrorCell` 在 UI 里没有 `fatal` 区分；`SkillCell` 失败仍显示 "skill loaded" | `transcript.py:93-97` |
| P2-9 | 无 transcript 搜索、无"查看完整工具输出/差异"的独立视图、无导出（历史只在内存），长输出只能靠终端 scrollback 碰运气 | `terminal_output_research.md` 第 3–4 条未落地 |
| P2-10 | `refresh_interval=0.1` 的 10 Hz 重绘在空闲时也持续（无动画需求） | `app.py:765` |
| P2-11 | 交互路径的 `_render_markdown` 自建**裸 Console**、没传 `theme.MARKDOWN_THEME` ⇒ 项目特意设计的"轻 Markdown 主题"在主路径不生效（实测：引用块在主题路径是 dim、在交互路径不是；h1 在交互路径带下划线） | `transcript.py:110-116` vs `render/theme.py:61-69` |
| P2-12 | 多行用户输入回显只有第一行有 `› ` 前缀，后续行无标记，与助手正文难以区分 | `transcript.py:66` |
| P2-13 | ~~工具运行中/完成都用 `●`（只有颜色不同）~~ → 已改为运行中 `◐`、完成 `●`、失败 `✗` | `cells/base.py` glyph |

---

## 4. 与 README / 设计文档的口径差异（文档需同步）

| 文档声明 | 实现现状 |
|---|---|
| "Every call is announced as it starts (`● read_file  {"path": …} …`)"（README **Tool trace** 行） | 交互模式运行中**不显示**；完成后也不显示参数与耗时（P0-3） |
| "Composer 支持 Home/End …"（README CLI 表） | `End` 被全局抢去滚 transcript（P1-5） |
| "Raw Markdown 分为 stable region + mutable tail，仅尾部重画（throttled）"（README / CLI_Design §19–20） | 交互路径未使用，改为每帧整段重渲染（P1-2、P0-5） |
| `terminal_output_research.md`：展开/折叠、工具间导航、详细 transcript 视图 | 仅"折叠到 3 行"落地；展开被 8 行上限卡住，无导航、无详细视图（P0-4、P2-9） |
| `CLI_Design.md` 未定义滚动/鼠标/resize 的交互模型 | 实现为临时方案（伪光标 + ±8 行 + End），缺少规格（P0-1/P0-2） |

---

## 5. 建议的修复顺序

1. **批次 A（可感知的核心体验）**：P0-1 滚轮接线 → P0-2 视口模型（相对底部偏移 + 上翻提示）→ P0-3 运行中 cell 可见 → P0-4 真展开 + 高度感知预览 → P0-7 Markdown 链接/尖括号乱码（主路径天天可见）。
2. **批次 B（正确性与一致性）**：P1-1 统一 formatter + markup escape/`markup=False`（顺带消掉 `MarkupError` 崩溃路径）→ P0-5 宽度/缓存/重绘频率 → P0-6 CoT 透传与展示 → P1-3 命令输出进 transcript → P1-5 键位冲突 → P1-6 批处理重复渲染 → P2-5 中断崩溃。
3. **批次 C（打磨）**：P1-2 stable/tail 流式、P1-4 状态栏与按键文档、P2 其余项、搜索/详细视图/导出。

## 6. 验证方式

**复现命令**

```bash
cd /home/catalyst259/Personal-Files/Agent
./miniagent.sh --mock                 # 交互模式（滚轮 / 运行中工具 / Ctrl+O 都在这里复现）
./miniagent.sh --mock "list files"    # 批处理模式走 rich 路径（markup 吃字/崩溃在这里复现）
.venv/bin/python -m pytest tests/test_cli_render.py -q
```

**量化实验（E1–E7；括号内为修复后的期望值）**

```python
# E1 Markdown 现在按调用方给的宽度排版（不再是固定 120）
import os; os.environ["TERM"] = "xterm-256color"   # TERM=dumb 时 rich 会走 80×25 回退
from harness.cli.render.transcript import _render_markdown
from harness.cli.sanitize import display_width
p = ("a long paragraph a model would emit. " * 6)
widths = [max(display_width(l) for l in "".join(t for _s, t in _render_markdown(p, width=w)).split("\n"))
          for w in (60, 120)]
print(widths)                     # 修复前: [80, 80]（固定）→ 修复后: [59, 117]

# E2 显式上限=真展开；真实宽度下的裁剪让一行只占一行
from harness.cli.cells import ToolCell
c = ToolCell(tool="shell"); c.output = "A"*5000 + "\n" + "\n".join(f"l{i}" for i in range(40))
print(c.preview_body(3)[1], c.preview_body(40)[1], c.preview_body(400)[1])   # 38 1 0

# E3 rich 路径不再把方括号当 markup（改成 Text 渲染）
import io; from rich.console import Console
from harness.cli.cells import ErrorCell
out = io.StringIO()
ErrorCell(message="bad [/] tag").render(Console(file=out, width=100))
print(repr(out.getvalue().strip()))   # 修复前: MarkupError → 修复后: '!! bad [/] tag'

# E4 bridge 透传 reasoning
from harness.cli.events_bridge import translate
from harness.agent.events import Event
print(translate(Event(type="assistant_message", message="x",
                      data={"reasoning": "CoT"}), None).reasoning)   # 修复前: None → 修复后: 'CoT'

# E5 逻辑行 ≠ 可视行（这就是旧"伪光标 + ±8 行"方案失效的原因）
from types import SimpleNamespace
from harness.cli.cells import AssistantCell, UserCell
from harness.cli.render.transcript import TranscriptControl
st = SimpleNamespace(history_cells=[UserCell(text="q"),
                     AssistantCell(source="long line " * 6, complete=True)],
                     expanded_tool_ids=set(), activity="")
ctl = TranscriptControl(st); ct = ctl.create_content(40, None)
print(ctl.line_count, sum(ct.get_height_for_line(i, 40, None) for i in range(ct.line_count)))  # 2 4

# E6 链接与尖括号内容保留
for s in ["see [docs](https://example.com) for details", "Use tool --flag <value> now"]:
    print(repr("".join(t for _st, t in _render_markdown(s)).strip()))
# 修复前: 'see 8;id=…;https://example.comdocs8;; for details' / 'Use tool --flag  now'
# 修复后: 'see docs for details'                                / 'Use tool --flag <value> now'

# E7 批处理 + TTY：答案只出现一次（预览行在最终渲染前被擦除）
# 见 tests/test_cli_render.py::test_finished_answer_is_not_printed_twice_on_a_terminal
```

**回归测试**（覆盖本轮修复）

```bash
.venv/bin/python -m pytest tests/test_cli_render.py -q   # 视口/滚轮/展开/CoT/markdown/markup
.venv/bin/python -m pytest -q                            # 全量：248 passed
.venv/bin/python .probe/wheel_probe.py                   # 无头：滚轮与翻页真的改变视口
.venv/bin/python .probe/ui_verify_probe.py               # 无头：运行中工具 / CoT / 命令输出
# E6 Markdown 链接变 OSC 乱码、尖括号文本被吞（走的就是交互 transcript 的渲染函数）
import os; os.environ["TERM"] = "xterm-256color"
from harness.cli.render.transcript import _render_markdown
for s in ["see [docs](https://example.com) for details",
          "Use tool --flag <value> to continue.",
          "Write to <stdout> and read <stdin>."]:
    print(repr("".join(t for _st, t in _render_markdown(s)).strip()))
# 'see 8;id=…;https://example.comdocs8;; for details' / 'Use tool --flag  to continue.' / 'Write to  and read .'

# E7 批处理 + TTY：答案渲染两次（需 isatty=True 的假 stdout，直接跑 async_main 最真实）
import asyncio, io, sys
class TTY(io.StringIO):
    def isatty(self): return True
buf = TTY(); real = sys.stdout; sys.stdout = buf
try:
    from harness.cli.app import async_main
    asyncio.run(async_main(["--mock", "--no-memory", "--no-checkpoint", "list the files here"]))
finally:
    sys.stdout = real
print("出现次数:", buf.getvalue().count("This run used the offline deterministic model"),
      buf.getvalue().count("── final answer ──"))  # 修复前 2 1（答案打印两次）→ 修复后屏幕上只可见一次
```

**实测附录（无头字节流抓包，替代 PTY）**

PTY 在本沙箱被禁用（`os.openpty()` → `OSError: [Errno 13] Permission denied`，`/dev/ptmx` 被拦），因此改用**无头但真实**的方式驱动 `MiniAgentApp.create_application()` 本身：输入 `create_pipe_input()`（注入终端会发的原始字节），输出用真正的 `Vt100_Output` 写进记录器，再用最小 ANSI 屏幕模拟器回放并回应该渲染器的 CPR 查询（等价于真实终端）。脚本与原始抓包在 `.probe/`（`headless_app_driver.py`、`state_scroll_probe.py`、`raw_*.bin`、`screen_*.json`）。

会话：4 个 mock 回合，24×100，`TERM=xterm-256color`（修复前为默认 `refresh_interval=0.1`，修复后为 `None`）。

| 观测 | 修复前 | 修复后 |
|---|---|---|
| 鼠标模式序列 `?1000h/1006h` | **0 次** | **各 1 次**（退出时正确关闭） |
| 注入 SGR 滚轮上滚 `\x1b[<64;12;6M` | 屏幕无反应 | `vertical_scroll` 10 → 7（上翻 3 行），屏幕变化 |
| 注入滚轮 release `\x1b[<64;12;6m` | — | 正确忽略（不是滚轮） |
| `PageUp` #1 | 屏幕**无变化**（假光标死区） | 整页上翻（scroll 10 → 0），屏幕立即变化 |
| `Ctrl+End` | 无此键（`End` 抢占输入框） | 回到底部并恢复跟随 |
| 空闲时写出 | 50 次写/秒（10 Hz 空转） | 2.5 s 内 **0 字节** |
| 终端 scrollback | 0 行（历史只在窗口里，但现在可滚） | 不变，但窗口内可滚 + 滚动条 |
| 交互屏内容 | `● list_dir` 无参数、无 CoT、答案里 `<your task>` 丢失 | `● shell {"command": "…"} …` + 运行中尾部预览 + `∴ thinking` 块 + `<your task>` 保留 + `/help` 输出在 transcript 内 |

对照假设的结论（修复前 → 修复后）：

- **H1 滚轮收不到事件 → 已解决**：`mouse_support=True` + 应用级 `Vt100MouseEvent` 拦截，不再依赖 prompt_toolkit 的 CPR 门槛。
- **H2 空闲 10 Hz 空转 → 已解决**：`refresh_interval=None`，空闲 2.5 s 内 0 字节写出。
- **H3 "终端 scrollback 堆满重复帧" → 设计上不再依赖终端 scrollback**：历史由应用内可滚动 pane 承载（滚动条 + 滚轮 + 翻页键），终端 scrollback 仍为 0 行。
- **H4 翻页键 → 已解决**：PageUp/PageDown 按窗口高度整页滚动（第一次按就有反馈）；`End` 归还输入框，新增 Ctrl+Home/Ctrl+End。
- **H5 固定 120 列 → 已解决**：Markdown 按 pane 实际宽度排版（60 列窗口 → 59 列行宽；120 列 → 117）。
- **H6 交互模式没有 CoT → 已解决**：`assistant_message` 事件携带 reasoning，bridge 透传，transcript 渲染 `∴ thinking` 折叠块。

**现有测试的盲区（本轮已补）**

- 旧测试曾把缺陷写成断言（"光标跟最后一行"、"展开后仍有 4 行被隐藏"），已替换为视口/展开回归测试；
- 新增覆盖：滚动分页与跟随语义、滚轮解码（SGR/X10/release）、运行中 cell 可见、参数与耗时、超长行裁剪、CoT 折叠与展开、markdown 链接/尖括号/宽度/代码段转义、reasoning 透传、命令输出进 transcript、批处理不重复渲染。
