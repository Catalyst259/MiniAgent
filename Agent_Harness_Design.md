# Agent Harness 技术选型与组件复用设计

## 1. 目标

本项目实现一个轻量但结构清晰的 **Agent Harness**，目标不是依赖某个“大而全”的 Agent 框架完成全部能力，而是：

- 复用成熟的基础设施组件；
- 保留 Agent Harness 的核心控制权；
- 明确区分状态、上下文、工具、技能、子 Agent、长期记忆等边界；
- LLM 层保持 **OpenAI-compatible**，避免绑定特定模型或推理服务；
- 后续可自由切换 DeepSeek、OpenAI、vLLM、本地模型或其他兼容服务。

整体原则：

> **框架负责基础设施，自研部分负责 Harness 本身。**

---

## 2. 总体技术栈

| 层 | 技术方案 | 说明 |
|---|---|---|
| 编程语言 | Python 3.11 / 3.12 | Agent / MCP / LangGraph 生态成熟 |
| 包管理 | `uv` | 环境与依赖管理 |
| Agent 编排 | **LangGraph `StateGraph`** | 仅负责 State / Node / Edge / Conditional Routing |
| 数据模型 | **Pydantic v2** | DTO、配置、Tool Call、Delegate Call 等 |
| LLM 抽象 | **自研 `ModelGateway`** | Agent 核心只依赖统一接口 |
| LLM 协议 | **OpenAI-compatible API** | 支持 DeepSeek / OpenAI / vLLM / 其他兼容服务 |
| LLM Client | `openai.AsyncOpenAI` | 统一请求兼容接口 |
| Tool 协议 | **MCP Python SDK** | Tool Schema、Tool Call、Tool Runtime |
| Runtime State | **LangGraph Checkpoint + SQLite** | 单任务运行态与恢复 |
| Global Memory | **Qdrant Local** | 长期语义记忆 |
| Embedding | 独立 Embedding Backend | 可用 Qwen3-Embedding 等 |
| Token Counting | `transformers.AutoTokenizer` 或 provider adapter | 避免强绑定某一模型 tokenizer |
| Skill | 自研 Filesystem Skill Loader | Progressive Disclosure |
| Context Builder | 自研 | Harness 核心逻辑 |
| Compact | 自研 | 当前上下文压缩 |
| Summary / Memory Formation | 自研 | 长期记忆写入 |
| Termination Guard | 自研 | 任务结束与异常终止判断 |
| Logging | `structlog` / stdlib `logging` | 可观测性 |
| CLI | `rich` | 本地调试与运行展示 |
| Test | `pytest + pytest-asyncio` | 单测 / 异步测试 |

---

# 3. 架构原则

## 3.1 不使用“大一统 Agent Framework”

本项目不希望：

```text
Agent Framework
├─ 自动管理 Context
├─ 自动管理 Tool
├─ 自动管理 Memory
├─ 自动管理 SubAgent
├─ 自动管理 Termination
└─ 自动管理 Loop
```

否则 Agent Harness 的核心机制会被框架隐藏。

因此选择：

```text
LangGraph
    ↓
只负责：
State
Node
Edge
Conditional Edge
Checkpoint
```

Agent Harness 自己负责：

```text
Context Construction
Token Budget
Compaction
Tool Routing
Skill Expansion
Delegation
Memory Formation
Termination
```

---

# 4. LangGraph：只负责 Agent Loop 编排

逻辑链路可抽象为：

```text
Context Builder
      ↓
Token Guard
   ┌──┴──────────┐
   │             │
Compact        LLM Reasoning
                 │
     ┌───────────┼───────────┬───────────┐
     ↓           ↓           ↓           ↓
 Tool Call   Skill Load   Delegate    Terminate
     │           │           │
     └───────────┴───────────┘
                 ↓
             Agent State
                 ↓
          Context Builder
```

LangGraph 负责描述这个状态机：

```python
graph.add_node(...)
graph.add_edge(...)
graph.add_conditional_edges(...)
```

但每个节点内部逻辑由 Harness 自己实现。

---

# 5. LLM 层：OpenAI-Compatible + ModelGateway

## 5.1 设计目标

LLM 层不能绑定：

- Qwen
- vLLM
- DeepSeek
- OpenAI
- 某个云平台

Agent Harness 只应该认识：

```text
ModelGateway
```

而不知道后面具体调用的是谁。

---

## 5.2 推荐结构

```text
Agent Harness
      ↓
ModelGateway
      ↓
OpenAICompatibleGateway
      ↓
OpenAI-compatible API
      ↓
┌────────────┬────────────┬────────────┬────────────┐
│ DeepSeek   │ OpenAI     │ vLLM       │ Other API  │
└────────────┴────────────┴────────────┴────────────┘
```

---

## 5.3 接口定义

建议使用 Protocol：

```python
from typing import Protocol

class ModelGateway(Protocol):
    async def chat(
        self,
        request: "ModelRequest",
    ) -> "ModelResponse":
        ...
```

输入：

```python
class ModelRequest(BaseModel):
    messages: list[Message]
    tools: list[ToolSchema] | None = None

    temperature: float | None = None
    max_tokens: int | None = None

    model: str | None = None
```

输出统一归一化为：

```python
class ModelResponse(BaseModel):
    text: str | None = None
    reasoning: str | None = None

    tool_calls: list[ToolCall] = []

    finish_reason: str | None = None
    usage: TokenUsage | None = None
```

---

## 5.4 为什么必须做 Normalize

即使各家都声称兼容 OpenAI API，仍可能存在细节差异：

```text
tool_calls
finish_reason
reasoning_content
usage
stream chunks
JSON arguments
error format
```

因此必须：

```text
Provider Response
        ↓
OpenAICompatibleGateway
        ↓
Normalize
        ↓
ModelResponse
        ↓
Agent Harness
```

Agent Harness 永远不要直接解析不同厂商原始响应。

---

## 5.5 配置示例

```yaml
models:
  main:
    provider: openai_compatible
    base_url: https://api.deepseek.com
    api_key_env: DEEPSEEK_API_KEY
    model: deepseek-chat

  coding:
    provider: openai_compatible
    base_url: http://localhost:8000/v1
    api_key: EMPTY
    model: Qwen2.5-Coder-7B-Instruct
```

切模型只修改配置。

---

# 6. Tool：MCP 作为标准协议

图中的：

```text
Tool Registry
Tool Schema
Tool Call
Tool Runtime
```

统一落到 MCP。

映射关系：

| Harness 概念 | MCP 对应 |
|---|---|
| Tool Registry | `list_tools()` |
| Tool Schema | MCP Input Schema |
| Tool Call | Structured Tool Invocation |
| Tool Runtime | MCP Client |
| Observation | MCP Tool Result |

---

## 6.1 Tool Runtime

推荐架构：

```text
LLM
 ↓
ToolCall DTO
 ↓
ToolRuntime
 ↓
MCP Client
 ↓
MCP Server
 ↓
Tool
```

`ToolRuntime` 保持很薄，只负责：

```text
找到 Tool
校验参数
执行 Tool
规范化结果
转换为 Observation
```

---

## 6.2 不重复定义私有 Tool 协议

不建议：

```python
class Tool:
    name: str
    description: str
    parameters: dict

    def execute(...):
        ...
```

这部分 MCP 已经解决。

可以有：

```python
class ToolRegistry:
    ...
```

但它应该只是：

```text
MCP Tool Metadata Cache / Adapter
```

而不是重新定义协议。

---

# 7. Agent State：运行态，不等于 Memory

必须严格区分：

```text
Agent State != Global Memory
```

Agent State 表示**当前任务正在发生什么**。

例如：

```python
class AgentState(TypedDict):
    messages: list[Message]

    iteration: int

    loaded_skills: list[str]

    active_subagent: str | None

    tool_results: list[Observation]

    compact_summary: str | None

    token_usage: int

    termination_status: str | None
```

---

## 7.1 State 存储

使用：

```text
LangGraph Checkpoint
        ↓
SQLite
```

作用：

```text
持久化运行态
支持 Resume
支持 Debug
支持 Thread / Session
```

不需要自己实现：

```text
save_state()
load_state()
resume_state()
```

---

# 8. Global Memory：Qdrant

Global Memory 存的是跨任务长期信息：

```text
某个 Repository 的结构
以前修过什么 Bug
某个问题最后怎么解决
成功的 Agent Trajectory
项目长期事实
过去任务的总结
```

这些信息需要：

```text
Semantic Retrieval
Metadata Filtering
Top-K Search
Namespace
```

因此推荐：

```text
Qdrant Local
```

初期直接本地持久化即可。

以后可以平滑迁移到独立 Qdrant Server。

---

## 8.1 Memory 数据结构

可以统一为：

```python
class MemoryRecord(BaseModel):
    id: str

    content: str

    memory_type: str

    source: str | None = None

    task_id: str | None = None
    repo: str | None = None

    timestamp: datetime

    metadata: dict[str, Any] = {}
```

---

# 9. Embedding 独立于主模型

不要使用主 Coding LLM 去做 embedding。

推荐架构：

```text
Memory Query
     ↓
EmbeddingBackend
     ↓
Embedding Model
     ↓
Qdrant
     ↓
Relevant Memories
```

接口：

```python
class EmbeddingBackend(Protocol):
    async def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        ...
```

初期可以：

```text
sentence-transformers
+
Qwen3-Embedding
```

未来也可以换成远程 Embedding API。

---

# 10. Context Builder：Harness 核心

Context Builder 是整个 Agent Harness 最重要的模块之一。

其输入：

```text
System Prompt
Relevant Memory
Tool Schema
Skill Metadata
Agent State
Recent Messages
Current Observation
```

输出：

```text
Model Context
```

可以抽象为：

```text
System Prompt
      +
Relevant Memory
      +
Available Tool Schemas
      +
Available Skill Metadata
      +
Current Agent State
      +
Recent Conversation
      +
Current Observation
      ↓
Context Builder
      ↓
Model Request
```

---

## 10.1 Context Builder 应是确定性组件

建议：

```python
class ContextBuilder:

    async def build(
        self,
        state: AgentState,
    ) -> ModelRequest:
        ...
```

Context Builder 不负责：

```text
执行模型
调用工具
执行 Skill
决定路由
```

它只负责：

```text
把当前信息组装成下一轮模型输入
```

---

# 11. Token Budget / Context Manager

`Token Exceed Limit` 本质不是 Service，而是 Policy。

例如：

```python
class TokenBudgetPolicy:

    def exceeded(
        self,
        context: ModelRequest,
    ) -> bool:
        ...
```

Context Manager 负责：

```text
Token Counting
Message Selection
Compaction Strategy
Context Window Budget
```

---

# 12. Compact 和 Summary 必须分开

这是整个设计里很重要的一条边界。

---

## 12.1 Compact

目标：

> 解决当前 Context 太大。

属于：

```text
Context Management
```

流程：

```text
Context Too Large
      ↓
Compact
      ↓
缩短历史消息
      ↓
继续当前任务
```

---

## 12.2 Summary

目标：

> 当前任务结束后，提取值得长期保留的信息。

属于：

```text
Memory Formation
```

流程：

```text
Task Finished
      ↓
Summary
      ↓
Extract Valuable Memory
      ↓
Embedding
      ↓
Qdrant
```

两者不能合并。

---

# 13. Skill：Filesystem + Progressive Disclosure

Skill 不需要额外框架。

推荐结构：

```text
skills/
├── code_review/
│   └── SKILL.md
├── debugging/
│   └── SKILL.md
├── testing/
│   └── SKILL.md
└── refactor/
    └── SKILL.md
```

---

## 13.1 启动阶段只加载 Metadata

例如：

```text
name
description
trigger
```

注入 Context：

```text
Available Skills:

- code_review
  Review implementation correctness and maintainability.

- debugging
  Diagnose failing tests and runtime errors.
```

而不是直接把完整 SKILL.md 塞进 Prompt。

---

## 13.2 Skill Loader

当模型请求：

```text
load_skill("debugging")
```

才真正读取：

```text
skills/debugging/SKILL.md
```

然后把 Skill Detail 加入 Agent State。

流程：

```text
Skill Metadata
      ↓
LLM selects Skill
      ↓
Skill Loader
      ↓
Skill Detail
      ↓
Agent State
      ↓
Context Builder
```

---

# 14. SubAgent：LangGraph Subgraph

不需要额外引入 AutoGen / CrewAI。

每个 SubAgent 本质上是：

```text
独立 State
+
独立 Context
+
独立 Graph
```

例如：

```text
PlanningAgent
SearchingAgent
CodingAgent
```

---

## 14.1 Delegate Call

主 Agent：

```python
class DelegateRequest(BaseModel):
    agent: str
    task: str
    context: str | None = None
```

子 Agent 返回：

```python
class DelegateResult(BaseModel):
    summary: str

    artifacts: list[str] = []

    metadata: dict = {}
```

---

## 14.2 Context Isolation

原则：

```text
Main Agent Context
        │
        │ selected context only
        ↓
SubAgent Context
        ↓
SubAgent Execution
        ↓
Compact Result
        ↓
Main Agent
```

不要：

```text
Main Agent 所有 messages
        ↓
完整复制给 SubAgent
```

这样会导致：

```text
上下文膨胀
角色污染
无关信息干扰
Token 浪费
```

---

# 15. Termination Guard

Termination Guard 不只是：

```text
模型没有 Tool Call → 结束
```

建议至少处理：

```text
Final Answer
Maximum Iterations
Fatal Tool Error
Repeated Tool Call
Context Failure
SubAgent Failure
Explicit Stop
```

例如：

```python
class TerminationDecision(BaseModel):
    terminate: bool

    reason: str | None = None

    final_answer: str | None = None
```

---

# 16. 逻辑图节点分类

不要把逻辑图每个方框都做成 Python Module。

应该先分类。

---

## 16.1 Service

真正具有行为和生命周期的组件：

```text
ContextBuilder
ContextManager
CompactService
ModelGateway
ToolRuntime
SkillLoader
SubAgentRuntime
MemoryService
SummaryService
```

---

## 16.2 Policy

只是规则判断：

```text
TokenBudgetPolicy
TerminationPolicy
RetryPolicy
ToolPermissionPolicy
```

---

## 16.3 DTO / Event / Value Object

只是结构化数据：

```text
ToolCall
DelegateCall
Observation
FinalAnswer
ModelRequest
ModelResponse
MemoryRecord
TokenUsage
```

---

## 16.4 Adapter

封装外部技术：

```text
OpenAICompatibleGateway
MCPClientAdapter
QdrantMemoryStore
SQLiteCheckpointStore
EmbeddingAdapter
TokenizerAdapter
```

---

# 17. 推荐模块分层

最终建议：

```text
src/
├── orchestration/
│   ├── graph.py
│   ├── nodes.py
│   └── routing.py
│
├── agent/
│   ├── state.py
│   ├── events.py
│   ├── dto.py
│   └── termination.py
│
├── context/
│   ├── builder.py
│   ├── manager.py
│   ├── token_budget.py
│   └── compact.py
│
├── inference/
│   ├── gateway.py
│   ├── openai_compatible.py
│   ├── dto.py
│   ├── tokenizer.py
│   └── config.py
│
├── tools/
│   ├── registry.py
│   ├── runtime.py
│   ├── mcp_client.py
│   └── dto.py
│
├── skills/
│   ├── registry.py
│   ├── loader.py
│   └── metadata.py
│
├── subagents/
│   ├── runtime.py
│   ├── registry.py
│   ├── planning/
│   └── searching/
│
├── memory/
│   ├── service.py
│   ├── store.py
│   ├── qdrant_store.py
│   ├── embedding.py
│   └── summary.py
│
├── infra/
│   ├── checkpoint.py
│   ├── config.py
│   └── logging.py
│
└── main.py
```

---

# 18. 依赖方向

核心原则：

```text
orchestration
     ↓
agent
context
tools
skills
subagents
memory
inference
     ↓
infra / external adapters
```

上层可以依赖抽象接口。

下层不能反向依赖 orchestration。

例如：

```text
ContextBuilder
    ↓
MemoryService interface
```

而不是：

```text
ContextBuilder
    ↓
QdrantClient
```

同理：

```text
LLM Node
    ↓
ModelGateway
```

而不是：

```text
LLM Node
    ↓
DeepSeekClient
```

---

# 19. 核心接口建议

整个系统可以优先稳定这些接口：

```python
ModelGateway
ToolRuntime
SkillLoader
SubAgentRuntime
MemoryStore
EmbeddingBackend
ContextBuilder
CompactService
TerminationPolicy
```

一旦这些边界稳定，底层组件都可以自由替换。

---

# 20. MVP 实现顺序

建议不要一次把整张图全部实现。

按照链路逐渐扩展。

## Phase 1：最小 Agent Loop

```text
User Input
 ↓
Context Builder
 ↓
LLM
 ↓
Final Answer
```

实现：

```text
AgentState
ModelGateway
ContextBuilder
LangGraph
```

---

## Phase 2：Tool

```text
LLM
 ↓
Tool Call
 ↓
MCP
 ↓
Observation
 ↓
LLM
```

实现：

```text
ToolRuntime
MCP Client
Tool DTO
```

---

## Phase 3：Skill

加入：

```text
Skill Metadata
Skill Loader
Skill Detail Injection
```

---

## Phase 4：Context Management

加入：

```text
Token Counting
TokenBudgetPolicy
Compact
```

---

## Phase 5：SubAgent

加入：

```text
DelegateCall
SubAgentRuntime
Context Isolation
```

---

## Phase 6：Memory

最后加入：

```text
Qdrant
Embedding
Memory Retrieval
Task Summary
Memory Formation
```

---

# 21. 暂时不要引入的东西

第一版不建议加入：

```text
CrewAI
AutoGen
Deep Agents
LangMem
Redis
Kafka
PostgreSQL
Elasticsearch
复杂 Event Bus
复杂 Dependency Injection Framework
```

原因不是这些组件不好，而是当前规模用不上。

尤其：

```text
LangGraph 已负责 orchestration
MCP 已负责 Tool Protocol
SQLite 已负责 runtime state
Qdrant 已负责 semantic memory
```

再叠框架只会增加复杂度。

---

# 22. 最终架构

```text
                        ┌─────────────────────┐
                        │    Global Memory    │
                        │       Qdrant        │
                        └──────────▲──────────┘
                                   │
                              Memory Service
                                   │
User Input                         │
    │                              │
    ▼                              │
┌──────────────────────────────────────────────┐
│                 Agent Harness                │
│                                              │
│   Context Builder                            │
│        │                                     │
│        ▼                                     │
│   Token Budget ─── exceed ──→ Compact       │
│        │                                     │
│      normal                                  │
│        ▼                                     │
│   ModelGateway                               │
│        │                                     │
│        ▼                                     │
│ OpenAI-Compatible API                       │
│        │                                     │
│ ┌──────┼─────────┬─────────┬────────────┐   │
│ │      │         │         │            │   │
│ Tool  Skill   Delegate   Final Answer   │   │
│ │      │         │                      │   │
│ ▼      ▼         ▼                      │   │
│ MCP  Loader   SubAgent                  │   │
│ │      │         │                      │   │
│ └──────┴─────────┴──────→ Agent State   │   │
│                           │              │   │
│                           └── loop ──────┘   │
│                                              │
└──────────────────────────────────────────────┘
                  │
                  ▼
        LangGraph Checkpoint
                  │
                SQLite
```

---

# 23. 定稿

当前技术栈可以确定为：

```text
LangGraph
+
Pydantic
+
OpenAI-compatible ModelGateway
+
MCP
+
SQLite Checkpoint
+
Qdrant
+
Independent Embedding Backend
+
Custom Context / Skill / Compact / Summary / Termination
```

核心边界：

```text
LangGraph          → 编排
ModelGateway       → 模型抽象
MCP                → 工具协议
SQLite             → 当前任务状态
Qdrant             → 长期语义记忆
Filesystem         → Skill
ContextBuilder     → 上下文组装
Compact            → 当前上下文压缩
Summary            → 长期记忆形成
TerminationGuard   → 生命周期结束
```

后续模块分层和类设计都应建立在这些边界上。
