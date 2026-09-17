"""MiniAgent: a lightweight, terminal-first coding agent.

Boundaries (see Agent_Harness_Design.md):

* ``orchestration`` - LangGraph state machine (nodes, edges, checkpointing)
* ``inference``     - ModelGateway abstraction over OpenAI-compatible APIs
* ``tools``         - MCP tool protocol, runtime and built-in tools
* ``skills``        - filesystem skills with progressive disclosure
* ``subagents``     - context-isolated delegation
* ``memory``        - long-term semantic memory (Qdrant) and memory formation
* ``context``       - context building, token budget and compaction
* ``agent``         - state, DTOs, events and the termination guard
* ``infra``         - config, logging, checkpoint adapters
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
