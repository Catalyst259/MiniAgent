"""Harness configuration: one YAML file, environment overrides, no hidden state.

Environment overrides use ``MINIAGENT_`` plus the nested path, e.g.::

    MINIAGENT_MODELS__MAIN__API_KEY=sk-...
    MINIAGENT_MODELS__MAIN__MODEL=deepseek-reasoner
    MINIAGENT_CONTEXT__MAX_INPUT_TOKENS=128000
    MINIAGENT_RUNTIME__MAX_ITERATIONS=60
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from harness.agent.errors import ConfigError
from harness.inference.config import ModelConfig

ENV_PREFIX = "MINIAGENT_"


class ContextConfig(BaseModel):
    max_input_tokens: int = 96_000
    reserve_output_tokens: int = 4_096
    compact_trigger_ratio: float = 0.8
    keep_recent_messages: int = 8
    max_recent_messages: int = 40
    target_summary_tokens: int = 900


class RuntimeConfig(BaseModel):
    max_iterations: int = 40
    max_repeated_tool_calls: int = 3
    max_consecutive_tool_errors: int = 4
    tool_timeout_seconds: float = 180.0
    workspace_root: str = "."
    recursion_limit: int = 400


class ToolsConfig(BaseModel):
    enabled: list[str] = Field(
        default_factory=lambda: [
            "list_dir",
            "glob",
            "grep",
            "read_file",
            "write_file",
            "apply_patch",
            "shell",
            "git_diff",
        ]
    )
    max_output_chars: int = 30_000
    shell_timeout_seconds: int = 120
    shell_max_output_chars: int = 20_000
    command_allowlist: list[str] = Field(default_factory=list)
    command_denylist: list[str] = Field(default_factory=list)
    transport: str = "local"  # local | mcp-stdio
    include_native_tools: bool = True


class EmbeddingConfig(BaseModel):
    backend: str = "hashing"  # hashing | openai_compatible | sentence_transformers
    dimensions: int = 256
    base_url: str | None = None
    api_key_env: str | None = "EMBEDDING_API_KEY"
    model: str | None = None


class MemoryConfig(BaseModel):
    enabled: bool = True
    backend: str = "qdrant_local"  # qdrant_local | memory
    path: str = ".harness/memory"
    collection: str = "miniagent_memory"
    top_k: int = 5
    score_threshold: float = 0.05
    write_on_finish: bool = True
    max_facts: int = 5


class SkillsConfig(BaseModel):
    paths: list[str] = Field(default_factory=lambda: ["skills"])
    enabled: bool = True


class PermissionsConfig(BaseModel):
    """The permission layer (design document section 3).

    ``mode`` decides what happens when no rule has an opinion:

    * ``off``  - no permission layer at all (the historical behaviour),
    * ``ask``  - ask the user (a headless run has no approver and fails closed),
    * ``auto`` - deny, for non-interactive runs that must not block.

    The sandbox (workspace escape, ``protected_paths``, ``denied_tools``) is
    enforced in every mode and can never be overridden by a rule or an approval.
    """

    mode: str = "off"
    #: fallback verdict for actions no rule matches: allow | ask | deny
    default: str = "ask"
    #: shorthand lists, equivalent to ``rules`` entries
    allow: list[str] = Field(default_factory=list)
    ask: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)
    #: full rule objects: {tool, target, program, prefix, permission}
    rules: list[Any] = Field(default_factory=list)
    #: globs that are denied outright, in every mode
    protected_paths: list[str] = Field(default_factory=list)
    #: tools the agent may not use at all
    denied_tools: list[str] = Field(default_factory=list)
    #: remember "Always allow" across runs
    persistent: bool = False
    #: where those grants live; must be outside the workspace
    persistent_path: str | None = None


class SubAgentsConfig(BaseModel):
    paths: list[str] = Field(default_factory=lambda: ["subagents"])
    enabled: bool = True
    max_iterations: int = 20


class CheckpointConfig(BaseModel):
    path: str = ".harness/checkpoints.sqlite"
    enabled: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str | None = None
    json_output: bool = Field(default=False, alias="json")

    model_config = {"populate_by_name": True}


class HarnessConfig(BaseModel):
    """The whole harness configuration."""

    models: dict[str, ModelConfig] = Field(
        default_factory=lambda: {"main": ModelConfig(provider="mock", model="mock-main")}
    )
    default_model: str = "main"
    context: ContextConfig = Field(default_factory=ContextConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    subagents: SubAgentsConfig = Field(default_factory=SubAgentsConfig)
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    system_prompt_extra: str = ""

    config_path: str | None = None
    base_dir: str = "."

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path | None = None, *, apply_env: bool = True) -> "HarnessConfig":
        data: dict[str, Any] = {}
        base_dir = Path.cwd()
        resolved: Path | None = None

        if path is not None:
            resolved = Path(path).expanduser()
        else:
            for candidate in (Path.cwd() / "config.yaml", Path.cwd() / "harness.yaml"):
                if candidate.exists():
                    resolved = candidate
                    break

        if resolved is not None and resolved.exists():
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - dependency is required
                raise ConfigError("PyYAML is required to read config files") from exc
            raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
            if not isinstance(raw, dict):
                raise ConfigError(f"config file {resolved} must contain a mapping")
            data = raw
            base_dir = resolved.parent.resolve()

        load_dotenv(dotenv_path=base_dir / ".env", override=False)

        if apply_env:
            data = _apply_env_overrides(data)

        config = cls(**data)
        config.config_path = str(resolved) if resolved else None
        config.base_dir = str(base_dir)
        return config

    # ------------------------------------------------------------------ helpers
    def resolve_model(self, name: str | None = None) -> ModelConfig:
        key = name or self.default_model
        if key not in self.models:
            raise ConfigError(
                f"unknown model `{key}`; configured models: {', '.join(sorted(self.models))}"
            )
        return self.models[key]

    def resolve_path(self, value: str | Path) -> Path:
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else (Path(self.base_dir) / candidate)

    @property
    def workspace_root(self) -> Path:
        return self.resolve_path(self.runtime.workspace_root).resolve()

    @property
    def checkpoint_path(self) -> Path:
        return self.resolve_path(self.checkpoint.path)

    def skill_paths(self) -> list[Path]:
        return [self.resolve_path(path) for path in self.skills.paths]

    def subagent_paths(self) -> list[Path]:
        return [self.resolve_path(path) for path in self.subagents.paths]

    def memory_path(self) -> Path:
        return self.resolve_path(self.memory.path)

    def approvals_path(self) -> Path:
        """Where "Always allow" grants are stored.

        Defaults to ``~/.config/miniagent/approvals.json`` - deliberately outside
        the workspace, because the agent can write inside the workspace and must
        not be able to edit the file that decides what it may do.
        """

        from harness.permission.memory import DEFAULT_PERSISTENT_PATH

        return self.resolve_path(self.permissions.persistent_path or DEFAULT_PERSISTENT_PATH)

    def describe(self) -> dict[str, Any]:
        model = self.resolve_model()
        return {
            "config": self.config_path or "(defaults)",
            "model": model.label,
            "models": ", ".join(sorted(self.models)),
            "workspace": str(self.workspace_root),
            "max_iterations": self.runtime.max_iterations,
            "context_budget": self.context.max_input_tokens,
            "memory": f"{self.memory.backend} ({'on' if self.memory.enabled else 'off'})",
            "embedding": self.embedding.backend,
            "permissions": (
                f"{self.permissions.mode} ({len(self.permissions.rules)} rules, "
                f"{len(self.permissions.allow) + len(self.permissions.ask) + len(self.permissions.deny)} shortcuts)"
            ),
            "checkpoint": str(self.checkpoint_path) if self.checkpoint.enabled else "(off)",
        }


#: Config paths whose value is a *word*, not a flag.  ``_coerce`` turns "off" into
#: ``False`` because that is what ``enabled: off`` means, but
#: ``permissions.mode: off`` is a mode name and must stay a string.
_VERBATIM_PATHS: frozenset[tuple[str, ...]] = frozenset(
    {
        ("permissions", "mode"),
        ("permissions", "default"),
        ("tools", "transport"),
        ("memory", "backend"),
        ("embedding", "backend"),
        ("logging", "level"),
        ("default_model",),
    }
)


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    merged = {key: value for key, value in data.items()}
    for env_key, raw_value in os.environ.items():
        if not env_key.startswith(ENV_PREFIX):
            continue
        path = env_key[len(ENV_PREFIX) :].lower().split("__")
        if not path or not path[0]:
            continue
        # ``MINIAGENT_MOCK=1`` style variables are flags, not config paths.
        if len(path) == 1 and path[0] in ("mock", "debug", "verbose"):
            continue
        value = raw_value if tuple(path) in _VERBATIM_PATHS else _coerce(raw_value)
        _assign(merged, path, value)
    return merged


def _assign(target: dict[str, Any], path: list[str], value: Any) -> None:
    cursor = target
    for key in path[:-1]:
        existing = cursor.get(key)
        if not isinstance(existing, dict):
            existing = {}
            cursor[key] = existing
        cursor = existing
    cursor[path[-1]] = value


def _coerce(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "none", ""):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [item.strip() for item in inner.split(",") if item.strip()]
    return value


__all__ = [
    "HarnessConfig",
    "ModelConfig",
    "ContextConfig",
    "RuntimeConfig",
    "ToolsConfig",
    "MemoryConfig",
    "EmbeddingConfig",
    "SkillsConfig",
    "SubAgentsConfig",
    "PermissionsConfig",
    "CheckpointConfig",
    "LoggingConfig",
    "ENV_PREFIX",
]
