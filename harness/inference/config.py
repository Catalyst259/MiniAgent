"""Inference configuration: one place where a model endpoint is described.

Switching models is a config change, never a code change.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field, model_validator

from harness.agent.errors import ConfigError


class ModelConfig(BaseModel):
    """Description of one OpenAI-compatible endpoint."""

    provider: str = "openai_compatible"
    base_url: str | None = None
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    api_key_env: str | None = None

    temperature: float | None = 0.2
    max_tokens: int | None = None
    timeout_seconds: float = 120.0
    max_retries: int = 2
    #: Honour ``HTTP_PROXY``/``HTTPS_PROXY``/``SSL_CERT_FILE`` from the environment.
    #: Turn it off when the ambient proxy variables are broken for this endpoint.
    trust_env: bool = True
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _fill_key_from_env(self) -> "ModelConfig":
        if self.api_key is None and self.api_key_env:
            value = os.environ.get(self.api_key_env)
            if value:
                self.api_key = value
        if self.api_key == "EMPTY":
            self.api_key = "EMPTY"
        return self

    @property
    def resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        raise ConfigError(
            f"No API key for model `{self.model}`. Set the `{self.api_key_env or 'api_key'}` "
            "environment variable or put `api_key` in the config."
        )

    @property
    def label(self) -> str:
        return f"{self.model} @ {self.base_url or 'default'}"


__all__ = ["ModelConfig"]
