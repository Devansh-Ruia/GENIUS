"""Anthropic client access.

The model name always comes from ANTHROPIC_MODEL. Nothing in this package
names a model. Nothing calls the LLM yet; agents will use these helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vus_panel.config import get_env

if TYPE_CHECKING:
    import anthropic


class LLMConfigError(RuntimeError):
    """Raised when the LLM settings are missing."""


def get_model() -> str:
    model = get_env("ANTHROPIC_MODEL")
    if model is None:
        raise LLMConfigError(
            "ANTHROPIC_MODEL is not set. Set it in your environment (see .env.example). "
            "The model name is never hardcoded."
        )
    return model


def get_client() -> "anthropic.Anthropic":
    """Build a client on demand. Constructing it sends no request."""
    api_key = get_env("ANTHROPIC_API_KEY")
    if api_key is None:
        raise LLMConfigError("ANTHROPIC_API_KEY is not set. Set it in your environment (see .env.example).")
    import anthropic

    return anthropic.Anthropic(api_key=api_key)
