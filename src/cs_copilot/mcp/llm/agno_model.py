"""Opt-in model loading for MCP internal-model mode.

This module intentionally avoids static imports of ``cs_copilot.model_config``.
The MCP default path must not import or run the Agno team or model stack.
"""

from __future__ import annotations

from typing import Any


def load_configured_model() -> Any:
    """Load the configured Agno model without importing the Agno team."""

    module = __import__("cs_copilot.model_config", fromlist=["load_model_from_config"])
    return module.load_model_from_config()
