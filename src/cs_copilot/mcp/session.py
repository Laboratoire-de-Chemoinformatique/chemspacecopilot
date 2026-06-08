"""Bootstrap helpers for the MCP server's storage and session lifecycle.

The cs_copilot storage client (`src/cs_copilot/storage/client.py`) resolves
``SESSION_ID`` at import time. This module exists so the CLI entry point can
set the desired ``SESSION_ID`` *before* the storage module is imported, then
bind the workflow layout and the agent-context singleton.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class BootstrapConfig:
    """Parsed values describing how to bootstrap the MCP server process."""

    session_id: Optional[str] = None
    workflow_slug: Optional[str] = None
    log_level: str = "info"


_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def configure_logging(level: str) -> None:
    """Configure the root logger for the MCP process.

    stdio MCP servers must keep stdout clean for JSON-RPC traffic, so logging
    is sent to stderr only.
    """

    numeric = _LEVELS.get(level.lower(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=__import__("sys").stderr,
        force=True,
    )


def apply_session_id(session_id: Optional[str]) -> None:
    """Persist ``session_id`` into the environment if provided.

    Must be called *before* any ``cs_copilot.storage`` or ``cs_copilot.tools``
    module is imported, because the storage client resolves ``SESSION_ID`` at
    import time.
    """

    if not session_id:
        return
    os.environ["SESSION_ID"] = session_id


def bootstrap(config: BootstrapConfig):
    """Bootstrap storage, layout, and the MCP agent context.

    Returns the singleton :class:`~cs_copilot.mcp.context.MCPAgentContext`.
    """

    # Imports below intentionally happen after ``apply_session_id`` has had a
    # chance to seed ``os.environ``; the storage client snapshots SESSION_ID
    # at import time.
    from cs_copilot.storage import S3, ensure_output_context

    from .context import MCPAgentContext, set_current_context

    requested = os.environ.get("SESSION_ID", "").strip()
    if requested:
        # If the caller forced a SESSION_ID, ensure the storage prefix matches
        # it even when the storage module had already auto-generated one.
        S3.set_session_prefix(f"sessions/{requested}")

    ctx = MCPAgentContext()
    ensure_output_context(ctx.session_state, workflow_slug=config.workflow_slug)
    set_current_context(ctx)
    return ctx
