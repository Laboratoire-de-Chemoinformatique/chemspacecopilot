"""Optional Model Context Protocol (MCP) server for cs_copilot.

The cs_copilot default runtime (Chainlit / CLI / Agno multi-agent team)
does not import anything from this package and is unaffected by it.

The MCP server exposes cs_copilot primitives — toolkit methods, agent prompt
constants, and session artifacts — to external MCP clients (Codex,
Claude Code). In its default mode it never executes the Agno team: the
external client is the reasoning engine.

The optional :mod:`mcp` Python SDK is only required at server build time, so
``import cs_copilot.mcp`` is safe even when the ``[mcp]`` extra is not
installed. Importers should call :func:`build_server` (which lazily imports
``mcp.server.fastmcp``) or invoke the ``cscopilot-mcp`` console script.
"""

from __future__ import annotations

__all__ = [
    "MCPAgentContext",
    "MCPToolError",
    "build_server",
]


def __getattr__(name: str):
    if name == "build_server":
        from .server import build_server

        return build_server
    if name == "MCPAgentContext":
        from .context import MCPAgentContext

        return MCPAgentContext
    if name == "MCPToolError":
        from .errors import MCPToolError

        return MCPToolError
    raise AttributeError(f"module 'cs_copilot.mcp' has no attribute {name!r}")
