"""Error types for the ChemSpace MCP server.

A small umbrella class lets unit tests assert on error semantics without
having to import the optional ``mcp`` package. The server boundary converts
:class:`MCPToolError` into the protocol-level ``ToolError`` so messages reach
the MCP client.
"""

from __future__ import annotations


class MCPToolError(RuntimeError):
    """Raised by the MCP tool adapter when a toolkit call cannot be served."""
