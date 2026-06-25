"""Lazy-import helper for the optional ``mcp`` dependency."""

from __future__ import annotations

_INSTALL_HINT = (
    "cscopilot-mcp requires the optional 'mcp' extra. Install it with:\n"
    "    uv pip install 'cs_copilot[mcp]'\n"
    "    # or, in a checkout of this repo:\n"
    "    uv sync --extra mcp"
)


def require_mcp() -> None:
    """Import the ``mcp`` SDK eagerly, raising a clear message on failure."""

    try:
        import mcp  # noqa: F401
        from mcp.server.fastmcp import FastMCP  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised in test
        raise SystemExit(f"{_INSTALL_HINT}\n(underlying ImportError: {exc})") from exc
