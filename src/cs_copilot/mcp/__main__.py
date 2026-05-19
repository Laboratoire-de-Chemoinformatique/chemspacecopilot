"""Entry point for the ``cscopilot-mcp`` console script.

Run with::

    cscopilot-mcp [--session-id SID] [--workflow-slug SLUG] [--log-level LEVEL]

Or::

    python -m cs_copilot.mcp [--session-id SID]
"""

from __future__ import annotations

import argparse
from typing import Sequence

from .lazy import require_mcp
from .session import BootstrapConfig, apply_session_id, bootstrap, configure_logging


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cscopilot-mcp",
        description=(
            "ChemSpace Copilot MCP server (stdio). Exposes ChemSpace toolkits, "
            "prompts, and session artifacts to external MCP clients."
        ),
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Session id used as the storage prefix. Defaults to the SESSION_ID "
        "env var if set, otherwise auto-generated.",
    )
    parser.add_argument(
        "--workflow-slug",
        default=None,
        help="Workflow slug stored in the session output layout.",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=("debug", "info", "warning", "error"),
        help="Logger level for the MCP process (logs to stderr).",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Skip registering MCP tools (useful for prompt/resource debugging).",
    )
    parser.add_argument(
        "--no-prompts",
        action="store_true",
        help="Skip registering MCP prompts.",
    )
    parser.add_argument(
        "--no-resources",
        action="store_true",
        help="Skip exposing session artifacts as MCP resources.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the MCP server. Blocks until the client disconnects."""

    require_mcp()
    args = _parse_args(argv)

    config = BootstrapConfig(
        session_id=args.session_id,
        workflow_slug=args.workflow_slug,
        log_level=args.log_level,
    )
    configure_logging(config.log_level)
    apply_session_id(config.session_id)
    ctx = bootstrap(config)

    # Import build_server only after bootstrap so storage / context vars are
    # set up before any toolkit / FastMCP module is touched.
    from .server import build_server

    server = build_server(
        ctx,
        include_tools=not args.no_tools,
        include_prompts=not args.no_prompts,
        include_resources=not args.no_resources,
    )
    server.run("stdio")


if __name__ == "__main__":
    main()
