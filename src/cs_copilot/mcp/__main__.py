"""Entry points for the cs_copilot MCP server.

Run local stdio clients with::

    cscopilot-mcp [--session-id SID] [--workflow-slug SLUG]

Run a remote HTTP endpoint for ChatGPT / browser-hosted MCP clients with::

    cscopilot-mcp-serve --host 127.0.0.1 --port 8000
    cscopilot-mcp --transport streamable-http --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import os
from typing import Sequence
from urllib.parse import quote

from .auth import DEFAULT_AUTH_CLIENT_ID, DEFAULT_AUTH_TOKEN_ENV
from .lazy import require_mcp
from .session import BootstrapConfig, apply_session_id, bootstrap, configure_logging

_TRANSPORTS = ("stdio", "sse", "streamable-http")
_LLM_POLICIES = ("external", "agno-model", "disabled")


def _parse_args(
    argv: Sequence[str] | None,
    *,
    default_transport: str = "stdio",
    prog: str = "cscopilot-mcp",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "cs_copilot MCP server. Exposes cs_copilot toolkits, "
            "prompts, and session artifacts to external MCP clients over "
            "stdio, SSE, or streamable HTTP."
        ),
    )
    parser.add_argument(
        "--transport",
        default=default_transport,
        choices=_TRANSPORTS,
        help=(
            "MCP transport to serve. Use stdio for local clients such as Codex "
            "or Claude Code; use streamable-http or sse for ChatGPT remote apps."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for HTTP transports. Use 0.0.0.0 only behind trusted access control.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP transports.",
    )
    parser.add_argument(
        "--mount-path",
        default="/",
        help="Mount path used when composing SSE message endpoints.",
    )
    parser.add_argument(
        "--sse-path",
        default="/sse",
        help="SSE endpoint path for --transport sse.",
    )
    parser.add_argument(
        "--message-path",
        default="/messages/",
        help="SSE client message endpoint path for --transport sse.",
    )
    parser.add_argument(
        "--streamable-http-path",
        default="/mcp",
        help="HTTP endpoint path for --transport streamable-http.",
    )
    parser.add_argument(
        "--json-response",
        action="store_true",
        help="Use JSON responses instead of SSE streams for streamable HTTP where supported.",
    )
    parser.add_argument(
        "--stateless-http",
        action="store_true",
        help="Create a fresh streamable-HTTP transport per request.",
    )
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help=(
            "Host header allowed by MCP DNS-rebinding protection. Supply the "
            "public HTTPS host when serving behind a reverse proxy. Can be "
            "supplied multiple times."
        ),
    )
    parser.add_argument(
        "--allowed-origin",
        action="append",
        default=[],
        help=(
            "Origin header allowed by MCP DNS-rebinding protection. Supply the "
            "browser/client origin, for example https://chatgpt.com. Can be "
            "supplied multiple times."
        ),
    )
    parser.add_argument(
        "--disable-dns-rebinding-protection",
        action="store_true",
        help=(
            "Disable MCP DNS-rebinding protection. Use only behind a trusted "
            "proxy or tunnel that performs equivalent Host/Origin checks."
        ),
    )
    parser.add_argument(
        "--auth-token-env",
        default=DEFAULT_AUTH_TOKEN_ENV,
        help=(
            "Environment variable containing a bearer token required by HTTP "
            "transports. Leave unset for no built-in bearer auth."
        ),
    )
    parser.add_argument(
        "--auth-token",
        default=None,
        help=(
            "Bearer token required by HTTP transports. Prefer --auth-token-env "
            "so the token is not visible in process listings."
        ),
    )
    parser.add_argument(
        "--auth-client-id",
        default=DEFAULT_AUTH_CLIENT_ID,
        help="Client id exposed to the MCP SDK after bearer-token verification.",
    )
    parser.add_argument(
        "--auth-scope",
        action="append",
        default=[],
        help="Required bearer-token scope. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--auth-issuer-url",
        default=None,
        help="Issuer URL advertised in MCP protected-resource metadata.",
    )
    parser.add_argument(
        "--auth-resource-url",
        default=None,
        help="Public MCP resource URL advertised in auth metadata.",
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
        help=(
            "Workflow slug used only to label the session output layout. It "
            "does not fetch, inject, or execute a workflow contract."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=("debug", "info", "warning", "error"),
        help="Logger level for the MCP process (logs to stderr).",
    )
    parser.add_argument(
        "--llm-policy",
        default=os.getenv("CS_COPILOT_MCP_LLM_POLICY", "external"),
        choices=_LLM_POLICIES,
        help=(
            "MCP LLM behavior. 'external' stores pending LLM tasks for the MCP "
            "client to complete; 'agno-model' loads only the configured Agno "
            "model for in-process toolkit LLM calls; 'disabled' rejects "
            "LLM-dependent work."
        ),
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Skip registering MCP tools (useful for prompt/resource debugging).",
    )
    parser.add_argument(
        "--no-chatgpt-compat",
        action="store_true",
        help="Skip the read-only ChatGPT-compatible search/fetch tools.",
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
    parser.add_argument(
        "--enable-agno-team-tool",
        action="store_true",
        help=(
            "Register private agno_team_run delegation tool. Use only for "
            "trusted clients; default MCP mode keeps Agno reasoning disabled."
        ),
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    default_transport: str = "stdio",
    prog: str = "cscopilot-mcp",
) -> None:
    """Run the MCP server. Blocks until the client disconnects."""

    require_mcp()
    args = _parse_args(argv, default_transport=default_transport, prog=prog)

    config = BootstrapConfig(
        session_id=args.session_id,
        workflow_slug=args.workflow_slug,
        log_level=args.log_level,
        llm_policy=args.llm_policy,
    )
    configure_logging(config.log_level)
    apply_session_id(config.session_id)
    ctx = bootstrap(config)

    auth_token = None
    if args.transport != "stdio":
        auth_token = args.auth_token or os.getenv(args.auth_token_env)

    # Import build_server only after bootstrap so storage / context vars are
    # set up before any toolkit / FastMCP module is touched.
    from .server import build_server

    server = build_server(
        ctx,
        include_tools=not args.no_tools,
        include_chatgpt_compat=not args.no_chatgpt_compat,
        include_prompts=not args.no_prompts,
        include_resources=not args.no_resources,
        enable_agno_team_tool=args.enable_agno_team_tool,
        host=args.host,
        port=args.port,
        mount_path=args.mount_path,
        sse_path=args.sse_path,
        message_path=args.message_path,
        streamable_http_path=args.streamable_http_path,
        json_response=args.json_response,
        stateless_http=args.stateless_http,
        log_level=args.log_level,
        auth_token=auth_token,
        auth_token_client_id=args.auth_client_id,
        auth_token_scopes=args.auth_scope,
        auth_issuer_url=args.auth_issuer_url,
        auth_resource_url=args.auth_resource_url or _default_auth_resource_url(args),
        allowed_hosts=args.allowed_host,
        allowed_origins=args.allowed_origin,
        disable_dns_rebinding_protection=args.disable_dns_rebinding_protection,
    )

    if args.transport == "sse":
        server.run("sse", mount_path=args.mount_path)
    else:
        server.run(args.transport)


def _default_auth_resource_url(args: argparse.Namespace) -> str | None:
    if args.transport == "stdio":
        return None
    path = args.sse_path if args.transport == "sse" else args.streamable_http_path
    host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    quoted_path = quote(path if path.startswith("/") else f"/{path}", safe="/")
    return f"http://{host}:{args.port}{quoted_path}"


def serve_main(argv: Sequence[str] | None = None) -> None:
    """Entry point for ``cscopilot-mcp-serve`` remote HTTP serving."""

    main(argv, default_transport="streamable-http", prog="cscopilot-mcp-serve")


if __name__ == "__main__":
    main()
