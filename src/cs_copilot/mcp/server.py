"""FastMCP server wiring for cs_copilot.

This is the only module that imports from ``mcp.server.fastmcp``. All other
modules in the package stay pure-Python so they can be unit-tested without
the optional ``[mcp]`` extra installed.

The server is intentionally a thin orchestrator: it instantiates the toolkit
singletons, wraps each via :mod:`cs_copilot.mcp.tool_adapter`, registers
prompts from :mod:`cs_copilot.mcp.prompts_registry`, and overrides resource
list / read on a FastMCP subclass. It never imports
``cs_copilot.agents.teams``, ``cs_copilot.agents.factories``, or any other
module that would launch the Agno multi-agent system.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

from .context import MCPAgentContext
from .lazy import require_mcp

logger = logging.getLogger(__name__)

_LOCAL_ALLOWED_HOSTS = ("127.0.0.1:*", "localhost:*", "[::1]:*")
_LOCAL_ALLOWED_ORIGINS = ("http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*")

SERVER_INSTRUCTIONS = (
    "cs_copilot MCP: external MCP client is the reasoning layer. "
    "Do not invoke the Agno team unless agno_team_run is enabled. "
    "Start with mcp_bootstrap and prompt cs_copilot_mcp_workflow; "
    "cs_copilot_workflow remains legacy. Fetch workflow/skills. "
    "Use chembl_prepare_retrieval / chemspace_plan_analysis before vague "
    "ChEMBL or chemical-space writes. Choose chembl_*, gtm_*, session_*, "
    "report_*, design, or other tools. Use llm_* for needs_external_llm. "
    "Use chembl_retrieval_judge for row filtering. Review write actions."
)


def build_server(
    ctx: MCPAgentContext,
    *,
    include_tools: bool = True,
    include_chatgpt_compat: bool = True,
    include_prompts: bool = True,
    include_resources: bool = True,
    enable_agno_team_tool: bool = False,
    host: str = "127.0.0.1",
    port: int = 8000,
    mount_path: str = "/",
    sse_path: str = "/sse",
    message_path: str = "/messages/",
    streamable_http_path: str = "/mcp",
    json_response: bool = False,
    stateless_http: bool = False,
    log_level: str = "INFO",
    auth_token: str | None = None,
    auth_token_client_id: str = "cs_copilot-mcp-client",
    auth_token_scopes: list[str] | None = None,
    auth_issuer_url: str | None = None,
    auth_resource_url: str | None = None,
    allowed_hosts: list[str] | None = None,
    allowed_origins: list[str] | None = None,
    disable_dns_rebinding_protection: bool = False,
) -> Any:
    """Create and configure the FastMCP server instance."""

    require_mcp()

    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.prompts.base import Prompt
    from mcp.server.lowlevel.helper_types import ReadResourceContents
    from mcp.server.transport_security import TransportSecuritySettings
    from mcp.types import Resource as MCPResource
    from mcp.types import ToolAnnotations

    from . import resources as _resources

    class _CsCopilotFastMCP(FastMCP):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._cs_include_resources = include_resources

        async def list_resources(self) -> list[MCPResource]:  # type: ignore[override]
            if not self._cs_include_resources:
                return await super().list_resources()
            return [
                MCPResource(
                    uri=entry.uri,  # type: ignore[arg-type]
                    name=entry.name,
                    description=f"cs_copilot session artifact ({entry.mime_type}).",
                    mimeType=entry.mime_type,
                )
                for entry in _resources.list_entries()
            ]

        async def read_resource(self, uri: Any):  # type: ignore[override]
            if not self._cs_include_resources:
                return await super().read_resource(uri)
            uri_str = str(uri)
            mime = _resources.resource_mime(uri_str)
            if _resources.is_text_resource(uri_str) or mime == "application/json":
                return [ReadResourceContents(content=_resources.read_text(uri_str), mime_type=mime)]
            return [ReadResourceContents(content=_resources.read_blob(uri_str), mime_type=mime)]

    auth_settings = None
    token_verifier = None
    if auth_token:
        from .auth import StaticBearerTokenVerifier, build_auth_settings

        protected_endpoint = auth_resource_url or _endpoint_url(
            host=host,
            port=port,
            path=streamable_http_path,
        )
        auth_settings = build_auth_settings(
            issuer_url=auth_issuer_url or protected_endpoint,
            resource_server_url=protected_endpoint,
            required_scopes=auth_token_scopes or (),
        )
        token_verifier = StaticBearerTokenVerifier(
            expected_token=auth_token,
            client_id=auth_token_client_id,
            scopes=tuple(auth_token_scopes or ()),
        )

    server = _CsCopilotFastMCP(
        name="cs_copilot",
        instructions=SERVER_INSTRUCTIONS,
        host=host,
        port=port,
        mount_path=mount_path,
        sse_path=sse_path,
        message_path=message_path,
        streamable_http_path=streamable_http_path,
        json_response=json_response,
        stateless_http=stateless_http,
        log_level=log_level.upper(),
        auth=auth_settings,
        token_verifier=token_verifier,
        transport_security=_transport_security_settings(
            TransportSecuritySettings,
            host=host,
            allowed_hosts=allowed_hosts,
            allowed_origins=allowed_origins,
            disable_dns_rebinding_protection=disable_dns_rebinding_protection,
        ),
    )

    if include_tools:
        if include_chatgpt_compat:
            _register_chatgpt_compat_tools(server, ToolAnnotations)
        _register_tools(server, ctx, ToolAnnotations)
        if enable_agno_team_tool:
            _register_agno_team_tool(server, ctx, ToolAnnotations)
    if include_prompts:
        _register_prompts(server, Prompt)

    return server


def _register_tools(server: Any, ctx: MCPAgentContext, tool_annotations_cls: Any) -> None:
    from .tool_adapter import build_tool
    from .tools_registry import iter_specs

    registered: set[str] = set()
    for spec in iter_specs():
        if spec.mcp_name in registered:
            logger.warning("Duplicate MCP tool name skipped: %s", spec.mcp_name)
            continue
        try:
            instance = spec.toolkit_factory()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Skipping MCP tool %s: factory %s raised %s",
                spec.mcp_name,
                spec.toolkit_factory,
                exc,
            )
            continue
        tool_fn = build_tool(spec, instance, ctx)
        server.add_tool(
            tool_fn,
            name=spec.mcp_name,
            description=spec.summary or tool_fn.__doc__,
            annotations=_tool_annotations(
                tool_annotations_cls,
                read_only=spec.read_only,
                destructive=spec.destructive,
                open_world=spec.open_world,
            ),
            structured_output=False,
        )
        registered.add(spec.mcp_name)
    logger.info("Registered %d MCP tools", len(registered))


def _dedupe(values: tuple[str, ...] | list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _transport_security_settings(
    TransportSecuritySettings: Any,
    *,
    host: str,
    allowed_hosts: list[str] | None,
    allowed_origins: list[str] | None,
    disable_dns_rebinding_protection: bool,
) -> Any | None:
    if disable_dns_rebinding_protection:
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)

    if not allowed_hosts and not allowed_origins:
        return None

    host_defaults = _LOCAL_ALLOWED_HOSTS if host in {"127.0.0.1", "localhost", "::1"} else ()
    origin_defaults = _LOCAL_ALLOWED_ORIGINS if host in {"127.0.0.1", "localhost", "::1"} else ()
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_dedupe((*host_defaults, *(allowed_hosts or ()))),
        allowed_origins=_dedupe((*origin_defaults, *(allowed_origins or ()))),
    )


def _tool_annotations(
    tool_annotations_cls: Any,
    *,
    read_only: bool,
    destructive: bool = False,
    open_world: bool = False,
) -> Any:
    return tool_annotations_cls(
        readOnlyHint=read_only,
        destructiveHint=destructive,
        openWorldHint=open_world,
    )


def _endpoint_url(*, host: str, port: int, path: str) -> str:
    connect_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    quoted_path = quote(path if path.startswith("/") else f"/{path}", safe="/")
    return f"http://{connect_host}:{port}{quoted_path}"


def _register_chatgpt_compat_tools(server: Any, tool_annotations_cls: Any) -> None:
    from .chatgpt_compat import fetch, search

    server.add_tool(
        search,
        name="search",
        description=(
            "Search the cs_copilot MCP tool, prompt, skill, workflow, and active "
            "session artifact catalogs. This read-only compatibility tool is "
            "intended for ChatGPT data-only apps, company knowledge, and deep research."
        ),
        annotations=_tool_annotations(tool_annotations_cls, read_only=True),
        structured_output=True,
    )
    server.add_tool(
        fetch,
        name="fetch",
        description=(
            "Fetch a cs_copilot MCP search result by id. Returns tool/prompt "
            "documentation or the text content of a session artifact."
        ),
        annotations=_tool_annotations(tool_annotations_cls, read_only=True),
        structured_output=True,
    )


def _register_agno_team_tool(server: Any, ctx: MCPAgentContext, tool_annotations_cls: Any) -> None:
    from .agno_delegate import build_agno_team_tool

    server.add_tool(
        build_agno_team_tool(ctx),
        name="agno_team_run",
        description=(
            "Private trusted-client escape hatch: delegate one prompt to the "
            "cs_copilot Agno team, using the configured Agno model. Disabled "
            "by default; prefer fine-grained MCP skills and tools for external clients."
        ),
        annotations=_tool_annotations(tool_annotations_cls, read_only=False),
        structured_output=True,
    )


def _register_prompts(server: Any, Prompt: Any) -> None:
    from .prompts_registry import iter_specs

    count = 0
    for spec in iter_specs():
        prompt = Prompt.from_function(
            spec.render,
            name=spec.mcp_name,
            description=spec.summary,
        )
        server.add_prompt(prompt)
        count += 1
    logger.info("Registered %d MCP prompts", count)
