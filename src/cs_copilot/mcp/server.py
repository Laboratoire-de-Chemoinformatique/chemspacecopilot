"""FastMCP server wiring for ChemSpace Copilot.

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

from .context import MCPAgentContext
from .lazy import require_mcp

logger = logging.getLogger(__name__)


def build_server(
    ctx: MCPAgentContext,
    *,
    include_tools: bool = True,
    include_prompts: bool = True,
    include_resources: bool = True,
) -> Any:
    """Create and configure the FastMCP server instance."""

    require_mcp()

    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.prompts.base import Prompt
    from mcp.server.lowlevel.helper_types import ReadResourceContents
    from mcp.types import Resource as MCPResource

    from . import resources as _resources

    class _ChemSpaceFastMCP(FastMCP):
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
                    description=f"ChemSpace session artifact ({entry.mime_type}).",
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

    server = _ChemSpaceFastMCP(
        name="chemspace-copilot",
        instructions=(
            "ChemSpace Copilot exposes ChemSpace chemistry, chemography, "
            "ChEMBL, design, and reporting toolkits as MCP tools. Use the "
            "`chemspace_workflow` prompt to adopt the canonical orchestration "
            "persona. Session artifacts written by tools appear as resources "
            "under cscopilot://session/."
        ),
    )

    if include_tools:
        _register_tools(server, ctx)
    if include_prompts:
        _register_prompts(server, Prompt)

    return server


def _register_tools(server: Any, ctx: MCPAgentContext) -> None:
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
            structured_output=False,
        )
        registered.add(spec.mcp_name)
    logger.info("Registered %d MCP tools", len(registered))


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
