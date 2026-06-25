"""General LLM broker MCP tool specs."""

from __future__ import annotations

from typing import List

from ..facades.llm import llm_facade
from ..tool_adapter import ToolSpec

SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name="llm_create_task",
        toolkit_factory=llm_facade,
        method="create_task",
        summary=(
            "Create a general pending LLM task in the MCP session. Use this "
            "for workflow-specific reasoning requests that should be completed "
            "by the external MCP client."
        ),
    ),
    ToolSpec(
        mcp_name="llm_list_pending_tasks",
        toolkit_factory=llm_facade,
        method="list_pending_tasks",
        summary="List pending LLM tasks created by MCP tools in this session.",
        read_only=True,
    ),
    ToolSpec(
        mcp_name="llm_get_task",
        toolkit_factory=llm_facade,
        method="get_task",
        summary="Return the prompt, input payload, and output schema for one LLM task.",
        read_only=True,
    ),
    ToolSpec(
        mcp_name="llm_submit_task_result",
        toolkit_factory=llm_facade,
        method="submit_task_result",
        summary="Submit an external LLM result for a pending MCP LLM task.",
    ),
    ToolSpec(
        mcp_name="llm_cancel_task",
        toolkit_factory=llm_facade,
        method="cancel_task",
        summary="Cancel a pending MCP LLM task.",
    ),
]
