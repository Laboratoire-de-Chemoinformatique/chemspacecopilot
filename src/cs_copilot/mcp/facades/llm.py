"""MCP facade for general LLM task lifecycle tools."""

from __future__ import annotations

import functools
from typing import Any

from ..errors import MCPToolError


def _broker(agent: Any | None) -> Any:
    broker = getattr(agent, "llm", None)
    if broker is None:
        raise MCPToolError("MCP LLM broker is not configured for this session.")
    return broker


class LLMFacade:
    """Expose broker task operations as MCP tools."""

    def create_task(
        self,
        task_type: str,
        prompt_text: str,
        prompt_name: str | None = None,
        input_payload: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        consumer_tool: str | None = None,
        metadata: dict[str, Any] | None = None,
        agent: Any | None = None,
    ) -> dict[str, Any]:
        """Create a pending external LLM task."""

        return _broker(agent).create_task(
            task_type=task_type,
            prompt_name=prompt_name,
            prompt_text=prompt_text,
            input_payload=input_payload,
            output_schema=output_schema,
            consumer_tool=consumer_tool,
            metadata=metadata,
        )

    def list_pending_tasks(
        self,
        task_type: str | None = None,
        agent: Any | None = None,
    ) -> list[dict[str, Any]]:
        """List pending LLM tasks for the active MCP session."""

        return _broker(agent).list_tasks(status="pending", task_type=task_type)

    def get_task(self, task_id: str, agent: Any | None = None) -> dict[str, Any]:
        """Return one LLM task by id."""

        return _broker(agent).get_task(task_id)

    def submit_task_result(
        self,
        task_id: str,
        result: Any | None = None,
        error: str | None = None,
        agent: Any | None = None,
    ) -> dict[str, Any]:
        """Complete or fail a pending LLM task."""

        return _broker(agent).submit_task_result(
            task_id=task_id,
            result=result,
            error=error,
        )

    def cancel_task(
        self,
        task_id: str,
        reason: str | None = None,
        agent: Any | None = None,
    ) -> dict[str, Any]:
        """Cancel a pending LLM task."""

        return _broker(agent).cancel_task(task_id=task_id, reason=reason)


@functools.lru_cache(maxsize=1)
def llm_facade() -> LLMFacade:
    return LLMFacade()
