"""General MCP LLM broker.

The broker gives MCP tools one shared way to request LLM work:

* ``external``: create a pending task for the MCP client to complete.
* ``agno-model``: allow toolkit code to use ``ctx.model`` directly.
* ``disabled``: reject LLM-dependent work with a clear protocol error.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Literal, Mapping

from ..errors import MCPToolError

LLMPolicy = Literal["external", "agno-model", "disabled"]
LLM_POLICIES: tuple[LLMPolicy, ...] = ("external", "agno-model", "disabled")
LLM_TASKS_KEY = "_mcp_llm_tasks"
LLM_TASK_ORDER_KEY = "_mcp_llm_task_order"


def normalize_llm_policy(value: str | None) -> LLMPolicy:
    """Normalize and validate an MCP LLM policy value."""

    normalized = str(value or "external").strip().lower().replace("_", "-")
    if normalized not in LLM_POLICIES:
        raise ValueError(
            f"Unsupported MCP LLM policy {value!r}. " f"Use one of: {', '.join(LLM_POLICIES)}."
        )
    return normalized  # type: ignore[return-value]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump())
        except Exception:
            pass
    return str(value)


class LLMBroker:
    """Task store and policy helper bound to one MCP agent context."""

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx

    @property
    def policy(self) -> LLMPolicy:
        return normalize_llm_policy(getattr(self._ctx, "llm_policy", "external"))

    def require_enabled(self, *, task_type: str) -> None:
        """Raise when an LLM-dependent task is disallowed by policy."""

        if self.policy == "disabled":
            raise MCPToolError(f"LLM task '{task_type}' is disabled by MCP llm_policy='disabled'.")

    def create_task(
        self,
        *,
        task_type: str,
        prompt_text: str,
        prompt_name: str | None = None,
        input_payload: Mapping[str, Any] | None = None,
        output_schema: Mapping[str, Any] | None = None,
        consumer_tool: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Create a pending external-LLM task in session state."""

        self.require_enabled(task_type=task_type)
        task_id = f"llm_{uuid.uuid4().hex[:12]}"
        task = {
            "task_id": task_id,
            "task_type": str(task_type),
            "prompt_name": prompt_name,
            "prompt_text": str(prompt_text),
            "input_payload": _json_safe(dict(input_payload or {})),
            "output_schema": _json_safe(dict(output_schema or {})),
            "consumer_tool": consumer_tool,
            "metadata": _json_safe(dict(metadata or {})),
            "status": "pending",
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "result": None,
            "error": None,
        }
        tasks = self._tasks()
        order = self._order()
        tasks[task_id] = task
        order.append(task_id)
        return dict(task)

    def list_tasks(
        self,
        *,
        status: str | None = "pending",
        task_type: str | None = None,
    ) -> list[Dict[str, Any]]:
        """List LLM tasks, newest last, optionally filtered by status/type."""

        tasks = self._tasks()
        selected: Iterable[Dict[str, Any]] = (
            tasks[task_id] for task_id in self._order() if task_id in tasks
        )
        if status:
            selected = (task for task in selected if task.get("status") == status)
        if task_type:
            selected = (task for task in selected if task.get("task_type") == task_type)
        return [dict(task) for task in selected]

    def get_task(self, task_id: str) -> Dict[str, Any]:
        """Return one task by id."""

        task = self._tasks().get(str(task_id))
        if task is None:
            raise MCPToolError(f"Unknown LLM task id: {task_id}")
        return dict(task)

    def submit_task_result(
        self,
        *,
        task_id: str,
        result: Any | None = None,
        error: str | None = None,
    ) -> Dict[str, Any]:
        """Mark a pending LLM task completed or errored."""

        task = self._mutable_task(task_id)
        if task.get("status") == "cancelled":
            raise MCPToolError(f"LLM task {task_id} is cancelled and cannot be completed.")
        task["status"] = "error" if error else "completed"
        task["result"] = _json_safe(result)
        task["error"] = str(error) if error else None
        task["updated_at"] = _now_iso()
        return dict(task)

    def cancel_task(self, *, task_id: str, reason: str | None = None) -> Dict[str, Any]:
        """Cancel a pending LLM task."""

        task = self._mutable_task(task_id)
        if task.get("status") == "completed":
            raise MCPToolError(f"LLM task {task_id} is completed and cannot be cancelled.")
        task["status"] = "cancelled"
        task["error"] = str(reason or "cancelled")
        task["updated_at"] = _now_iso()
        return dict(task)

    def _tasks(self) -> Dict[str, Dict[str, Any]]:
        tasks = self._ctx.session_state.setdefault(LLM_TASKS_KEY, {})
        if not isinstance(tasks, dict):
            raise MCPToolError("MCP LLM task store is corrupted: expected a dict.")
        return tasks

    def _order(self) -> list[str]:
        order = self._ctx.session_state.setdefault(LLM_TASK_ORDER_KEY, [])
        if not isinstance(order, list):
            raise MCPToolError("MCP LLM task order is corrupted: expected a list.")
        return order

    def _mutable_task(self, task_id: str) -> Dict[str, Any]:
        task = self._tasks().get(str(task_id))
        if task is None:
            raise MCPToolError(f"Unknown LLM task id: {task_id}")
        return task
