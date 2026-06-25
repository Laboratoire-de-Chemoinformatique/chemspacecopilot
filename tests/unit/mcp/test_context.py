"""Tests for the lightweight MCPAgentContext shim used by the MCP server."""

from cs_copilot.mcp.context import MCPAgentContext, get_current_context, set_current_context
from cs_copilot.mcp.llm import LLMBroker
from cs_copilot.tools.io.session_memory import update_state_targets


def test_default_fields():
    ctx = MCPAgentContext()
    assert ctx.name == "mcp-client"
    assert ctx.model is None
    assert ctx.llm_policy == "external"
    assert ctx.llm is None
    assert ctx.session_state == {}


def test_llm_broker_uses_session_state():
    ctx = MCPAgentContext()
    ctx.llm = LLMBroker(ctx)

    task = ctx.llm.create_task(
        task_type="unit.test",
        prompt_text="Return JSON.",
        output_schema={"type": "object"},
    )

    assert task["status"] == "pending"
    assert ctx.llm.get_task(task["task_id"])["prompt_text"] == "Return JSON."
    assert len(ctx.llm.list_tasks()) == 1


def test_session_state_mutation_is_independent_per_instance():
    a = MCPAgentContext()
    b = MCPAgentContext()
    a.session_state["dataset_id"] = "abc"
    assert "dataset_id" not in b.session_state


def test_update_state_targets_accepts_context_shim():
    ctx = MCPAgentContext()
    targets = update_state_targets(ctx, None)
    assert ctx.session_state in targets
    # `update_state_targets` should have initialised an empty dict, not None.
    assert isinstance(ctx.session_state, dict)


def test_set_and_get_current_context_round_trip():
    ctx = MCPAgentContext(name="round-trip")
    set_current_context(ctx)
    fetched = get_current_context()
    assert fetched is ctx
