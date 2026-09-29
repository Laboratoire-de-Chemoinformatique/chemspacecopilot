"""Runtime profiles, ad-hoc runs, and event labelling."""

from __future__ import annotations

from types import SimpleNamespace

from cs_copilot.execution.context import (
    AD_HOC_WORKFLOW_SLUGS,
    AGNO_RUNTIME,
    MCP_RUNTIME,
    BasicExecutionContext,
    ExecutionContext,
    is_ad_hoc_run,
    resolve_runtime,
)
from cs_copilot.execution.events import record_tool_progress
from cs_copilot.execution.runner import bind_sync_invoker, execute_sync
from cs_copilot.mcp.context import MCPAgentContext
from cs_copilot.storage import S3
from cs_copilot.workflows import list_workflows

from ._kernel_helpers import make_spec, payloads


class _Toolkit:
    def echo(self, text: str) -> str:
        return text


def test_runtime_resolution_defaults_to_mcp():
    assert resolve_runtime(MCPAgentContext()) is MCP_RUNTIME
    assert resolve_runtime(SimpleNamespace(session_state={})) is MCP_RUNTIME
    assert resolve_runtime(BasicExecutionContext(execution_runtime=AGNO_RUNTIME)) is AGNO_RUNTIME
    assert isinstance(MCPAgentContext(), ExecutionContext)


def test_ad_hoc_slugs_are_not_catalog_workflows():
    assert AD_HOC_WORKFLOW_SLUGS == {"mcp-session", "agno-session"}
    assert not {workflow.slug for workflow in list_workflows()} & AD_HOC_WORKFLOW_SLUGS
    assert is_ad_hoc_run(SimpleNamespace(workflow_slug="agno-session"))
    assert not is_ad_hoc_run(SimpleNamespace(workflow_slug="chembl-to-gtm-report"))


def test_agno_session_runs_are_exempt_like_mcp_sessions(bound_context):
    ctx = bound_context("context-agno-exempt")
    assert ctx.run_context.run.status.value != "running"
    spec = make_spec("echo", _Toolkit)

    outcome = execute_sync(
        spec, ctx, {"text": "ok"}, invoke=bind_sync_invoker(spec, _Toolkit().echo)
    )

    assert outcome.ok


def test_mcp_runtime_label_through_execute_sync(bound_context):
    ctx = bound_context("context-mcp-label", workflow_slug="mcp-session", runtime=MCP_RUNTIME)
    spec = make_spec("echo", _Toolkit)

    execute_sync(spec, ctx, {"text": "ok"}, invoke=bind_sync_invoker(spec, _Toolkit().echo))

    assert {payload["runtime"] for payload in payloads(ctx, "tool_progress")} == {"mcp"}


def test_lazily_created_run_uses_the_runtime_ad_hoc_slug(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    S3.set_session_prefix("sessions/context-lazy-run")
    ctx = BasicExecutionContext(
        execution_runtime=AGNO_RUNTIME,
        session_state={
            "output_context": {
                "session_id": "context-lazy-run",
                "run_id": "lazy-run",
                "trace_id": "trace-lazy",
            }
        },
    )

    event_path = record_tool_progress(
        ctx=ctx,
        tool_name="kernel_echo",
        trace={"trace_id": "trace-lazy", "span_id": "span-lazy"},
        stage="started",
        attempt=0,
        max_attempts=1,
    )

    assert event_path is not None
    assert ctx.run_context.run.workflow_slug == "agno-session"
    [payload] = payloads(ctx, "tool_progress")
    assert payload["runtime"] == "agno"
