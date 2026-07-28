"""Tests for the MCP tool adapter — schema stripping, injection, errors."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pytest

from cs_copilot.mcp.context import MCPAgentContext, bind_active_task_scope
from cs_copilot.mcp.tool_adapter import ToolSpec, build_tool
from cs_copilot.storage import S3
from cs_copilot.workflows import (
    HandoffEnvelope,
    InvalidTransitionError,
    RunContext,
    TaskRecord,
    ToolError,
)


class _DummyToolkit:
    """Plain class used as a stand-in for an Agno-style toolkit instance."""

    def __init__(self) -> None:
        self.echo_calls = 0
        self.flaky_calls = 0
        self.structured_calls = 0
        self.write_calls = 0
        self.write_args: dict[str, Any] = {}

    def echo(
        self,
        text: str,
        count: int = 1,
        flag: bool = True,
        agent: Optional[Any] = None,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Repeat *text* *count* times and mutate session state."""

        self.echo_calls += 1
        assert agent is not None and session_state is not None
        agent.session_state["last_text"] = text
        session_state["last_count"] = count
        return text * count

    def boom(self) -> str:
        raise RuntimeError("boom")

    def big_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({"x": list(range(5))})

    def secret_echo(self, api_key: str, token_value: str) -> dict[str, str]:
        return {"status": "ok", "api_key": api_key, "token_value": token_value}

    def flaky(self) -> str:
        self.flaky_calls += 1
        if self.flaky_calls == 1:
            raise ConnectionError("temporary outage")
        return "recovered"

    def structured(self) -> dict[str, Any]:
        self.structured_calls += 1
        return {"can_proceed": True, "plan": ["retrieve", "analyze"]}

    def legacy_text(self, text: str) -> str:
        return text

    def write_output(
        self,
        output_path: str | None = None,
        options: Any | None = None,
    ) -> dict[str, Any]:
        self.write_calls += 1
        self.write_args = {"output_path": output_path, "options": options}
        if output_path is not None:
            with S3.open(output_path, "w") as handle:
                handle.write("bounded")
        return dict(self.write_args)

    def operation(
        self,
        operation: str,
        operation_parameters: Any | None = None,
        function_parameters: Any | None = None,
        dataframe_name: str | None = None,
    ) -> dict[str, Any]:
        self.write_calls += 1
        self.write_args = {
            "operation": operation,
            "operation_parameters": operation_parameters,
            "function_parameters": function_parameters,
            "dataframe_name": dataframe_name,
        }
        return dict(self.write_args)

    def create(
        self,
        create_using_function: str,
        function_parameters: Any | None = None,
    ) -> dict[str, Any]:
        self.write_calls += 1
        self.write_args = {
            "create_using_function": create_using_function,
            "function_parameters": function_parameters,
        }
        return dict(self.write_args)

    def read_dataset(self, path_to_dataset: str) -> dict[str, str]:
        self.structured_calls += 1
        return {"path_to_dataset": path_to_dataset}

    def save_named(self, dataset_name: str, gtm_name: str) -> dict[str, str]:
        self.write_calls += 1
        self.write_args = {"dataset_name": dataset_name, "gtm_name": gtm_name}
        return dict(self.write_args)


class _PublishingTextToolkit:
    def __init__(self, paths: list[str], content: str) -> None:
        self.paths = paths
        self.content = content

    def legacy_text(self, text: str) -> str:
        for path in self.paths:
            with S3.open(path, "w") as handle:
                handle.write(self.content)
        return text


class _HiddenWriteToolkit:
    def write_hidden(self, path: str) -> dict[str, bool]:
        with S3.open(path, "w") as handle:
            handle.write("hidden")
        return {"ok": True}


class _ConcurrentToolkit:
    def __init__(self) -> None:
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def slow_echo(self, text: str) -> str:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return text


class _ConcurrentArtifactToolkit:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def write_output(
        self,
        output_path: str,
        content: str,
        wait_for_release: bool = False,
    ) -> dict[str, str]:
        if wait_for_release:
            self.started.set()
            await self.release.wait()
        with S3.open(output_path, "w") as handle:
            handle.write(content)
        return {"output_path": output_path}


class _BlockingToolkit:
    def __init__(self) -> None:
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()

    def mutate(self) -> str:
        self.started.set()
        self.release.wait(timeout=2)
        self.calls += 1
        return "done"


class _AsyncToolkit:
    async def slow(self) -> str:
        await asyncio.sleep(1)
        return "too late"


class _SlowAsyncToolkit:
    def __init__(self) -> None:
        self.started = False

    async def structured(self) -> dict[str, bool]:
        self.started = True
        await asyncio.sleep(0.75)
        return {"can_proceed": True}


class _WaitingCatalogToolkit:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def structured(self) -> dict[str, Any]:
        self.started.set()
        await self.release.wait()
        return {"can_proceed": True, "plan": ["retrieve"]}


class _SlowSyncToolkit:
    def __init__(self) -> None:
        self.started = False

    def structured(self) -> dict[str, bool]:
        self.started = True
        time.sleep(0.75)
        return {"can_proceed": True}


class _SymlinkSwapToolkit:
    def __init__(self, external: Path) -> None:
        self.external = external

    def write_output(self, output_path: str) -> str:
        destination = Path(S3.path(output_path))
        destination.parent.rmdir()
        destination.parent.symlink_to(self.external, target_is_directory=True)
        with S3.open(output_path, "w") as handle:
            handle.write("escaped")
        return output_path


class _SessionPointerSwapToolkit:
    """Mutate live pointers immediately before consuming adapter-injected values."""

    def __init__(self, live_state: dict[str, Any], replacement: str) -> None:
        self.live_state = live_state
        self.replacement = replacement

    def load_dataframe_from_session(
        self,
        dataframe_name: str,
        session_key: str,
        session_state: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        self.live_state["dataset"]["csv_path"] = self.replacement
        assert session_state is not None
        source = str(session_state[session_key])
        with S3.open(source, "r") as handle:
            content = handle.read()
        return {"dataframe_name": dataframe_name, "source": source, "content": content}

    def load_candidate_set_artifact(
        self,
        reference: str = "top candidates",
        include_candidates: bool = False,
        session_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        pointer = self.live_state["candidate_pointer"]
        pointer["artifact_rel_path"] = self.replacement
        assert session_state is not None
        pinned_pointer = session_state[reference]
        return {
            "reference": reference,
            "consumed_path": pinned_pointer["artifact_rel_path"],
            "include_candidates": include_candidates,
            "has_session_state": True,
        }

    def load_peptide_design_candidates(
        self,
        reference: str = "designed_peptides",
        include_candidates: bool = True,
        session_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.load_candidate_set_artifact(
            reference=reference,
            include_candidates=include_candidates,
            session_state=session_state,
        )

    def materialize_candidate_set_dataset(
        self,
        reference: str = "generated compounds",
        top_n: int | None = None,
        session_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        memory = self.live_state["session_objects"]
        candidate_set_id = self.live_state["candidate_alias"]["candidate_set_id"]
        memory["candidate_sets"][candidate_set_id]["artifact_rel_path"] = self.replacement
        return {
            "reference": reference,
            "top_n": top_n,
            "has_session_state": session_state is not None,
        }

    def save_rich(
        self,
        title: str,
        figures: list[dict[str, Any]] | None = None,
        session_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from cs_copilot.tools.io.figure_metadata import session_figure_metadata

        assert figures and session_state is not None
        figure_id = figures[0]["figure_id"]
        live_record = self.live_state["session_objects"]["figures"][figure_id]
        live_record["paths"]["png_path"] = self.replacement
        metadata = session_figure_metadata(session_state, figure_id)
        session_state["report_snapshot_write"] = "preserved"
        return {"title": title, "png_path": metadata["paths"]["png_path"]}


def _spec(method: str, **kwargs: Any) -> ToolSpec:
    return ToolSpec(
        mcp_name=f"dummy_{method}",
        toolkit_factory=lambda: _DummyToolkit(),
        method=method,
        summary=f"Dummy {method}",
        **kwargs,
    )


def test_adapter_strips_injected_params_from_public_signature():
    ctx = MCPAgentContext()
    tool = build_tool(_spec("echo"), _DummyToolkit(), ctx)
    sig = inspect.signature(tool)
    assert "agent" not in sig.parameters
    assert "session_state" not in sig.parameters
    assert "text" in sig.parameters
    assert "count" in sig.parameters


def test_adapter_injects_agent_and_session_state():
    ctx = MCPAgentContext()
    instance = _DummyToolkit()
    tool = build_tool(_spec("echo"), instance, ctx)
    result = asyncio.run(tool(text="ab", count=3))
    assert result["status"] == "success"
    assert result["data"] == "ababab"
    # State mutated via both the agent and session_state injection points.
    assert ctx.session_state["last_text"] == "ab"
    assert ctx.session_state["last_count"] == 3


def test_adapter_routes_worker_process_specs(monkeypatch):
    ctx = MCPAgentContext(session_state={"existing": True})
    instance = _DummyToolkit()
    calls: list[tuple[str, dict[str, Any], MCPAgentContext]] = []

    def fake_run_tool_job(
        spec: ToolSpec,
        kwargs: Dict[str, Any],
        job_ctx: MCPAgentContext,
        *,
        before_state_merge=None,
        defer_commit=False,
    ):
        from cs_copilot.mcp.jobs import DeferredToolJob

        calls.append((spec.mcp_name, dict(kwargs), job_ctx))
        if before_state_merge is not None:
            before_state_merge()
        if defer_commit:
            return DeferredToolJob(
                result="from worker",
                ctx=job_ctx,
                base_session_state=dict(job_ctx.session_state),
                worker_session_state={
                    **job_ctx.session_state,
                    "worker_ran": True,
                },
                tool_name=spec.mcp_name,
                retryable=spec.idempotent,
                publications={},
                write_boundary=None,
                staging_id=None,
            )
        job_ctx.session_state["worker_ran"] = True
        return "from worker"

    import cs_copilot.mcp.jobs as jobs

    monkeypatch.setattr(jobs, "run_tool_job", fake_run_tool_job)
    spec = _spec("echo", run_in_worker_process=True, forces={"flag": False})
    tool = build_tool(spec, instance, ctx)
    result = asyncio.run(tool(text="ab", count=2))

    assert result["status"] == "success"
    assert result["data"] == "from worker"
    assert calls == [(spec.mcp_name, {"text": "ab", "count": 2, "flag": False}, ctx)]
    assert ctx.session_state["worker_ran"] is True
    assert "last_text" not in ctx.session_state


def test_adapter_rejects_uncancellable_sync_timeout():
    with pytest.raises(ValueError, match="cannot safely cancel"):
        build_tool(
            _spec("echo", timeout_s=0.01),
            _DummyToolkit(),
            MCPAgentContext(),
        )


def test_adapter_enforces_timeout_for_async_method():
    instance = _AsyncToolkit()
    spec = ToolSpec(
        mcp_name="dummy_async_timeout",
        toolkit_factory=_AsyncToolkit,
        method="slow",
        summary="Slow async call.",
        timeout_s=0.01,
    )

    result = asyncio.run(build_tool(spec, instance, MCPAgentContext())())

    assert result["status"] == "error"
    assert result["error"]["code"] == "timeout"


def test_adapter_waits_for_sync_mutation_before_propagating_cancellation():
    instance = _BlockingToolkit()
    spec = ToolSpec(
        mcp_name="dummy_blocking_mutation",
        toolkit_factory=_BlockingToolkit,
        method="mutate",
        summary="Blocking mutation.",
        write_scope="session",
    )
    tool = build_tool(spec, instance, MCPAgentContext())

    async def exercise() -> None:
        task = asyncio.create_task(tool())
        while not instance.started.is_set():
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        assert instance.calls == 0
        instance.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert instance.calls == 1


def test_deferred_worker_lease_survives_repeated_cancellation(tmp_path, monkeypatch):
    from cs_copilot.mcp.jobs import DeferredToolJob
    from cs_copilot.mcp.tool_adapter import _run_deferred_worker_to_completion

    monkeypatch.chdir(tmp_path)
    S3.set_session_prefix("sessions/repeated-worker-cancel")
    started = threading.Event()
    release = threading.Event()
    ctx = MCPAgentContext(session_state={})
    job = DeferredToolJob(
        result={"ok": True},
        ctx=ctx,
        base_session_state={},
        worker_session_state={},
        tool_name="dummy_worker",
        retryable=False,
        publications={},
        write_boundary="workflows/run-1",
        staging_id="job-repeated-cancel",
    )

    def finish_worker() -> DeferredToolJob:
        started.set()
        release.wait(timeout=2)
        return job

    async def exercise() -> None:
        task = asyncio.create_task(_run_deferred_worker_to_completion(finish_worker))
        while not started.is_set():
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert job.settled is True


def test_adapter_forces_override_kwargs_and_hides_them():
    ctx = MCPAgentContext()
    instance = _DummyToolkit()
    spec = _spec("echo", forces={"flag": False})
    tool = build_tool(spec, instance, ctx)
    sig = inspect.signature(tool)
    assert "flag" not in sig.parameters


def test_adapter_normalizes_exceptions_into_error_envelope():
    ctx = MCPAgentContext()
    instance = _DummyToolkit()
    tool = build_tool(_spec("boom"), instance, ctx)
    result = asyncio.run(tool())
    assert result["status"] == "error"
    assert result["error"]["code"] == "internal"
    assert result["error"]["retryable"] is False
    assert "boom" in result["error"]["message"]


def test_adapter_coerces_dataframe_return():
    ctx = MCPAgentContext()
    instance = _DummyToolkit()
    tool = build_tool(_spec("big_dataframe"), instance, ctx)
    result = asyncio.run(tool())
    assert isinstance(result, dict)
    assert result["data"]["row_count"] == 5
    assert result["data"]["columns"] == ["x"]


def _bind_run(ctx: MCPAgentContext, session_name: str) -> RunContext:
    runtime = RunContext.create(
        "mcp-session",
        session_state=ctx.session_state,
        run_id=session_name,
    )
    ctx.run_context = runtime
    return runtime


def _bind_catalog_preflight(
    ctx: MCPAgentContext,
    session_name: str,
    *,
    max_tool_calls: int,
    timeout_seconds: float,
) -> RunContext:
    runtime = RunContext.create(
        "chembl-to-gtm-report",
        session_state=ctx.session_state,
        run_id=session_name,
    )
    ctx.run_context = runtime
    runtime.transition_run("running")
    runtime.add_task(
        TaskRecord(
            task_id="chembl-preflight",
            role="chembl_downloader",
            profile="chembl-retrieval",
            step="Run ChEMBL retrieval preflight.",
        )
    )
    request_path = runtime.layout.artifact_rel_path("inputs/retrieval-request.json")
    with S3.open(request_path, "w") as handle:
        json.dump({"request": "retrieve CDK2 inhibitors"}, handle)
    request = runtime.register_artifact(
        "inputs/retrieval-request.json",
        artifact_type="retrieval_request",
        mime_type="application/json",
        producer_tool="workflow_start_run",
    )
    runtime.record_handoff(
        HandoffEnvelope.create(
            run_id=runtime.run.run_id,
            workflow_slug=runtime.run.workflow_slug,
            task_id="chembl-preflight",
            sender_role="supervisor",
            receiver_role="chembl_downloader",
            objective="Validate the persisted retrieval request.",
            required_capabilities=("chembl_prepare_retrieval",),
            acceptance_criteria=(
                "Preflight either permits retrieval or records all required "
                "clarification questions.",
            ),
            input_artifact_ids=(request.artifact_id,),
            expected_output_artifacts=("retrieval_plan",),
            budget={
                "max_tokens": 1_000,
                "max_tool_calls": max_tool_calls,
                "timeout_seconds": timeout_seconds,
            },
        )
    )
    runtime.transition_task("chembl-preflight", "running")
    bind_active_task_scope(
        ctx.session_state,
        runtime.run.tasks["chembl-preflight"],
        run=runtime.run,
    )
    return runtime


def _append_persisted_catalog_tool_start(
    runtime: RunContext,
    *,
    span_id: str,
) -> None:
    task = runtime.run.tasks["chembl-preflight"]
    handoff = next(item for item in reversed(runtime.run.handoffs) if item.task_id == task.task_id)
    runtime.append_event(
        "tool_progress",
        {
            "runtime": "mcp",
            "session_id": runtime.run.session_id,
            "run_id": runtime.run.run_id,
            "workflow_slug": runtime.run.workflow_slug,
            "trace_id": runtime.run.trace_id,
            "span_id": span_id,
            "parent_span_id": None,
            "tool_name": "chembl_prepare_retrieval",
            "task_id": task.task_id,
            "role": task.role,
            "profile": task.profile,
            "stage": "started",
            "attempt": 0,
            "max_attempts": 1,
            "cached": False,
            "task_attempt": task.attempts,
            "handoff_id": handoff.handoff_id,
        },
    )


def _tool_payloads(ctx: MCPAgentContext):
    return [
        event.payload
        for event in ctx.run_context.events
        if event.event_type == "tool_call_recorded"
    ]


def _progress_payloads(ctx: MCPAgentContext):
    return [
        event.payload for event in ctx.run_context.events if event.event_type == "tool_progress"
    ]


def test_adapter_writes_success_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_name = "manifest-success"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    _bind_run(ctx, session_name)
    ctx.session_state.update(
        {
            "active_task_id": "analyze",
            "active_role": "gtm_agent",
            "active_profile": "gtm-analysis",
            "mcp_profile": "standard",
        }
    )
    tool = build_tool(_spec("echo"), _DummyToolkit(), ctx)

    asyncio.run(tool(text="ab", count=2))

    payloads = _tool_payloads(ctx)
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["runtime"] == "mcp"
    assert payload["tool_name"] == "dummy_echo"
    assert payload["status"] == "success"
    assert payload["task_id"] == "analyze"
    assert payload["role"] == "gtm_agent"
    assert payload["profile"] == "gtm-analysis"
    assert payload["attempts"] == 1
    assert payload["retries"] == 0
    assert payload["public_args"]["text"] == "ab"
    assert payload["output_summary"]["type"] == "dict"


def test_manifest_scope_uses_authoritative_task_assignment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_name = "manifest-authoritative-task"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    runtime = _bind_run(ctx, session_name)
    runtime.transition_run("running")
    runtime.add_task(
        TaskRecord(
            task_id="analyze",
            role="gtm_agent",
            profile="gtm-analysis",
            step="Analyze",
        )
    )
    runtime.transition_task("analyze", "running")
    ctx.session_state.update(
        {
            "active_task_id": "analyze",
            "active_role": "forged_role",
            "active_profile": "forged-profile",
        }
    )

    asyncio.run(build_tool(_spec("echo"), _DummyToolkit(), ctx)(text="result"))

    payload = _tool_payloads(ctx)[0]
    assert payload["task_id"] == "analyze"
    assert payload["role"] == "gtm_agent"
    assert payload["profile"] == "gtm-analysis"


def test_invocation_scope_survives_concurrent_active_task_switch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_name = "manifest-concurrent-task"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    runtime = _bind_run(ctx, session_name)
    runtime.transition_run("running")
    for task_id, role, profile in (
        ("first", "gtm_agent", "gtm-analysis"),
        ("second", "report_generator", "reporting"),
    ):
        runtime.add_task(
            TaskRecord(
                task_id=task_id,
                role=role,
                profile=profile,
                step=f"Run {task_id}.",
            )
        )
        runtime.transition_task(task_id, "running")
    ctx.session_state.update(
        {
            "active_task_id": "first",
            "active_role": "gtm_agent",
            "active_profile": "gtm-analysis",
        }
    )
    instance = _ConcurrentToolkit()
    spec = ToolSpec(
        mcp_name="dummy_concurrent_scope",
        toolkit_factory=_ConcurrentToolkit,
        method="slow_echo",
        summary="Concurrent scoped output.",
        result_artifact_type="analysis_result",
    )
    tool = build_tool(spec, instance, ctx)

    async def exercise():
        pending = asyncio.create_task(tool(text="result"))
        await instance.started.wait()
        ctx.session_state.update(
            {
                "active_task_id": "second",
                "active_role": "report_generator",
                "active_profile": "reporting",
            }
        )
        instance.release.set()
        return await pending

    result = asyncio.run(exercise())

    artifact = runtime.run.artifacts[result["artifact_ids"][0]]
    assert artifact.producer_task_id == "first"
    payload = _tool_payloads(ctx)[0]
    assert payload["task_id"] == "first"
    assert payload["role"] == "gtm_agent"
    assert payload["profile"] == "gtm-analysis"


def test_adapter_writes_error_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_name = "manifest-error"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    _bind_run(ctx, session_name)
    tool = build_tool(_spec("boom"), _DummyToolkit(), ctx)

    result = asyncio.run(tool())

    assert result["status"] == "error"
    payloads = _tool_payloads(ctx)
    assert len(payloads) == 1
    assert payloads[0]["status"] == "error"
    assert "boom" in payloads[0]["error"]


def test_session_write_scope_rewrites_relative_output_into_active_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USE_S3", "false")
    session_name = "write-boundary"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    runtime = _bind_run(ctx, session_name)
    instance = _DummyToolkit()
    tool = build_tool(
        _spec("write_output", write_scope="session"),
        instance,
        ctx,
    )

    result = asyncio.run(tool(output_path="reports/analysis"))

    expected = f"workflows/{session_name}/reports/analysis"
    assert result["status"] == "success"
    assert result["data"]["output_path"] == expected
    assert instance.write_args["output_path"] == expected
    with S3.open(expected, "r") as handle:
        assert handle.read() == "bounded"
    assert runtime.layout.run_root == f"workflows/{session_name}"


@pytest.mark.parametrize(
    "destination",
    [
        "/tmp/escaped.txt",
        "file:///tmp/escaped.txt",
        "../escaped.txt",
        "reports/../../escaped.txt",
        "workflows/another-run/escaped.txt",
        "https://example.test/escaped.txt",
        "reports/%2e%2e/escaped.txt",
        r"reports\..\escaped.txt",
    ],
)
def test_session_write_scope_rejects_local_escape_before_tool_execution(
    destination, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USE_S3", "false")
    session_name = "write-boundary-denied"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    _bind_run(ctx, session_name)
    instance = _DummyToolkit()
    tool = build_tool(
        _spec("write_output", write_scope="session"),
        instance,
        ctx,
    )

    result = asyncio.run(tool(output_path=destination))

    assert result["status"] == "error"
    assert result["error"]["code"] == "permission_denied"
    assert "active workflow run/session" in result["error"]["message"]
    assert instance.write_calls == 0


def test_session_write_scope_rejects_symlink_escape(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USE_S3", "false")
    session_name = "write-boundary-symlink"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    _bind_run(ctx, session_name)
    run_root = Path(S3.path(f"workflows/{session_name}"))
    external = tmp_path / "external"
    external.mkdir()
    (run_root / "reports").symlink_to(external, target_is_directory=True)
    instance = _DummyToolkit()
    tool = build_tool(
        _spec("write_output", write_scope="session"),
        instance,
        ctx,
    )

    result = asyncio.run(tool(output_path="reports/escaped.txt"))

    assert result["status"] == "error"
    assert result["error"]["code"] == "permission_denied"
    assert instance.write_calls == 0
    assert not (external / "escaped.txt").exists()


def test_session_write_scope_blocks_reserved_file_symlink_alias(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USE_S3", "false")
    session_name = "write-boundary-reserved-alias"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    runtime = _bind_run(ctx, session_name)
    manifest = Path(S3.path(runtime.layout.manifest_rel_path))
    original_manifest = json.loads(manifest.read_text())
    alias = manifest.parent / "manifest-alias.json"
    alias.symlink_to(manifest.name)
    instance = _DummyToolkit()
    tool = build_tool(
        _spec("write_output", write_scope="session"),
        instance,
        ctx,
    )

    result = asyncio.run(tool(output_path="manifest-alias.json"))

    assert result["status"] == "error"
    assert result["error"]["code"] == "permission_denied"
    updated_manifest = json.loads(manifest.read_text())
    assert updated_manifest["run_id"] == original_manifest["run_id"]
    assert updated_manifest["workflow_slug"] == original_manifest["workflow_slug"]
    assert "bounded" not in manifest.read_text()


def test_session_write_scope_closes_directory_swap_race(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USE_S3", "false")
    session_name = "write-boundary-directory-race"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    runtime = _bind_run(ctx, session_name)
    run_root = Path(S3.path(runtime.layout.run_root))
    (run_root / "reports").mkdir()
    external = tmp_path / "external"
    external.mkdir()
    instance = _SymlinkSwapToolkit(external)
    tool = build_tool(
        ToolSpec(
            mcp_name="dummy_write_output",
            toolkit_factory=lambda: instance,
            method="write_output",
            summary="Swap a validated output parent before opening it.",
            write_scope="session",
        ),
        instance,
        ctx,
    )

    result = asyncio.run(tool(output_path="reports/escaped.txt"))

    assert result["status"] == "error"
    assert result["error"]["code"] == "permission_denied"
    assert not (external / "escaped.txt").exists()


def test_concurrent_writers_cannot_replace_a_newly_registered_artifact(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USE_S3", "false")
    session_name = "write-boundary-concurrent-registration"
    S3.set_session_prefix(f"sessions/{session_name}")
    first_ctx = MCPAgentContext()
    runtime = _bind_run(first_ctx, session_name)
    second_ctx = MCPAgentContext()
    second_ctx.run_context = RunContext.load(session_name)
    instance = _ConcurrentArtifactToolkit()
    spec = ToolSpec(
        mcp_name="dummy_concurrent_writer",
        toolkit_factory=lambda: instance,
        method="write_output",
        summary="Write a result after an optional gate.",
        write_scope="session",
    )
    first_tool = build_tool(
        spec,
        instance,
        first_ctx,
    )
    second_tool = build_tool(
        spec,
        instance,
        second_ctx,
    )

    async def invoke_concurrently():
        first_task = asyncio.create_task(
            first_tool(
                output_path="shared.txt",
                content="first",
                wait_for_release=True,
            )
        )
        await instance.started.wait()
        second_task = asyncio.create_task(
            second_tool(
                output_path="shared.txt",
                content="second",
            )
        )
        await asyncio.sleep(0)
        instance.release.set()
        return await asyncio.gather(first_task, second_task)

    first, second = asyncio.run(invoke_concurrently())

    assert first["status"] == "success"
    assert second["status"] == "error"
    assert second["error"]["code"] == "permission_denied"
    artifact_id = first["artifact_ids"][0]
    assert runtime.verify_artifact(artifact_id).relative_path == "shared.txt"
    assert Path(S3.path(f"workflows/{session_name}/shared.txt")).read_text() == "first"


def test_control_plane_registration_shares_domain_artifact_write_lock(
    tmp_path,
    monkeypatch,
):
    from cs_copilot.mcp.facades.workflows import WorkflowRuntimeFacade

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USE_S3", "false")
    session_name = "write-boundary-control-plane-registration"
    S3.set_session_prefix(f"sessions/{session_name}")
    domain_ctx = MCPAgentContext()
    runtime = _bind_run(domain_ctx, session_name)
    control_ctx = MCPAgentContext()
    control_ctx.run_context = RunContext.load(session_name)
    with S3.open(f"workflows/{session_name}/shared.txt", "w") as handle:
        handle.write("initial")

    instance = _ConcurrentArtifactToolkit()
    domain_tool = build_tool(
        ToolSpec(
            mcp_name="dummy_control_plane_race_writer",
            toolkit_factory=lambda: instance,
            method="write_output",
            summary="Write while registration is attempted.",
            write_scope="session",
        ),
        instance,
        domain_ctx,
    )
    facade = WorkflowRuntimeFacade()
    register_tool = build_tool(
        ToolSpec(
            mcp_name="workflow_register_artifact",
            toolkit_factory=lambda: facade,
            method="register_artifact",
            summary="Register an existing artifact.",
            write_scope="session",
        ),
        facade,
        control_ctx,
    )

    async def register_while_domain_call_is_running():
        domain_task = asyncio.create_task(
            domain_tool(
                output_path="shared.txt",
                content="domain",
                wait_for_release=True,
            )
        )
        await instance.started.wait()
        registration_task = asyncio.create_task(
            register_tool(
                path="shared.txt",
                artifact_type="manually_registered",
                mime_type="text/plain",
            )
        )
        await asyncio.sleep(0)
        assert not registration_task.done()
        instance.release.set()
        return await asyncio.gather(domain_task, registration_task)

    domain_result, registration_result = asyncio.run(register_while_domain_call_is_running())

    assert domain_result["status"] == "error"
    assert domain_result["error"]["code"] == "permission_denied"
    assert registration_result["status"] == "success"
    artifact_id = registration_result["data"]["artifact_id"]
    runtime.refresh()
    assert runtime.verify_artifact(artifact_id).relative_path == "shared.txt"
    assert Path(S3.path(f"workflows/{session_name}/shared.txt")).read_text() == "initial"


def test_session_write_scope_handles_nested_and_json_encoded_path_sinks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USE_S3", "false")
    session_name = "write-boundary-nested"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    _bind_run(ctx, session_name)
    instance = _DummyToolkit()
    output_tool = build_tool(
        _spec("write_output", write_scope="session"),
        instance,
        ctx,
    )
    operation_tool = build_tool(
        ToolSpec(
            mcp_name="pandas_run_operation",
            toolkit_factory=lambda: instance,
            method="operation",
            summary="Dummy pandas operation.",
            write_scope="session",
        ),
        instance,
        ctx,
    )

    nested = asyncio.run(
        output_tool(options={"delivery": {"outputs": {"html": "reports/result.html"}}})
    )
    denied = asyncio.run(
        operation_tool(
            operation="to_csv",
            operation_parameters={"transport": {"path_or_buf": "../escaped.csv"}},
        )
    )
    denied_operation = asyncio.run(
        operation_tool(
            dataframe_name=str(tmp_path / "outside.csv"),
            operation="head",
        )
    )
    encoded = asyncio.run(
        operation_tool(
            operation="to_csv",
            function_parameters=json.dumps({"path_or_buf": "tables/result.csv"}),
        )
    )

    expected_html = f"workflows/{session_name}/reports/result.html"
    expected_csv = f"workflows/{session_name}/tables/result.csv"
    assert nested["status"] == "success"
    assert nested["data"]["options"]["delivery"]["outputs"]["html"] == expected_html
    assert denied["status"] == "error"
    assert denied["error"]["code"] == "permission_denied"
    assert denied_operation["status"] == "error"
    assert denied_operation["error"]["code"] == "permission_denied"
    assert encoded["status"] == "success"
    assert json.loads(encoded["data"]["function_parameters"])["path_or_buf"] == expected_csv


def test_pandas_python_literal_parameter_bags_and_path_aliases_are_confined(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USE_S3", "false")
    session_name = "write-boundary-python-literal"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    _bind_run(ctx, session_name)
    instance = _DummyToolkit()
    tool = build_tool(
        ToolSpec(
            mcp_name="pandas_run_operation",
            toolkit_factory=lambda: instance,
            method="operation",
            summary="Dummy pandas operation.",
            write_scope="session",
        ),
        instance,
        ctx,
    )

    denied = asyncio.run(
        tool(
            operation="to_csv",
            operation_parameters="{'path_or_buffer': '../escaped.csv'}",
        )
    )
    allowed = asyncio.run(
        tool(
            operation="to_csv",
            operation_parameters="{'file_path': 'tables/result.csv'}",
        )
    )

    assert denied["status"] == "error"
    assert denied["error"]["code"] == "permission_denied"
    assert allowed["status"] == "success"
    decoded = json.loads(allowed["data"]["operation_parameters"])
    assert decoded["file_path"] == f"workflows/{session_name}/tables/result.csv"


@pytest.mark.parametrize(
    "creation_function",
    ["ExcelWriter", "HDFStore", "read_pickle", "read_html"],
)
def test_pandas_create_dataframe_rejects_unapproved_constructor_or_reader(
    creation_function, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USE_S3", "false")
    session_name = "write-boundary-pandas-constructor"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    _bind_run(ctx, session_name)
    instance = _DummyToolkit()
    tool = build_tool(
        ToolSpec(
            mcp_name="pandas_create_dataframe",
            toolkit_factory=lambda: instance,
            method="create",
            summary="Dummy pandas create.",
            write_scope="session",
        ),
        instance,
        ctx,
    )

    result = asyncio.run(
        tool(
            create_using_function=creation_function,
            function_parameters={"path": "reports/escaped.bin"},
        )
    )

    assert result["status"] == "error"
    assert result["error"]["code"] == "permission_denied"
    assert instance.write_calls == 0
    assert not Path(S3.path(f"workflows/{session_name}/reports/escaped.bin")).exists()


def test_pandas_write_alias_is_confined_and_non_scoped_serializers_are_denied(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USE_S3", "false")
    session_name = "write-boundary-pandas-alias"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    _bind_run(ctx, session_name)
    instance = _DummyToolkit()
    tool = build_tool(
        ToolSpec(
            mcp_name="pandas_run_operation",
            toolkit_factory=lambda: instance,
            method="operation",
            summary="Dummy pandas operation.",
            write_scope="session",
        ),
        instance,
        ctx,
    )

    alias = asyncio.run(
        tool(
            operation="export_csv",
            operation_parameters={"filename": "tables/aliased.csv"},
        )
    )
    unsupported = asyncio.run(
        tool(
            operation="to_pickle",
            operation_parameters={"path": "tables/result.pkl"},
        )
    )
    external = asyncio.run(
        tool(
            operation="to_sql",
            operation_parameters={"name": "results", "con": "sqlite:////tmp/results.db"},
        )
    )

    expected = f"workflows/{session_name}/tables/aliased.csv"
    assert alias["status"] == "success"
    assert alias["data"]["operation_parameters"]["filename"] == expected
    assert unsupported["status"] == "error"
    assert unsupported["error"]["code"] == "permission_denied"
    assert external["status"] == "error"
    assert external["error"]["code"] == "permission_denied"
    assert instance.write_calls == 1


@pytest.mark.parametrize(
    "field_name, malicious_name",
    [
        ("dataset_name", "/tmp/dataset"),
        ("dataset_name", "s3://other-bucket/dataset"),
        ("dataset_name", "workflows/other-run/dataset"),
        ("gtm_name", "../model"),
        ("gtm_name", "file:///tmp/model"),
    ],
)
def test_gtm_derived_artifact_names_cannot_escape_run(
    field_name, malicious_name, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USE_S3", "false")
    session_name = "write-boundary-gtm-name"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    _bind_run(ctx, session_name)
    instance = _DummyToolkit()
    tool = build_tool(
        ToolSpec(
            mcp_name="gtm_save_model_and_data",
            toolkit_factory=lambda: instance,
            method="save_named",
            summary="Dummy GTM save.",
            write_scope="session",
        ),
        instance,
        ctx,
    )
    arguments = {"dataset_name": "dataset", "gtm_name": "model"}
    arguments[field_name] = malicious_name

    denied = asyncio.run(tool(**arguments))
    allowed = asyncio.run(tool(dataset_name="dataset.v1", gtm_name="model-1"))

    assert denied["status"] == "error"
    assert denied["error"]["code"] == "invalid_input"
    assert "derive a run artifact filename" in denied["error"]["message"]
    assert allowed["status"] == "success"
    assert instance.write_calls == 1


def test_pandas_file_reads_require_verified_registered_run_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USE_S3", "false")
    S3.set_session_prefix("sessions/write-boundary-read")
    ctx = MCPAgentContext()
    runtime = _bind_run(ctx, "write-boundary-read")
    with S3.open("workflows/write-boundary-read/inputs/uploaded.csv", "w") as handle:
        handle.write("value\n1\n")
    runtime.register_artifact(
        "inputs/uploaded.csv",
        artifact_type="source_table",
        mime_type="text/csv",
    )
    instance = _DummyToolkit()
    tool = build_tool(
        ToolSpec(
            mcp_name="pandas_create_dataframe",
            toolkit_factory=lambda: instance,
            method="create",
            summary="Dummy pandas create.",
            write_scope="session",
        ),
        instance,
        ctx,
    )

    allowed = asyncio.run(
        tool(
            create_using_function="pd.read_csv",
            function_parameters={"path_or_buf": "inputs/uploaded.csv"},
        )
    )
    expanded = asyncio.run(
        tool(
            create_using_function="read_csv",
            function_parameters={
                "path_or_buf": S3.path("workflows/write-boundary-read/inputs/uploaded.csv")
            },
        )
    )
    denied = asyncio.run(
        tool(
            create_using_function="read_csv",
            function_parameters={"path_or_buf": str(tmp_path / "outside.csv")},
        )
    )

    assert allowed["status"] == "success"
    assert allowed["data"]["create_using_function"] == "read_csv"
    assert (
        allowed["data"]["function_parameters"]["path_or_buf"]
        == "workflows/write-boundary-read/inputs/uploaded.csv"
    )
    assert expanded["status"] == "success"
    assert (
        expanded["data"]["function_parameters"]["path_or_buf"]
        == "workflows/write-boundary-read/inputs/uploaded.csv"
    )
    assert denied["status"] == "error"
    assert denied["error"]["code"] == "permission_denied"


def test_declared_scientific_read_fields_require_registered_run_artifacts(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USE_S3", "false")
    session_name = "registered-scientific-read"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    runtime = _bind_run(ctx, session_name)
    with S3.open(f"workflows/{session_name}/inputs/activity.csv", "w") as handle:
        handle.write("smiles,value\nCCO,1\n")
    runtime.register_artifact(
        "inputs/activity.csv",
        artifact_type="clean_dataset_path",
        mime_type="text/csv",
    )
    outside = tmp_path / "outside.csv"
    outside.write_text("private,value\nsecret,1\n", encoding="utf-8")
    instance = _DummyToolkit()
    tool = build_tool(
        ToolSpec(
            mcp_name="chembl_describe_dataset",
            toolkit_factory=lambda: instance,
            method="read_dataset",
            summary="Read a registered dataset.",
            read_only=True,
            read_artifact_fields=("path_to_dataset",),
        ),
        instance,
        ctx,
    )

    allowed = asyncio.run(tool(path_to_dataset="inputs/activity.csv"))
    denied = asyncio.run(tool(path_to_dataset=str(outside)))

    assert allowed["status"] == "success"
    assert allowed["data"]["path_to_dataset"] == f"workflows/{session_name}/inputs/activity.csv"
    assert denied["status"] == "error"
    assert denied["error"]["code"] == "permission_denied"
    assert instance.structured_calls == 1


def test_catalog_task_reads_only_handed_off_or_same_task_artifacts(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USE_S3", "false")
    session_name = "task-read-least-privilege"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    runtime = _bind_catalog_preflight(
        ctx,
        session_name,
        max_tool_calls=4,
        timeout_seconds=60,
    )
    with S3.open(f"workflows/{session_name}/inputs/unrelated.csv", "w") as handle:
        handle.write("smiles,value\nCCC,2\n")
    runtime.register_artifact(
        "inputs/unrelated.csv",
        artifact_type="unrelated_dataset",
        mime_type="text/csv",
        trust="external",
    )
    instance = _DummyToolkit()
    tool = build_tool(
        ToolSpec(
            mcp_name="chembl_prepare_retrieval",
            toolkit_factory=lambda: instance,
            method="read_dataset",
            summary="Read one catalog artifact.",
            group="chembl",
            roles=("chembl_downloader",),
            profiles=("chembl-retrieval",),
            read_only=True,
            read_artifact_fields=("path_to_dataset",),
        ),
        instance,
        ctx,
    )

    denied = asyncio.run(tool(path_to_dataset="inputs/unrelated.csv"))

    assert denied["status"] == "error"
    assert denied["error"]["code"] == "permission_denied"
    assert "was not handed off" in denied["error"]["message"]
    assert instance.structured_calls == 0


def test_pandas_session_loader_denies_unregistered_csv_but_allows_dataframe(
    tmp_path,
    monkeypatch,
):
    from cs_copilot.mcp.facades.pandas import PointerPandasFacade

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USE_S3", "false")
    session_name = "registered-session-pandas-read"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    _bind_run(ctx, session_name)
    outside = tmp_path / "outside.csv"
    outside.write_text("private\nsecret\n", encoding="utf-8")
    ctx.session_state["outside_csv"] = str(outside)
    ctx.session_state["in_memory"] = pd.DataFrame({"value": [1]})
    facade = PointerPandasFacade()
    tool = build_tool(
        ToolSpec(
            mcp_name="pandas_load_dataframe_from_session",
            toolkit_factory=lambda: facade,
            method="load_dataframe_from_session",
            summary="Load a safe session DataFrame.",
            write_scope="session",
        ),
        facade,
        ctx,
    )

    denied = asyncio.run(tool(dataframe_name="outside", session_key="outside_csv"))
    allowed = asyncio.run(tool(dataframe_name="memory", session_key="in_memory"))

    assert denied["status"] == "error"
    assert denied["error"]["code"] == "permission_denied"
    assert allowed["status"] == "success"
    assert allowed["data"]["dataframe_name"] == "memory"


def test_pandas_session_loader_consumes_pinned_registered_pointer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USE_S3", "false")
    session_name = "pinned-session-pandas-read"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    runtime = _bind_run(ctx, session_name)
    run_path = f"workflows/{session_name}/inputs/activity.csv"
    with S3.open(run_path, "w") as handle:
        handle.write("value\nsafe\n")
    runtime.register_artifact(
        "inputs/activity.csv",
        artifact_type="source_table",
        mime_type="text/csv",
    )
    outside = tmp_path / "outside.csv"
    outside.write_text("value\nprivate\n", encoding="utf-8")
    ctx.session_state["dataset"] = {"csv_path": "inputs/activity.csv"}
    instance = _SessionPointerSwapToolkit(ctx.session_state, str(outside))
    tool = build_tool(
        ToolSpec(
            mcp_name="pandas_load_dataframe_from_session",
            toolkit_factory=lambda: instance,
            method="load_dataframe_from_session",
            summary="Load a pinned session dataset.",
            write_scope="session",
        ),
        instance,
        ctx,
    )

    result = asyncio.run(tool(dataframe_name="activity", session_key="dataset"))

    assert result["status"] == "success"
    assert result["data"]["source"] == run_path
    assert result["data"]["content"] == "value\nsafe\n"
    assert ctx.session_state["dataset"]["csv_path"] == str(outside)


@pytest.mark.parametrize(
    ("mcp_name", "method"),
    [
        ("session_load_candidate_set_artifact", "load_candidate_set_artifact"),
        ("peptide_load_design_candidates", "load_peptide_design_candidates"),
    ],
)
def test_candidate_loader_consumes_pinned_registered_pointer(
    mcp_name,
    method,
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USE_S3", "false")
    session_name = "pinned-session-candidate-read"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    runtime = _bind_run(ctx, session_name)
    run_path = f"workflows/{session_name}/candidate_sets/candidates.json"
    with S3.open(run_path, "w") as handle:
        json.dump({"candidates": [{"smiles": "CCO"}]}, handle)
    runtime.register_artifact(
        "candidate_sets/candidates.json",
        artifact_type="candidate_set",
        mime_type="application/json",
    )
    outside = tmp_path / "outside.json"
    outside.write_text('{"candidates": [{"smiles": "private"}]}', encoding="utf-8")
    ctx.session_state["candidate_pointer"] = {"artifact_rel_path": "candidate_sets/candidates.json"}
    instance = _SessionPointerSwapToolkit(ctx.session_state, str(outside))
    tool = build_tool(
        ToolSpec(
            mcp_name=mcp_name,
            toolkit_factory=lambda: instance,
            method=method,
            summary="Load a pinned candidate artifact.",
            read_only=True,
        ),
        instance,
        ctx,
    )

    result = asyncio.run(tool(reference="candidate_pointer"))

    assert result["status"] == "success"
    assert result["data"]["reference"] == "candidate_pointer"
    assert result["data"]["consumed_path"] == run_path
    assert ctx.session_state["candidate_pointer"]["artifact_rel_path"] == str(outside)


def test_session_candidate_materializer_consumes_pinned_candidate_set_artifact(
    tmp_path,
    monkeypatch,
):
    from cs_copilot.tools.io.session_memory import register_session_object

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USE_S3", "false")
    session_name = "pinned-session-candidate-materialize"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    runtime = _bind_run(ctx, session_name)
    run_path = f"workflows/{session_name}/candidate_sets/candidates.json"
    with S3.open(run_path, "w") as handle:
        json.dump({"candidates": [{"smiles": "CCO"}]}, handle)
    runtime.register_artifact(
        "candidate_sets/candidates.json",
        artifact_type="candidate_set",
        mime_type="application/json",
    )
    candidate_set_id = register_session_object(
        ctx.session_state,
        "candidate_set",
        {
            "artifact_rel_path": "candidate_sets/candidates.json",
            "count_returned": 1,
        },
    )
    ctx.session_state["candidate_alias"] = {"candidate_set_id": candidate_set_id}
    outside = tmp_path / "outside.json"
    outside.write_text('{"candidates": [{"smiles": "private"}]}', encoding="utf-8")
    instance = _SessionPointerSwapToolkit(ctx.session_state, str(outside))
    tool = build_tool(
        ToolSpec(
            mcp_name="session_materialize_candidate_set_dataset",
            toolkit_factory=lambda: instance,
            method="materialize_candidate_set_dataset",
            summary="Materialize a pinned candidate artifact.",
        ),
        instance,
        ctx,
    )

    result = asyncio.run(tool(reference="candidate_alias", top_n=1))

    assert result["status"] == "success"
    assert result["data"]["reference"] == run_path
    record = ctx.session_state["session_objects"]["candidate_sets"][candidate_set_id]
    assert record["artifact_rel_path"] == str(outside)


def test_session_write_scope_allows_only_active_s3_run(monkeypatch):
    monkeypatch.setenv("USE_S3", "true")
    monkeypatch.setenv("ASSETS_BUCKET", "experiment-artifacts")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.delenv("S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    monkeypatch.delenv("MINIO_ENDPOINT_URL", raising=False)
    session_name = "write-boundary-s3"
    run_id = "run-s3"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext(
        session_state={
            "output_context": {
                "session_id": session_name,
                "run_id": run_id,
                "workflow_slug": "mcp-session",
            }
        }
    )
    instance = _DummyToolkit()
    tool = build_tool(
        ToolSpec(
            mcp_name="pandas_run_operation",
            toolkit_factory=lambda: instance,
            method="operation",
            summary="Dummy pandas operation.",
            write_scope="session",
        ),
        instance,
        ctx,
    )
    active = (
        f"s3://experiment-artifacts/sessions/{session_name}/"
        f"workflows/{run_id}/tables/result.csv"
    )
    foreign = [
        active.replace("experiment-artifacts", "other-bucket"),
        active.replace(f"sessions/{session_name}", "sessions/other-session"),
        active.replace(f"workflows/{run_id}", "workflows/other-run"),
    ]

    allowed = asyncio.run(
        tool(
            operation="to_csv",
            operation_parameters={"path_or_buf": active},
        )
    )
    denied = [
        asyncio.run(
            tool(
                operation="to_csv",
                operation_parameters={"path_or_buf": destination},
            )
        )
        for destination in foreign
    ]

    assert allowed["status"] == "success"
    assert allowed["data"]["operation_parameters"]["path_or_buf"] == active
    assert all(result["status"] == "error" for result in denied)
    assert all(result["error"]["code"] == "permission_denied" for result in denied)
    assert instance.write_calls == 1


def test_adapter_redacts_secret_manifest_args(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_name = "manifest-redaction"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    _bind_run(ctx, session_name)
    tool = build_tool(_spec("secret_echo"), _DummyToolkit(), ctx)

    asyncio.run(tool(api_key="secret", token_value="token"))

    payload = _tool_payloads(ctx)[0]
    assert payload["public_args"]["api_key"] == "<redacted>"
    assert payload["public_args"]["token_value"] == "<redacted>"


def test_tool_spec_retries_require_idempotence():
    with pytest.raises(ValueError, match="retries require an idempotent tool"):
        _spec("flaky", max_retries=1)


def test_adapter_retries_retryable_idempotent_failures_and_records_progress(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_name = "retry-success"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    _bind_run(ctx, session_name)
    instance = _DummyToolkit()
    tool = build_tool(
        _spec("flaky", idempotent=True, max_retries=1, retry_backoff_s=0),
        instance,
        ctx,
    )

    result = asyncio.run(tool())

    assert result["status"] == "success"
    assert result["data"] == "recovered"
    assert result["metrics"]["attempts"] == 2
    assert result["metrics"]["retries"] == 1
    assert result["metrics"]["output_bytes"] == len(json.dumps("recovered").encode())
    assert instance.flaky_calls == 2
    assert [payload["stage"] for payload in _progress_payloads(ctx)] == [
        "started",
        "retrying",
        "result_accepted",
        "completed",
    ]
    manifest = _tool_payloads(ctx)[0]
    assert manifest["attempts"] == 2
    assert manifest["retries"] == 1


def test_adapter_idempotency_key_cache_is_bounded_to_same_request(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_name = "idempotency-cache"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    _bind_run(ctx, session_name)
    instance = _DummyToolkit()
    tool = build_tool(_spec("echo", idempotent=True), instance, ctx)

    assert "idempotency_key" in inspect.signature(tool).parameters
    first = asyncio.run(tool(text="ab", count=2, idempotency_key="request-1"))
    second = asyncio.run(tool(text="ab", count=2, idempotency_key="request-1"))
    conflict = asyncio.run(tool(text="different", idempotency_key="request-1"))

    assert first["metrics"]["cached"] is False
    assert first["metrics"]["attempts"] == 1
    assert second["data"] == first["data"]
    assert second["metrics"]["cached"] is True
    assert second["metrics"]["attempts"] == 0
    assert second["trace"]["span_id"] != first["trace"]["span_id"]
    assert instance.echo_calls == 1
    assert conflict["status"] == "error"
    assert conflict["error"]["code"] == "invalid_input"
    manifests = _tool_payloads(ctx)
    assert manifests[1]["idempotency_fingerprint"].startswith("sha256:")
    assert "request-1" not in manifests[1]["idempotency_fingerprint"]
    assert manifests[1]["attempts"] == 0
    assert manifests[1]["cached"] is True


def test_adapter_idempotency_cache_is_scoped_to_active_task(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_name = "idempotency-task-scope"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    runtime = _bind_run(ctx, session_name)
    runtime.transition_run("running")
    for task_id in ("first", "second"):
        runtime.add_task(
            TaskRecord(
                task_id=task_id,
                role="gtm_agent",
                profile="gtm-analysis",
                step=f"Run {task_id}.",
            )
        )
        runtime.transition_task(task_id, "running")
    instance = _DummyToolkit()
    tool = build_tool(_spec("echo", idempotent=True), instance, ctx)

    ctx.session_state["active_task_id"] = "first"
    first = asyncio.run(tool(text="same", idempotency_key="request-1"))
    ctx.session_state["active_task_id"] = "second"
    second = asyncio.run(tool(text="same", idempotency_key="request-1"))

    assert first["status"] == second["status"] == "success"
    assert first["metrics"]["cached"] is False
    assert second["metrics"]["cached"] is False
    assert instance.echo_calls == 2


def test_adapter_coalesces_concurrent_idempotent_calls():
    ctx = MCPAgentContext()
    instance = _ConcurrentToolkit()
    spec = ToolSpec(
        mcp_name="dummy_concurrent",
        toolkit_factory=_ConcurrentToolkit,
        method="slow_echo",
        summary="Concurrent echo.",
        idempotent=True,
    )
    tool = build_tool(spec, instance, ctx)

    async def exercise():
        first = asyncio.create_task(tool(text="same", idempotency_key="request-1"))
        await instance.started.wait()
        second = asyncio.create_task(tool(text="same", idempotency_key="request-1"))
        await asyncio.sleep(0)
        instance.release.set()
        return await asyncio.gather(first, second)

    first, second = asyncio.run(exercise())

    assert instance.calls == 1
    assert first["data"] == second["data"] == "same"
    assert sorted((first["metrics"]["cached"], second["metrics"]["cached"])) == [False, True]
    assert sorted((first["metrics"]["attempts"], second["metrics"]["attempts"])) == [0, 1]


def test_adapter_rejects_concurrent_idempotency_digest_conflict():
    ctx = MCPAgentContext()
    instance = _ConcurrentToolkit()
    spec = ToolSpec(
        mcp_name="dummy_concurrent_conflict",
        toolkit_factory=_ConcurrentToolkit,
        method="slow_echo",
        summary="Concurrent echo.",
        idempotent=True,
    )
    tool = build_tool(spec, instance, ctx)

    async def exercise():
        first = asyncio.create_task(tool(text="first", idempotency_key="request-1"))
        await instance.started.wait()
        conflict = await tool(text="different", idempotency_key="request-1")
        instance.release.set()
        return await first, conflict

    first, conflict = asyncio.run(exercise())

    assert first["status"] == "success"
    assert conflict["status"] == "error"
    assert conflict["error"]["code"] == "invalid_input"
    assert instance.calls == 1


def test_adapter_registers_backticked_report_path_with_contract_type(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_name = "report-artifact"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    _bind_run(ctx, session_name)
    rel_path = f"workflows/{session_name}/reports/final.html"
    full_path = S3.path(rel_path)
    instance = _PublishingTextToolkit([rel_path], "<html>result</html>")
    spec = ToolSpec(
        mcp_name="report_save_rich",
        toolkit_factory=lambda: instance,
        method="legacy_text",
        summary="Return a legacy report result.",
        write_scope="session",
    )

    result = asyncio.run(
        build_tool(spec, instance, ctx)(text=f"Rich report saved to S3:\n- HTML: `{full_path}`")
    )

    assert result["status"] == "success"
    assert len(result["artifact_ids"]) == 1
    artifact = ctx.run_context.run.artifacts[result["artifact_ids"][0]]
    assert artifact.relative_path == "reports/final.html"
    assert artifact.artifact_type == "html_report_path"
    assert artifact.mime_type == "text/html"
    assert artifact.producer_tool == "report_save_rich"


@pytest.mark.parametrize("nested_in_section", [False, True])
def test_rich_report_rejects_unregistered_explicit_figure_paths(
    nested_in_section,
    tmp_path,
    monkeypatch,
):
    from cs_copilot.mcp.facades.reporting import ReportExportFacade

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USE_S3", "false")
    session_name = f"report-read-boundary-{int(nested_in_section)}"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    _bind_run(ctx, session_name)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"private-image")
    figure = {
        "image_path": str(outside),
        "caption": "A private image must not be embedded.",
    }
    arguments = (
        {
            "title": "Boundary test",
            "sections": [{"heading": "Results", "figures": [figure]}],
            "formats": ["html"],
        }
        if nested_in_section
        else {
            "title": "Boundary test",
            "figures": [figure],
            "formats": ["html"],
        }
    )
    facade = ReportExportFacade()
    tool = build_tool(
        ToolSpec(
            mcp_name="report_save_rich",
            toolkit_factory=lambda: facade,
            method="save_rich",
            summary="Save a rich report.",
            write_scope="session",
        ),
        facade,
        ctx,
    )

    result = asyncio.run(tool(**arguments))

    assert result["status"] == "error"
    assert result["error"]["code"] == "permission_denied"
    assert not any(
        artifact.artifact_type == "html_report_path"
        for artifact in ctx.run_context.run.artifacts.values()
    )


def test_rich_report_verifies_session_figure_metadata_paths(tmp_path, monkeypatch):
    from cs_copilot.mcp.facades.reporting import ReportExportFacade
    from cs_copilot.tools.io.figure_metadata import (
        build_figure_metadata,
        register_figure_metadata,
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USE_S3", "false")
    session_name = "report-session-figure-boundary"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    _bind_run(ctx, session_name)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"private-image")
    figure_id = register_figure_metadata(
        ctx.session_state,
        build_figure_metadata(
            figure_kind="density",
            renderer="static",
            report_role="static_inline",
            title_subject="Private density map",
            paths={"png_path": str(outside)},
            caption_facts=["Private data."],
        ),
    )
    facade = ReportExportFacade()
    tool = build_tool(
        ToolSpec(
            mcp_name="report_save_rich",
            toolkit_factory=lambda: facade,
            method="save_rich",
            summary="Save a rich report.",
            write_scope="session",
        ),
        facade,
        ctx,
    )

    result = asyncio.run(
        tool(
            title="Boundary test",
            figures=[{"figure_id": figure_id}],
            formats=["html"],
        )
    )

    assert result["status"] == "error"
    assert result["error"]["code"] == "permission_denied"


def test_rich_report_consumes_pinned_session_figure_metadata(tmp_path, monkeypatch):
    from cs_copilot.tools.io.figure_metadata import (
        build_figure_metadata,
        register_figure_metadata,
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USE_S3", "false")
    session_name = "pinned-report-session-figure"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    runtime = _bind_run(ctx, session_name)
    run_path = f"workflows/{session_name}/figures/density.png"
    with S3.open(run_path, "wb") as handle:
        handle.write(b"safe-image")
    runtime.register_artifact(
        "figures/density.png",
        artifact_type="density_figure",
        mime_type="image/png",
    )
    figure_id = register_figure_metadata(
        ctx.session_state,
        build_figure_metadata(
            figure_kind="density",
            renderer="static",
            report_role="inline_static",
            title_subject="Density map",
            paths={"png_path": "figures/density.png"},
            caption_facts=["Registered density figure."],
        ),
    )
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"private-image")
    instance = _SessionPointerSwapToolkit(ctx.session_state, str(outside))
    tool = build_tool(
        ToolSpec(
            mcp_name="report_save_rich",
            toolkit_factory=lambda: instance,
            method="save_rich",
            summary="Inspect a pinned rich-report figure.",
            read_only=True,
        ),
        instance,
        ctx,
    )

    result = asyncio.run(
        tool(
            title="Pinned figure",
            figures=[{"figure_id": figure_id}],
        )
    )

    assert result["status"] == "success"
    assert result["data"]["png_path"] == run_path
    live_record = ctx.session_state["session_objects"]["figures"][figure_id]
    assert live_record["paths"]["png_path"] == str(outside)
    assert ctx.session_state["report_snapshot_write"] == "preserved"


def test_adapter_maps_chembl_labeled_paths_to_contract_types(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_name = "chembl-artifacts"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    _bind_run(ctx, session_name)
    labeled_files = {
        "📄 Clean dataset (CSV)": ("clean.csv", "clean_dataset_path"),
        "📄 Raw dataset": ("raw.csv", "raw_dataset_path"),
        "🧮 Descriptor Parquet": ("descriptors.parquet", "descriptor_parquet_path"),
        "🧾 Standardization report": (
            "standardization.json",
            "standardization_report_path",
        ),
        "Filtered rows": ("filtered.csv", "filtered_rows_path"),
    }
    lines = []
    paths = []
    for label, (filename, _artifact_type) in labeled_files.items():
        rel_path = f"workflows/{session_name}/01_chemical_space/{filename}"
        paths.append(rel_path)
        lines.append(f"{label}: `{S3.path(rel_path)}`")
    instance = _PublishingTextToolkit(paths, "result")
    spec = ToolSpec(
        mcp_name="chembl_fetch_compounds",
        toolkit_factory=lambda: instance,
        method="legacy_text",
        summary="Return legacy ChEMBL paths.",
        write_scope="session",
    )

    result = asyncio.run(build_tool(spec, instance, ctx)(text="\n".join(lines)))

    assert result["status"] == "success"
    artifact_types = {
        ctx.run_context.run.artifacts[artifact_id].artifact_type
        for artifact_id in result["artifact_ids"]
    }
    assert artifact_types == {artifact_type for _filename, artifact_type in labeled_files.values()}


def test_adapter_materializes_structured_result_as_json_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_name = "structured-artifact"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    _bind_run(ctx, session_name)
    spec = _spec("structured", result_artifact_type="retrieval_plan")

    result = asyncio.run(build_tool(spec, _DummyToolkit(), ctx)())

    assert result["status"] == "success"
    assert len(result["artifact_ids"]) == 1
    artifact = ctx.run_context.run.artifacts[result["artifact_ids"][0]]
    assert artifact.artifact_type == "retrieval_plan"
    assert artifact.mime_type == "application/json"
    with S3.open(ctx.run_context.layout.artifact_rel_path(artifact.relative_path), "r") as handle:
        assert json.load(handle) == result["data"]


def test_structured_result_rejects_conflicting_deterministic_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_name = "structured-artifact-conflict"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    runtime = _bind_run(ctx, session_name)
    expected = {"can_proceed": True, "plan": ["retrieve", "analyze"]}
    serialized = json.dumps(
        expected,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    relative = f"artifacts/contracts/retrieval_plan-{digest[:16]}.json"
    with S3.open(runtime.layout.artifact_rel_path(relative), "w") as handle:
        handle.write("attacker content")

    result = asyncio.run(
        build_tool(
            _spec("structured", result_artifact_type="retrieval_plan"),
            _DummyToolkit(),
            ctx,
        )()
    )

    assert result["status"] == "error"
    assert result["error"]["code"] == "scientific_validation"
    assert not runtime.run.artifacts


def test_adapter_rejects_claiming_a_preexisting_unregistered_result(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    session_name = "unowned-result-path"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    _bind_run(ctx, session_name)
    path = f"workflows/{session_name}/reports/unowned.html"
    with S3.open(path, "w") as handle:
        handle.write("<html>unowned</html>")
    instance = _DummyToolkit()
    spec = ToolSpec(
        mcp_name="report_save_rich",
        toolkit_factory=lambda: instance,
        method="legacy_text",
        summary="Return a path without publishing it.",
        write_scope="session",
    )

    result = asyncio.run(
        build_tool(spec, instance, ctx)(text=f"Rich report saved to S3:\n- HTML: `{S3.path(path)}`")
    )

    assert result["status"] == "error"
    assert result["error"]["code"] == "scientific_validation"
    assert not ctx.run_context.run.artifacts


def test_adapter_releases_unreturned_invocation_publications(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_name = "hidden-tool-publication"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    runtime = _bind_run(ctx, session_name)
    path = f"workflows/{session_name}/reports/hidden.txt"
    instance = _HiddenWriteToolkit()
    spec = ToolSpec(
        mcp_name="dummy_hidden_write",
        toolkit_factory=lambda: instance,
        method="write_hidden",
        summary="Write an output without returning its path.",
        write_scope="session",
    )

    result = asyncio.run(build_tool(spec, instance, ctx)(path=path))

    assert result["status"] == "success"
    assert result["artifact_ids"] == []
    assert not Path(S3.path(path)).exists()
    assert not runtime.run.artifacts


def test_required_structured_artifact_registration_failure_removes_materialized_file(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    session_name = "structured-artifact-registration-failure"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    runtime = _bind_catalog_preflight(
        ctx,
        session_name,
        max_tool_calls=4,
        timeout_seconds=60,
    )
    original_append_event = runtime.append_event

    def fail_artifact_registration(event_type, *args, **kwargs):
        if event_type == "artifact_registered":
            raise OSError("simulated artifact event-store failure")
        return original_append_event(event_type, *args, **kwargs)

    monkeypatch.setattr(runtime, "append_event", fail_artifact_registration)
    spec = ToolSpec(
        mcp_name="chembl_prepare_retrieval",
        toolkit_factory=_DummyToolkit,
        method="structured",
        summary="Dummy preflight.",
        group="chembl",
        roles=("chembl_downloader",),
        profiles=("chembl-retrieval",),
        result_artifact_type="retrieval_plan",
    )

    result = asyncio.run(build_tool(spec, _DummyToolkit(), ctx)())

    assert result["status"] == "error"
    assert result["error"]["code"] == "internal"
    contracts_dir = Path(S3.path(runtime.layout.artifact_rel_path("artifacts/contracts")))
    assert not contracts_dir.exists() or not any(contracts_dir.iterdir())
    authoritative = RunContext.load(session_name)
    assert not any(
        artifact.artifact_type == "retrieval_plan"
        for artifact in authoritative.run.artifacts.values()
    )


def test_worker_publication_is_rolled_back_when_required_registration_fails(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    session_name = "worker-registration-failure"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    runtime = _bind_catalog_preflight(
        ctx,
        session_name,
        max_tool_calls=4,
        timeout_seconds=60,
    )
    original_append_event = runtime.append_event

    def fail_artifact_registration(event_type, *args, **kwargs):
        if event_type == "artifact_registered":
            raise OSError("simulated artifact event-store failure")
        return original_append_event(event_type, *args, **kwargs)

    def publish_worker_result(*, job_path, result_path, timeout_s):
        del timeout_s
        job = json.loads(job_path.read_text(encoding="utf-8"))
        final_path = f"{job['write_boundary']}/results/worker-output.csv"
        staged_path = (
            f"{job['write_boundary']}/.staging/{job['staging_id']}" "/results/worker-output.csv"
        )
        payload = b"value\n1\n"
        staged_file = Path(S3.path(staged_path))
        staged_file.parent.mkdir(parents=True, exist_ok=True)
        staged_file.write_bytes(payload)
        result_path.write_text(
            json.dumps(
                {
                    "ok": True,
                    "result": {"output_path": final_path},
                    "session_state": job["session_state"],
                    "staged_publications": {
                        final_path: {
                            "staged_path": staged_path,
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "size_bytes": len(payload),
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return "", ""

    monkeypatch.setattr(runtime, "append_event", fail_artifact_registration)
    monkeypatch.setattr(
        "cs_copilot.mcp.jobs._run_worker_process",
        publish_worker_result,
    )
    spec = ToolSpec(
        mcp_name="chembl_prepare_retrieval",
        toolkit_factory=_DummyToolkit,
        method="structured",
        summary="Worker preflight.",
        group="chembl",
        roles=("chembl_downloader",),
        profiles=("chembl-retrieval",),
        run_in_worker_process=True,
        write_scope="session",
        result_artifact_type="retrieval_plan",
    )

    result = asyncio.run(build_tool(spec, _DummyToolkit(), ctx)())

    assert result["status"] == "error"
    assert result["error"]["code"] == "internal"
    run_root = Path(S3.path(runtime.layout.run_root))
    assert not (run_root / "results" / "worker-output.csv").exists()
    staging_root = run_root / ".staging"
    assert not staging_root.exists() or not any(path.is_file() for path in staging_root.rglob("*"))
    authoritative = RunContext.load(session_name)
    assert not any(
        artifact.artifact_type == "retrieval_plan"
        for artifact in authoritative.run.artifacts.values()
    )


def test_catalog_domain_tool_requires_active_running_allowlisted_task(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_name = "authorization"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    runtime = RunContext.create(
        "chembl-to-gtm-report",
        session_state=ctx.session_state,
        run_id=session_name,
    )
    ctx.run_context = runtime
    runtime.transition_run("running")
    spec = ToolSpec(
        mcp_name="chembl_prepare_retrieval",
        toolkit_factory=lambda: _DummyToolkit(),
        method="structured",
        summary="Dummy preflight.",
        group="chembl",
        roles=("chembl_downloader",),
        profiles=("chembl-retrieval",),
    )
    tool = build_tool(spec, _DummyToolkit(), ctx)

    denied = asyncio.run(tool())

    assert denied["status"] == "error"
    assert denied["error"]["code"] == "permission_denied"
    runtime.add_task(
        TaskRecord(
            task_id="chembl-preflight",
            role="chembl_downloader",
            profile="chembl-retrieval",
            step="Run ChEMBL retrieval preflight.",
        )
    )
    request_path = runtime.layout.artifact_rel_path("inputs/retrieval-request.json")
    with S3.open(request_path, "w") as handle:
        json.dump({"request": "retrieve CDK2 inhibitors"}, handle)
    runtime.register_artifact(
        "inputs/retrieval-request.json",
        artifact_type="retrieval_request",
        mime_type="application/json",
        producer_tool="workflow_start_run",
    )
    runtime.record_handoff(
        HandoffEnvelope.create(
            run_id=runtime.run.run_id,
            workflow_slug=runtime.run.workflow_slug,
            task_id="chembl-preflight",
            sender_role="supervisor",
            receiver_role="chembl_downloader",
            objective="Validate the persisted retrieval request.",
            required_capabilities=("chembl_prepare_retrieval",),
            input_artifact_ids=tuple(runtime.run.workflow_inputs.values()),
            expected_output_artifacts=("retrieval_plan",),
            acceptance_criteria=(
                "Preflight either permits retrieval or records all required "
                "clarification questions.",
            ),
            budget={
                "max_tokens": 1_000,
                "max_tool_calls": 4,
                "timeout_seconds": 60,
            },
        )
    )
    runtime.transition_task("chembl-preflight", "running")
    ctx.session_state.update(
        {
            "active_task_id": "chembl-preflight",
            "active_role": "chembl_downloader",
            "active_profile": "chembl-retrieval",
        }
    )

    allowed = asyncio.run(tool())

    assert allowed["status"] == "success"


def test_catalog_handoff_tool_call_budget_is_enforced(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_name = "authorization-call-budget"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    _bind_catalog_preflight(
        ctx,
        session_name,
        max_tool_calls=1,
        timeout_seconds=60,
    )
    instance = _DummyToolkit()
    spec = ToolSpec(
        mcp_name="chembl_prepare_retrieval",
        toolkit_factory=lambda: instance,
        method="structured",
        summary="Dummy preflight.",
        group="chembl",
        roles=("chembl_downloader",),
        profiles=("chembl-retrieval",),
    )
    tool = build_tool(spec, instance, ctx)

    first = asyncio.run(tool())
    exhausted = asyncio.run(tool())

    assert first["status"] == "success"
    assert exhausted["status"] == "error"
    assert exhausted["error"]["code"] == "resource_limit"
    assert "exhausted its handoff budget" in exhausted["error"]["message"]


def test_catalog_tool_call_budget_is_atomic_across_loaded_contexts(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    session_name = "authorization-cross-context-budget"
    S3.set_session_prefix(f"sessions/{session_name}")
    first_ctx = MCPAgentContext()
    runtime = _bind_catalog_preflight(
        first_ctx,
        session_name,
        max_tool_calls=1,
        timeout_seconds=60,
    )
    second_ctx = MCPAgentContext()
    second_ctx.run_context = RunContext.load(session_name)
    second_ctx.run_context.bind_session_state(second_ctx.session_state)
    bind_active_task_scope(
        second_ctx.session_state,
        second_ctx.run_context.run.tasks["chembl-preflight"],
        run=second_ctx.run_context.run,
    )
    spec = ToolSpec(
        mcp_name="chembl_prepare_retrieval",
        toolkit_factory=_DummyToolkit,
        method="structured",
        summary="Dummy preflight.",
        group="chembl",
        roles=("chembl_downloader",),
        profiles=("chembl-retrieval",),
    )
    first_tool = build_tool(spec, _DummyToolkit(), first_ctx)
    second_tool = build_tool(spec, _DummyToolkit(), second_ctx)

    first = asyncio.run(first_tool())
    exhausted = asyncio.run(second_tool())

    assert first["status"] == "success"
    assert exhausted["status"] == "error"
    assert exhausted["error"]["code"] == "resource_limit"
    authoritative = RunContext.load(session_name)
    starts = [
        event
        for event in authoritative.events
        if event.event_type == "tool_progress" and event.payload.get("stage") == "started"
    ]
    assert len(starts) == 1
    assert runtime.run.run_id == authoritative.run.run_id


def test_catalog_domain_tool_is_denied_while_workflow_run_requires_input(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_name = "authorization-paused-run"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    runtime = _bind_catalog_preflight(
        ctx,
        session_name,
        max_tool_calls=4,
        timeout_seconds=60,
    )
    runtime.transition_run("input_required")
    instance = _DummyToolkit()
    spec = ToolSpec(
        mcp_name="chembl_prepare_retrieval",
        toolkit_factory=lambda: instance,
        method="structured",
        summary="Dummy preflight.",
        group="chembl",
        roles=("chembl_downloader",),
        profiles=("chembl-retrieval",),
    )

    result = asyncio.run(build_tool(spec, instance, ctx)())

    assert result["status"] == "error"
    assert result["error"]["code"] == "permission_denied"
    assert "RUNNING workflow run" in result["error"]["message"]
    assert instance.structured_calls == 0


def test_catalog_pending_domain_span_blocks_failed_task_and_run_transitions(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    session_name = "authorization-pending-failure-transition"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    _bind_catalog_preflight(
        ctx,
        session_name,
        max_tool_calls=4,
        timeout_seconds=60,
    )
    instance = _WaitingCatalogToolkit()
    spec = ToolSpec(
        mcp_name="chembl_prepare_retrieval",
        toolkit_factory=lambda: instance,
        method="structured",
        summary="Waiting preflight.",
        group="chembl",
        roles=("chembl_downloader",),
        profiles=("chembl-retrieval",),
    )
    failure = ToolError(code="internal", message="simulated concurrent failure")

    async def exercise_transitions():
        pending = asyncio.create_task(build_tool(spec, instance, ctx)())
        await instance.started.wait()
        independent = RunContext.load(session_name)
        try:
            with pytest.raises(InvalidTransitionError, match="in flight"):
                independent.transition_task(
                    "chembl-preflight",
                    "failed",
                    error=failure,
                )
            with pytest.raises(InvalidTransitionError, match="in flight"):
                independent.transition_run("failed", error=failure)
        finally:
            instance.release.set()
        return await pending

    result = asyncio.run(exercise_transitions())

    assert result["status"] == "success"
    authoritative = RunContext.load(session_name)
    assert authoritative.run.status.value == "running"
    assert authoritative.run.tasks["chembl-preflight"].status.value == "running"


def test_abandoning_persisted_catalog_span_releases_terminal_transitions(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    session_name = "authorization-abandon-persisted-span"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    runtime = _bind_catalog_preflight(
        ctx,
        session_name,
        max_tool_calls=4,
        timeout_seconds=60,
    )
    span_id = "orphaned-catalog-span"
    _append_persisted_catalog_tool_start(runtime, span_id=span_id)

    authoritative = RunContext.load(session_name)
    failure = ToolError(code="internal", message="worker process crashed")
    assert authoritative.pending_tool_invocations(domain_only=True) == (
        f"chembl_prepare_retrieval ({span_id})",
    )
    with pytest.raises(InvalidTransitionError, match="in flight"):
        authoritative.transition_task(
            "chembl-preflight",
            "failed",
            error=failure,
        )
    with pytest.raises(InvalidTransitionError, match="in flight"):
        authoritative.transition_run("failed", error=failure)

    abandoned = authoritative.abandon_tool_invocation(
        span_id,
        reason="confirmed that the crashed worker process is no longer running",
    )

    assert abandoned.event_type == "tool_progress"
    assert abandoned.payload["stage"] == "abandoned"
    assert abandoned.payload["recovery"] == {
        "confirmed_not_running": True,
        "reason": "confirmed that the crashed worker process is no longer running",
    }
    assert authoritative.pending_tool_invocations(domain_only=True) == ()
    authoritative.transition_task(
        "chembl-preflight",
        "failed",
        error=failure,
    )
    authoritative.transition_run("failed", error=failure)
    assert authoritative.run.tasks["chembl-preflight"].status.value == "failed"
    assert authoritative.run.status.value == "failed"
    with pytest.raises(InvalidTransitionError, match="is not pending"):
        authoritative.abandon_tool_invocation(
            span_id,
            reason="duplicate recovery attempt",
        )


def test_result_accepted_is_terminal_when_completed_progress_persistence_fails(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    session_name = "authorization-accepted-without-completed"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    runtime = _bind_catalog_preflight(
        ctx,
        session_name,
        max_tool_calls=4,
        timeout_seconds=60,
    )
    original_append_event = runtime.append_event

    def fail_completed_progress(event_type, payload, *args, **kwargs):
        if event_type == "tool_progress" and payload.get("stage") == "completed":
            raise OSError("simulated completed-event persistence failure")
        return original_append_event(event_type, payload, *args, **kwargs)

    monkeypatch.setattr(runtime, "append_event", fail_completed_progress)
    instance = _DummyToolkit()
    spec = ToolSpec(
        mcp_name="chembl_prepare_retrieval",
        toolkit_factory=lambda: instance,
        method="structured",
        summary="Dummy preflight.",
        group="chembl",
        roles=("chembl_downloader",),
        profiles=("chembl-retrieval",),
    )

    result = asyncio.run(build_tool(spec, instance, ctx)())

    assert result["status"] == "success"
    authoritative = RunContext.load(session_name)
    stages = [
        event.payload.get("stage")
        for event in authoritative.events
        if event.event_type == "tool_progress"
        and event.payload.get("tool_name") == "chembl_prepare_retrieval"
    ]
    assert stages == ["started", "result_accepted"]
    assert authoritative.pending_tool_invocations(domain_only=True) == ()
    failure = ToolError(code="internal", message="later workflow failure")
    authoritative.transition_task(
        "chembl-preflight",
        "failed",
        error=failure,
    )
    authoritative.transition_run("failed", error=failure)


def test_catalog_tool_discards_result_after_task_attempt_and_handoff_change(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    session_name = "authorization-stale-task-attempt"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    runtime = _bind_catalog_preflight(
        ctx,
        session_name,
        max_tool_calls=4,
        timeout_seconds=60,
    )
    instance = _WaitingCatalogToolkit()
    spec = ToolSpec(
        mcp_name="chembl_prepare_retrieval",
        toolkit_factory=lambda: instance,
        method="structured",
        summary="Waiting preflight.",
        group="chembl",
        roles=("chembl_downloader",),
        profiles=("chembl-retrieval",),
        write_scope="session",
        result_artifact_type="retrieval_plan",
    )
    request_id = next(
        artifact.artifact_id
        for artifact in runtime.run.artifacts.values()
        if artifact.artifact_type == "retrieval_request"
    )

    async def change_attempt_while_running():
        pending = asyncio.create_task(build_tool(spec, instance, ctx)())
        await instance.started.wait()
        # Simulate an independently committed lifecycle event that bypasses
        # this context's high-level in-flight transition guard. The adapter
        # must still discard the stale result defensively.
        runtime.append_event(
            "task_status_changed",
            {"task_id": "chembl-preflight", "status": "input_required"},
        )
        runtime.record_handoff(
            HandoffEnvelope.create(
                run_id=runtime.run.run_id,
                workflow_slug=runtime.run.workflow_slug,
                task_id="chembl-preflight",
                sender_role="supervisor",
                receiver_role="chembl_downloader",
                objective="Retry validation with the clarified request.",
                required_capabilities=("chembl_prepare_retrieval",),
                acceptance_criteria=(
                    "Preflight either permits retrieval or records all required "
                    "clarification questions.",
                ),
                input_artifact_ids=(request_id,),
                expected_output_artifacts=("retrieval_plan",),
                budget={
                    "max_tokens": 1_000,
                    "max_tool_calls": 4,
                    "timeout_seconds": 60,
                },
            )
        )
        task = runtime.transition_task("chembl-preflight", "running")
        bind_active_task_scope(ctx.session_state, task, run=runtime.run)
        instance.release.set()
        return await pending

    result = asyncio.run(change_attempt_while_running())

    assert result["status"] == "error"
    assert result["error"]["code"] == "permission_denied"
    assert "changed while" in result["error"]["message"]
    assert not any(
        artifact.artifact_type == "retrieval_plan" for artifact in runtime.run.artifacts.values()
    )


def test_catalog_domain_tool_rechecks_selected_input_integrity_before_execution(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    session_name = "authorization-input-integrity"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    runtime = _bind_catalog_preflight(
        ctx,
        session_name,
        max_tool_calls=4,
        timeout_seconds=60,
    )
    request_id = runtime.run.workflow_inputs["retrieval_request"]
    request = runtime.run.artifacts[request_id]
    Path(S3.path(runtime.layout.artifact_rel_path(request.relative_path))).write_text(
        '{"request":"tampered"}',
        encoding="utf-8",
    )
    instance = _DummyToolkit()
    spec = ToolSpec(
        mcp_name="chembl_prepare_retrieval",
        toolkit_factory=lambda: instance,
        method="structured",
        summary="Dummy preflight.",
        group="chembl",
        roles=("chembl_downloader",),
        profiles=("chembl-retrieval",),
    )

    result = asyncio.run(build_tool(spec, instance, ctx)())

    assert result["status"] == "error"
    assert result["error"]["code"] == "scientific_validation"
    assert instance.structured_calls == 0


def test_catalog_domain_tool_cannot_overwrite_registered_input_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_name = "authorization-input-immutability"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    runtime = _bind_catalog_preflight(
        ctx,
        session_name,
        max_tool_calls=4,
        timeout_seconds=60,
    )
    request_id = runtime.run.workflow_inputs["retrieval_request"]
    instance = _DummyToolkit()
    spec = ToolSpec(
        mcp_name="chembl_prepare_retrieval",
        toolkit_factory=lambda: instance,
        method="write_output",
        summary="Attempt to replace the registered request.",
        group="chembl",
        roles=("chembl_downloader",),
        profiles=("chembl-retrieval",),
        write_scope="session",
    )

    result = asyncio.run(
        build_tool(spec, instance, ctx)(
            output_path="inputs/retrieval-request.json",
        )
    )

    assert result["status"] == "error"
    assert result["error"]["code"] == "permission_denied"
    assert "immutable registered workflow artifact" in result["error"]["message"]
    assert runtime.verify_artifact(request_id).artifact_id == request_id


def test_catalog_handoff_timeout_budget_is_enforced(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_name = "authorization-time-budget"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    runtime = _bind_catalog_preflight(
        ctx,
        session_name,
        max_tool_calls=4,
        timeout_seconds=1,
    )
    runtime.run.handoffs[-1] = replace(
        runtime.run.handoffs[-1],
        created_at="2000-01-01T00:00:00+00:00",
    )
    instance = _DummyToolkit()
    spec = ToolSpec(
        mcp_name="chembl_prepare_retrieval",
        toolkit_factory=lambda: instance,
        method="structured",
        summary="Dummy preflight.",
        group="chembl",
        roles=("chembl_downloader",),
        profiles=("chembl-retrieval",),
    )

    expired = asyncio.run(build_tool(spec, instance, ctx)())

    assert expired["status"] == "error"
    assert expired["error"]["code"] == "timeout"
    assert "exceeded its handoff timeout" in expired["error"]["message"]
    assert instance.structured_calls == 0


def test_catalog_handoff_timeout_cancels_running_async_tool(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_name = "authorization-async-deadline"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    _bind_catalog_preflight(
        ctx,
        session_name,
        max_tool_calls=4,
        timeout_seconds=0.5,
    )
    instance = _SlowAsyncToolkit()
    spec = ToolSpec(
        mcp_name="chembl_prepare_retrieval",
        toolkit_factory=lambda: instance,
        method="structured",
        summary="Slow async preflight.",
        group="chembl",
        roles=("chembl_downloader",),
        profiles=("chembl-retrieval",),
    )

    result = asyncio.run(build_tool(spec, instance, ctx)())

    assert result["status"] == "error"
    assert result["error"]["code"] == "timeout"
    assert "exceeded its handoff timeout" in result["error"]["message"]
    assert instance.started is True


def test_catalog_handoff_timeout_detects_sync_overrun_after_safe_drain(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_name = "authorization-sync-deadline"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    _bind_catalog_preflight(
        ctx,
        session_name,
        max_tool_calls=4,
        timeout_seconds=0.5,
    )
    instance = _SlowSyncToolkit()
    spec = ToolSpec(
        mcp_name="chembl_prepare_retrieval",
        toolkit_factory=lambda: instance,
        method="structured",
        summary="Slow synchronous preflight.",
        group="chembl",
        roles=("chembl_downloader",),
        profiles=("chembl-retrieval",),
    )

    result = asyncio.run(build_tool(spec, instance, ctx)())

    assert result["status"] == "error"
    assert result["error"]["code"] == "timeout"
    assert "exceeded its handoff timeout" in result["error"]["message"]
    assert instance.started is True
