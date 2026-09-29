"""``agno_team_run`` runs the in-process team under the MCP kernel.

A stand-in team keeps these tests about the delegation contract: policy
checks, state hand-off, span linkage, cancellation, and the one-run-at-a-time
rule. ``test_production_team_tool_calls_are_linked_to_the_outer_span`` drives
the real team with a scripted model.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

import pytest
from agno.models.base import Model
from agno.models.response import ModelResponse

from cs_copilot.mcp.agno_delegate import AgnoTeamFacade
from cs_copilot.mcp.context import MCPAgentContext
from cs_copilot.mcp.errors import MCPToolError
from cs_copilot.mcp.tool_adapter import build_tool
from cs_copilot.mcp.tools_registry import OPT_IN_GROUPS, all_specs
from cs_copilot.storage import S3
from cs_copilot.workflows import RunContext, TaskStatus


class FakeTeam:
    """The part of ``agno.team.Team`` the facade uses."""

    def __init__(self, behaviour: Callable[["FakeTeam"], Awaitable[Any]]) -> None:
        self.session_state: dict[str, Any] = {"resource_profile": {"cpu": "test"}}
        self.behaviour = behaviour
        self.entered = asyncio.Event()
        self.cancelled: list[str] = []
        self.seen_output_context: dict[str, Any] = {}

    async def arun(self, prompt: str, *, stream: bool, session_id: str) -> Any:
        self.session_state["current_run_id"] = "agno-run-1"
        self.seen_output_context = dict(self.session_state.get("output_context") or {})
        self.entered.set()
        return await self.behaviour(self)

    def cancel_run(self, run_id: str) -> bool:
        self.cancelled.append(run_id)
        return True


async def _answer(team: FakeTeam) -> Any:
    team.session_state["notes"] = {"summary": "two clusters"}
    team.session_state["frame"] = object()
    team.session_state["active_task_id"] = "forged"
    return SimpleNamespace(content="team answer")


async def _hang(team: FakeTeam) -> Any:
    await asyncio.Event().wait()


async def _explode(team: FakeTeam) -> Any:
    team.session_state["notes"] = "partial"
    raise RuntimeError("model exploded")


def _session(tmp_path, monkeypatch, name: str, *, workflow_slug: str = "mcp-session"):
    monkeypatch.chdir(tmp_path)
    S3.set_session_prefix(f"sessions/{name}")
    ctx = MCPAgentContext(llm_policy="agno-model", model=object())
    ctx.run_context = RunContext.create(
        workflow_slug,
        session_state=ctx.session_state,
        run_id=name,
    )
    ctx.session_state["mcp_profile"] = "standard"
    return ctx


def _tool(ctx: MCPAgentContext, team: Any, **spec_overrides: Any):
    [spec] = [spec for spec in all_specs(opt_in_groups=OPT_IN_GROUPS) if spec.group == "agno"]
    facade = AgnoTeamFacade()
    if team is not None:
        facade._team_for = lambda _ctx, _run_context: team  # type: ignore[method-assign]
    return build_tool(replace(spec, **spec_overrides), facade, ctx)


def _calls(ctx: MCPAgentContext) -> list[dict[str, Any]]:
    ctx.run_context.refresh()
    return [
        event.payload
        for event in ctx.run_context.events
        if event.event_type == "tool_call_recorded"
    ]


# -- policy -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("policy", "model"),
    [("external", None), ("disabled", None), ("agno-model", None)],
)
def test_the_team_needs_the_servers_own_model(tmp_path, monkeypatch, policy, model):
    ctx = _session(tmp_path, monkeypatch, f"team-policy-{policy}")
    ctx.llm_policy, ctx.model = policy, model
    team = FakeTeam(_answer)

    envelope = asyncio.run(_tool(ctx, team)(prompt="map the dataset"))

    assert envelope["error"]["code"] == "permission_denied"
    assert "--llm-policy agno-model" in envelope["error"]["message"]
    assert not team.entered.is_set()


def test_catalog_runs_are_refused(tmp_path, monkeypatch):
    ctx = _session(tmp_path, monkeypatch, "team-catalog", workflow_slug="chembl-to-gtm-report")

    with pytest.raises(MCPToolError, match="catalog workflow"):
        asyncio.run(AgnoTeamFacade().run("map the dataset", agent=ctx))


def test_an_empty_prompt_is_invalid(tmp_path, monkeypatch):
    ctx = _session(tmp_path, monkeypatch, "team-empty")

    envelope = asyncio.run(_tool(ctx, FakeTeam(_answer))(prompt="  "))

    assert envelope["error"]["code"] == "invalid_input"


# -- execution --------------------------------------------------------------


def test_the_answer_and_shareable_state_come_back(tmp_path, monkeypatch):
    ctx = _session(tmp_path, monkeypatch, "team-answer")
    output_context = dict(ctx.session_state["output_context"])
    team = FakeTeam(_answer)

    envelope = asyncio.run(_tool(ctx, team)(prompt="summarize the map"))

    assert envelope["status"] == "success", envelope["error"]
    data = envelope["data"]
    assert data["content"] == "team answer"
    assert data["runtime"] == "agno_team"
    assert data["run_id"] == ctx.run_context.run.run_id
    assert ctx.session_state["notes"] == {"summary": "two clusters"}
    for private in ("frame", "active_task_id", "current_run_id", "resource_profile"):
        assert private not in ctx.session_state, private
    assert ctx.session_state["output_context"] == output_context
    assert ctx.session_state["mcp_profile"] == "standard"
    assert any("frame" in warning for warning in data["warnings"])
    [outer] = [call for call in _calls(ctx) if call["tool_name"] == "agno_team_run"]
    assert outer["runtime"] == "mcp" and outer["status"] == "success"
    assert team.seen_output_context["parent_span_id"] == outer["span_id"]
    assert "mcp_profile" not in team.session_state


def test_in_memory_session_objects_are_shared_by_reference(tmp_path, monkeypatch):
    ctx = _session(tmp_path, monkeypatch, "team-objects")
    fitted_map, replaced = object(), object()
    ctx.session_state["ligand_map"] = fitted_map
    ctx.session_state["scratch_model"] = replaced
    seen: dict[str, Any] = {}

    async def use_the_map(team: FakeTeam) -> Any:
        seen["map"] = team.session_state["ligand_map"]
        team.session_state["scratch_model"] = object()
        team.session_state["notes"] = "used the map"
        return SimpleNamespace(content="done")

    envelope = asyncio.run(_tool(ctx, FakeTeam(use_the_map))(prompt="project the ligands"))

    assert envelope["status"] == "success", envelope["error"]
    assert seen["map"] is fitted_map
    assert ctx.session_state["ligand_map"] is fitted_map
    assert ctx.session_state["scratch_model"] is replaced
    assert ctx.session_state["notes"] == "used the map"
    [warning] = envelope["data"]["warnings"]
    assert "scratch_model" in warning and "ligand_map" not in warning


def test_delegated_work_can_start_the_session_run(tmp_path, monkeypatch):
    ctx = _session(tmp_path, monkeypatch, "team-start-run")
    seen: dict[str, Any] = {}

    async def delegate(team: FakeTeam) -> Any:
        # What the delegation guard does when the team hands off its first task.
        seen["domain"] = ctx.run_context.pending_tool_invocations(domain_only=True)
        seen["all"] = ctx.run_context.pending_tool_invocations()
        await asyncio.to_thread(ctx.run_context.transition_run, "running", reason="delegated")
        return SimpleNamespace(content="started")

    envelope = asyncio.run(_tool(ctx, FakeTeam(delegate))(prompt="map the dataset"))

    assert envelope["status"] == "success", envelope["error"]
    assert seen["domain"] == () and len(seen["all"]) == 1
    assert ctx.run_context.refresh().status.value == "running"
    [started] = [
        event.payload
        for event in ctx.run_context.events
        if event.event_type == "tool_progress"
        and event.payload["tool_name"] == "agno_team_run"
        and event.payload["stage"] == "started"
    ]
    assert started["delegates_execution"] is True


def test_an_orphaned_delegating_span_is_not_domain_work_but_can_be_abandoned(tmp_path, monkeypatch):
    ctx = _session(tmp_path, monkeypatch, "team-orphan")
    run = ctx.run_context
    run.append_event(
        "tool_progress",
        {
            "runtime": "mcp",
            "session_id": run.run.session_id,
            "run_id": run.run.run_id,
            "workflow_slug": run.run.workflow_slug,
            "trace_id": run.run.trace_id,
            "span_id": "outerspan",
            "parent_span_id": None,
            "tool_name": "agno_team_run",
            "task_id": None,
            "role": "supervisor",
            "profile": "standard",
            "stage": "started",
            "attempt": 0,
            "max_attempts": 1,
            "cached": False,
            "task_attempt": None,
            "handoff_id": None,
            "delegates_execution": True,
        },
    )

    assert run.pending_tool_invocations(domain_only=True) == ()
    assert run.pending_tool_invocations() == ("agno_team_run (outerspan)",)
    run.abandon_tool_invocation("outerspan", reason="the MCP server restarted")
    assert run.pending_tool_invocations() == ()


def test_a_failed_team_run_merges_nothing(tmp_path, monkeypatch):
    ctx = _session(tmp_path, monkeypatch, "team-failure")

    envelope = asyncio.run(_tool(ctx, FakeTeam(_explode))(prompt="summarize the map"))

    assert envelope["status"] == "error"
    assert "model exploded" in envelope["error"]["message"]
    assert "notes" not in ctx.session_state


def test_a_timeout_cancels_the_team_and_frees_the_session(tmp_path, monkeypatch):
    ctx = _session(tmp_path, monkeypatch, "team-timeout")
    team = FakeTeam(_hang)
    tool = _tool(ctx, team, timeout_s=0.2)

    async def scenario() -> tuple[dict[str, Any], dict[str, Any]]:
        timed_out = await tool(prompt="never finishes")
        team.behaviour = _answer
        return timed_out, await tool(prompt="summarize the map")

    timed_out, retried = asyncio.run(scenario())

    assert timed_out["error"]["code"] == "timeout"
    assert team.cancelled == ["agno-run-1"]
    assert retried["status"] == "success"


def test_one_team_run_at_a_time(tmp_path, monkeypatch):
    ctx = _session(tmp_path, monkeypatch, "team-busy")
    release = asyncio.Event()

    async def wait_for_release(team: FakeTeam) -> Any:
        await release.wait()
        return SimpleNamespace(content="first")

    team = FakeTeam(wait_for_release)
    tool = _tool(ctx, team)

    async def scenario() -> tuple[dict[str, Any], dict[str, Any]]:
        first = asyncio.create_task(tool(prompt="first"))
        await team.entered.wait()
        second = await tool(prompt="second")
        release.set()
        return await first, second

    first, second = asyncio.run(scenario())

    assert second["error"]["code"] == "resource_limit"
    assert "already running" in second["error"]["message"]
    assert first["data"]["content"] == "first"


# -- the production team ----------------------------------------------------


@dataclass
class _ScriptedModel(Model):
    id: str = "scripted"
    provider: str = "test"
    script: list = field(default_factory=list)

    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self.script.pop(0)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self.script.pop(0)

    def invoke_stream(self, *args: Any, **kwargs: Any):
        raise NotImplementedError
        yield

    async def ainvoke_stream(self, *args: Any, **kwargs: Any):
        raise NotImplementedError
        yield

    def _parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:
        raise NotImplementedError

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        raise NotImplementedError


def _offline_production_team(monkeypatch) -> None:
    from cs_copilot.agents import teams
    from cs_copilot.tools.chemistry.autoencoder_toolkit import AutoencoderToolkit
    from cs_copilot.tools.chemistry.peptide_designer_toolkit import PeptideDesignerToolkit

    for cls in (AutoencoderToolkit, PeptideDesignerToolkit):
        for name in ("_ensure_model_exists", "_load_model"):
            if hasattr(cls, name):
                monkeypatch.setattr(cls, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(teams, "analyze_resources", lambda: {"cpu": "test"})


def _tool_call(call_id: str, name: str, /, **arguments: Any) -> ModelResponse:
    return ModelResponse(
        role="assistant",
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    )


def test_production_team_tool_calls_are_linked_to_the_outer_span(tmp_path, monkeypatch):
    _offline_production_team(monkeypatch)
    ctx = _session(tmp_path, monkeypatch, "team-production")
    ctx.model = _ScriptedModel(
        script=[
            _tool_call("call-1", "list_skills"),
            ModelResponse(role="assistant", content="The catalog has skills."),
        ]
    )

    envelope = asyncio.run(_tool(ctx, None)(prompt="Which skills are available?"))

    assert envelope["status"] == "success", envelope["error"]
    assert envelope["data"]["content"] == "The catalog has skills."
    calls = {call["tool_name"]: call for call in _calls(ctx)}
    outer, inner = calls["agno_team_run"], calls["skill_list"]
    assert (outer["runtime"], inner["runtime"]) == ("mcp", "agno")
    assert inner["parent_span_id"] == outer["span_id"]
    assert inner["role"] == "coordinator" and inner["status"] == "success"


def test_production_team_delegation_registers_the_members_report(tmp_path, monkeypatch):
    _offline_production_team(monkeypatch)
    ctx = _session(tmp_path, monkeypatch, "team-delegation")
    handoff = {
        "schema_version": 2,
        "run_id": "guessed-by-the-model",
        "workflow_slug": "ad-hoc",
        "task_id": "ethanol-report",
        "sender_role": "coordinator",
        "receiver_role": "report_generator",
        "objective": "Save a short markdown report on ethanol.",
        "constraints": [],
        "required_capabilities": ["report_save_markdown"],
        "input_artifact_ids": [],
        "expected_output_artifacts": ["markdown_report_path"],
        "expected_output_schema": {"type": "object"},
        "acceptance_criteria": ["The report is saved."],
        "context_summary": "Ethanol weighs 46.07 g/mol.",
        "budget": {"max_tokens": 2_000, "max_tool_calls": 4, "timeout_seconds": 120},
        "trace_id": "trace-guess",
        "span_id": "span-1",
    }
    ctx.model = _ScriptedModel(
        script=[
            _tool_call(
                "call-1",
                "delegate_task_to_member",
                member_id="report-generator-agent",
                task_description=json.dumps(handoff),
            ),
            _tool_call(
                "call-2",
                "save_markdown_report",
                content="# Ethanol\n\nMolecular weight: 46.07 g/mol\n",
                filename="ethanol",
                report_type="descriptors",
            ),
            ModelResponse(role="assistant", content="Saved the report."),
            ModelResponse(role="assistant", content="The ethanol report is saved."),
        ]
    )

    envelope = asyncio.run(_tool(ctx, None)(prompt="Save a short ethanol report."))

    assert envelope["status"] == "success", envelope["error"]
    assert envelope["data"]["content"] == "The ethanol report is saved."
    run = ctx.run_context.refresh()
    assert run.status.value == "running"
    assert run.tasks["ethanol-report"].status is TaskStatus.COMPLETED
    [report] = run.artifacts.values()
    assert report.relative_path == "reports/descriptors/ethanol.md"
    assert (report.producer_tool, report.producer_task_id) == (
        "report_save_markdown",
        "ethanol-report",
    )
    assert envelope["data"]["artifact_ids"] == [report.artifact_id]
