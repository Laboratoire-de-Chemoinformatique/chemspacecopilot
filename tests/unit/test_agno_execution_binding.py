"""Routing in-process Agno tool calls through the execution kernel."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest
from agno.agent import Agent
from agno.tools import Toolkit
from agno.tools.function import Function, FunctionCall

from cs_copilot.agents.execution_binding import (
    ExecutionMode,
    attach_execution,
    execution_mode_from_env,
    get_binding,
)
from cs_copilot.storage import S3
from cs_copilot.tools.io.skill_toolkit import SkillToolkit
from cs_copilot.workflows import RunContext


class ProbeToolkit(Toolkit):
    def __init__(self) -> None:
        super().__init__("probe")
        self.seen: list[dict[str, Any]] = []
        self.register(self.echo)
        self.register(self.remember)

    def echo(self, text: str, repeat: int = 1) -> str:
        """Repeat text.

        Args:
            text: Text to repeat.
            repeat: How many times.
        """
        return text * repeat

    def remember(
        self,
        key: str,
        agent: Optional[Any] = None,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record which Agno context the toolkit received.

        Args:
            key: Name to record.
        """
        self.seen.append({"key": key, "agent": agent, "state": session_state})
        return key


def plot_probe(title: str) -> str:
    """Pretend to save a plot.

    Args:
        title: Plot title.
    """
    return f"saved {title}"


@pytest.fixture
def chat(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    S3.set_session_prefix("sessions/binding-chat")
    state: dict[str, Any] = {"agentic_contracts": {}}
    run = RunContext.create("agno-session", session_state=state, session_id="binding-chat")
    run.transition_run("running")
    toolkit = ProbeToolkit()
    member = Agent(name="gtm member", tools=[toolkit, plot_probe])
    member.agentic_role = "gtm_agent"
    coordinator_toolkit = SkillToolkit()
    team = SimpleNamespace(
        session_state=state, members=[member], tools=[coordinator_toolkit], run_context=run
    )
    return SimpleNamespace(
        team=team,
        member=member,
        toolkit=toolkit,
        coordinator_toolkit=coordinator_toolkit,
        run=run,
        state=state,
    )


def _execute(function: Function, arguments: dict[str, Any], *, agent=None, team=None, state=None):
    function.process_entrypoint()
    function._agent = agent
    function._team = team
    function._session_state = state
    return FunctionCall(function=function, arguments=arguments).execute()


def _tool_calls(run: RunContext) -> list[dict[str, Any]]:
    run.refresh()
    return [event.payload for event in run.events if event.event_type == "tool_call_recorded"]


def test_wrapping_keeps_the_model_facing_schema(chat):
    before = {
        name: Function(name=name, entrypoint=function.entrypoint)
        for name, function in chat.toolkit.functions.items()
    }
    for function in before.values():
        function.process_entrypoint()

    attach_execution(chat.team, run_context=chat.run, mode="observe")

    for name, function in chat.toolkit.functions.items():
        function.process_entrypoint()
        assert function.parameters == before[name].parameters, name


def test_member_calls_are_recorded_with_role_and_default_identity(chat):
    attach_execution(chat.team, run_context=chat.run, mode="observe")

    result = _execute(
        chat.toolkit.functions["echo"],
        {"text": "ab", "repeat": "2"},
        agent=chat.member,
        state=chat.state,
    )

    assert result.status == "success" and result.result == "abab"
    [call] = _tool_calls(chat.run)
    assert call["runtime"] == "agno"
    assert call["tool_name"] == "agno.ProbeToolkit.echo"
    assert call["role"] == "gtm_agent"
    assert call["profile"] == "gtm-analysis"
    assert call["public_args"] == {"text": "ab", "repeat": 2}


def test_toolkits_still_receive_agno_context(chat):
    attach_execution(chat.team, run_context=chat.run, mode="observe")

    _execute(chat.toolkit.functions["remember"], {"key": "k"}, agent=chat.member, state=chat.state)

    [seen] = chat.toolkit.seen
    assert seen["agent"] is chat.member
    assert seen["state"] is not None and seen["state"]["output_context"]["run_id"] == (
        chat.run.run.run_id
    )
    [call] = _tool_calls(chat.run)
    assert "agent" not in call["public_args"] and "session_state" not in call["public_args"]


def test_coordinator_toolkits_use_catalog_identity(chat):
    attach_execution(chat.team, run_context=chat.run, mode="observe")

    result = _execute(
        chat.coordinator_toolkit.functions["list_skills"],
        {},
        team=chat.team,
        state=chat.state,
    )

    assert result.status == "success"
    [call] = _tool_calls(chat.run)
    assert call["tool_name"] == "skill_list"
    assert call["role"] == "coordinator"


def test_bare_function_tools_are_wrapped(chat):
    attach_execution(chat.team, run_context=chat.run, mode="observe")

    [wrapped] = [
        tool for tool in chat.member.tools if callable(tool) and not isinstance(tool, Toolkit)
    ]
    assert wrapped.__name__ == "plot_probe"
    function = Function.from_callable(wrapped)
    original = Function.from_callable(plot_probe)
    function.process_entrypoint()
    original.process_entrypoint()
    assert function.parameters == original.parameters

    assert _execute(function, {"title": "map"}, agent=chat.member, state=chat.state).result == (
        "saved map"
    )
    [call] = _tool_calls(chat.run)
    assert call["tool_name"] == "agno.test_agno_execution_binding.plot_probe"


def test_attach_is_idempotent(chat):
    first = attach_execution(chat.team, run_context=chat.run, mode="observe")
    entrypoint = chat.toolkit.functions["echo"].entrypoint
    second = attach_execution(chat.team, run_context=chat.run, mode="observe")

    assert first is second is get_binding(chat.team)
    assert chat.toolkit.functions["echo"].entrypoint is entrypoint
    _execute(chat.toolkit.functions["echo"], {"text": "x"}, agent=chat.member, state=chat.state)
    assert len(_tool_calls(chat.run)) == 1


def test_off_mode_changes_nothing(chat):
    entrypoint = chat.toolkit.functions["echo"].entrypoint

    assert attach_execution(chat.team, run_context=chat.run, mode="off") is None
    assert chat.toolkit.functions["echo"].entrypoint == entrypoint
    assert get_binding(chat.team) is None


def test_calls_dispatched_to_a_worker_thread_are_recorded(chat):
    attach_execution(chat.team, run_context=chat.run, mode="observe")
    function = chat.toolkit.functions["echo"]
    function.process_entrypoint()
    function._agent = chat.member
    function._session_state = chat.state

    async def dispatch_like_agno() -> Any:
        return await asyncio.to_thread(
            FunctionCall(function=function, arguments={"text": "t"}).execute
        )

    assert asyncio.run(dispatch_like_agno()).result == "t"
    assert len(_tool_calls(chat.run)) == 1


def test_agentic_state_updates_cannot_replace_run_identity(chat):
    attach_execution(chat.team, run_context=chat.run, mode="observe")
    run_identity = dict(chat.state["output_context"])

    message = chat.member.update_session_state(
        chat.state,
        {"output_context": {"run_id": "forged"}, "active_task_id": "x", "notes": "kept"},
    )

    assert chat.state["output_context"] == run_identity
    assert "active_task_id" not in chat.state
    assert chat.state["notes"] == "kept"
    assert "active_task_id, output_context" in message


def test_execution_mode_resolution(monkeypatch):
    monkeypatch.delenv("CS_COPILOT_AGNO_EXECUTION", raising=False)
    assert execution_mode_from_env() is ExecutionMode.OBSERVE
    monkeypatch.setenv("CS_COPILOT_AGNO_EXECUTION", "OFF")
    assert execution_mode_from_env() is ExecutionMode.OFF
    monkeypatch.setenv("CS_COPILOT_AGNO_EXECUTION", "sometimes")
    with pytest.raises(ValueError, match="CS_COPILOT_AGNO_EXECUTION"):
        execution_mode_from_env()
