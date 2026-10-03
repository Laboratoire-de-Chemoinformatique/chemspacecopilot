"""End to end: a scripted Agno model drives tool calls through the kernel."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from agno.agent import Agent
from agno.models.base import Model
from agno.models.response import ModelResponse
from agno.tools import Toolkit

from cs_copilot.agents.execution_binding import attach_execution
from cs_copilot.storage import S3
from cs_copilot.workflows import RunContext


@dataclass
class _ScriptedModel(Model):
    """Returns queued responses: tool calls first, then a final answer."""

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


class NotesToolkit(Toolkit):
    def __init__(self) -> None:
        super().__init__("notes")
        self.register(self.save_note)

    def save_note(self, name: str, text: str, session_state: Any = None) -> str:
        """Save a note inside the chat's workflow run.

        Args:
            name: File name of the note.
            text: Note contents.
        """
        run_root = session_state["output_context"]
        path = f"workflows/{run_root['run_id']}/notes/{name}.txt"
        with S3.open(path, "w") as handle:
            handle.write(text)
        return path


def _tool_call(call_id: str, function_name: str, /, **arguments: Any) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": function_name, "arguments": json.dumps(arguments)},
    }


@pytest.fixture
def chat(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    S3.set_session_prefix("sessions/e2e-chat")
    state: dict[str, Any] = {}
    run = RunContext.create("agno-session", session_state=state, session_id="e2e-chat")
    run.transition_run("running")

    def make_agent(script: list[ModelResponse]) -> Agent:
        agent = Agent(
            name="notes member",
            model=_ScriptedModel(script=script),
            tools=[NotesToolkit()],
            session_state=state,
            telemetry=False,
        )
        agent.agentic_role = "report_generator"
        team = SimpleNamespace(session_state=state, members=[agent], tools=[], run_context=run)
        attach_execution(team, run_context=run, mode="observe")
        return agent

    return SimpleNamespace(run=run, make_agent=make_agent)


def _recorded(run: RunContext) -> list[dict[str, Any]]:
    run.refresh()
    return [event.payload for event in run.events if event.event_type == "tool_call_recorded"]


def test_sync_agent_run_records_tool_calls_and_artifacts(chat):
    agent = chat.make_agent(
        [
            ModelResponse(
                role="assistant",
                tool_calls=[_tool_call("c1", "save_note", name="first", text="hello")],
            ),
            ModelResponse(role="assistant", content="done"),
        ]
    )

    output = agent.run("save a note", session_id="e2e-chat")

    assert output.content == "done"
    [call] = _recorded(chat.run)
    assert call["tool_name"] == "agno.NotesToolkit.save_note"
    assert call["role"] == "report_generator"
    assert call["status"] == "success"
    [artifact] = chat.run.run.artifacts.values()
    assert artifact.relative_path == "notes/first.txt"
    assert artifact.producer_tool == "agno.NotesToolkit.save_note"


def test_async_agent_run_records_parallel_tool_calls(chat):
    agent = chat.make_agent(
        [
            ModelResponse(
                role="assistant",
                tool_calls=[
                    _tool_call("c1", "save_note", name="a", text="1"),
                    _tool_call("c2", "save_note", name="b", text="2"),
                ],
            ),
            ModelResponse(role="assistant", content="done"),
        ]
    )

    output = asyncio.run(agent.arun("save two notes", session_id="e2e-chat"))

    assert output.content == "done"
    calls = _recorded(chat.run)
    assert len(calls) == 2
    assert len({call["span_id"] for call in calls}) == 2
    assert {artifact.relative_path for artifact in chat.run.run.artifacts.values()} == {
        "notes/a.txt",
        "notes/b.txt",
    }
    assert chat.run.pending_tool_invocations() == ()
