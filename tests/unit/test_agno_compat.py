"""Compatibility fix: async tool pre-hook retries reach the model."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from agno.agent import Agent
from agno.exceptions import RetryAgentRun
from agno.models.base import Model
from agno.models.response import ModelResponse
from agno.tools.function import Function

from cs_copilot.agents.agno_compat import patch_async_function_call_retry


@dataclass
class _ScriptedModel(Model):
    id: str = "scripted"
    provider: str = "test"
    script: list = field(default_factory=list)
    seen: list = field(default_factory=list)

    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self._next(kwargs)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self._next(kwargs)

    def _next(self, kwargs: dict[str, Any]) -> ModelResponse:
        self.seen.append([str(message.content) for message in kwargs.get("messages", [])])
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


def _guarded_tool() -> Function:
    def delegate(task: str) -> str:
        """Delegate a task.

        Args:
            task: The task.
        """
        return f"delegated {task}"

    function = Function.from_callable(delegate)

    def reject(fc: Any) -> None:
        raise RetryAgentRun("Delegation rejected: pass a structured handoff.")

    function.pre_hook = reject
    return function


def test_patch_is_active_and_idempotent():
    assert patch_async_function_call_retry() is True
    assert patch_async_function_call_retry() is True


def test_async_pre_hook_retry_is_returned_to_the_model():
    patch_async_function_call_retry()
    model = _ScriptedModel(
        script=[
            ModelResponse(
                role="assistant",
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "delegate", "arguments": json.dumps({"task": "x"})},
                    }
                ],
            ),
            ModelResponse(role="assistant", content="recovered"),
        ]
    )
    agent = Agent(name="coordinator", model=model, tools=[_guarded_tool()], telemetry=False)

    output = asyncio.run(agent.arun("go"))

    assert output.content == "recovered"
    assert any("pass a structured handoff" in message for message in model.seen[-1])


def test_literal_parameters_reach_the_model_as_enums_not_opaque_objects():
    """Agno 2.1.9 has no Literal branch, so such parameters lose their values.

    The model is then rejected at runtime for a constraint it was never shown,
    and a model that obeys the advertised object schema sends a dict and fails
    the same way. Nested Optional/List forms resolve through the same function,
    so they are covered too.
    """
    from typing import Literal, Optional

    from agno.utils import json_schema as agno_json_schema

    from cs_copilot.agents.agno_compat import patch_literal_enum_schema

    assert patch_literal_enum_schema()

    assert agno_json_schema.get_json_schema_for_arg(Literal["text", "dataframe"]) == {
        "type": "string",
        "enum": ["text", "dataframe"],
    }
    assert agno_json_schema.get_json_schema_for_arg(Literal[1, 2]) == {
        "type": "integer",
        "enum": [1, 2],
    }
    optional = agno_json_schema.get_json_schema_for_arg(Optional[Literal["a", "b"]])
    assert {"type": "string", "enum": ["a", "b"]} in optional["anyOf"]

    # Non-Literal hints keep agno's own behaviour.
    assert agno_json_schema.get_json_schema_for_arg(str) == {"type": "string"}


def test_the_literal_patch_is_idempotent():
    from agno.utils import json_schema as agno_json_schema

    from cs_copilot.agents.agno_compat import patch_literal_enum_schema

    assert patch_literal_enum_schema()
    once = agno_json_schema.get_json_schema_for_arg
    assert patch_literal_enum_schema()
    assert agno_json_schema.get_json_schema_for_arg is once
