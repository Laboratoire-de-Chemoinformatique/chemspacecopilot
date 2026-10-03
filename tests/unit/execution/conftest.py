"""Shared fixtures for execution-kernel tests."""

from __future__ import annotations

import pytest

from cs_copilot.execution.context import AGNO_RUNTIME, BasicExecutionContext, RuntimeProfile
from cs_copilot.storage import S3
from cs_copilot.workflows import RunContext


@pytest.fixture
def bound_context(tmp_path, monkeypatch):
    """Return a factory for execution contexts bound to a fresh v2 run."""

    monkeypatch.chdir(tmp_path)

    def make(
        name: str,
        *,
        workflow_slug: str = "agno-session",
        runtime: RuntimeProfile = AGNO_RUNTIME,
    ) -> BasicExecutionContext:
        S3.set_session_prefix(f"sessions/{name}")
        ctx = BasicExecutionContext(execution_runtime=runtime)
        ctx.run_context = RunContext.create(
            workflow_slug,
            session_state=ctx.session_state,
            run_id=name,
        )
        return ctx

    return make
