"""Paths a tool returns are registered only when the call produced them."""

from __future__ import annotations

from typing import Any

import pytest

from cs_copilot.execution.runner import bind_sync_invoker, execute_sync
from cs_copilot.storage import S3

from ._kernel_helpers import make_spec


class _Toolkit:
    def write_output(self, output_path: str) -> dict[str, Any]:
        with S3.open(output_path, "w") as handle:
            handle.write("table")
        return {"output_path": output_path}

    def list_outputs(self, output_path: str) -> dict[str, Any]:
        return {"objects": [{"id": "table", "output_path": output_path}]}

    def write_and_list(self, output_path: str, listed: str) -> dict[str, Any]:
        with S3.open(output_path, "w") as handle:
            handle.write("summary")
        return {"output_path": output_path, "listed_path": listed}


def _produce(ctx) -> str:
    toolkit = _Toolkit()
    spec = make_spec("write_output", _Toolkit, write_scope="session")
    outcome = execute_sync(
        spec,
        ctx,
        {"output_path": "tables/result.csv"},
        invoke=bind_sync_invoker(spec, toolkit.write_output),
    )
    assert outcome.ok
    return outcome.value["output_path"]


def test_read_only_results_only_refer_to_existing_artifacts(bound_context):
    ctx = bound_context("result-paths-read-only")
    path = _produce(ctx)
    spec = make_spec("list_outputs", _Toolkit, read_only=True)

    outcome = execute_sync(
        spec,
        ctx,
        {"output_path": path},
        invoke=bind_sync_invoker(spec, _Toolkit().list_outputs),
    )

    assert outcome.ok
    assert outcome.envelope["warnings"] == []
    assert len(ctx.run_context.refresh().artifacts) == 1


@pytest.mark.parametrize("strict", [True, False])
def test_listing_another_calls_artifact(bound_context, strict):
    ctx = bound_context(f"result-paths-listing-{strict}")
    path = _produce(ctx)
    spec = make_spec("write_and_list", _Toolkit, write_scope="session")

    outcome = execute_sync(
        spec,
        ctx,
        {"output_path": "tables/summary.csv", "listed": path},
        invoke=bind_sync_invoker(spec, _Toolkit().write_and_list),
        publication_policy="result_paths" if strict else "all_published",
    )

    assert outcome.ok
    run = ctx.run_context.refresh()
    produced = {record.relative_path: record for record in run.artifacts.values()}
    assert produced["tables/summary.csv"].producer_tool == "kernel_write_and_list"
    assert produced["tables/result.csv"].producer_tool == "kernel_write_output"
    # MCP keeps reporting a foreign result path; in-process runs, which
    # register every published file anyway, treat it as a reference.
    assert bool(outcome.envelope["warnings"]) is strict
