"""Offline regression tests for benchmark isolation and honest failure accounting."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROBUSTNESS_DIR = Path(__file__).parents[1] / "robustness"
sys.path.insert(0, str(ROBUSTNESS_DIR))

from reliability.reporting import save_reliability_bundle, summarize_records  # noqa: E402
from reliability.telemetry import normalize_agno_output  # noqa: E402
from robustness_minimal_example import (  # noqa: E402
    RobustnessConfig,
    RobustnessRunner,
)
from robustness_minimal_example import (  # noqa: E402
    TestConfig as RunnerTestConfig,
)

from cs_copilot.storage import S3  # noqa: E402
from cs_copilot.storage import client as storage_client  # noqa: E402


def output(*, tools=(), tokens=None, content="completed"):
    metrics = (
        None
        if tokens is None
        else SimpleNamespace(
            input_tokens=tokens, output_tokens=0, total_tokens=tokens, duration=0.1
        )
    )
    return SimpleNamespace(content=content, tools=list(tools), metrics=metrics)


def tool(name="test_tool", result=None, error=False):
    return SimpleNamespace(tool_name=name, result=result, tool_call_error=error)


@pytest.fixture
def local_storage(monkeypatch, tmp_path):
    monkeypatch.setattr(storage_client, "is_s3_enabled", lambda: False)
    monkeypatch.setattr(storage_client, "LOCAL_STORAGE_ROOT", tmp_path / "storage")
    original = S3.prefix
    yield
    S3.prefix = original


def runner(tmp_path, repetitions=2):
    return RobustnessRunner(
        RobustnessConfig(
            output_dir=str(tmp_path / "reports"),
            repetitions=repetitions,
            s3_session_isolation=False,
            reliability_enabled=True,
        )
    )


def case(**kwargs):
    return RunnerTestConfig(name="case", enabled=True, prompt_key="", **kwargs)


def test_local_repetitions_write_to_distinct_sessions_and_restore_prefix(
    monkeypatch,
    tmp_path,
    local_storage,
):
    harness = runner(tmp_path)
    observed = []
    original_prefix = S3.prefix

    class Agent:
        session_state = {}

        def run(self, prompt, stream=False):
            observed.append((self.session_id, S3.prefix, S3.path("result.txt")))
            with S3.open("result.txt", "w") as handle:
                handle.write(self.session_id)
            return output()

    monkeypatch.setattr(harness, "_build_system", Agent)
    runs = harness._run_independent_test(case(), ["run"])
    assert len({item[0] for item in observed}) == 2
    assert len({item[2] for item in observed}) == 2
    for record, (session_id, prefix, path) in zip(runs, observed, strict=True):
        assert record["session_id"] == session_id
        assert prefix == f"sessions/{session_id}"
        assert Path(path).read_text() == session_id
    assert S3.prefix == original_prefix


def test_chain_shares_storage_and_agent_session_only_within_each_repetition(
    monkeypatch,
    tmp_path,
    local_storage,
):
    harness = runner(tmp_path)
    observed = []

    class Agent:
        def __init__(self):
            self.session_state = {}

        def run(self, prompt, stream=False):
            observed.append((id(self), self.session_id, S3.prefix))
            if prompt == "first":
                with S3.open("first.txt", "w") as handle:
                    handle.write(self.session_id)
            else:
                with S3.open("first.txt") as handle:
                    assert handle.read().decode() == self.session_id
            return output()

    monkeypatch.setattr(harness, "_build_system", Agent)
    records = harness._run_chain_test(case(steps=[{"prompt": "first"}, {"prompt": "second"}]))
    assert all(record["status"] == "success" for record in records)
    assert observed[0] == observed[1]
    assert observed[2] == observed[3]
    assert observed[0][1:] != observed[2][1:]


def test_frozen_source_files_are_never_used_as_writable_run_pointers(
    monkeypatch,
    tmp_path,
    local_storage,
):
    source = tmp_path / "dataset.csv"
    source.write_text("SMILES\nCCO\n")
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps({"data_file_paths": {"dataset_path": str(source)}}))
    harness = runner(tmp_path)
    seen = []

    class Agent:
        def __init__(self):
            self.session_state = {}

        def run(self, prompt, stream=False):
            path = Path(self.session_state["data_file_paths"]["dataset_path"])
            seen.append(path)
            assert path != source
            assert path.read_text() == "SMILES\nCCO\n"
            path.write_text("run-specific mutation")
            return output()

    monkeypatch.setattr(harness, "_build_system", Agent)
    records = harness._run_independent_test(
        case(
            fixture={
                "required": True,
                "session_state_path": str(fixture_path),
                "sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            }
        ),
        ["analyze"],
    )
    assert all(record["status"] == "success" for record in records)
    assert source.read_text() == "SMILES\nCCO\n"
    assert len(set(seen)) == 2
    for record, path in zip(records, seen, strict=True):
        assert str(path) in record["artifact_baseline"]
        assert (
            record["artifact_baseline"][str(path)]
            == hashlib.sha256(source.read_bytes()).hexdigest()
        )
        assert record["initial_session_state"]["data_file_paths"]["dataset_path"] == str(path)


def test_exception_retains_current_partial_output_and_artifacts(tmp_path):
    harness = runner(tmp_path)
    artifact = tmp_path / "partial.csv"
    artifact.write_text("SMILES\nCCO\n")

    class Agent:
        session_state = {"result_path": str(artifact)}

        def run(self, prompt, stream=False):
            run_response = output(  # noqa: F841 - mirrors Agno in-flight local
                tokens=21, tools=[tool(result={"error": "network timeout"})]
            )
            raise RuntimeError("later failure")

    record = harness._run_single_variation("run", "test", 0, agent=Agent())
    assert record["status"] == "failed"
    assert record["session_state"]["result_path"] == str(artifact)
    assert str(artifact) in record["generated_files"].values()
    assert record["telemetry"]["telemetry_status"] == "partial"
    assert record["telemetry"]["total_tokens"] == 21
    assert record["telemetry"]["failed_tool_call_count"] == 1
    normalized = harness._to_reliability_record(record)
    assert normalized["telemetry_status"] == "partial"
    assert normalized["total_tokens"] == 21


def test_failure_without_current_output_does_not_reuse_previous_run_or_zero_usage(tmp_path):
    harness = runner(tmp_path)

    class Agent:
        session_state = {}
        run_response = output(tokens=999, tools=[tool()])

        def run(self, prompt, stream=False):
            raise TimeoutError("no exposed current output")

    record = harness._run_single_variation("run", "test", 0, agent=Agent())
    telemetry = record["telemetry"]
    assert telemetry["telemetry_status"] == "unavailable"
    assert telemetry["total_tokens"] is None
    assert telemetry["tool_call_count"] is None
    assert harness._to_reliability_record(record)["total_tokens"] is None


@pytest.mark.parametrize(
    "result,expected",
    [
        ({"status": "error", "message": "bad request"}, True),
        ('{"error": "bad request"}', True),
        ({"success": False, "error": "bad request"}, True),
        ({"status": "no_route", "success": False}, False),
        ({"routes": [], "route_found": False}, False),
        ({"status": "ok", "error": None}, False),
        ("No route found; do not claim a synthesis is feasible.", False),
        ({"summary": {"error": "quoted historical error"}}, False),
    ],
)
def test_explicit_application_errors_without_false_no_route_failures(result, expected):
    telemetry = normalize_agno_output(output(tools=[tool(result=result)]))
    assert telemetry["failed_tool_call_count"] == int(expected)


def test_missing_token_usage_stays_unknown_even_when_tools_are_known():
    telemetry = normalize_agno_output(output(tools=[tool()]))
    assert telemetry["telemetry_status"] == "complete"
    assert telemetry["token_metrics_status"] == "unavailable"
    assert telemetry["tool_call_count"] == 1
    assert telemetry["total_tokens"] is None


def test_partial_usage_is_retained_but_not_a_complete_total_or_distribution(tmp_path):
    records = [
        {
            "case_name": "case",
            "task_success": True,
            "telemetry_status": "complete",
            "token_metrics_status": "complete",
            "tool_call_count": 2,
            "failed_tool_call_count": 0,
            "total_tokens": 100,
        },
        {
            "case_name": "case",
            "task_success": False,
            "telemetry_status": "partial",
            "token_metrics_status": "partial",
            "tool_call_count": 1,
            "failed_tool_call_count": 1,
            "total_tokens": 50,
        },
        {
            "case_name": "case",
            "task_success": False,
            "telemetry_status": "unavailable",
            "token_metrics_status": "unavailable",
            "tool_call_count": None,
            "failed_tool_call_count": None,
            "total_tokens": None,
        },
    ]
    summary = summarize_records(records)
    assert summary["tool_calls"] is None
    assert summary["failed_tool_calls_per_100"] is None
    assert summary["total_tokens_sum"] is None
    assert summary["observed_total_tokens_sum"] == 150
    assert summary["observed_tool_calls"] == 3
    assert summary["tool_metrics_complete_runs"] == 1
    assert summary["total_tokens"]["median"] == 100
    save_reliability_bundle(tmp_path / "report", records, environment_manifest={})
    assert "unavailable" in (tmp_path / "report" / "reliability_report.md").read_text()


def test_relative_fixture_paths_resolve_next_to_json_even_with_s3_enabled(tmp_path):
    harness = runner(tmp_path)
    harness._s3_config = {"bucket": "unused"}
    data = tmp_path / "data.csv"
    data.write_text("SMILES\nCCO\n")
    staged = harness._stage_fixture_state({"dataset_path": "data.csv"}, tmp_path)
    copied = Path(staged["dataset_path"])
    assert copied.is_absolute()
    assert copied != data
    assert copied.read_bytes() == data.read_bytes()


def test_model_directories_are_copied_without_preserving_writable_source_alias(tmp_path):
    harness = runner(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"test weights")
    state = harness._stage_fixture_state({"model_path": str(model)}, tmp_path)
    copied = Path(state["model_path"])
    assert copied != model
    assert (copied / "weights.bin").read_bytes() == b"test weights"
    (copied / "weights.bin").write_bytes(b"changed")
    assert (model / "weights.bin").read_bytes() == b"test weights"


def test_missing_member_usage_is_partial_even_when_coordinator_has_metrics():
    root = output(tokens=10)
    root.member_responses = [output()]
    telemetry = normalize_agno_output(root)
    assert telemetry["total_tokens"] == 10
    assert telemetry["token_metrics_status"] == "partial"
    assert telemetry["telemetry_status"] == "complete"


def test_unknown_traces_do_not_become_perfect_sequence_repeatability():
    summary = summarize_records(
        [
            {
                "case_name": "case",
                "prompt_variant": 0,
                "task_success": False,
                "telemetry_status": "unavailable",
            },
            {
                "case_name": "case",
                "prompt_variant": 0,
                "task_success": False,
                "telemetry_status": "unavailable",
            },
        ]
    )
    assert summary["repeatability"]["task_outcome_agreement"]["median"] == 1
    assert summary["repeatability"]["exact_tool_sequence_agreement"]["median"] is None


def test_persisted_transcript_is_linked_in_records_and_human_review(tmp_path, monkeypatch):
    harness = runner(tmp_path, repetitions=1)

    class Agent:
        session_state = {}

        def run(self, prompt, stream=False):
            return output(content="Saved scientific result with explicit uncertainty.")

    monkeypatch.setattr(harness, "_build_system", Agent)
    monkeypatch.setattr(harness, "_compare_outputs", lambda *_args: {})
    harness.run_test(case(prompt_variants=["Please complete this task."]))
    record = harness.reliability_records[0]
    assert record["response_path"] == "case/run_0/response.txt"
    assert (harness.output_dir / record["response_path"]).is_file()
    bundle_dir = harness.output_dir / "reliability"
    save_reliability_bundle(bundle_dir, [record], environment_manifest={})
    packets = list((bundle_dir / "human_review_packets").glob("*.md"))
    assert len(packets) == 1
    assert "Saved scientific result with explicit uncertainty." in packets[0].read_text()
