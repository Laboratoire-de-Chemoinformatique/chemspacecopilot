"""Unit tests for the publication-facing reliability benchmark."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest
import yaml

ROBUSTNESS_DIR = Path(__file__).parents[1] / "robustness"
sys.path.insert(0, str(ROBUSTNESS_DIR))

from config_schema import ConfigValidator  # noqa: E402
from reliability.human_review import (  # noqa: E402
    SCORE_FIELDS,
    aggregate_reviews,
    load_reviews,
)
from reliability.reporting import (  # noqa: E402
    build_environment_manifest,
    build_system_comparison,
    save_reliability_bundle,
    save_system_comparison,
    summarize_records,
    wilson_interval,
)
from reliability.telemetry import normalize_agno_output  # noqa: E402
from reliability.validators import evaluate_run  # noqa: E402
from robustness_minimal_example import (  # noqa: E402
    FixtureLoadError,
    RobustnessConfig,
    RobustnessRunner,
    parse_args,
)
from robustness_minimal_example import TestConfig as RunnerTestConfig  # noqa: E402


@dataclass
class FakeMetrics:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    duration: float = 0.0


@dataclass
class FakeTool:
    tool_name: str
    tool_args: dict = field(default_factory=dict)
    result: Any = None
    tool_call_error: bool = False
    metrics: Any = None
    child_run_id: str | None = None
    created_at: int | None = None


@dataclass
class FakeOutput:
    content: str = ""
    metrics: Any = None
    tools: List[Any] = field(default_factory=list)
    member_responses: List[Any] = field(default_factory=list)
    agent_name: str | None = None
    team_name: str | None = None
    model: str | None = None
    model_provider: str | None = None


def _tool_call(name: str, *, error: bool = False) -> dict:
    return {
        "sequence": 0,
        "tool_name": name,
        "tool_args": {},
        "error": error,
    }


def _output_with_state(state: dict, tool_calls: list[dict], response: str = "") -> dict:
    return {
        "status": "success",
        "prompt": "",
        "response": response,
        "session_state": state,
        "generated_files": {},
        "telemetry": {
            "tool_calls": tool_calls,
            "tool_call_count": len(tool_calls),
            "failed_tool_call_count": sum(call["error"] for call in tool_calls),
        },
    }


def test_telemetry_aggregates_team_members_and_redacts_secrets():
    member = FakeOutput(
        agent_name="GTM Agent",
        metrics=FakeMetrics(
            input_tokens=30,
            output_tokens=10,
            total_tokens=40,
            duration=0.5,
        ),
        tools=[
            FakeTool(
                "gtm_optimization",
                {"api_key": "secret", "nested": {"password": "hidden"}, "objective": "LLH"},
                result={"status": "ok"},
                metrics=SimpleNamespace(duration=0.25),
                created_at=10,
            )
        ],
        model="deepseek-chat",
        model_provider="deepseek",
    )
    root = FakeOutput(
        team_name="Cs_copilot Team",
        metrics=FakeMetrics(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            reasoning_tokens=5,
            duration=1.5,
        ),
        tools=[
            FakeTool(
                "delegate_task_to_member",
                tool_call_error=True,
                result="failed",
                created_at=20,
            )
        ],
        member_responses=[member, member],
        model="deepseek-chat",
        model_provider="deepseek",
    )

    telemetry = normalize_agno_output(
        root,
        pricing={"input_per_million": 1.0, "output_per_million": 2.0},
    )

    assert telemetry["input_tokens"] == 130
    assert telemetry["output_tokens"] == 30
    assert telemetry["total_tokens"] == 160
    assert telemetry["reasoning_tokens"] == 5
    assert telemetry["model_call_count"] == 2
    assert telemetry["tool_call_count"] == 2
    assert telemetry["failed_tool_call_count"] == 1
    assert telemetry["estimated_cost"] == pytest.approx(0.00019)
    assert [call["tool_name"] for call in telemetry["tool_calls"]] == [
        "gtm_optimization",
        "delegate_task_to_member",
    ]
    assert telemetry["tool_calls"][0]["tool_args"]["api_key"] == "[REDACTED]"
    assert telemetry["tool_calls"][0]["tool_args"]["nested"]["password"] == "[REDACTED]"
    assert telemetry["tool_calls"][0]["result_sha256"]


def test_molecular_validator_requires_real_candidates_and_seed_provenance():
    empty_state = {
        "session_objects": {
            "candidate_sets": {
                "cset_001": {
                    "count_returned": 0,
                    "compound_ids": [],
                    "seed_smiles": None,
                }
            }
        }
    }
    output = _output_with_state(
        empty_state,
        [_tool_call("generate_analogs")],
        response="Candidate generation finished.",
    )
    output["prompt"] = "Generate analogues of CHEMBL3327073."

    failed = evaluate_run("molecular_generation", output)

    assert failed["task_success"] is False
    failed_names = {check["name"] for check in failed["checks"] if not check["passed"]}
    assert "valid_candidates_returned" in failed_names
    assert "parent_provenance_present" in failed_names

    candidate = empty_state["session_objects"]["candidate_sets"]["cset_001"]
    candidate.update(
        {
            "count_returned": 2,
            "compound_ids": ["cmp_001", "cmp_002"],
            "seed_smiles": "CCO",
        }
    )
    passed = evaluate_run("molecular_generation", output)
    assert passed["task_success"] is True


def test_retrosynthesis_no_route_is_a_valid_reported_outcome():
    state = {
        "synplanner_plan": {
            "smiles": "CCO",
            "attempts": [{"max_iterations": 100}],
            "routes": [],
        },
        "session_objects": {"routes": {}},
    }
    output = _output_with_state(
        state,
        [_tool_call("plan_synthesis")],
        response="SynPlanner completed the search but no route was found.",
    )

    result = evaluate_run("retrosynthesis", output)

    assert result["task_success"] is True
    assert result["scientific_outcome"] == {"route_found": False, "route_count": 0}


def test_recovery_validator_counts_inappropriate_calls():
    output = _output_with_state(
        {},
        [_tool_call("design_molecules")],
        response="Which parent molecule should I use?",
    )

    result = evaluate_run("missing_design_seed", output)

    assert result["task_success"] is False
    assert "incorrect_tool_selection" in result["failure_categories"]


def _record(
    run_id: str,
    *,
    success: bool,
    sequence: tuple[str, ...],
    failed_tools: int = 0,
    categories: list[str] | None = None,
    case_name: str = "case_1",
    prompt_variant: int = 0,
    repetition: int | None = None,
    wall_time_seconds: float | None = None,
    total_tokens: int | None = None,
    estimated_cost: float | None = None,
) -> dict:
    run_number = int(run_id)
    return {
        "benchmark_run_id": "benchmark",
        "case_name": case_name,
        "run_id": run_id,
        "prompt": "Analyze chemical space.",
        "prompt_variant": prompt_variant,
        "repetition": run_number if repetition is None else repetition,
        "tier": "frozen",
        "task_success": success,
        "execution_status": "success",
        "wall_time_seconds": (1.0 + run_number if wall_time_seconds is None else wall_time_seconds),
        "total_tokens": 100 + run_number if total_tokens is None else total_tokens,
        "estimated_cost": estimated_cost,
        "tool_call_count": len(sequence),
        "failed_tool_call_count": failed_tools,
        "tool_calls": [
            {"sequence": index, "tool_name": name, "error": False}
            for index, name in enumerate(sequence)
        ],
        "validations": [{"name": "execution_completed", "passed": True}],
        "failure_categories": categories or [],
        "response_path": f"{case_name}/run_{run_id}/response.txt",
    }


def test_summary_reports_confidence_intervals_and_repeatability():
    records = [
        _record("0", success=True, sequence=("fetch", "gtm")),
        _record("1", success=True, sequence=("fetch", "gtm")),
        _record(
            "2",
            success=False,
            sequence=("fetch", "wrong_tool"),
            failed_tools=1,
            categories=["incorrect_tool_selection"],
        ),
    ]

    summary = summarize_records(records)
    low, high = wilson_interval(2, 3)

    assert summary["success_rate"] == pytest.approx(2 / 3)
    assert summary["wilson_95"] == pytest.approx([low, high])
    assert summary["failed_tool_calls_per_100"] == pytest.approx(100 / 6)
    assert summary["incorrect_tool_selection_runs_per_100"] == pytest.approx(100 / 3)
    assert summary["wall_time_seconds_total"] == 6
    assert summary["total_tokens_sum"] == 303
    assert summary["estimated_cost_total"] is None
    repeatability = summary["repeatability"]
    assert repeatability["groups_with_repeats"] == 1
    assert repeatability["task_outcome_agreement"]["median"] == pytest.approx(2 / 3)
    assert repeatability["exact_tool_sequence_agreement"]["median"] == pytest.approx(2 / 3)


def test_report_bundle_writes_normalized_and_blinded_artifacts(tmp_path):
    records = [_record("0", success=True, sequence=("fetch", "gtm"))]
    source_response = tmp_path / records[0]["response_path"]
    source_response.parent.mkdir(parents=True)
    source_response.write_text("Grounded response")
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text("general: {}\n")
    manifest = build_environment_manifest(
        project_root=tmp_path,
        model_provider="deepseek",
        model_id="deepseek-chat",
        config_path=config_path,
    )

    summary = save_reliability_bundle(
        tmp_path / "out",
        records,
        environment_manifest=manifest,
    )

    expected = {
        "runs.jsonl",
        "tool_calls.jsonl",
        "validations.jsonl",
        "reliability_summary.json",
        "reliability_report.md",
        "environment_manifest.json",
        "human_review.csv",
    }
    assert expected.issubset(path.name for path in (tmp_path / "out").iterdir())
    assert summary["runs"] == 1
    run = json.loads((tmp_path / "out" / "runs.jsonl").read_text().splitlines()[0])
    assert "response" not in run
    with (tmp_path / "out" / "human_review.csv").open() as handle:
        review_rows = list(csv.DictReader(handle))
    assert review_rows[0]["review_id"]
    assert "prompt" not in review_rows[0]
    assert review_rows[0]["review_packet_path"].startswith("human_review_packets/")
    review_packet = tmp_path / "out" / review_rows[0]["review_packet_path"]
    packet_text = review_packet.read_text()
    assert "Analyze chemical space." in packet_text
    assert "Grounded response" in packet_text


def test_system_comparison_pairs_runs_and_reports_objective_deltas(tmp_path):
    records_by_arm = {
        "team": [
            _record(
                "0",
                success=True,
                sequence=("delegate", "gtm"),
                case_name="seh_analysis",
                repetition=0,
                wall_time_seconds=4,
                total_tokens=200,
            ),
            _record(
                "1",
                success=True,
                sequence=("delegate", "design"),
                case_name="molecular_generation",
                repetition=0,
                wall_time_seconds=6,
                total_tokens=300,
            ),
            _record(
                "2",
                success=True,
                sequence=("delegate", "report"),
                case_name="team_only_unmatched",
                repetition=0,
                wall_time_seconds=8,
                total_tokens=400,
            ),
        ],
        "single_agent": [
            _record(
                "0",
                success=True,
                sequence=("gtm",),
                case_name="seh_analysis",
                repetition=0,
                wall_time_seconds=3,
                total_tokens=150,
            ),
            _record(
                "1",
                success=False,
                sequence=("wrong_tool",),
                failed_tools=1,
                categories=["incorrect_tool_selection"],
                case_name="molecular_generation",
                repetition=0,
                wall_time_seconds=5,
                total_tokens=250,
            ),
        ],
    }
    robustness_summaries = {
        "team": {
            "total_tests": 2,
            "passed": 2,
            "pass_rate": 1.0,
            "average_robustness_score": 0.9,
            "overall_rating": "Excellent",
            "results": {},
        },
        "single_agent": {
            "total_tests": 2,
            "passed": 1,
            "pass_rate": 0.5,
            "average_robustness_score": 0.7,
            "overall_rating": "Acceptable",
            "results": {},
        },
    }

    comparison = build_system_comparison(
        records_by_arm,
        robustness_summaries=robustness_summaries,
    )

    assert comparison["pairing"]["paired_runs"] == 2
    assert comparison["pairing"]["outcomes"] == {
        "both_success": 1,
        "team_only_success": 1,
        "single_agent_only_success": 0,
        "both_failed": 0,
    }
    assert comparison["per_arm"]["team"]["success_rate"] == 1
    assert comparison["per_arm"]["single_agent"]["success_rate"] == 0.5
    assert comparison["delta"]["success_rate"] == pytest.approx(0.5)
    assert comparison["delta"]["wall_time_seconds_median"] == pytest.approx(1)
    assert comparison["warnings"]
    assert len(comparison["pairing"]["unmatched_runs"]["team"]) == 1

    comparison_md = save_system_comparison(
        tmp_path,
        records_by_arm,
        robustness_summaries=robustness_summaries,
    )
    persisted = json.loads((tmp_path / "comparison.json").read_text())
    report = comparison_md.read_text()

    assert persisted["pairing"]["paired_runs"] == 2
    assert "Objective task success is the primary outcome" in report
    assert "Results by representative case" in report
    assert "Secondary prompt-robustness results" in report
    assert "n/a" in report


def test_system_comparison_rejects_duplicate_pairing_keys():
    duplicate = _record("0", success=True, sequence=("gtm",), repetition=0)

    with pytest.raises(ValueError, match="Duplicate comparison record"):
        build_system_comparison(
            {
                "team": [duplicate, dict(duplicate, run_id="another")],
                "single_agent": [duplicate],
            }
        )


def test_system_comparison_excludes_common_prerequisite_failures(tmp_path):
    team_success = _record("0", success=True, sequence=("delegate", "gtm"))
    single_success = _record("0", success=True, sequence=("gtm",))
    team_blocked = _record(
        "1",
        success=False,
        sequence=(),
        case_name="peptide_design",
        repetition=0,
    )
    single_blocked = dict(team_blocked)
    team_blocked["execution_status"] = "prerequisite_error"
    single_blocked["execution_status"] = "prerequisite_error"

    records = {
        "team": [team_success, team_blocked],
        "single_agent": [single_success, single_blocked],
    }
    comparison = build_system_comparison(records)

    assert comparison["per_arm"]["team"]["runs"] == 1
    assert comparison["per_arm"]["single_agent"]["runs"] == 1
    assert comparison["pairing"]["paired_runs"] == 1
    assert comparison["pairing"]["outcomes"]["both_success"] == 1
    assert comparison["pairing"]["excluded_runs"]["team"] == [
        {
            "tier": "frozen",
            "case_name": "peptide_design",
            "prompt_variant": 0,
            "repetition": 0,
            "execution_status": "prerequisite_error",
        }
    ]

    report_path = save_system_comparison(tmp_path, records)
    report = report_path.read_text()
    assert "Not evaluated" in report
    assert "peptide_design" in report


def test_ablation_arm_order_cli_option(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["robustness_minimal_example.py", "--system", "both", "--arm-order", "single-agent-first"],
    )

    args = parse_args()

    assert args.system == "both"
    assert args.arm_order == "single-agent-first"


def test_human_review_aggregation_reports_ordinal_agreement(tmp_path):
    sheets = []
    for reviewer, scores in (
        ("reviewer_a", (2, 2)),
        ("reviewer_b", (2, 1)),
    ):
        path = tmp_path / f"{reviewer}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["review_id", "reviewer_id", *SCORE_FIELDS],
            )
            writer.writeheader()
            for review_id, score in zip(
                ("response_1", "response_2"),
                scores,
                strict=True,
            ):
                writer.writerow(
                    {
                        "review_id": review_id,
                        "reviewer_id": reviewer,
                        **dict.fromkeys(SCORE_FIELDS, score),
                    }
                )
        sheets.append(path)

    summary = aggregate_reviews(load_reviews(sheets))

    assert summary["responses"] == 2
    assert summary["responses_with_at_least_two_reviews"] == 2
    factuality = summary["dimensions"]["artifact_grounded_factuality_0_2"]
    assert factuality["mean"] == pytest.approx(1.75)
    assert factuality["exact_agreement"] == pytest.approx(0.5)
    assert factuality["mean_pairwise_quadratic_weighted_kappa"] is not None


def _valid_config() -> dict:
    return {
        "general": {
            "n_variations": 3,
            "repetitions": 2,
            "tier": "both",
            "timeout_seconds": 30,
            "reliability_min_success_rate": 0.8,
        },
        "model": {
            "provider": "ollama",
            "model_id": "test-model",
            "api_key_env": "UNUSED",
        },
        "metrics": {
            "weights": {
                "data_similarity": 0.4,
                "semantic_similarity": 0.3,
                "process_consistency": 0.2,
                "visual_similarity": 0.1,
            },
            "thresholds": {
                "excellent": 0.9,
                "good": 0.8,
                "acceptable": 0.7,
            },
            "pass_threshold": 0.75,
        },
        "tests": {
            "explicit_prompts": {
                "enabled": True,
                "prompt_variants": ["one", "two"],
                "validator": "execution_only",
                "tier": "frozen",
                "required_files": [
                    {
                        "name": "example input",
                        "env": "EXAMPLE_INPUT_PATH",
                        "default_path": "fixtures/example.csv",
                    }
                ],
            },
            "chain": {
                "enabled": True,
                "steps": [
                    {"name": "first", "prompt": "one", "validator": "execution_only"},
                    {"name": "second", "prompt": "two", "validator": "execution_only"},
                ],
                "tier": "live",
            },
        },
        "reporting": {},
    }


def test_config_schema_accepts_explicit_prompts_and_chains(tmp_path):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(yaml.safe_dump(_valid_config()))

    loaded = ConfigValidator.load_and_validate(config_path)

    assert loaded["general"]["repetitions"] == 2
    assert loaded["tests"]["chain"]["steps"][1]["name"] == "second"
    assert loaded["tests"]["explicit_prompts"]["required_files"][0]["env"] == ("EXAMPLE_INPUT_PATH")


def test_config_schema_reads_nested_legacy_prompt_catalog(tmp_path):
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    (fixtures_dir / "prompt_templates.yaml").write_text(
        yaml.safe_dump({"prompts": {"legacy_prompt": {"variations": ["one"]}}})
    )
    config = _valid_config()
    config["tests"] = {
        "legacy_test": {
            "enabled": True,
            "prompt_key": "legacy_prompt",
        }
    }
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(yaml.safe_dump(config))

    loaded = ConfigValidator.load_and_validate(config_path)

    assert loaded["tests"]["legacy_test"]["prompt_key"] == "legacy_prompt"


def test_fixture_loader_verifies_hash_and_fails_closed(tmp_path):
    fixture_path = tmp_path / "state.json"
    fixture_path.write_text(json.dumps({"session_state": {"marker": "fixture"}}))
    digest = __import__("hashlib").sha256(fixture_path.read_bytes()).hexdigest()
    runner = RobustnessRunner(
        RobustnessConfig(
            output_dir=str(tmp_path / "reports"),
            s3_session_isolation=False,
        )
    )

    state = runner._load_fixture_state(
        {
            "required": True,
            "session_state_path": str(fixture_path),
            "sha256": digest,
        }
    )
    assert state == {"marker": "fixture"}

    with pytest.raises(FixtureLoadError, match="SHA-256 mismatch"):
        runner._load_fixture_state(
            {
                "required": True,
                "session_state_path": str(fixture_path),
                "sha256": "0" * 64,
            }
        )


def test_completed_run_uses_in_memory_state_without_persisted_session(tmp_path):
    class MemoryDisabledAgent:
        def __init__(self):
            self.session_state = {"marker": "in-memory"}

        def run(self, prompt, stream=False):  # noqa: ARG002
            return FakeOutput(
                content="completed",
                metrics=FakeMetrics(input_tokens=3, output_tokens=2, total_tokens=5),
            )

        def get_session_state(self):
            raise RuntimeError("Session not found")

    runner = RobustnessRunner(
        RobustnessConfig(
            output_dir=str(tmp_path / "reports"),
            s3_session_isolation=False,
        )
    )

    output = runner._run_single_variation(
        prompt="run",
        test_name="memory_disabled",
        run_id=0,
        agent=MemoryDisabledAgent(),
    )

    assert output["status"] == "success"
    assert output["response"] == "completed"
    assert output["session_state"] == {"marker": "in-memory"}
    assert output["telemetry"]["total_tokens"] == 5


def test_missing_required_file_fails_before_building_agent(monkeypatch, tmp_path):
    runner = RobustnessRunner(
        RobustnessConfig(
            output_dir=str(tmp_path / "reports"),
            s3_session_isolation=False,
        )
    )
    monkeypatch.setattr(
        runner,
        "_build_system",
        lambda: pytest.fail("prerequisite failure must not build or call an agent"),
    )

    output = runner._run_single_variation(
        prompt="run",
        test_name="missing_input",
        run_id=0,
        required_files=[
            {
                "name": "DBAASP dataset",
                "env": "TEST_DBAASP_DATA_PATH",
                "default_path": str(tmp_path / "missing.csv"),
            }
        ],
    )

    assert output["status"] == "prerequisite_error"
    assert "DBAASP dataset" in output["error"]
    assert "prerequisite_failure" in output["validation"]["failure_categories"]


def test_runner_retains_records_when_optional_comparison_fails(monkeypatch, tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session_state = {}

        def run(self, prompt, stream=False):  # noqa: ARG002
            return FakeOutput(content="completed")

    runner = RobustnessRunner(
        RobustnessConfig(
            n_variations=1,
            repetitions=1,
            output_dir=str(tmp_path / "reports"),
            s3_session_isolation=False,
            reliability_enabled=True,
        )
    )
    monkeypatch.setattr(runner, "_build_system", FakeAgent)
    monkeypatch.setattr(
        runner,
        "_compare_outputs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ImportError("optional comparator")),
    )
    test_config = RunnerTestConfig(
        name="comparison_failure",
        enabled=True,
        prompt_key="",
        prompt_variants=["run"],
    )

    result = runner.run_test(test_config)

    assert result["n_runs"] == 1
    assert result["comparison"]["error"].endswith("optional comparator")
    assert len(runner.reliability_records) == 1
    assert runner.reliability_records[0]["execution_status"] == "success"


def test_runner_repeats_explicit_prompts(monkeypatch, tmp_path):
    class FakeAgent:
        def __init__(self):
            self.session_state = {}

        def run(self, prompt, stream=False):  # noqa: ARG002
            return FakeOutput(content=f"completed: {prompt}")

        def get_session_state(self):
            return self.session_state

    config = RobustnessConfig(
        n_variations=2,
        repetitions=3,
        output_dir=str(tmp_path / "reports"),
        save_artifacts=True,
        s3_session_isolation=False,
        reliability_enabled=True,
        include_run_details=False,
    )
    runner = RobustnessRunner(config)
    monkeypatch.setattr(runner, "_build_system", FakeAgent)
    monkeypatch.setattr(
        runner,
        "_compare_outputs",
        lambda outputs, test_name: {
            "process": {"completion_rate": 1.0, "tool_sequence_similarity": 1.0}
        },
    )
    test_config = RunnerTestConfig(
        name="repeat_test",
        enabled=True,
        prompt_key="",
        prompt_variants=["one", "two", "three"],
        validator="execution_only",
        tier="frozen",
    )

    result = runner.run_test(test_config)

    assert result["n_variations"] == 2
    assert result["n_runs"] == 6
    assert result["task_success_rate"] == 1
    assert len(runner.reliability_records) == 6
    snapshots = list((runner.output_dir / "repeat_test").glob("run_*/session_state.json"))
    assert len(snapshots) == 6


def test_runner_counts_missing_required_fixture_without_live_fallback(
    monkeypatch,
    tmp_path,
):
    config = RobustnessConfig(
        n_variations=1,
        repetitions=2,
        output_dir=str(tmp_path / "reports"),
        s3_session_isolation=False,
        reliability_enabled=True,
        include_run_details=True,
    )
    runner = RobustnessRunner(config)
    monkeypatch.setattr(
        runner,
        "_build_system",
        lambda: pytest.fail("missing frozen fixture must not build a live system"),
    )
    monkeypatch.setattr(
        runner,
        "_compare_outputs",
        lambda outputs, test_name: {
            "process": {"completion_rate": 0.0, "tool_sequence_similarity": 0.0}
        },
    )
    test_config = RunnerTestConfig(
        name="missing_fixture",
        enabled=True,
        prompt_key="",
        prompt_variants=["analyze fixture"],
        validator="execution_only",
        tier="frozen",
        fixture={
            "required": True,
            "session_state_path": str(tmp_path / "missing.json"),
            "sha256": "0" * 64,
        },
    )

    result = runner.run_test(test_config)

    assert result["n_runs"] == 2
    assert result["successful_runs"] == 0
    assert {output["status"] for output in result["outputs"]} == {"fixture_error"}
    assert all(
        "fixture_failure" in record["failure_categories"] for record in runner.reliability_records
    )


def test_runner_summary_serializes_non_json_session_keys(tmp_path):
    runner = RobustnessRunner(
        RobustnessConfig(
            output_dir=str(tmp_path / "reports"),
            generate_markdown=False,
        )
    )
    summary = {
        "test_run_id": "tuple-state",
        "results": {
            "case": {
                "outputs": [
                    {
                        "session_state": {
                            ("autoencoder", 256): {"api_key": "must-not-leak"},
                        }
                    }
                ]
            }
        },
    }

    runner._save_reports(summary)

    saved = json.loads((runner.output_dir / "summary.json").read_text())
    state = saved["results"]["case"]["outputs"][0]["session_state"]
    assert state["('autoencoder', 256)"]["api_key"] == "[REDACTED]"
