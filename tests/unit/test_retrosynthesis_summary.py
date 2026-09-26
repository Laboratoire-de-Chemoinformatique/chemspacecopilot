"""Scientific accounting checks for the offline retrosynthesis supplement."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "retrosynthesis_summary", Path(__file__).parents[2] / "scripts/summarize_retrosynthesis.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _save(tmp_path, name, **overrides):
    plan = {"smiles": "CCO", "routes": [], "attempts": []}
    plan.update(overrides)
    path = tmp_path / name
    path.write_text(json.dumps(plan))
    return path


def test_no_route_is_separate_from_error_and_missing_evidence(tmp_path):
    no_route = _save(tmp_path, "no_route.json", attempts=[{"stop_reason": "no_routes"}])
    failed = _save(tmp_path, "error.json", attempts=[{"stop_reason": "error", "error": "OOM"}])
    empty = _save(tmp_path, "empty.json")
    assert MODULE.summarize_plan(no_route)["outcome"] == "no_route"
    assert MODULE.summarize_plan(failed)["outcome"] == "error"
    assert MODULE.summarize_plan(empty)["outcome"] == "unavailable"


def test_partial_search_timing_does_not_become_complete_runtime(tmp_path):
    path = _save(
        tmp_path,
        "plan.json",
        attempts=[{"search_time": 2.5}, {"stop_reason": "error", "error": "OOM"}],
    )
    row = MODULE.summarize_plan(path)
    assert row["search_seconds"] is None
    assert row["observed_search_seconds"] == 2.5
    assert row["search_time_complete"] is False
    assert row["error_attempt_count"] == 1


def test_mixed_no_route_and_failed_attempts_remain_visible_in_summary(tmp_path):
    path = _save(
        tmp_path,
        "mixed.json",
        attempts=[{"stop_reason": "no_routes"}, {"stop_reason": "error", "error": "OOM"}],
    )
    summary = MODULE.write_summary([path], tmp_path / "out")
    assert summary["outcomes"]["no_route_with_errors"] == 1
    assert summary["outcomes"]["no_route"] == 0


def test_repeated_targets_remain_executions_and_route_order_is_preserved(tmp_path):
    first = _save(
        tmp_path,
        "one.json",
        routes=[
            {"score": 0.2, "num_steps": 2, "steps": [{}, {}]},
            {"score": 0.9, "num_steps": 1, "steps": [{}]},
        ],
        attempts=[{"stop_reason": "routes_found", "search_time": 3.0}],
    )
    second = _save(tmp_path, "two.json", smiles="OCC", attempts=[{"stop_reason": "no_routes"}])
    summary = MODULE.write_summary([first, second], tmp_path / "out")
    assert summary["execution_count"] == 2
    assert summary["distinct_target_count"] == 1
    assert summary["repeated_target_executions"] == 1
    assert summary["route_found_fraction_of_supplied_executions"] == 0.5
    with (tmp_path / "out/targets.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["first_route_score"] == "0.2"
    assert rows[0]["first_route_num_steps"] == "2"
    assert rows[0]["route_length_check"] == "consistent"
    assert summary["inputs"][0]["sha256"]


def test_zero_step_route_and_conflicting_length_are_flagged(tmp_path):
    zero = _save(tmp_path, "zero.json", routes=[{"num_steps": 0, "steps": []}])
    mismatch = _save(tmp_path, "mismatch.json", routes=[{"num_steps": 3, "steps": [{}]}])
    assert MODULE.summarize_plan(zero)["route_length_check"] == "zero_steps_requires_review"
    assert MODULE.summarize_plan(mismatch)["route_length_check"] == "inconsistent"


def test_bad_input_and_duplicate_file_do_not_create_results(tmp_path):
    valid = _save(tmp_path, "valid.json")
    bad = _save(tmp_path, "bad.json", routes="not a route list")
    with pytest.raises(ValueError, match="routes must"):
        MODULE.write_summary([valid, bad], tmp_path / "out")
    assert not (tmp_path / "out").exists()
    with pytest.raises(ValueError, match="more than once"):
        MODULE.write_summary([valid, valid], tmp_path / "out")
