"""Absence of an automatic tool-selection flag is not a human review result."""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "robustness"))
from reliability.reporting import (
    _markdown_report,
    _write_human_review,
    summarize_records,
)  # noqa: E402


def test_no_automatic_flags_keeps_human_assessment_unavailable(tmp_path):
    records = [{"case_name": "example", "task_success": True, "failure_categories": []}]
    summary = summarize_records(records)
    assert summary["incorrect_tool_selection_runs"] == 0
    assert summary["human_reviewed_incorrect_tool_selection_rate"] is None
    assert "Automatic validator flags only" in summary["tool_selection_assessment_scope"]
    rendered = _markdown_report(summary)
    assert "automatically flagged for tool selection" in rendered
    assert "human-reviewed incorrect-selection frequency remains unavailable" in rendered
    path = tmp_path / "human_review.csv"
    _write_human_review(path, records)
    with path.open() as handle:
        row = next(csv.DictReader(handle))
    assert row["incorrect_tool_selection_present_yes_no"] == ""
    assert row["tool_selection_evidence"] == ""


def test_detected_flag_remains_separate_from_expert_assessment():
    summary = summarize_records(
        [
            {
                "case_name": "example",
                "task_success": False,
                "failure_categories": ["incorrect_tool_selection"],
            }
        ]
    )
    assert summary["incorrect_tool_selection_runs"] == 1
    assert summary["incorrect_tool_selection_runs_per_100"] == 100.0
    assert summary["human_reviewed_incorrect_tool_selection_rate"] is None
