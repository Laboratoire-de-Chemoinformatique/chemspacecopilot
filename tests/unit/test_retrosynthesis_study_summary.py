"""Full-denominator exports must not hide failures that produced no route file."""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
from _study_common import file_identity  # noqa: E402
from summarize_retrosynthesis_study import summarize_study  # noqa: E402


def fixture_study(tmp_path):
    root = tmp_path / "study"
    root.mkdir()
    jobs = [
        {
            "id": f"target_{i}",
            "target_id": f"declared_{i}",
            "seed": i,
            "smiles": s,
            "canonical_smiles": s,
        }
        for i, s in enumerate(("CCO", "CCN", "CCC"), 1)
    ]
    specification = {
        key: {}
        for key in (
            "configuration",
            "software",
            "assets",
            "implementation_files",
            "launcher",
            "ledger_code",
            "target_source",
        )
    }
    specification.update(study="retrosynthesis", jobs=jobs, case_timeout_seconds=300)
    (root / "manifest.json").write_text(
        json.dumps({"created_at": "2026-09-28", "specification": specification})
    )
    for job in jobs[:2]:
        path = root / job["id"]
        path.mkdir()
        record = dict(job, status="error", error="failed before plan", wall_seconds=2)
        (path / "execution.json").write_text(json.dumps(record))
    case = root / jobs[0]["id"]
    plan = {
        "smiles": "CCO",
        "routes": [{"num_steps": 1, "steps": [{}], "score": 0.4}],
        "attempts": [{"stop_reason": "routes_found", "search_time": 1.2, "iterations": 3}],
    }
    (case / "plan.json").write_text(json.dumps(plan))
    record = dict(
        jobs[0],
        status="completed",
        wall_seconds=3,
        result={
            "scientific_outcome": "route_found",
            "artifacts": {"plan.json": file_identity(case / "plan.json")},
        },
    )
    (case / "execution.json").write_text(json.dumps(record))
    return root


def test_retains_failed_and_unstarted_without_plan(tmp_path):
    root = fixture_study(tmp_path)
    result = summarize_study(root, tmp_path / "out")
    assert result["declared_target_count"] == 3
    assert result["attempted_target_count"] == 2
    assert result["outcomes"] == {"route_found": 1, "error": 1, "not_started": 1}
    assert result["route_found_fraction_of_declared_targets"] == 1 / 3
    assert result["finalized"] is False
    rows = list(csv.DictReader((tmp_path / "out" / "targets.csv").open()))
    assert rows[1]["route_count"] == ""
    assert rows[2]["worker_wall_seconds"] == ""
    assert rows[0]["backend_reported_search_seconds"] == "1.2"


def test_rejects_changed_plan_and_missing_completed_plan(tmp_path):
    root = fixture_study(tmp_path)
    plan = root / "target_1" / "plan.json"
    plan.write_text(plan.read_text() + "\n")
    with pytest.raises(ValueError, match="artifact hash"):
        summarize_study(root, tmp_path / "out")
    plan.unlink()
    with pytest.raises(ValueError, match="missing its plan"):
        summarize_study(root, tmp_path / "out")


def test_partial_plan_never_promoted_from_interrupted_worker(tmp_path):
    root = fixture_study(tmp_path)
    path = root / "target_1" / "execution.json"
    record = json.loads(path.read_text())
    record["status"] = "interrupted"
    path.write_text(json.dumps(record))
    result = summarize_study(root, tmp_path / "out")
    assert result["route_found_count"] == 0
    assert result["outcomes"]["interrupted"] == 1


def test_rejects_output_inside_source_and_target_mismatch(tmp_path):
    root = fixture_study(tmp_path)
    with pytest.raises(ValueError, match="outside the source"):
        summarize_study(root, root / "out")
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["specification"]["jobs"][0]["canonical_smiles"] = "CC"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="target disagrees"):
        summarize_study(root, tmp_path / "out")
