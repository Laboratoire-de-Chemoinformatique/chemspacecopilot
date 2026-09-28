"""The declared matrix must never accidentally run 72 or duplicate60 studies."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "study_plan", ROOT / "scripts/plan_revision_reliability.py"
)
planner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(planner)


def test_counterbalanced_matrix_is_exactly_48_frozen_plus_12_live(tmp_path):
    base = yaml.safe_load((ROOT / "tests/robustness/manuscript_reliability.yaml").read_text())
    fixtures = {
        key: {"required": True, "session_state_path": key, "sha256": "a" * 64}
        for key in ("seh_input", "seh_analysis", "seh_candidates", "peptide_input", "live_model")
    }
    batches = planner.build_matrix(
        base, fixtures, tmp_path / "bundle", tmp_path / "study", tmp_path / "synplanner"
    )
    assert len(batches) == 9
    assert sum(batch["expected_executions"] for batch in batches[:6]) == 48
    assert sum(batch["expected_executions"] for batch in batches[6:]) == 12
    assert [b["arm_order"] for b in batches[:6]] == ["team-first", "single-agent-first"] * 3
    cells = set()
    for batch in batches[:6]:
        config = batch["configuration"]
        assert config["general"]["n_variations"] == config["general"]["repetitions"] == 1
        assert config["general"]["scientific_seed"] == (11, 22, 33)[batch["global_repetition"]]
        assert config["general"]["stop_on_timeout"] is True
        for case, _ in planner.FROZEN:
            prompt = config["tests"][case]["prompt_variants"]
            assert len(prompt) == 1
            assert (
                prompt[0] == base["tests"][case]["prompt_variants"][batch["global_prompt_variant"]]
            )
            for arm in ("team", "single_agent"):
                cells.add((case, batch["global_prompt_variant"], batch["global_repetition"], arm))
    assert len(cells) == 48
    for batch in batches[6:]:
        tests = batch["configuration"]["tests"]
        assert len(tests["live_seh_workflow"]["steps"]) == 3
        assert len(tests["live_peptide_design"]["prompt_variants"]) == 1
        assert tests["live_peptide_design"]["fixture"] == fixtures["peptide_input"]
        assert all(
            "DBAASP_DATA" not in str(item)
            for item in tests["live_peptide_design"]["required_files"]
        )


def test_frozen_plan_guard_detects_changed_source_input_and_config(tmp_path):
    source = tmp_path / "runtime.py"
    artifact = tmp_path / "model.pt"
    config = tmp_path / "batch.yaml"
    for path in (source, artifact, config):
        path.write_text("original")
    plan = {
        "runtime_sha256": {"runtime.py": planner.sha256(source)},
        "input_sha256": {str(artifact): planner.sha256(artifact)},
        "batches": [
            {
                "batch_id": "batch",
                "config_path": str(config),
                "config_sha256": planner.sha256(config),
            }
        ],
    }
    assert planner.verify_plan(plan, tmp_path) == []
    for path in (source, artifact, config):
        path.write_text("changed")
    assert len(planner.verify_plan(plan, tmp_path)) == 3
