"""The declared matrix is 48 frozen + 24 live; it must never silently change size."""

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


def test_counterbalanced_matrix_is_exactly_48_frozen_plus_24_live(tmp_path):
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
    assert sum(batch["expected_executions"] for batch in batches[6:]) == 24
    assert [b["arm_order"] for b in batches[:6]] == ["team-first", "single-agent-first"] * 3
    # Live runs both arms, counterbalanced across repetitions, so that the
    # architecture comparison is not confounded with ChEMBL drift between arms.
    assert all(b["system"] == "both" for b in batches[6:])
    assert [b["arm_order"] for b in batches[6:]] == [
        "team-first",
        "single-agent-first",
        "team-first",
    ]
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


def test_live_model_fixture_takes_its_descriptor_family_from_the_same_fixture_as_its_map(tmp_path):
    """The live tier must not pair one family's manifold with another family's descriptors.

    Both latents are 256 wide, so a cross-family projection returns meaningless
    coordinates without raising. Reading the map path and the descriptor family
    from the same session state keeps them consistent by construction.
    """
    import json

    seh_input = tmp_path / "seh_input.json"
    seh_input.write_text(
        json.dumps(
            {
                "session_state": {
                    "_current_gtm_model_path": "/models/gtm-chembed/GTM_manifold_chembed_900_225.pkl.gz",
                    "default_descriptor": "chembed",
                    "map_type": "default_map",
                }
            }
        )
    )

    state = planner.live_model_state(seh_input)

    assert state["default_descriptor"] == "chembed"
    assert state["_current_gtm_model_path"].endswith("GTM_manifold_chembed_900_225.pkl.gz")
    assert state["map_type"] == "default_map"


def test_live_model_fixture_refuses_a_fixture_with_no_descriptor_family(tmp_path):
    """A snapshot that records no family cannot be silently defaulted to one."""
    import json

    import pytest

    seh_input = tmp_path / "seh_input.json"
    seh_input.write_text(
        json.dumps({"session_state": {"_current_gtm_model_path": "/models/m.pkl.gz"}})
    )

    with pytest.raises(KeyError):
        planner.live_model_state(seh_input)


def test_planning_a_retrosynthesis_study_refuses_a_runtime_without_synplan():
    """_python_path falls back to src alone when the vendored directory is absent.

    That fallback is silent, and every route search then fails for a reason
    unrelated to what the study measures -- 0/18 task success that looks like a
    result. Refuse at plan time instead.
    """
    import pytest

    config = {"tests": {"case_3": {"validator": "retrosynthesis"}}}

    with pytest.raises(ValueError, match="synplan"):
        planner.verify_retrosynthesis_runtime(config, "/only/src", probe=lambda path: False)


def test_planning_accepts_a_runtime_that_can_import_synplan():
    config = {"tests": {"case_3": {"validator": "retrosynthesis"}}}
    planner.verify_retrosynthesis_runtime(config, "/src:/vendored", probe=lambda path: True)


def test_a_study_without_retrosynthesis_does_not_need_synplan():
    config = {"tests": {"case_1": {"validator": "seh_analysis"}}}
    planner.verify_retrosynthesis_runtime(config, "/only/src", probe=lambda path: False)
