"""Check study denominators, local-only preflight, RNG application, and resume safety."""

import json
import os
import platform
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[2] / "scripts"
# Launchers share sibling helpers and are normally executed as script files.
sys.path.insert(0, str(SCRIPTS))
import _study_common as common  # noqa: E402
import run_generation_study as generation  # noqa: E402
import run_retrosynthesis_study as retrosynthesis  # noqa: E402


def _spec(timeout=3):
    return {
        "study": "execution_ledger_unit_test",
        "case_timeout_seconds": timeout,
        "jobs": [{"id": "case_1", "seed": 11}, {"id": "case_2", "seed": 22}],
    }


def _worker(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text("""import json, pathlib, sys
spec = json.loads(pathlib.Path(sys.argv[2]).read_text())
out = pathlib.Path(spec['output_dir'])
(out/'called').write_text('once')
if spec['job']['seed'] == 11:
    raise RuntimeError('intentional unit-test worker failure before a plan exists')
(out/'result.json').write_text(json.dumps({'unit_test': True, 'artifacts': {}}))
""")
    return worker


def test_failed_workers_remain_in_denominator_and_are_not_retried(tmp_path):
    out = tmp_path / "study"
    worker = _worker(tmp_path)
    first = common.run_jobs(_spec(), out, worker_script=worker)
    assert first["planned_case_count"] == first["attempted_case_count"] == 2
    assert first["status_counts"]["error"] == first["status_counts"]["completed"] == 1
    timestamps = [(out / job["id"] / "called").stat().st_mtime_ns for job in _spec()["jobs"]]
    second = common.run_jobs(_spec(), out, worker_script=worker, resume=True)
    assert second == first
    assert timestamps == [
        (out / job["id"] / "called").stat().st_mtime_ns for job in _spec()["jobs"]
    ]


def test_prepare_declares_denominator_without_creating_scientific_results(tmp_path):
    out = tmp_path / "study"
    summary = common.run_jobs(_spec(), out, worker_script=_worker(tmp_path), prepare_only=True)
    assert summary["planned_case_count"] == 2
    assert summary["attempted_case_count"] == 0
    assert summary["status_counts"]["not_started"] == 2
    assert not list(out.glob("*/result.json"))


def test_running_controller_cannot_be_reclassified_as_interrupted(tmp_path):
    out = tmp_path / "study"
    worker = _worker(tmp_path)
    common.run_jobs(_spec(), out, worker_script=worker, prepare_only=True)
    common.write_json(
        out / "case_1" / "execution.json",
        {
            "id": "case_1",
            "status": "running",
            "hostname": platform.node(),
            "controller_pid": os.getpid(),
        },
    )
    with pytest.raises(ValueError, match="still alive"):
        common.run_jobs(_spec(), out, worker_script=worker, resume=True)


def test_proven_dead_running_job_is_terminal_without_retry(tmp_path, monkeypatch):
    out = tmp_path / "study"
    worker = _worker(tmp_path)
    common.run_jobs(_spec(), out, worker_script=worker, prepare_only=True)
    common.write_json(
        out / "case_1" / "execution.json",
        {
            "id": "case_1",
            "status": "running",
            "hostname": platform.node(),
            "controller_pid": 42,
        },
    )
    monkeypatch.setattr(common, "_alive", lambda _pid: False)
    result = common.run_jobs(_spec(), out, worker_script=worker, resume=True)
    assert result["status_counts"]["interrupted"] == 1
    assert result["status_counts"]["completed"] == 1
    assert not (out / "case_1/called").exists()


def test_resume_rejects_changed_parameters(tmp_path):
    out = tmp_path / "study"
    worker = _worker(tmp_path)
    common.run_jobs(_spec(), out, worker_script=worker, prepare_only=True)
    with pytest.raises(ValueError, match="differs"):
        common.run_jobs(_spec(timeout=6), out, worker_script=worker, resume=True)


def test_worker_timeout_is_terminal_and_keeps_full_denominator(tmp_path):
    worker = tmp_path / "slow.py"
    worker.write_text("import time\ntime.sleep(10)\n")
    summary = common.run_jobs(_spec(timeout=0.1), tmp_path / "study", worker_script=worker)
    assert summary["status_counts"]["timeout"] == 2
    assert summary["attempted_case_count"] == 2


def test_lfs_pointers_are_not_accepted_as_model_assets(tmp_path):
    pointer = tmp_path / "model.pt"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abcd\nsize 123\n")
    with pytest.raises(ValueError, match="LFS pointer"):
        common.file_identity(pointer)


def test_seed_control_applies_to_all_three_local_rngs():
    import random

    import numpy as np
    import torch

    common.apply_seed(123)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    settings = common.apply_seed(123)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert first == second
    assert settings["seed"] == 123
    assert settings["torch_deterministic_algorithms"] is True
    assert settings["torch_num_threads"] == 1


def test_targets_are_predeclared_and_canonical_duplicates_rejected(tmp_path):
    targets = tmp_path / "targets.json"
    targets.write_text(json.dumps(["CCO", "OCC"]))
    with pytest.raises(ValueError, match="duplicates"):
        retrosynthesis.load_targets(targets, 2)
    with pytest.raises(ValueError, match="exactly 10"):
        retrosynthesis.load_targets(targets, 10)


def test_target_ids_do_not_control_output_paths(tmp_path):
    targets = tmp_path / "targets.csv"
    targets.write_text("target_id,smiles\n../user-label,CCO\n")
    assert retrosynthesis.load_targets(targets, 1) == [
        {"target_id": "../user-label", "smiles": "CCO", "canonical_smiles": "CCO"}
    ]


def test_generation_defaults_match_bounded_manuscript_protocol():
    assert generation.PARENT_ID == "CHEMBL3327073"
    from rdkit import Chem

    assert Chem.MolFromSmiles(generation.PARENT_SMILES) is not None


def test_controller_lock_blocks_concurrent_execution(tmp_path):
    out = tmp_path / "study"
    with common._controller_lock(out):
        with pytest.raises(ValueError, match="controller lock"):
            common.run_jobs(_spec(), out, worker_script=_worker(tmp_path))


def test_resume_does_not_trust_changed_artifacts(tmp_path):
    out = tmp_path / "study"
    worker = _worker(tmp_path)
    common.run_jobs(_spec(), out, worker_script=worker)
    artifact = out / "case_2" / "called"
    execution = out / "case_2" / "execution.json"
    record = json.loads(execution.read_text())
    record["result"]["artifacts"] = {"called": common.file_identity(artifact)}
    common.write_json(execution, record)
    artifact.write_text("altered")
    with pytest.raises(ValueError, match="artifact changed"):
        common.run_jobs(_spec(), out, worker_script=worker, resume=True)


def test_provenance_records_actual_distribution_and_redacts_source_credentials(monkeypatch):
    class Distribution:
        version = "4.1.35"

        def locate_file(self, _path):
            return Path("/installed/compatibility/runtime")

        def read_text(self, name):
            assert name == "direct_url.json"
            return json.dumps(
                {
                    "url": "https://user:secret@example.org/repo.git?token=secret#secret",
                    "vcs_info": {"vcs": "git", "commit_id": "abc123"},
                }
            )

    def distribution(name):
        if name == "CGRtools":
            return Distribution()
        raise common.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(common.importlib.metadata, "distribution", distribution)
    identity = common.software_identity()
    assert identity["packages"]["CGRtools"] == "4.1.35"
    assert identity["packages"]["cgrtools-stable"] is None
    recorded = identity["distributions"]["CGRtools"]
    assert recorded["source_url"] == "https://example.org/repo.git"
    assert recorded["commit_id"] == "abc123"
    assert "secret" not in json.dumps(identity)
