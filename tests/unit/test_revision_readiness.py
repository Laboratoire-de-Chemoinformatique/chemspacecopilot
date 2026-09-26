"""Offline readiness checks must detect unusable assets and never reveal secrets."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).parents[2] / "scripts/check_revision_readiness.py"
SPEC = importlib.util.spec_from_file_location("check_revision_readiness", SCRIPT)
readiness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(readiness)


def write_config(tmp_path, tests=None):
    path = tmp_path / "benchmark.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "general": {"repetitions": 3, "n_variations": 3, "timeout_seconds": 30},
                "model": {
                    "provider": "deepseek",
                    "model_id": "deepseek-chat",
                    "api_key_env": "TEST_PROVIDER_KEY",
                },
                "tests": tests
                or {
                    "recovery": {
                        "enabled": True,
                        "tier": "frozen",
                        "validator": "clarification",
                        "prompt_variants": ["clarify"],
                    }
                },
            }
        )
    )
    return path


def create_models(tmp_path):
    for folder, filenames in {
        "autoencoder": ["config.pt", "vocab.pt", "model.pt"],
        "peptide_designer": ["model.pt", "vocab.dict"],
    }.items():
        directory = tmp_path / ".cache/cs_copilot/models" / folder
        directory.mkdir(parents=True)
        for name in filenames:
            (directory / name).write_bytes(b"offline-presence-test")


@pytest.fixture
def versions(monkeypatch):
    monkeypatch.setattr(
        readiness.importlib.metadata, "version", lambda name: "2.1.9" if name == "agno" else "1.0"
    )


def test_detects_git_lfs_pointer_and_empty_checkpoint(tmp_path):
    pointer = tmp_path / "model.pt"
    pointer.write_bytes(readiness.LFS_HEADER + b"\nsize 1000\n")
    assert (
        readiness.artifact(str(pointer), base=tmp_path, env={}, home=tmp_path)["status"]
        == "git_lfs_pointer"
    )
    pointer.write_bytes(b"")
    assert (
        readiness.artifact(str(pointer), base=tmp_path, env={}, home=tmp_path)["status"] == "empty"
    )


def test_recovery_requires_eager_design_checkpoints_and_redacts_provider_key(tmp_path, versions):
    secret = "unique-test-key-never-print"
    report = readiness.check_readiness(
        write_config(tmp_path),
        project_root=tmp_path,
        env={"TEST_PROVIDER_KEY": secret},
        home=tmp_path,
    )
    assert report["ready_for_pilot"] is False
    assert report["model"]["credential"]["present"] is True
    assert secret not in json.dumps(report)
    assert any("autoencoder checkpoint" in blocker for blocker in report["blockers"])
    assert any("peptide checkpoint" in blocker for blocker in report["blockers"])


def test_recovery_ready_when_shared_inputs_present(tmp_path, versions):
    create_models(tmp_path)
    report = readiness.check_readiness(
        write_config(tmp_path),
        project_root=tmp_path,
        env={"TEST_PROVIDER_KEY": "present"},
        home=tmp_path,
        system="both",
    )
    assert report["ready_for_pilot"] is True
    assert report["planned_task_executions"] == 6
    assert report["configured_execution_timeout_ceiling_seconds"] == 180


def test_snapshot_hash_and_referenced_artifacts_are_checked(tmp_path):
    dataset = tmp_path / "data.csv"
    dataset.write_text("SMILES\nCCO\n")
    snapshot = tmp_path / "state.json"
    snapshot.write_text(json.dumps({"session_state": {"dataset_path": "data.csv"}}))
    fixture = {
        "session_state_path": str(snapshot),
        "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
    }
    kwargs = {"config_dir": tmp_path, "env": {}, "home": tmp_path}
    assert readiness.check_fixture(fixture, **kwargs)["status"] == "verified"
    dataset.unlink()
    assert readiness.check_fixture(fixture, **kwargs)["status"] == "artifact_unavailable"
    fixture["sha256"] = "0" * 64
    assert readiness.check_fixture(fixture, **kwargs)["status"] == "checksum_mismatch"


def test_unresolved_fixtures_and_remote_credentials_are_not_echoed(tmp_path):
    result = readiness.artifact("${MISSING_FIXTURE}", base=tmp_path, env={}, home=tmp_path)
    assert result["status"] == "unresolved_variable"
    result = readiness.artifact(
        "https://example.test/data?token=do-not-record", base=tmp_path, env={}, home=tmp_path
    )
    assert "do-not-record" not in json.dumps(result)
    assert result["status"] == "remote_not_verified_offline"


def test_task_execution_counts_respect_chains_and_case_selection(tmp_path, versions):
    config = write_config(
        tmp_path,
        tests={
            "chain": {
                "enabled": True,
                "tier": "live",
                "steps": [{"prompt": "one"}, {"prompt": "two"}, {"prompt": "three"}],
            },
            "frozen": {
                "enabled": True,
                "tier": "frozen",
                "prompt_variants": ["one", "two", "three"],
            },
        },
    )
    create_models(tmp_path)
    kwargs = {
        "project_root": tmp_path,
        "env": {"TEST_PROVIDER_KEY": "present"},
        "home": tmp_path,
        "system": "both",
        "n_variations": 2,
    }
    assert readiness.check_readiness(config, tier="live", **kwargs)["planned_task_executions"] == 18
    assert (
        readiness.check_readiness(config, tier="frozen", **kwargs)["planned_task_executions"] == 12
    )
    report = readiness.check_readiness(config, tests=["unknown"], **kwargs)
    assert report["ready_for_pilot"] is False
    assert "Unknown test: unknown" in report["blockers"]


def test_missing_dependency_is_reported_without_importing_it(tmp_path, versions, monkeypatch):
    def version(name):
        if name == "torch":
            raise readiness.importlib.metadata.PackageNotFoundError(name)
        return "2.1.9" if name == "agno" else "1.0"

    monkeypatch.setattr(readiness.importlib.metadata, "version", version)
    report = readiness.check_readiness(
        write_config(tmp_path),
        project_root=tmp_path,
        env={"TEST_PROVIDER_KEY": "present"},
        home=tmp_path,
    )
    assert "Missing package: torch" in report["blockers"]
