#!/usr/bin/env python
# coding: utf-8
"""Unit tests for storage backend resolution and local session paths."""

import contextvars
import hashlib
import io
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cs_copilot.storage import (
    OUTPUT_CONTEXT_KEY,
    S3,
    OutputLayout,
    OutputOperation,
    StorageConfigError,
    ensure_output_context,
    get_s3_config,
    is_s3_enabled,
    normalize_run_relative_path,
    operation_rel_path,
    scoped_artifact_path,
)


@pytest.fixture
def clean_storage_env(monkeypatch):
    """Clear storage-related environment variables for deterministic tests."""
    for key in (
        "USE_S3",
        "S3_ENDPOINT_URL",
        "MINIO_ENDPOINT",
        "MINIO_ENDPOINT_URL",
        "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "ASSETS_BUCKET",
        "S3_BUCKET_NAME",
        "AWS_REGION",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def fixed_session_prefix():
    """Force a stable session prefix so expected paths stay deterministic."""
    original_prefix = S3.prefix
    S3.prefix = "sessions/test-session"
    try:
        yield
    finally:
        S3.prefix = original_prefix


def test_s3_disabled_by_default(clean_storage_env):
    """Relative storage should stay local when USE_S3 is unset."""
    config = get_s3_config()

    assert config.use_s3 is False
    assert config.storage_backend() == "local"
    assert is_s3_enabled() is False


def test_aws_credentials_do_not_enable_s3_without_flag(
    clean_storage_env, fixed_session_prefix, monkeypatch
):
    """Ambient AWS credentials should not switch the default backend to S3."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setenv("ASSETS_BUCKET", "test-bucket")

    assert is_s3_enabled() is False
    assert S3.path("dataset.csv") == os.fspath(
        Path("data") / "sessions" / "test-session" / "dataset.csv"
    )


def test_valid_aws_config_enables_s3(clean_storage_env, fixed_session_prefix, monkeypatch):
    """USE_S3=true without an endpoint should resolve to AWS S3."""
    monkeypatch.setenv("USE_S3", "true")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setenv("ASSETS_BUCKET", "test-bucket")

    assert get_s3_config().storage_backend() == "aws"
    assert is_s3_enabled() is True
    assert (
        S3.path("nested/dataset.csv") == "s3://test-bucket/sessions/test-session/nested/dataset.csv"
    )


def test_valid_minio_config_enables_s3(clean_storage_env, monkeypatch):
    """USE_S3=true with an explicit endpoint should resolve to S3-compatible storage."""
    monkeypatch.setenv("USE_S3", "true")
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://localhost:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "minio-key")
    monkeypatch.setenv("MINIO_SECRET_KEY", "minio-secret")
    monkeypatch.setenv("ASSETS_BUCKET", "test-bucket")

    assert get_s3_config().storage_backend() == "s3-compatible"
    assert is_s3_enabled() is True


def test_incomplete_explicit_aws_config_raises(
    clean_storage_env, fixed_session_prefix, monkeypatch
):
    """Explicit S3 mode should fail clearly when AWS credentials are incomplete."""
    monkeypatch.setenv("USE_S3", "true")
    monkeypatch.setenv("ASSETS_BUCKET", "test-bucket")

    with pytest.raises(StorageConfigError, match="AWS_ACCESS_KEY_ID"):
        is_s3_enabled()

    with pytest.raises(StorageConfigError, match="AWS_ACCESS_KEY_ID"):
        S3.path("dataset.csv")


def test_relative_local_paths_are_session_scoped(
    clean_storage_env, fixed_session_prefix, monkeypatch, tmp_path
):
    """Relative local paths should resolve under data/sessions/{SESSION_ID}."""
    monkeypatch.chdir(tmp_path)

    with S3.open("nested/output.csv", "w") as handle:
        handle.write("value\n1\n")

    saved_path = tmp_path / "data" / "sessions" / "test-session" / "nested" / "output.csv"
    assert saved_path.exists()
    assert saved_path.read_text() == "value\n1\n"
    assert S3.path("nested/output.csv") == os.fspath(
        Path("data") / "sessions" / "test-session" / "nested" / "output.csv"
    )

    with S3.open("nested/output.csv", "r") as handle:
        assert handle.read() == "value\n1\n"


def test_session_qualified_local_paths_are_idempotent(
    clean_storage_env, fixed_session_prefix, monkeypatch, tmp_path
):
    """Pointers returned by S3.path must remain readable through S3.open."""
    monkeypatch.chdir(tmp_path)
    qualified_path = S3.path("nested/output.csv")

    assert S3.path(qualified_path) == qualified_path
    with S3.open(qualified_path, "w") as handle:
        handle.write("value\n1\n")
    with S3.open(qualified_path, "r") as handle:
        assert handle.read() == "value\n1\n"

    assert not (tmp_path / "data/sessions/test-session/data").exists()


def test_confined_local_writes_are_descriptor_safe_and_context_local(
    clean_storage_env, fixed_session_prefix, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    boundary = "workflows/bounded-run"
    immutable = f"{boundary}/inputs/request.json"
    with S3.open(immutable, "w") as handle:
        handle.write("original")

    with S3.confine_writes(boundary, protected_paths=(immutable,)):
        assert S3.current_write_boundary() == boundary
        with S3.open(f"{boundary}/reports/result.txt", "w") as handle:
            handle.write("bounded")
        with pytest.raises(PermissionError, match="outside the active boundary"):
            S3.open("workflows/another-run/result.txt", "w")
        with pytest.raises(PermissionError, match="reserved"):
            S3.open(f"{boundary}/manifest.json", "w")
        with pytest.raises(PermissionError, match="reserved"):
            S3.open(f"{boundary}/.staging/not-a-worker/result.txt", "w")
        with pytest.raises(PermissionError, match="immutable registered"):
            S3.open(immutable, "w")

    assert S3.current_write_boundary() is None
    assert S3.current_write_protected_paths() == frozenset()
    result = tmp_path / "data/sessions/test-session/workflows/bounded-run/reports/result.txt"
    assert result.read_text() == "bounded"
    immutable_path = (
        tmp_path / "data/sessions/test-session/workflows/bounded-run/inputs/request.json"
    )
    assert immutable_path.read_text() == "original"


def test_confined_writes_are_transactional_and_support_read_your_writes(
    clean_storage_env,
    fixed_session_prefix,
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    boundary = "workflows/transaction-run"
    key = f"{boundary}/reports/result.txt"
    expanded = Path(S3.path(key))
    publications: dict[str, dict[str, object]] = {}

    with S3.confine_writes(boundary, publication_receipt=publications):
        with S3.open(key, "w") as handle:
            handle.write("complete")
        assert not expanded.exists()
        assert publications == {}
        with S3.open(S3.path(key), "r") as handle:
            assert handle.read() == "complete"

    assert expanded.read_text() == "complete"
    assert publications == {
        key: {
            "staged_path": key,
            "sha256": hashlib.sha256(b"complete").hexdigest(),
            "size_bytes": len(b"complete"),
        }
    }


def test_failed_confined_write_is_aborted_and_same_path_can_be_retried(
    clean_storage_env,
    fixed_session_prefix,
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    boundary = "workflows/retry-run"
    key = f"{boundary}/result.txt"

    with pytest.raises(RuntimeError, match="later stage"):
        with S3.confine_writes(boundary):
            with S3.open(key, "w") as handle:
                handle.write("partial")
            raise RuntimeError("later stage failed")

    assert not Path(S3.path(key)).exists()
    with S3.confine_writes(boundary):
        with S3.open(key, "w") as handle:
            handle.write("retry")
    assert Path(S3.path(key)).read_text() == "retry"


def test_runtime_atomic_create_bypasses_domain_artifact_staging(
    clean_storage_env,
    fixed_session_prefix,
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    boundary = "workflows/event-run"
    event_key = f"{boundary}/events/00000001.jsonl"

    with S3.confine_writes(boundary):
        with S3.open_atomic(event_key, "x") as handle:
            handle.write('{"event":"accepted"}\n')
        assert Path(S3.path(event_key)).read_text() == '{"event":"accepted"}\n'


def test_worker_publications_require_explicit_parent_promotion(
    clean_storage_env,
    fixed_session_prefix,
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    boundary = "workflows/worker-run"
    final_key = f"{boundary}/result.csv"

    with S3.stage_publications("job-123"):
        with S3.confine_writes(boundary):
            with S3.open(final_key, "w") as handle:
                handle.write("value\n1\n")
        staged_paths = S3.current_staged_publications()
        publications = S3.staged_publication_metadata()

    assert not Path(S3.path(final_key)).exists()
    assert staged_paths == {
        final_key: f"{boundary}/.staging/job-123/result.csv",
    }
    assert publications[final_key]["staged_path"] == staged_paths[final_key]
    assert publications[final_key]["size_bytes"] == len("value\n1\n")
    assert Path(S3.path(staged_paths[final_key])).exists()

    S3.promote_staged_publications(publications)

    assert Path(S3.path(final_key)).read_text() == "value\n1\n"
    assert not Path(S3.path(staged_paths[final_key])).exists()


def test_staging_namespace_cannot_be_registered_as_a_run_artifact():
    with pytest.raises(ValueError, match="reserved"):
        normalize_run_relative_path("run-1", ".staging/job-1/result.csv")


def test_local_staging_cleanup_rejects_session_prefix_symlink(
    clean_storage_env,
    fixed_session_prefix,
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    external = tmp_path / "external"
    staged = external / "workflows/worker-run/.staging/job-123/result.csv"
    staged.parent.mkdir(parents=True)
    staged.write_text("must survive")
    session_parent = tmp_path / "data/sessions"
    session_parent.mkdir(parents=True)
    (session_parent / "test-session").symlink_to(external, target_is_directory=True)

    with pytest.raises(PermissionError, match="symlink"):
        S3.discard_staging_prefix("workflows/worker-run", "job-123")

    assert staged.read_text() == "must survive"


def test_verified_snapshot_fails_before_consuming_oversized_source(monkeypatch):
    class TrackingSource(io.BytesIO):
        def __init__(self):
            super().__init__(b"x" * (2 * 1024 * 1024))
            self.read_calls = 0

        def read(self, size=-1):
            self.read_calls += 1
            return super().read(size)

    source = TrackingSource()
    opened = MagicMock()
    opened.open.return_value = source
    config = MagicMock()
    config.to_storage_options.return_value = {}
    monkeypatch.setattr("cs_copilot.storage.client.is_s3_enabled", lambda: True)
    monkeypatch.setattr("cs_copilot.storage.client.get_s3_config", lambda: config)
    monkeypatch.setattr("cs_copilot.storage.client.fsspec.open", lambda *args, **kwargs: opened)

    with pytest.raises(PermissionError, match="exceeded its expected size"):
        S3._open_verified_snapshot(
            "workflows/run-1/result.csv",
            "rb",
            expected_sha256=hashlib.sha256(b"x").hexdigest(),
            expected_size=1,
        )

    assert source.read_calls == 1


def test_rollback_removes_only_unchanged_publication(
    clean_storage_env,
    fixed_session_prefix,
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    key = "workflows/rollback-run/result.csv"
    original = b"value\n1\n"
    metadata = {
        key: {
            "staged_path": key,
            "sha256": hashlib.sha256(original).hexdigest(),
            "size_bytes": len(original),
        }
    }
    path = Path(S3.path(key))
    path.parent.mkdir(parents=True)
    path.write_bytes(original)

    S3.rollback_promoted_publications(metadata)
    assert not path.exists()

    path.write_bytes(b"replacement")
    S3.rollback_promoted_publications(metadata)
    assert path.read_bytes() == b"replacement"


def test_relative_paths_use_context_local_session_prefix(clean_storage_env, monkeypatch, tmp_path):
    """Concurrent execution contexts should not share the mutable storage prefix."""
    monkeypatch.chdir(tmp_path)

    def path_for(prefix: str) -> str:
        S3.set_session_prefix(prefix)
        return S3.path("artifact.txt")

    context_a = contextvars.Context()
    context_b = contextvars.Context()

    path_a = context_a.run(path_for, "sessions/session-a")
    path_b = context_b.run(path_for, "sessions/session-b")

    assert path_a == os.fspath(Path("data") / "sessions" / "session-a" / "artifact.txt")
    assert path_b == os.fspath(Path("data") / "sessions" / "session-b" / "artifact.txt")
    assert context_a.run(S3.path, "artifact.txt") == path_a
    assert context_b.run(S3.path, "artifact.txt") == path_b


@pytest.mark.parametrize(
    "prefix",
    ["/tmp/session", "../session", "sessions/../../escape", r"sessions\\escape"],
)
def test_session_prefix_rejects_filesystem_escape(prefix):
    with pytest.raises(ValueError, match="safe relative path"):
        S3.set_session_prefix(prefix)


def test_output_layout_scopes_operation_artifacts_to_workflow():
    """Workflow output helpers should organize relative artifacts by operation."""
    old_prefix = S3.current_prefix()
    S3.set_session_prefix("sessions/layout-session")

    try:
        state = {}

        context = ensure_output_context(
            state,
            workflow_slug="EGFR SAR",
            run_id="egfr-run",
        )
        path = scoped_artifact_path(
            "../model.pkl.gz",
            OutputOperation.CHEMICAL_SPACE,
            "gtm",
            "models",
            session_state=state,
        )
        reports_path = operation_rel_path(
            OutputOperation.REPORTS,
            "gtm_activity",
            "report.html",
            session_state=state,
            workflow_slug="reports",
        )
    finally:
        S3.set_session_prefix(old_prefix)

    assert context == {
        "layout_version": 4,
        "session_id": "layout-session",
        "run_id": "egfr-run",
        "workflow_slug": "egfr_sar",
    }
    assert state[OUTPUT_CONTEXT_KEY] == context
    assert path == "workflows/egfr-run/01_chemical_space/gtm/models/model.pkl.gz"
    assert reports_path == "workflows/egfr-run/reports/gtm_activity/report.html"


def test_output_layout_rejects_unsafe_identity_components():
    with pytest.raises(ValueError, match="run_id"):
        OutputLayout(
            session_id="safe-session",
            run_id="../another-run",
            workflow_slug="workflow",
        )


def test_output_context_does_not_reuse_a_malformed_serialized_context():
    old_prefix = S3.current_prefix()
    S3.set_session_prefix("sessions/malformed-context-session")
    state = {
        OUTPUT_CONTEXT_KEY: {
            "layout_version": 4,
            "session_id": "malformed-context-session",
            "run_id": "old-run",
        }
    }
    try:
        context = ensure_output_context(
            state,
            workflow_slug="replacement",
            run_id="replacement-run",
        )
    finally:
        S3.set_session_prefix(old_prefix)

    assert context["run_id"] == "replacement-run"
    assert context["workflow_slug"] == "replacement"


def test_output_layout_uses_one_root_per_explicit_workflow_run():
    """Different tools in one session share a root only when run identity matches."""
    old_prefix = S3.current_prefix()
    S3.set_session_prefix("sessions/layout-session")

    try:
        state = {
            OUTPUT_CONTEXT_KEY: {
                "layout_version": 4,
                "session_id": "layout-session",
                "run_id": "shared-run",
                "workflow_slug": "chemical_space",
            }
        }
        context = ensure_output_context(state, workflow_slug="chemical_space")
        direct_path = operation_rel_path(
            OutputOperation.ANALOG_GENERATION,
            "candidate_sets",
            "cset_001",
            session_state=state,
        )
        state_path = operation_rel_path(
            OutputOperation.CHEMICAL_SPACE,
            "gtm",
            session_state=state,
            workflow_slug="chemical_space",
        )
    finally:
        S3.set_session_prefix(old_prefix)

    assert direct_path == "workflows/shared-run/02_analog_generation/candidate_sets/cset_001"
    assert state_path == "workflows/shared-run/01_chemical_space/gtm"
    assert context["session_id"] == "layout-session"
    assert context["run_id"] == "shared-run"
    assert state[OUTPUT_CONTEXT_KEY] == context


def test_output_layout_leaves_explicit_paths_untouched():
    """Absolute and S3 paths should not be moved into workflow folders."""
    state = {}

    assert (
        scoped_artifact_path(
            "s3://bucket/custom.csv",
            OutputOperation.ANALOG_GENERATION,
            "candidate_sets",
            session_state=state,
        )
        == "s3://bucket/custom.csv"
    )
    assert (
        scoped_artifact_path(
            "/tmp/custom.csv",
            OutputOperation.CHEMICAL_SPACE,
            "datasets",
            session_state=state,
        )
        == "/tmp/custom.csv"
    )


def test_explicit_s3_paths_stay_explicit(clean_storage_env):
    """Explicit s3:// paths should still go through fsspec regardless of default mode."""
    with patch("cs_copilot.storage.client.fsspec.open") as mock_open:
        S3.open("s3://bucket/key.csv", "r")

    mock_open.assert_called_once()
    assert mock_open.call_args.args[0] == "s3://bucket/key.csv"


def test_confined_s3_autocommit_is_a_file_open_option():
    """S3 transaction flags must not leak into filesystem construction."""

    filesystem = MagicMock()
    handle = MagicMock()
    filesystem.open.return_value = handle
    with patch(
        "cs_copilot.storage.client.fsspec.core.url_to_fs",
        return_value=(filesystem, "bucket/key.csv"),
    ) as resolve:
        opened = S3._open_confined_s3_path(
            "s3://bucket/key.csv",
            "wb",
            storage_options={"key": "test-key"},
            defer_transaction=False,
        )

    resolve.assert_called_once_with("s3://bucket/key.csv", key="test-key")
    filesystem.open.assert_called_once_with(
        "bucket/key.csv",
        mode="xb",
        autocommit=True,
    )
    opened._abort()
