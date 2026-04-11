#!/usr/bin/env python
# coding: utf-8
"""Unit tests for storage backend resolution and local session paths."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from cs_copilot.storage import S3, StorageConfigError, get_s3_config, is_s3_enabled


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
    assert S3.path("nested/dataset.csv") == "s3://test-bucket/sessions/test-session/nested/dataset.csv"


def test_valid_minio_config_enables_s3(clean_storage_env, monkeypatch):
    """USE_S3=true with an explicit endpoint should resolve to S3-compatible storage."""
    monkeypatch.setenv("USE_S3", "true")
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://localhost:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "minio-key")
    monkeypatch.setenv("MINIO_SECRET_KEY", "minio-secret")
    monkeypatch.setenv("ASSETS_BUCKET", "test-bucket")

    assert get_s3_config().storage_backend() == "s3-compatible"
    assert is_s3_enabled() is True


def test_incomplete_explicit_aws_config_raises(clean_storage_env, fixed_session_prefix, monkeypatch):
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


def test_explicit_s3_paths_stay_explicit(clean_storage_env):
    """Explicit s3:// paths should still go through fsspec regardless of default mode."""
    with patch("cs_copilot.storage.client.fsspec.open") as mock_open:
        S3.open("s3://bucket/key.csv", "r")

    mock_open.assert_called_once()
    assert mock_open.call_args.args[0] == "s3://bucket/key.csv"
