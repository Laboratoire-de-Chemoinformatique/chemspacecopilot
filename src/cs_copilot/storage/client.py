#!/usr/bin/env python
# coding: utf-8
"""
S3 storage client for unified file operations.

Provides transparent access to files stored in S3/MinIO or locally,
with automatic session-based path management.
"""

import builtins
import datetime
import logging
import os
import uuid
from pathlib import Path, PurePosixPath

import fsspec

from .config import get_s3_config, is_s3_enabled

logger = logging.getLogger(__name__)
LOCAL_STORAGE_ROOT = Path("data")

# Generate a per-run session ID when SESSION_ID is unset or blank
_ENV_SESSION_ID = os.getenv("SESSION_ID")
if _ENV_SESSION_ID is None or _ENV_SESSION_ID.strip() == "":
    logger.info("SESSION_ID is not set, generating new session ID")
    # Timestamp + short uuid for readability and uniqueness
    _ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    SESSION_ID = f"{_ts}-{uuid.uuid4().hex[:6]}"
else:
    logger.info("SESSION_ID is set from environment")
    SESSION_ID = _ENV_SESSION_ID


class S3:
    """
    Unified storage client for S3/MinIO and local file operations.

    All file operations are scoped to the current session by default,
    unless an absolute S3 URL or local path is provided.

    Class Attributes:
    -----------------
    prefix : str
        Session-scoped prefix for all relative paths

    Methods:
    --------
    path(rel: str) -> str
        Convert a relative path to an S3 URL

    open(rel: str, mode: str = "rb")
        Open a file for reading or writing

    Examples:
    ---------
    >>> # Write to session-scoped S3 path
    >>> with S3.open("results.csv", "w") as f:
    ...     f.write("data")

    >>> # Read from absolute S3 URL
    >>> with S3.open("s3://bucket/key.csv", "r") as f:
    ...     data = f.read()

    >>> # Read from local file
    >>> with S3.open("/tmp/local.csv", "r") as f:
    ...     data = f.read()
    """

    prefix = f"sessions/{SESSION_ID}"
    logger.info(f"Initialized S3 client with SESSION_ID: {SESSION_ID}")

    @classmethod
    def path(cls, rel: str) -> str:
        """
        Convert a relative path to an S3 URL or local session path.

        Explicit S3 and local paths are returned unchanged. Relative paths are
        resolved against the active backend.

        Args:
            rel: Relative path or absolute S3 URL

        Returns:
            str: Full S3 URL or local path

        Examples:
        ---------
        >>> S3.path("data.csv")
        's3://chatbot-assets/sessions/20250121-123456-abc123/data.csv'

        >>> S3.path("s3://mybucket/data.csv")
        's3://mybucket/data.csv'
        """
        if cls._is_s3_url(rel) or cls._is_explicit_local_path(rel):
            return rel

        config = get_s3_config()
        if not is_s3_enabled():
            return os.fspath(cls._local_session_path(rel))

        key = PurePosixPath(cls.prefix.strip("/")) / str(rel).lstrip("/")
        return f"s3://{config.bucket_name}/{key.as_posix()}"

    @classmethod
    def open(cls, rel: str, mode: str = "rb"):
        """
        Open a file for reading or writing.

        Supports three types of paths:
        1. Absolute S3 URLs (s3://...)
        2. Local absolute paths (/ or file://)
        3. Relative paths (scoped to current session)

        Args:
            rel: File path (relative, absolute, or S3 URL)
            mode: File mode ('r', 'w', 'rb', 'wb', etc.)

        Returns:
            File-like object opened with the specified mode

        Examples:
        ---------
        >>> # Read from session-scoped S3 path
        >>> with S3.open("data.csv", "r") as f:
        ...     df = pd.read_csv(f)

        >>> # Write to session-scoped S3 path
        >>> with S3.open("output.csv", "w") as f:
        ...     df.to_csv(f)

        >>> # Read from absolute S3 URL
        >>> with S3.open("s3://bucket/data.csv", "r") as f:
        ...     df = pd.read_csv(f)

        >>> # Read from local file
        >>> with S3.open("/tmp/data.csv", "r") as f:
        ...     df = pd.read_csv(f)
        """
        config = get_s3_config()

        if cls._is_s3_url(rel):
            return fsspec.open(rel, mode=mode, **config.to_storage_options())

        if cls._is_explicit_local_path(rel):
            if rel.startswith("file://"):
                return fsspec.open(rel, mode=mode)
            return cls._open_local_path(Path(rel), mode)

        if not is_s3_enabled():
            return cls._open_local_path(cls._local_session_path(rel), mode)

        return fsspec.open(cls.path(rel), mode=mode, **config.to_storage_options())

    @staticmethod
    def _is_s3_url(path: str) -> bool:
        return isinstance(path, str) and path.startswith("s3://")

    @staticmethod
    def _is_explicit_local_path(path: str) -> bool:
        return isinstance(path, str) and (path.startswith("/") or path.startswith("file://"))

    @classmethod
    def _local_session_path(cls, rel: str) -> Path:
        return LOCAL_STORAGE_ROOT / Path(cls.prefix.strip("/")) / Path(rel)

    @staticmethod
    def _is_write_mode(mode: str) -> bool:
        return any(flag in mode for flag in ("w", "a", "x", "+"))

    @classmethod
    def _open_local_path(cls, path: Path, mode: str):
        if cls._is_write_mode(mode):
            path.parent.mkdir(parents=True, exist_ok=True)
        return builtins.open(path, mode)
