#!/usr/bin/env python
# coding: utf-8
"""
Storage configuration module.

Manages S3/MinIO connection settings and environment variable fallbacks.
"""

import os
from dataclasses import dataclass
from typing import Optional


class StorageConfigError(ValueError):
    """Raised when S3 is explicitly enabled but the runtime config is incomplete."""


def _getenv_nonempty(*names: str) -> Optional[str]:
    """Return the first non-empty environment variable from the provided names."""
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue

        normalized = value.strip()
        if normalized:
            return normalized

    return None


@dataclass
class S3Config:
    """Configuration for S3/MinIO storage."""

    endpoint_url: Optional[str]
    access_key_id: Optional[str]
    secret_access_key: Optional[str]
    bucket_name: Optional[str]
    region_name: str = "us-east-1"
    use_s3: bool = False

    @classmethod
    def from_env(cls) -> "S3Config":
        """
        Create S3Config from environment variables.

        Environment Variables:
        ----------------------
        Endpoint (one of):
            - MINIO_ENDPOINT
            - MINIO_ENDPOINT_URL
            - S3_ENDPOINT_URL

        Access Key (one of):
            - MINIO_ACCESS_KEY
            - AWS_ACCESS_KEY_ID

        Secret Key (one of):
            - MINIO_SECRET_KEY
            - AWS_SECRET_ACCESS_KEY

        Bucket Name (one of):
            - ASSETS_BUCKET
            - S3_BUCKET_NAME

        Other:
            - AWS_REGION (default: us-east-1)
            - USE_S3 (default: false)

        Returns:
            S3Config: Configuration instance
        """
        endpoint = _getenv_nonempty("MINIO_ENDPOINT", "MINIO_ENDPOINT_URL", "S3_ENDPOINT_URL")

        minio_access_key = _getenv_nonempty("MINIO_ACCESS_KEY")
        minio_secret_key = _getenv_nonempty("MINIO_SECRET_KEY")
        aws_access_key = _getenv_nonempty("AWS_ACCESS_KEY_ID")
        aws_secret_key = _getenv_nonempty("AWS_SECRET_ACCESS_KEY")

        if endpoint:
            access_key = minio_access_key or aws_access_key
            secret_key = minio_secret_key or aws_secret_key
        else:
            access_key = aws_access_key
            secret_key = aws_secret_key

        bucket_name = _getenv_nonempty("ASSETS_BUCKET", "S3_BUCKET_NAME")
        region_name = os.getenv("AWS_REGION", "us-east-1")
        use_s3 = os.getenv("USE_S3", "false").lower() == "true"

        return cls(
            endpoint_url=endpoint,
            access_key_id=access_key,
            secret_access_key=secret_key,
            bucket_name=bucket_name,
            region_name=region_name,
            use_s3=use_s3,
        )

    def storage_backend(self) -> str:
        """
        Resolve the active storage backend.

        Returns:
            str: "local", "aws", or "s3-compatible"

        Raises:
            StorageConfigError: If S3 is explicitly enabled but missing required settings.
        """
        if not self.use_s3:
            return "local"

        missing_bucket = not self.bucket_name
        missing_key = not self.access_key_id
        missing_secret = not self.secret_access_key

        if self.endpoint_url:
            missing = []
            if missing_bucket:
                missing.append("ASSETS_BUCKET or S3_BUCKET_NAME")
            if missing_key:
                missing.append("MINIO_ACCESS_KEY or AWS_ACCESS_KEY_ID")
            if missing_secret:
                missing.append("MINIO_SECRET_KEY or AWS_SECRET_ACCESS_KEY")
            if missing:
                raise StorageConfigError(
                    "USE_S3=true with an S3-compatible endpoint requires "
                    + ", ".join(missing)
                    + "."
                )
            return "s3-compatible"

        missing = []
        if missing_bucket:
            missing.append("ASSETS_BUCKET or S3_BUCKET_NAME")
        if missing_key:
            missing.append("AWS_ACCESS_KEY_ID")
        if missing_secret:
            missing.append("AWS_SECRET_ACCESS_KEY")
        if missing:
            raise StorageConfigError(
                "USE_S3=true without an endpoint requires "
                + ", ".join(missing)
                + " for AWS S3."
            )
        return "aws"

    def to_storage_options(self) -> dict:
        """
        Convert config to fsspec/s3fs storage options.

        Returns:
            dict: Storage options for fsspec.open()
        """
        opts = {"config_kwargs": {"s3": {"addressing_style": "path"}}}

        if self.access_key_id:
            opts["key"] = self.access_key_id

        if self.secret_access_key:
            opts["secret"] = self.secret_access_key

        if self.endpoint_url:
            opts["client_kwargs"] = {"endpoint_url": self.endpoint_url}

        return opts


def get_s3_config() -> S3Config:
    """
    Get S3 configuration from environment variables.

    This is evaluated at call time (not import time) so that notebooks can
    call load_dotenv() after importing modules and still have up-to-date
    credentials and endpoint settings.

    Returns:
        S3Config: Current S3 configuration
    """
    return S3Config.from_env()


def is_s3_enabled() -> bool:
    """
    Check if S3 is enabled based on configuration.

    Returns:
        bool: True if S3 should be used
    """
    return get_s3_config().storage_backend() != "local"
