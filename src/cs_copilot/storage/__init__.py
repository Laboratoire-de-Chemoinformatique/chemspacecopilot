#!/usr/bin/env python
# coding: utf-8
"""
Cs_copilot Storage Module

Provides unified storage abstraction for local files and S3/MinIO.
Handles session management and file operations transparently.

Main Components:
----------------
- S3: Storage client class for file operations
- SESSION_ID: Current session identifier
- S3Config: Configuration dataclass for S3 settings

Usage:
------
    from cs_copilot.storage import S3

    # Read a file
    with S3.open("data/compounds.csv", "r") as f:
        df = pd.read_csv(f)

    # Write a file
    with S3.open("results/output.csv", "w") as f:
        df.to_csv(f)

    # Get S3 path
    path = S3.path("results/model.pkl.gz")
"""

from .client import S3, SESSION_ID
from .config import S3Config, StorageConfigError, get_s3_config, is_s3_enabled

__all__ = [
    "S3",
    "SESSION_ID",
    "S3Config",
    "StorageConfigError",
    "get_s3_config",
    "is_s3_enabled",
]
