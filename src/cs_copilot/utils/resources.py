#!/usr/bin/env python
# coding: utf-8
"""
Runtime resource detection for cs_copilot.

Probes the local environment (GPU, CPU, RAM, databases, cached models) and
returns a structured profile dict.  All checks are local -- no network I/O,
no model loading -- so the function is safe to call at startup (<100 ms).
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from cs_copilot.tools.constants import (
    DEFAULT_AUTOENCODER_MODEL_PATH,
    DEFAULT_DBAASP_DATA_PATH,
    DEFAULT_GTM_MODEL_PATH,
    DEFAULT_PEPTIDE_WAE_MODEL_PATH,
    HUGGINGFACE_AUTOENCODER_REPO,
    HUGGINGFACE_GTM_REPO,
    HUGGINGFACE_PEPTIDE_WAE_REPO,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_gpu() -> Dict[str, Any]:
    """Detect GPU/CUDA availability with a functional test."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False, "device_name": None, "cuda_functional": False}

        # Functional test (same pattern as gtm_operations.py:2322-2334)
        try:
            torch.tensor([1.0]).cuda()
            device_name = torch.cuda.get_device_name(0)
            return {
                "available": True,
                "device_name": device_name,
                "cuda_functional": True,
            }
        except Exception as exc:
            logger.warning("CUDA reported available but not functional: %s", exc)
            return {"available": True, "device_name": None, "cuda_functional": False}

    except ImportError:
        logger.info("PyTorch not installed -- GPU detection skipped")
        return {"available": False, "device_name": None, "cuda_functional": False}


def _detect_cpu() -> Dict[str, Any]:
    """Detect CPU core count and usable parallel workers."""
    core_count = os.cpu_count() or 1
    return {"core_count": core_count, "usable_workers": core_count}


def _detect_ram() -> Dict[str, Optional[float]]:
    """Detect total and available RAM by parsing /proc/meminfo (Linux).

    Returns ``None`` values on non-Linux platforms.
    """
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return {"total_gb": None, "available_gb": None}

    try:
        text = meminfo.read_text()
        info: Dict[str, float] = {}
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].rstrip(":")
                # Values in /proc/meminfo are in kB
                if key in ("MemTotal", "MemAvailable"):
                    info[key] = float(parts[1]) / (1024 * 1024)  # kB -> GB
        return {
            "total_gb": round(info.get("MemTotal", 0), 1) or None,
            "available_gb": round(info.get("MemAvailable", 0), 1) or None,
        }
    except Exception as exc:
        logger.warning("Failed to read /proc/meminfo: %s", exc)
        return {"total_gb": None, "available_gb": None}


def _detect_chembl_backend() -> Dict[str, str]:
    """Detect configured ChEMBL backend from environment variables.

    Mirrors the priority cascade in chembl.py _resolve_backend:
    SQLite > PostgreSQL > MySQL > REST API.
    """
    if os.getenv("CHEMBL_SQLITE_PATH"):
        return {
            "backend": "sqlite",
            "description": "Local SQLite ChEMBL database",
        }
    if os.getenv("CHEMBL_PG_HOST"):
        return {
            "backend": "postgresql",
            "description": "Local PostgreSQL ChEMBL database",
        }
    if os.getenv("CHEMBL_MYSQL_HOST"):
        return {
            "backend": "mysql",
            "description": "Local MySQL ChEMBL database",
        }
    return {
        "backend": "rest",
        "description": "ChEMBL REST API (no local database configured)",
    }


def _detect_storage_backend() -> Dict[str, Optional[str]]:
    """Detect configured storage backend from environment variables."""
    use_s3 = os.getenv("USE_S3", "").lower()
    if use_s3 == "false":
        return {"backend": "local", "bucket": None}

    endpoint = os.getenv("S3_ENDPOINT_URL")
    if endpoint:
        bucket = os.getenv("ASSETS_BUCKET", "chatbot-assets")
        return {"backend": "s3-compatible", "bucket": bucket}

    if os.getenv("AWS_ACCESS_KEY_ID"):
        bucket = os.getenv("ASSETS_BUCKET", "chatbot-assets")
        return {"backend": "aws", "bucket": bucket}

    return {"backend": "local", "bucket": None}


def _dir_has_files(path: str) -> bool:
    """Check if a directory exists and contains at least one file."""
    p = Path(path)
    if not p.is_dir():
        return False
    try:
        return any(p.iterdir())
    except OSError:
        return False


def _detect_cached_models() -> Dict[str, bool]:
    """Check which models are cached locally."""
    return {
        "autoencoder": _dir_has_files(DEFAULT_AUTOENCODER_MODEL_PATH),
        "peptide_wae": _dir_has_files(DEFAULT_PEPTIDE_WAE_MODEL_PATH),
        "gtm": _dir_has_files(DEFAULT_GTM_MODEL_PATH),
        "dbaasp_data": Path(DEFAULT_DBAASP_DATA_PATH).is_file(),
    }


def _build_recommendations(profile: Dict[str, Any]) -> List[str]:
    """Generate human-readable recommendations from the resource profile."""
    recs: List[str] = []

    # GPU
    gpu = profile["gpu"]
    if gpu["cuda_functional"]:
        name = gpu["device_name"] or "unknown device"
        recs.append(
            f"GPU detected ({name}) -- GTM optimization and autoencoder "
            "encoding will use CUDA acceleration."
        )
    elif gpu["available"]:
        recs.append(
            "CUDA reported available but not functional -- computations "
            "will fall back to CPU."
        )
    else:
        recs.append(
            "No GPU detected -- computations will run on CPU. Consider "
            "'low' GTM optimization strategy for faster results."
        )

    # CPU
    cpu = profile["cpu"]
    cores = cpu["core_count"]
    recs.append(
        f"{cores} CPU core{'s' if cores != 1 else ''} detected -- "
        "fingerprint and SMILES processing will be parallelized."
    )

    # RAM
    ram = profile["ram"]
    if ram["total_gb"] is not None:
        recs.append(f"{ram['total_gb']} GB RAM total, {ram['available_gb']} GB available.")

    # ChEMBL backend
    chembl = profile["chembl_backend"]
    if chembl["backend"] == "rest":
        recs.append(
            "No local ChEMBL database configured -- data download will use the "
            "REST API (functional but slower for large queries)."
        )
    else:
        recs.append(
            f"{chembl['description']} detected -- data queries will be fast and work offline."
        )

    # Cached models
    models = profile["cached_models"]
    not_cached = []
    if not models["autoencoder"]:
        not_cached.append(f"autoencoder ({HUGGINGFACE_AUTOENCODER_REPO})")
    if not models["peptide_wae"]:
        not_cached.append(f"peptide WAE ({HUGGINGFACE_PEPTIDE_WAE_REPO})")
    if not models["gtm"]:
        not_cached.append(f"GTM ({HUGGINGFACE_GTM_REPO})")

    if not_cached:
        recs.append(
            "Models not cached locally (will download from HuggingFace on first use): "
            + ", ".join(not_cached)
            + "."
        )

    return recs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_resources() -> Dict[str, Any]:
    """Probe the runtime environment and return a resource profile.

    Fast (no network I/O, no model loading). Safe to call at startup.

    Returns:
        Dict with keys: ``gpu``, ``cpu``, ``ram``, ``chembl_backend``,
        ``storage_backend``, ``cached_models``, ``recommendations``.
    """
    profile: Dict[str, Any] = {
        "gpu": _detect_gpu(),
        "cpu": _detect_cpu(),
        "ram": _detect_ram(),
        "chembl_backend": _detect_chembl_backend(),
        "storage_backend": _detect_storage_backend(),
        "cached_models": _detect_cached_models(),
    }
    profile["recommendations"] = _build_recommendations(profile)

    logger.info("Resource profile: %s", profile)
    return profile
