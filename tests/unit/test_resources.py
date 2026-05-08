#!/usr/bin/env python
"""Unit tests for the runtime resource detection module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cs_copilot.utils.resources import (
    _build_recommendations,
    _detect_cached_models,
    _detect_chembl_backend,
    _detect_cpu,
    _detect_ram,
    _detect_storage_backend,
    _dir_has_files,
    analyze_resources,
)

# ---------------------------------------------------------------------------
# _detect_gpu
# ---------------------------------------------------------------------------


class TestDetectGpu:
    def test_cuda_functional(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA A100"
        mock_torch.tensor.return_value.cuda.return_value = None

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("cs_copilot.utils.resources.torch", mock_torch, create=True):
                # Re-import to pick up the mock -- easier to just call the internal
                import importlib

                import cs_copilot.utils.resources as res_mod

                importlib.reload(res_mod)
                result = res_mod._detect_gpu()

        assert result["available"] is True
        assert result["cuda_functional"] is True
        assert result["device_name"] == "NVIDIA A100"

    def test_cuda_not_available(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            import importlib

            import cs_copilot.utils.resources as res_mod

            importlib.reload(res_mod)
            result = res_mod._detect_gpu()

        assert result["available"] is False
        assert result["cuda_functional"] is False
        assert result["device_name"] is None

    def test_cuda_available_but_not_functional(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.tensor.return_value.cuda.side_effect = RuntimeError("CUDA error")

        with patch.dict("sys.modules", {"torch": mock_torch}):
            import importlib

            import cs_copilot.utils.resources as res_mod

            importlib.reload(res_mod)
            result = res_mod._detect_gpu()

        assert result["available"] is True
        assert result["cuda_functional"] is False


# ---------------------------------------------------------------------------
# _detect_cpu
# ---------------------------------------------------------------------------


class TestDetectCpu:
    def test_returns_core_count(self):
        with patch("os.cpu_count", return_value=8):
            result = _detect_cpu()
        assert result["core_count"] == 8
        assert result["usable_workers"] == 8

    def test_returns_1_when_cpu_count_is_none(self):
        with patch("os.cpu_count", return_value=None):
            result = _detect_cpu()
        assert result["core_count"] == 1
        assert result["usable_workers"] == 1


# ---------------------------------------------------------------------------
# _detect_ram
# ---------------------------------------------------------------------------


class TestDetectRam:
    MEMINFO_CONTENT = (
        "MemTotal:       16384000 kB\n"
        "MemFree:         2048000 kB\n"
        "MemAvailable:    8192000 kB\n"
        "Buffers:          512000 kB\n"
    )

    def test_parses_proc_meminfo(self, tmp_path):
        meminfo = tmp_path / "meminfo"
        meminfo.write_text(self.MEMINFO_CONTENT)

        original_path = Path

        def patched_path(arg):
            if arg == "/proc/meminfo":
                return meminfo
            return original_path(arg)

        with patch("cs_copilot.utils.resources.Path", side_effect=patched_path):
            result = _detect_ram()

        assert result["total_gb"] is not None
        assert result["total_gb"] == pytest.approx(15.6, abs=0.1)
        assert result["available_gb"] == pytest.approx(7.8, abs=0.1)

    def test_returns_none_when_not_linux(self):
        with patch("cs_copilot.utils.resources.Path") as mock_path_cls:
            mock_path_cls.return_value.exists.return_value = False
            result = _detect_ram()

        assert result["total_gb"] is None
        assert result["available_gb"] is None


# ---------------------------------------------------------------------------
# _detect_chembl_backend
# ---------------------------------------------------------------------------


class TestDetectChemblBackend:
    def test_sqlite_priority(self):
        env = {
            "CHEMBL_SQLITE_PATH": "/data/chembl.db",
            "CHEMBL_PG_HOST": "localhost",
        }
        with patch.dict("os.environ", env, clear=False):
            result = _detect_chembl_backend()
        assert result["backend"] == "sqlite"

    def test_postgresql(self):
        env = {"CHEMBL_PG_HOST": "localhost"}
        with patch.dict("os.environ", env, clear=False):
            with patch("cs_copilot.utils.resources._optional_driver_available", return_value=True):
                with patch.dict("os.environ", {"CHEMBL_SQLITE_PATH": ""}, clear=False):
                    result = _detect_chembl_backend()
        assert result["backend"] == "postgresql"

    def test_postgresql_falls_back_to_mysql_when_pg_driver_missing(self):
        env = {"CHEMBL_PG_HOST": "localhost", "CHEMBL_MYSQL_HOST": "localhost"}

        def has_driver(module_name: str) -> bool:
            return module_name == "pymysql"

        with patch.dict("os.environ", env, clear=False):
            with patch(
                "cs_copilot.utils.resources._optional_driver_available",
                side_effect=has_driver,
            ):
                with patch.dict("os.environ", {"CHEMBL_SQLITE_PATH": ""}, clear=False):
                    result = _detect_chembl_backend()
        assert result["backend"] == "mysql"

    def test_mysql(self):
        env = {"CHEMBL_MYSQL_HOST": "localhost"}
        with patch.dict(
            "os.environ",
            {**env, "CHEMBL_SQLITE_PATH": "", "CHEMBL_PG_HOST": ""},
            clear=False,
        ):
            with patch("cs_copilot.utils.resources._optional_driver_available", return_value=True):
                result = _detect_chembl_backend()
        assert result["backend"] == "mysql"

    def test_mysql_falls_back_to_rest_when_driver_missing(self):
        env = {"CHEMBL_MYSQL_HOST": "localhost"}
        with patch.dict(
            "os.environ",
            {**env, "CHEMBL_SQLITE_PATH": "", "CHEMBL_PG_HOST": ""},
            clear=False,
        ):
            with patch("cs_copilot.utils.resources._optional_driver_available", return_value=False):
                result = _detect_chembl_backend()
        assert result["backend"] == "rest"
        assert "pymysql" in result["description"]

    def test_rest_fallback(self):
        with patch.dict(
            "os.environ",
            {"CHEMBL_SQLITE_PATH": "", "CHEMBL_PG_HOST": "", "CHEMBL_MYSQL_HOST": ""},
            clear=False,
        ):
            result = _detect_chembl_backend()
        assert result["backend"] == "rest"


# ---------------------------------------------------------------------------
# _detect_storage_backend
# ---------------------------------------------------------------------------


class TestDetectStorageBackend:
    def test_local_when_s3_disabled(self):
        with patch.dict("os.environ", {"USE_S3": "false"}, clear=False):
            result = _detect_storage_backend()
        assert result["backend"] == "local"
        assert result["bucket"] is None

    def test_s3_compatible(self):
        env = {"USE_S3": "true", "S3_ENDPOINT_URL": "http://minio:9000"}
        with patch.dict("os.environ", env, clear=False):
            result = _detect_storage_backend()
        assert result["backend"] == "s3-compatible"

    def test_aws(self):
        env = {"USE_S3": "true", "AWS_ACCESS_KEY_ID": "AKIA...", "S3_ENDPOINT_URL": ""}
        with patch.dict("os.environ", env, clear=False):
            result = _detect_storage_backend()
        assert result["backend"] == "aws"

    def test_local_fallback(self):
        with patch.dict(
            "os.environ",
            {"USE_S3": "", "S3_ENDPOINT_URL": "", "AWS_ACCESS_KEY_ID": ""},
            clear=False,
        ):
            result = _detect_storage_backend()
        assert result["backend"] == "local"


# ---------------------------------------------------------------------------
# _dir_has_files / _detect_cached_models
# ---------------------------------------------------------------------------


class TestDetectCachedModels:
    def test_dir_has_files_true(self, tmp_path):
        (tmp_path / "model.pt").touch()
        assert _dir_has_files(str(tmp_path)) is True

    def test_dir_has_files_false_empty(self, tmp_path):
        assert _dir_has_files(str(tmp_path)) is False

    def test_dir_has_files_missing_dir(self):
        assert _dir_has_files("/nonexistent/path") is False

    def test_detect_cached_models_structure(self):
        result = _detect_cached_models()
        assert set(result.keys()) == {"autoencoder", "peptide_designer", "gtm", "dbaasp_data"}
        for v in result.values():
            assert isinstance(v, bool)


# ---------------------------------------------------------------------------
# _build_recommendations
# ---------------------------------------------------------------------------


class TestBuildRecommendations:
    def test_gpu_available_recommendation(self):
        profile = {
            "gpu": {"available": True, "device_name": "NVIDIA A100", "cuda_functional": True},
            "cpu": {"core_count": 8, "usable_workers": 8},
            "ram": {"total_gb": 64.0, "available_gb": 48.0},
            "chembl_backend": {"backend": "sqlite", "description": "Local SQLite ChEMBL database"},
            "storage_backend": {"backend": "local", "bucket": None},
            "cached_models": {
                "autoencoder": True,
                "peptide_designer": True,
                "gtm": True,
                "dbaasp_data": True,
            },
        }
        recs = _build_recommendations(profile)
        assert any("GPU detected" in r and "NVIDIA A100" in r for r in recs)
        assert any("8 CPU" in r for r in recs)

    def test_no_gpu_recommendation(self):
        profile = {
            "gpu": {"available": False, "device_name": None, "cuda_functional": False},
            "cpu": {"core_count": 4, "usable_workers": 4},
            "ram": {"total_gb": 16.0, "available_gb": 8.0},
            "chembl_backend": {"backend": "rest", "description": "ChEMBL REST API"},
            "storage_backend": {"backend": "local", "bucket": None},
            "cached_models": {
                "autoencoder": False,
                "peptide_designer": False,
                "gtm": False,
                "dbaasp_data": False,
            },
        }
        recs = _build_recommendations(profile)
        assert any("No GPU" in r for r in recs)
        assert any("REST API" in r for r in recs)
        assert any("not cached" in r.lower() for r in recs)

    def test_all_models_cached_no_download_warning(self):
        profile = {
            "gpu": {"available": False, "device_name": None, "cuda_functional": False},
            "cpu": {"core_count": 4, "usable_workers": 4},
            "ram": {"total_gb": None, "available_gb": None},
            "chembl_backend": {"backend": "sqlite", "description": "Local SQLite ChEMBL database"},
            "storage_backend": {"backend": "local", "bucket": None},
            "cached_models": {
                "autoencoder": True,
                "peptide_designer": True,
                "gtm": True,
                "dbaasp_data": True,
            },
        }
        recs = _build_recommendations(profile)
        assert not any("not cached" in r.lower() for r in recs)


# ---------------------------------------------------------------------------
# analyze_resources (integration)
# ---------------------------------------------------------------------------


class TestAnalyzeResources:
    def test_returns_complete_profile(self):
        result = analyze_resources()
        expected_keys = {
            "gpu",
            "cpu",
            "ram",
            "chembl_backend",
            "storage_backend",
            "cached_models",
            "recommendations",
        }
        assert set(result.keys()) == expected_keys

    def test_gpu_section_structure(self):
        result = analyze_resources()
        gpu = result["gpu"]
        assert "available" in gpu
        assert "device_name" in gpu
        assert "cuda_functional" in gpu
        assert isinstance(gpu["available"], bool)

    def test_cpu_section_structure(self):
        result = analyze_resources()
        cpu = result["cpu"]
        assert cpu["core_count"] >= 1
        assert cpu["usable_workers"] >= 1

    def test_recommendations_is_nonempty_list(self):
        result = analyze_resources()
        assert isinstance(result["recommendations"], list)
        assert len(result["recommendations"]) >= 2  # at least GPU + CPU lines
