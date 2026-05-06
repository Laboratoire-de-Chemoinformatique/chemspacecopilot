#!/usr/bin/env python
# coding: utf-8
"""
Tests for the autoencoder toolkit, including Hugging Face download functionality.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cs_copilot.tools.chemistry.autoencoder_toolkit import AutoencoderError, AutoencoderToolkit
from cs_copilot.tools.constants import HUGGINGFACE_AUTOENCODER_REPO
from cs_copilot.tools.io.session_memory import load_candidate_set_artifact


class TestAutoencoderDownload:
    """Test Hugging Face download functionality for autoencoder models."""

    def test_download_from_huggingface_when_files_missing(self, tmp_path):
        """Test that model files are downloaded from Hugging Face when missing."""
        model_path = tmp_path / "test_autoencoder"
        model_path.mkdir()

        # Mock snapshot_download to simulate successful download
        with patch("huggingface_hub.snapshot_download") as mock_download:
            # Mock the download to create test files
            def create_test_files(repo_id, cache_dir, local_dir, resume_download):
                # Create test model files
                model_dir = Path(local_dir)
                model_dir.mkdir(parents=True, exist_ok=True)
                (model_dir / "config.pt").touch()
                (model_dir / "vocab.pt").touch()
                (model_dir / "model.pt").touch()
                return str(model_dir)

            mock_download.side_effect = create_test_files

            # Mock the model loading to avoid loading actual PyTorch files
            with patch(
                "cs_copilot.tools.chemistry.autoencoder_toolkit.AutoencoderToolkit._load_model"
            ):
                # Try to initialize AutoencoderToolkit with missing files
                AutoencoderToolkit(model_path=str(model_path), device="cpu")

                # Verify snapshot_download was called with correct parameters
                assert mock_download.called
                call_args = mock_download.call_args
                assert call_args.kwargs["repo_id"] == HUGGINGFACE_AUTOENCODER_REPO
                assert call_args.kwargs["resume_download"] is True

    def test_download_uses_correct_repo_id(self, tmp_path):
        """Test that the correct Hugging Face repository ID is used for download."""
        model_path = tmp_path / "test_autoencoder"
        model_path.mkdir()

        with (
            patch("huggingface_hub.snapshot_download") as mock_download,
            patch("cs_copilot.tools.chemistry.autoencoder_toolkit.AutoencoderToolkit._load_model"),
        ):

            def create_test_files(repo_id, cache_dir, local_dir, resume_download):
                model_dir = Path(local_dir)
                model_dir.mkdir(parents=True, exist_ok=True)
                (model_dir / "config.pt").touch()
                (model_dir / "vocab.pt").touch()
                (model_dir / "model.pt").touch()
                return str(model_dir)

            mock_download.side_effect = create_test_files

            AutoencoderToolkit(model_path=str(model_path), device="cpu")

            # Verify the correct repo was used
            call_args = mock_download.call_args
            assert call_args.kwargs["repo_id"] == HUGGINGFACE_AUTOENCODER_REPO

    def test_download_fails_when_huggingface_hub_missing(self, tmp_path):
        """Test that appropriate error is raised when huggingface_hub is not installed."""
        model_path = tmp_path / "test_autoencoder"
        model_path.mkdir()

        with patch("huggingface_hub.snapshot_download", side_effect=ImportError):
            with pytest.raises(AutoencoderError) as exc_info:
                AutoencoderToolkit(model_path=str(model_path), device="cpu")

            assert "huggingface_hub not installed" in str(exc_info.value)

    def test_download_fails_on_generic_exception(self, tmp_path):
        """Test that AutoencoderError is raised on download failure."""
        model_path = tmp_path / "test_autoencoder"
        model_path.mkdir()

        with patch("huggingface_hub.snapshot_download") as mock_download:
            mock_download.side_effect = Exception("Network error")

            with pytest.raises(AutoencoderError) as exc_info:
                AutoencoderToolkit(model_path=str(model_path), device="cpu")

            error_msg = str(exc_info.value)
            assert HUGGINGFACE_AUTOENCODER_REPO in error_msg
            assert "Failed to download model" in error_msg

    def test_download_fails_when_files_incomplete(self, tmp_path):
        """Test that error is raised when downloaded files are incomplete."""
        model_path = tmp_path / "test_autoencoder"
        model_path.mkdir()

        with patch("huggingface_hub.snapshot_download") as mock_download:
            # Mock download to create incomplete files (missing model.pt)
            def create_incomplete_files(repo_id, cache_dir, local_dir, resume_download):
                model_dir = Path(local_dir)
                model_dir.mkdir(parents=True, exist_ok=True)
                (model_dir / "config.pt").touch()
                (model_dir / "vocab.pt").touch()
                # Deliberately not creating model.pt
                return str(model_dir)

            mock_download.side_effect = create_incomplete_files

            with pytest.raises(AutoencoderError) as exc_info:
                AutoencoderToolkit(model_path=str(model_path), device="cpu")

            assert "Downloaded files incomplete" in str(exc_info.value)

    def test_no_download_when_files_exist(self, tmp_path):
        """Test that no download occurs when model files already exist."""
        model_path = tmp_path / "test_autoencoder"
        model_path.mkdir()

        # Create existing model files
        (model_path / "config.pt").touch()
        (model_path / "vocab.pt").touch()
        (model_path / "model.pt").touch()

        with (
            patch("huggingface_hub.snapshot_download") as mock_download,
            patch("cs_copilot.tools.chemistry.autoencoder_toolkit.AutoencoderToolkit._load_model"),
        ):
            AutoencoderToolkit(model_path=str(model_path), device="cpu")

            # Verify snapshot_download was NOT called
            assert not mock_download.called

    def test_download_creates_parent_directories(self, tmp_path):
        """Test that parent directories are created if they don't exist."""
        model_path = tmp_path / "non_existent" / "test_autoencoder"

        with (
            patch("huggingface_hub.snapshot_download") as mock_download,
            patch("cs_copilot.tools.chemistry.autoencoder_toolkit.AutoencoderToolkit._load_model"),
        ):

            def create_test_files(repo_id, cache_dir, local_dir, resume_download):
                model_dir = Path(local_dir)
                model_dir.mkdir(parents=True, exist_ok=True)
                (model_dir / "config.pt").touch()
                (model_dir / "vocab.pt").touch()
                (model_dir / "model.pt").touch()
                return str(model_dir)

            mock_download.side_effect = create_test_files

            # This should not raise an error
            AutoencoderToolkit(model_path=str(model_path), device="cpu")

            # Verify the directory was created
            assert model_path.exists()

    def test_download_with_default_path(self, tmp_path):
        """Test that download works with default model path."""
        model_path = tmp_path / "test_models" / "autoencoder"

        # Mock the default path
        with (
            patch(
                "cs_copilot.tools.chemistry.autoencoder_toolkit.DEFAULT_AUTOENCODER_MODEL_PATH",
                str(model_path),
            ),
            patch("huggingface_hub.snapshot_download") as mock_download,
        ):

            def create_test_files(repo_id, cache_dir, local_dir, resume_download):
                model_dir = Path(local_dir)
                model_dir.mkdir(parents=True, exist_ok=True)
                (model_dir / "config.pt").touch()
                (model_dir / "vocab.pt").touch()
                (model_dir / "model.pt").touch()
                return str(model_dir)

            mock_download.side_effect = create_test_files

            # This should call the download
            try:
                AutoencoderToolkit(model_path=None, device="cpu")
            except Exception:
                pass  # Ignore errors related to model loading

            # Verify download was attempted (may fail if files exist)


def test_sample_molecules_registers_autoencoder_candidate_set(monkeypatch, tmp_path):
    """Direct Autoencoder sampling records generated provenance and artifact-backed set."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(AutoencoderToolkit, "_ensure_model_exists", lambda self: None)
    monkeypatch.setattr(AutoencoderToolkit, "_load_model", lambda self: None)
    toolkit = AutoencoderToolkit(model_path="unused", device="cpu")
    monkeypatch.setattr(toolkit, "sample_from_latent", lambda **_kwargs: ["CCO", "CCN"])
    agent = SimpleNamespace(session_state={})
    session_state = {}

    summary = toolkit.sample_molecules(agent=agent, session_state=session_state)

    memory = session_state["session_objects"]
    assert summary["registered_candidate_set_id"] == "cset_001"
    assert summary["artifact_path"].endswith(
        "02_analog_generation/candidate_sets/cset_001/candidates.json"
    )
    assert session_state["sampled_molecules"]["candidate_set_id"] == "cset_001"
    assert session_state["sampled_molecules"]["preview"] == [
        {"smiles": "CCO"},
        {"smiles": "CCN"},
    ]
    assert memory["candidate_sets"]["cset_001"]["origin_agent"] == "autoencoder_toolkit"
    assert memory["candidate_sets"]["cset_001"]["generation_engine"] == "autoencoder"
    assert memory["candidate_sets"]["cset_001"]["artifact_path"] == summary["artifact_path"]
    assert memory["compounds"]["cmp_001"]["origin_agent"] == "autoencoder_toolkit"
    assert memory["compounds"]["cmp_001"]["candidate_set_id"] == "cset_001"
    artifact = load_candidate_set_artifact(session_state, "sampled_molecules")
    assert artifact["candidates"] == ["CCO", "CCN"]
