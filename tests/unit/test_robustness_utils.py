#!/usr/bin/env python
# coding: utf-8
"""
Unit tests for robustness testing utilities.

Tests the shared infrastructure introduced in Phase 1-4 refactoring:
- test_utils.py (ModelLoader, S3SessionManager, ResponseParser)
- tool_tracker.py (ToolSequenceComparator)
- config_schema.py (ConfigValidator)
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add robustness directory to path
robustness_dir = Path(__file__).parent.parent / "robustness"
sys.path.insert(0, str(robustness_dir))

from test_utils import ModelConfig, ResponseParser, S3SessionManager, TestValidation  # noqa: E402


class TestResponseParser:
    """Test ResponseParser utility class."""

    def test_extract_files_pattern1(self):
        """Test file extraction with 'Saved to S3:' pattern."""
        text = "Saved to S3: `chembl_12345.csv`"
        files = ResponseParser.extract_files(text)
        assert "chembl_12345.csv" in files

    def test_extract_files_pattern2(self):
        """Test file extraction with 'saved to' pattern."""
        text = "Data saved to `dataset.csv` successfully"
        files = ResponseParser.extract_files(text)
        assert "dataset.csv" in files

    def test_extract_files_pattern3(self):
        """Test chembl file pattern extraction."""
        text = "Created chembl_target_123_data.csv with results"
        files = ResponseParser.extract_files(text)
        assert "chembl_target_123_data.csv" in files

    def test_extract_files_empty(self):
        """Test file extraction from empty text."""
        files = ResponseParser.extract_files("")
        assert len(files) == 0

    def test_check_success_positive(self):
        """Test success detection with positive indicators."""
        text = "Successfully downloaded 1000 compounds to dataset.csv"
        assert ResponseParser.check_success(text) is True

    def test_check_success_negative(self):
        """Test success detection with no indicators."""
        text = "Hello, how can I help you?"
        assert ResponseParser.check_success(text) is False

    def test_check_success_partial(self):
        """Test success detection with partial indicators."""
        # Has action word but no file/quantity
        text = "I will download the data"
        assert ResponseParser.check_success(text) is False

    def test_extract_smiles_backticks(self):
        """Test SMILES extraction from backticks."""
        text = "Generated molecules: `CCO`, `CCCCCC`, `CC(=O)O`"
        smiles = ResponseParser.extract_smiles(text)
        assert "CCO" in smiles
        assert "CCCCCC" in smiles
        assert "CC(=O)O" in smiles

    def test_extract_row_count(self):
        """Test row count extraction from response."""
        text = "Dataset contains 1523 records"
        count = ResponseParser.extract_row_count(text)
        assert count == 1523

    def test_extract_row_count_none(self):
        """Test row count extraction when not present."""
        text = "No data available"
        count = ResponseParser.extract_row_count(text)
        assert count is None


class TestS3SessionManager:
    """Test S3SessionManager utility class."""

    def test_create_session_id_format(self):
        """Test session ID format."""
        manager = S3SessionManager()
        session_id = manager.create_session_id("test_run", 0, 0)

        # Should contain test_run_id
        assert "test_run" in session_id
        # Should contain robustness prefix
        assert session_id.startswith("robustness_")
        # Should contain prompt/variation indices
        assert "_p0_v0_" in session_id

    @patch("test_utils.S3")
    def test_context_manager_cleanup(self, mock_s3):
        """Test context manager ensures cleanup even on exception."""
        manager = S3SessionManager()
        mock_s3.prefix = "original_prefix"

        try:
            with manager.create_isolated_session("test", 0, 0) as session_id:
                assert session_id is not None
                # Simulate exception
                raise ValueError("Test error")
        except ValueError:
            pass

        # Verify restoration happened despite exception
        assert mock_s3.prefix == "original_prefix"

    @patch("test_utils.S3")
    def test_context_manager_sets_prefix(self, mock_s3):
        """Test context manager sets isolated S3 prefix."""
        manager = S3SessionManager()
        mock_s3.prefix = "original_prefix"

        with manager.create_isolated_session("test", 0, 0) as _session_id:
            # Prefix should be changed during context
            assert mock_s3.prefix.startswith("sessions/robustness_")

        # Prefix should be restored after context
        assert mock_s3.prefix == "original_prefix"


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_validate_valid_provider(self):
        """Test validation with valid provider."""
        with patch.dict(os.environ, {"TEST_API_KEY": "test_key"}):
            config = ModelConfig(
                provider="deepseek", model_id="deepseek-chat", api_key_env="TEST_API_KEY"
            )
            config.api_key = "test_key"
            # Should not raise
            config.validate()

    def test_validate_invalid_provider(self):
        """Test validation with invalid provider."""
        with patch.dict(os.environ, {"TEST_API_KEY": "test_key"}):
            config = ModelConfig(
                provider="invalid_provider",
                model_id="model",
                api_key_env="TEST_API_KEY",
                api_key="test_key",
            )
            with pytest.raises(ValueError, match="Invalid provider"):
                config.validate()

    def test_validate_missing_api_key(self):
        """Test validation with missing API key."""
        config = ModelConfig(
            provider="deepseek", model_id="deepseek-chat", api_key_env="NONEXISTENT_KEY"
        )
        with pytest.raises(ValueError, match="API key not found"):
            config.validate()


class TestTestValidation:
    """Test TestValidation utility class."""

    def test_validate_smiles_list_valid(self):
        """Test SMILES list validation with valid SMILES."""
        smiles_list = ["CCO", "CCCCCC", "CC(=O)O"]
        assert TestValidation.validate_smiles_list(smiles_list) is True

    def test_validate_smiles_list_empty(self):
        """Test SMILES list validation with empty list."""
        assert TestValidation.validate_smiles_list([]) is False

    def test_validate_smiles_list_min_count(self):
        """Test SMILES list validation with minimum count."""
        assert TestValidation.validate_smiles_list(["CCO"], min_count=2) is False
        assert TestValidation.validate_smiles_list(["CCO", "CCC"], min_count=2) is True


class TestToolSequenceComparator:
    """Test ToolSequenceComparator from tool_tracker.py."""

    def test_compare_sequences_identical(self):
        """Test sequence comparison with identical sequences."""
        from tool_tracker import ToolSequenceComparator

        sequences = [["tool1", "tool2", "tool3"], ["tool1", "tool2", "tool3"]]
        similarity = ToolSequenceComparator.compare_sequences(sequences)
        assert similarity == 1.0

    def test_compare_sequences_different(self):
        """Test sequence comparison with completely different sequences."""
        from tool_tracker import ToolSequenceComparator

        sequences = [["tool1", "tool2"], ["tool3", "tool4"]]
        similarity = ToolSequenceComparator.compare_sequences(sequences)
        assert similarity < 0.5

    def test_compare_sequences_partial_match(self):
        """Test sequence comparison with partial overlap."""
        from tool_tracker import ToolSequenceComparator

        sequences = [["tool1", "tool2", "tool3"], ["tool1", "tool2", "tool4"]]
        similarity = ToolSequenceComparator.compare_sequences(sequences)
        # Should be > 0 but < 1 (partial match)
        assert 0.5 < similarity < 1.0

    def test_compare_sequences_single(self):
        """Test sequence comparison with single sequence."""
        from tool_tracker import ToolSequenceComparator

        sequences = [["tool1", "tool2"]]
        similarity = ToolSequenceComparator.compare_sequences(sequences)
        assert similarity == 1.0  # Single sequence is perfectly similar to itself

    def test_analyze_sequence_patterns(self):
        """Test sequence pattern analysis."""
        from tool_tracker import ToolSequenceComparator

        sequences = [
            ["chembl_search", "download_data", "save_csv"],
            ["chembl_search", "download_data", "save_csv"],
            ["chembl_search", "filter_data", "save_csv"],
        ]
        patterns = ToolSequenceComparator.analyze_sequence_patterns(sequences)

        assert patterns["total_sequences"] == 3
        assert patterns["non_empty_sequences"] == 3
        assert "chembl_search" in patterns["tool_usage"]
        assert patterns["tool_usage"]["chembl_search"] == 3  # Used in all sequences

    def test_find_common_subsequence(self):
        """Test finding common subsequence."""
        from tool_tracker import ToolSequenceComparator

        sequences = [
            ["tool1", "tool2", "tool3", "tool4"],
            ["tool1", "tool2", "tool3", "tool5"],
            ["tool1", "tool2", "tool3", "tool6"],
        ]
        common = ToolSequenceComparator.find_common_subsequence(sequences)

        # Should find ["tool1", "tool2", "tool3"] as common
        assert len(common) >= 3
        assert common[:3] == ["tool1", "tool2", "tool3"]


class TestConfigValidator:
    """Test ConfigValidator from config_schema.py."""

    @pytest.fixture
    def valid_config_dict(self):
        """Provide a valid configuration dictionary."""
        return {
            "general": {
                "n_variations": 5,
                "debug_mode": False,
                "output_dir": "reports",
                "save_artifacts": True,
                "s3_session_isolation": True,
            },
            "model": {
                "provider": "deepseek",
                "model_id": "deepseek-chat",
                "api_key_env": "TEST_API_KEY",
            },
            "metrics": {
                "weights": {
                    "data_similarity": 0.4,
                    "semantic_similarity": 0.3,
                    "process_consistency": 0.2,
                    "visual_similarity": 0.1,
                },
                "thresholds": {"excellent": 0.90, "good": 0.80, "acceptable": 0.70},
                "pass_threshold": 0.75,
            },
            "reporting": {
                "generate_markdown": True,
                "generate_json": True,
                "include_run_details": True,
                "include_recommendations": True,
            },
            "tests": {
                "test1": {"enabled": True, "prompt_key": "chembl_download", "description": "Test 1"}
            },
        }

    def test_validate_valid_config(self, valid_config_dict, tmp_path):
        """Test validation with valid configuration."""
        # Write to temporary file
        import yaml
        from config_schema import ConfigValidator

        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml.dump(valid_config_dict))

        # Mock environment variable
        with patch.dict(os.environ, {"TEST_API_KEY": "test_key"}):
            # Should not raise
            validated = ConfigValidator.load_and_validate(config_path)
            assert validated is not None

    def test_validate_missing_weight(self, valid_config_dict, tmp_path):
        """Test validation with missing required weight."""
        from config_schema import ConfigValidator

        # Remove required weight
        del valid_config_dict["metrics"]["weights"]["data_similarity"]

        import yaml

        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml.dump(valid_config_dict))

        with patch.dict(os.environ, {"TEST_API_KEY": "test_key"}):
            with pytest.raises(ValueError, match="Missing required weight"):
                ConfigValidator.load_and_validate(config_path)

    def test_validate_invalid_threshold_order(self, valid_config_dict, tmp_path):
        """Test validation with incorrect threshold ordering."""
        from config_schema import ConfigValidator

        # Make thresholds out of order
        valid_config_dict["metrics"]["thresholds"] = {
            "excellent": 0.70,
            "good": 0.80,
            "acceptable": 0.90,
        }

        import yaml

        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml.dump(valid_config_dict))

        with patch.dict(os.environ, {"TEST_API_KEY": "test_key"}):
            with pytest.raises(ValueError, match="Thresholds must be ordered"):
                ConfigValidator.load_and_validate(config_path)
