#!/usr/bin/env python
# coding: utf-8
"""
Shared utilities for robustness testing infrastructure.

This module provides centralized implementations of common functionality
to eliminate code duplication across test files.
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Set

from cs_copilot.storage import S3
from cs_copilot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    """Configuration for LLM model."""

    provider: str
    model_id: str
    api_key_env: str
    api_key: Optional[str] = None

    def validate(self):
        """Validate model configuration."""
        valid_providers = ["deepseek", "openai", "anthropic", "ollama"]
        if self.provider not in valid_providers:
            raise ValueError(f"Invalid provider: {self.provider}. Must be one of {valid_providers}")

        # Ollama (local) does not require an API key
        if self.provider != "ollama" and not self.api_key:
            raise ValueError(f"API key not found in environment variable: {self.api_key_env}")


class ModelLoader:
    """Load and cache LLM models from configuration."""

    _cached_model: Optional[Any] = None
    _cached_config: Optional[ModelConfig] = None

    @classmethod
    def from_config(cls, config_path: Path) -> ModelLoader:
        """Create ModelLoader from robustness_config.yaml path."""
        instance = cls()
        instance._load_config(config_path)
        return instance

    def _load_config(self, config_path: Path):
        """Load configuration from YAML file."""
        try:
            import yaml
        except ImportError as e:
            raise ImportError("PyYAML is required to load robustness_config.yaml") from e

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        data = yaml.safe_load(config_path.read_text()) or {}
        model_config = data.get("model", {})

        provider = model_config.get("provider", "deepseek")
        model_id = model_config.get("model_id", "deepseek-chat")
        api_key_env = model_config.get("api_key_env", "DEEPSEEK_API_KEY")
        api_key = os.environ.get(api_key_env)

        self._cached_config = ModelConfig(
            provider=provider,
            model_id=model_id,
            api_key_env=api_key_env,
            api_key=api_key,
        )
        self._cached_config.validate()

    def load_model(self) -> Any:
        """Load model instance (cached)."""
        if self._cached_model is not None:
            return self._cached_model

        if self._cached_config is None:
            raise RuntimeError("Config not loaded. Call from_config() first.")

        config = self._cached_config

        if config.provider == "deepseek":
            from agno import DeepSeekChat

            self._cached_model = DeepSeekChat(id=config.model_id, api_key=config.api_key)
        elif config.provider == "openai":
            from agno import OpenAIChat

            self._cached_model = OpenAIChat(id=config.model_id, api_key=config.api_key)
        elif config.provider == "anthropic":
            from agno import Claude

            self._cached_model = Claude(id=config.model_id, api_key=config.api_key)
        elif config.provider == "ollama":
            from agno.models.ollama import Ollama

            host = os.environ.get("OLLAMA_HOST")
            self._cached_model = Ollama(id=config.model_id, host=host)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

        logger.info(f"Loaded model: {config.provider}/{config.model_id}")
        return self._cached_model

    @property
    def config(self) -> ModelConfig:
        """Get loaded configuration."""
        if self._cached_config is None:
            raise RuntimeError("Config not loaded. Call from_config() first.")
        return self._cached_config


class S3SessionManager:
    """Manage S3 session isolation with guaranteed cleanup."""

    def __init__(self):
        self._original_prefix: Optional[str] = None
        self._session_stack: List[str] = []

    def create_session_id(self, test_run_id: str, prompt_idx: int, variation_idx: int) -> str:
        """Generate unique session ID for isolated test run."""
        import uuid
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        session_id = (
            f"robustness_{test_run_id}_{timestamp}_p{prompt_idx}_v{variation_idx}_{unique_id}"
        )
        return session_id

    @contextmanager
    def create_isolated_session(self, test_run_id: str, prompt_idx: int, variation_idx: int):
        """
        Create isolated S3 session with automatic cleanup.

        Usage:
            manager = S3SessionManager()
            with manager.create_isolated_session("test", 0, 0) as session_id:
                # Run test with isolated S3 storage
                result = agent.run(prompt)
            # Automatic cleanup guaranteed
        """
        # Save original prefix on first call
        if self._original_prefix is None:
            self._original_prefix = S3.prefix

        session_id = self.create_session_id(test_run_id, prompt_idx, variation_idx)
        s3_prefix = f"sessions/{session_id}/"

        # Set new prefix
        S3.prefix = s3_prefix
        self._session_stack.append(s3_prefix)

        logger.debug(f"Created isolated S3 session: {session_id}")

        try:
            yield session_id
        finally:
            # Always restore, even on exception
            self.restore()

    def restore(self):
        """Restore S3 prefix to original value with verification."""
        if self._session_stack:
            self._session_stack.pop()

        if self._original_prefix is not None:
            S3.prefix = self._original_prefix
            logger.debug(f"Restored S3 prefix to: {self._original_prefix}")

            # Verify restoration
            if S3.prefix != self._original_prefix:
                raise RuntimeError(
                    f"S3 prefix restoration failed! "
                    f"Expected: {self._original_prefix}, Got: {S3.prefix}"
                )


class ResponseParser:
    """Centralized response parsing utilities."""

    # Regex patterns for file extraction
    FILE_PATTERNS = [
        r"Saved to S3:\s*[`'\"]([^`'\"]+)[`'\"]",  # Pattern 1: "Saved to S3: `filename`"
        r"saved (?:to|as|in)\s*[`'\"]([^`'\"]+\.csv)[`'\"]",  # Pattern 2: "saved to `file.csv`"
        r"[`'\"]?(chembl_[^`'\"<>\s]+\.csv)[`'\"]?",  # Pattern 3: chembl_*.csv
        r"`([^`]+\.csv)`",  # Pattern 4: Any .csv in backticks
    ]

    # Success action words
    SUCCESS_ACTIONS = [
        "downloaded",
        "retrieved",
        "fetched",
        "saved",
        "collected",
        "extracted",
        "obtained",
        "gathered",
        "found",
    ]

    # Data quantity indicators
    QUANTITY_PATTERNS = [
        "records",
        "compounds",
        "rows",
        "entries",
        "molecules",
        "activities",
        "bioactivities",
        "assays",
        "datapoints",
    ]

    # File/dataset references
    FILE_INDICATORS = [".csv", ".sdf", "dataset", "file", "s3://"]

    @classmethod
    def extract_files(cls, response_text: str) -> Set[str]:
        """
        Extract file paths from agent response text.

        Args:
            response_text: Agent response text to parse

        Returns:
            Set of extracted file paths
        """
        files = set()
        if not response_text:
            return files

        for pattern in cls.FILE_PATTERNS:
            for match in re.finditer(pattern, response_text, re.IGNORECASE):
                files.add(match.group(1))

        return files

    @classmethod
    def extract_smiles(cls, response_text: str) -> List[str]:
        """
        Extract SMILES strings from agent response.

        Args:
            response_text: Agent response text to parse

        Returns:
            List of extracted SMILES strings
        """
        smiles = []

        # Pattern 1: Backtick enclosed
        backtick_pattern = r"`([A-Za-z0-9@+\-\[\]\(\)=#$]+)`"
        smiles.extend(re.findall(backtick_pattern, response_text))

        # Pattern 2: Lines starting with SMILES-like strings
        for line in response_text.split("\n"):
            line = line.strip()
            if line and not line.startswith(("#", "-", "*", ">")):
                if re.match(
                    r"^[A-Za-z0-9@+\-\[\]\(\)=#$]+$",
                    line.split()[0] if line.split() else "",
                ):
                    smiles.append(line.split()[0])

        return smiles

    @classmethod
    def check_success(cls, response_text: str) -> bool:
        """
        Check if response indicates successful execution.

        Looks for indicators like:
        - Success action words (downloaded, retrieved, etc.)
        - Record/compound/row counts
        - Dataset/file references
        - Success confirmations

        Args:
            response_text: Agent response text to check

        Returns:
            True if response indicates success, False otherwise
        """
        if not response_text:
            return False

        lower = response_text.lower()

        # Check for success actions
        has_success_action = any(action in lower for action in cls.SUCCESS_ACTIONS)

        # Check for quantity indicators
        has_quantity = any(pattern in lower for pattern in cls.QUANTITY_PATTERNS)

        # Check for file indicators
        has_file = any(indicator in lower for indicator in cls.FILE_INDICATORS)

        # Consider successful if we have:
        # 1. Success action + quantity (e.g., "downloaded 1000 compounds")
        # 2. Success action + file (e.g., "saved to dataset.csv")
        # 3. All three indicators present
        if has_success_action and (has_quantity or has_file):
            return True

        # Also check for explicit success messages
        success_phrases = [
            "successfully",
            "completed",
            "finished",
            "done",
            "ready",
        ]
        if any(phrase in lower for phrase in success_phrases):
            return True

        return False

    @classmethod
    def extract_row_count(cls, response_text: str) -> Optional[int]:
        """
        Extract row/record count from response text.

        Args:
            response_text: Agent response text to parse

        Returns:
            Extracted count or None if not found
        """
        if not response_text:
            return None

        # Pattern: "N records/compounds/rows/etc."
        count_pattern = r"(\d+)\s*(?:records?|compounds?|rows?|entries|molecules?|datapoints?)"
        match = re.search(count_pattern, response_text, re.IGNORECASE)

        if match:
            return int(match.group(1))

        return None


def create_agent_team_factory(model: Any) -> Callable:
    """
    Create factory function for agent teams with memory disabled.

    Args:
        model: LLM model instance

    Returns:
        Factory function that creates agent teams
    """
    from cs_copilot.agents import get_cs_copilot_agent_team

    def _create_team(**overrides):
        """Create agent team with default parameters."""
        params = {"enable_memory": False, **overrides}
        return get_cs_copilot_agent_team(model=model, **params)

    return _create_team


class TestValidation:
    """Common validation utilities for robustness tests."""

    @staticmethod
    def validate_dataframe(df: Any) -> bool:
        """
        Validate that object is a valid non-empty DataFrame.

        Args:
            df: Object to validate

        Returns:
            True if valid DataFrame, False otherwise
        """
        try:
            import pandas as pd

            return isinstance(df, pd.DataFrame) and not df.empty
        except ImportError:
            return False

    @staticmethod
    def validate_file_exists(file_path: str) -> bool:
        """
        Check if file exists in S3 or local filesystem.

        Args:
            file_path: Path to check

        Returns:
            True if file exists, False otherwise
        """
        try:
            with S3.open(file_path, "r"):
                return True
        except Exception:
            return False

    @staticmethod
    def validate_smiles_list(smiles_list: List[str], min_count: int = 1) -> bool:
        """
        Validate that SMILES list is non-empty and contains valid SMILES.

        Args:
            smiles_list: List of SMILES strings to validate
            min_count: Minimum number of SMILES required

        Returns:
            True if valid, False otherwise
        """
        if not smiles_list or len(smiles_list) < min_count:
            return False

        # Basic validation: SMILES should contain valid characters
        valid_chars = set("CNOPSFIBrClcnops0123456789@+\\-=[]()#$%.")
        for smiles in smiles_list:
            if not smiles or not set(smiles).issubset(valid_chars):
                return False

        return True
