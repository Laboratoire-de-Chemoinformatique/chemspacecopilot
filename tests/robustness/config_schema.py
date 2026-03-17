#!/usr/bin/env python
# coding: utf-8
"""
Configuration schema validation for robustness testing.

This module provides dataclasses and validators for robustness_config.yaml
to ensure configuration is valid before running tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelConfigSchema:
    """Schema for model configuration."""

    provider: str
    model_id: str
    api_key_env: str

    def validate(self):
        """Validate model configuration."""
        valid_providers = ["deepseek", "openai", "anthropic", "ollama"]
        if self.provider not in valid_providers:
            raise ValueError(f"Invalid provider: {self.provider}. Must be one of {valid_providers}")

        # Ollama (local) does not require an API key
        if self.provider != "ollama":
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise ValueError(
                    f"API key not found in environment variable: {self.api_key_env}. "
                    f"Please set {self.api_key_env} before running tests."
                )

        # Validate model_id format (basic check)
        if not self.model_id or len(self.model_id) < 3:
            raise ValueError(f"Invalid model_id: {self.model_id}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelConfigSchema:
        """Create from dictionary."""
        return cls(
            provider=data.get("provider", "deepseek"),
            model_id=data.get("model_id", "deepseek-chat"),
            api_key_env=data.get("api_key_env", "DEEPSEEK_API_KEY"),
        )


@dataclass
class MetricsConfigSchema:
    """Schema for metrics configuration."""

    weights: Dict[str, float] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)
    pass_threshold: float = 0.75

    def validate(self):
        """Validate metrics configuration."""
        # Required weight categories
        required_weights = [
            "data_similarity",
            "semantic_similarity",
            "process_consistency",
            "visual_similarity",
        ]

        for weight in required_weights:
            if weight not in self.weights:
                raise ValueError(f"Missing required weight: {weight}")

            # Validate weight is a number between 0 and 1
            value = self.weights[weight]
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                raise ValueError(f"Weight {weight} must be between 0 and 1, got {value}")

        # Required thresholds
        required_thresholds = ["excellent", "good", "acceptable"]
        for threshold in required_thresholds:
            if threshold not in self.thresholds:
                raise ValueError(f"Missing required threshold: {threshold}")

            value = self.thresholds[threshold]
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                raise ValueError(f"Threshold {threshold} must be between 0 and 1, got {value}")

        # Validate threshold ordering (excellent > good > acceptable)
        if not (
            self.thresholds["excellent"] > self.thresholds["good"] > self.thresholds["acceptable"]
        ):
            raise ValueError(
                "Thresholds must be ordered: excellent > good > acceptable. "
                f"Got: excellent={self.thresholds['excellent']}, "
                f"good={self.thresholds['good']}, "
                f"acceptable={self.thresholds['acceptable']}"
            )

        # Validate pass_threshold
        if (
            not isinstance(self.pass_threshold, (int, float))
            or self.pass_threshold < 0
            or self.pass_threshold > 1
        ):
            raise ValueError(f"pass_threshold must be between 0 and 1, got {self.pass_threshold}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MetricsConfigSchema:
        """Create from dictionary."""
        return cls(
            weights=data.get("weights", {}),
            thresholds=data.get("thresholds", {}),
            pass_threshold=data.get("pass_threshold", 0.75),
        )


@dataclass
class TestConfigSchema:
    """Schema for individual test configuration."""

    name: str
    enabled: bool
    prompt_key: str
    description: str = ""
    depends_on: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)

    def validate(self, available_prompt_keys: Optional[List[str]] = None):
        """Validate test configuration."""
        # Validate name (no spaces, alphanumeric + underscore)
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(
                f"Invalid test name: {self.name}. " "Must be alphanumeric with underscores only."
            )

        # Validate prompt_key exists if list provided
        if available_prompt_keys and self.prompt_key not in available_prompt_keys:
            raise ValueError(
                f"Test '{self.name}' references undefined prompt_key: {self.prompt_key}. "
                f"Available keys: {', '.join(available_prompt_keys)}"
            )

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> TestConfigSchema:
        """Create from dictionary."""
        return cls(
            name=name,
            enabled=data.get("enabled", False),
            prompt_key=data.get("prompt_key", ""),
            description=data.get("description", ""),
            depends_on=data.get("depends_on", []),
            params=data.get("params", {}),
        )


@dataclass
class GeneralConfigSchema:
    """Schema for general settings."""

    n_variations: int = 5
    debug_mode: bool = False
    output_dir: str = "reports"
    save_artifacts: bool = True
    s3_session_isolation: bool = True

    def validate(self):
        """Validate general configuration."""
        if self.n_variations < 1 or self.n_variations > 100:
            raise ValueError(f"n_variations must be between 1 and 100, got {self.n_variations}")

        if not self.output_dir:
            raise ValueError("output_dir cannot be empty")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GeneralConfigSchema:
        """Create from dictionary."""
        return cls(
            n_variations=data.get("n_variations", 5),
            debug_mode=data.get("debug_mode", False),
            output_dir=data.get("output_dir", "reports"),
            save_artifacts=data.get("save_artifacts", True),
            s3_session_isolation=data.get("s3_session_isolation", True),
        )


@dataclass
class ReportingConfigSchema:
    """Schema for reporting settings."""

    generate_markdown: bool = True
    generate_json: bool = True
    include_run_details: bool = True
    include_recommendations: bool = True

    def validate(self):
        """Validate reporting configuration."""
        # All fields are booleans, no additional validation needed
        pass

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReportingConfigSchema:
        """Create from dictionary."""
        return cls(
            generate_markdown=data.get("generate_markdown", True),
            generate_json=data.get("generate_json", True),
            include_run_details=data.get("include_run_details", True),
            include_recommendations=data.get("include_recommendations", True),
        )


class ConfigValidator:
    """Validate robustness_config.yaml structure and content."""

    @staticmethod
    def load_and_validate(config_path: Path) -> Dict[str, Any]:
        """
        Load and validate robustness configuration file.

        Args:
            config_path: Path to robustness_config.yaml

        Returns:
            Validated configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
            yaml.YAMLError: If YAML is malformed
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load YAML
        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}") from e

        if not isinstance(data, dict):
            raise ValueError("Configuration file must contain a YAML dictionary")

        # Validate each section
        errors = []

        # 1. General settings
        try:
            general = GeneralConfigSchema.from_dict(data.get("general", {}))
            general.validate()
        except Exception as e:
            errors.append(f"[general] {e}")

        # 2. Model configuration
        try:
            model = ModelConfigSchema.from_dict(data.get("model", {}))
            model.validate()
        except Exception as e:
            errors.append(f"[model] {e}")

        # 3. Metrics configuration
        try:
            metrics = MetricsConfigSchema.from_dict(data.get("metrics", {}))
            metrics.validate()
        except Exception as e:
            errors.append(f"[metrics] {e}")

        # 4. Reporting configuration
        try:
            reporting = ReportingConfigSchema.from_dict(data.get("reporting", {}))
            reporting.validate()
        except Exception as e:
            errors.append(f"[reporting] {e}")

        # 5. Test configurations
        tests_data = data.get("tests", {})
        if not isinstance(tests_data, dict):
            errors.append("[tests] Must be a dictionary")
        else:
            # Get available prompt keys from fixtures (if fixtures file exists)
            available_prompt_keys = ConfigValidator._get_available_prompt_keys(config_path.parent)

            for test_name, test_data in tests_data.items():
                try:
                    test_config = TestConfigSchema.from_dict(test_name, test_data)
                    test_config.validate(available_prompt_keys)
                except Exception as e:
                    errors.append(f"[tests.{test_name}] {e}")

        # If there are errors, raise with all error messages
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {err}" for err in errors
            )
            raise ValueError(error_msg)

        return data

    @staticmethod
    def _get_available_prompt_keys(config_dir: Path) -> Optional[List[str]]:
        """Get available prompt keys from prompt_templates.yaml if it exists."""
        fixtures_path = config_dir / "fixtures" / "prompt_templates.yaml"
        if not fixtures_path.exists():
            return None

        try:
            with open(fixtures_path, "r") as f:
                templates = yaml.safe_load(f)
            if isinstance(templates, dict):
                return list(templates.keys())
        except Exception:
            pass

        return None

    @staticmethod
    def validate_dependencies(config_data: Dict[str, Any]) -> List[str]:
        """
        Validate test dependencies form valid DAG (no cycles).

        Args:
            config_data: Configuration dictionary

        Returns:
            List of test names in dependency order

        Raises:
            ValueError: If circular dependencies detected
        """
        tests_data = config_data.get("tests", {})

        # Build dependency graph
        dependencies = {}
        for test_name, test_data in tests_data.items():
            if test_data.get("enabled", False):
                dependencies[test_name] = test_data.get("depends_on", [])

        # Check for undefined dependencies
        all_tests = set(dependencies.keys())
        for test_name, deps in dependencies.items():
            for dep in deps:
                if dep not in all_tests:
                    raise ValueError(f"Test '{test_name}' depends on undefined test '{dep}'")

        # Topological sort to detect cycles and get execution order
        execution_order = []
        visited = set()
        visiting = set()

        def visit(test_name):
            if test_name in visited:
                return
            if test_name in visiting:
                raise ValueError(f"Circular dependency detected involving test '{test_name}'")

            visiting.add(test_name)
            for dep in dependencies.get(test_name, []):
                visit(dep)
            visiting.remove(test_name)
            visited.add(test_name)
            execution_order.append(test_name)

        for test_name in dependencies:
            visit(test_name)

        return execution_order
