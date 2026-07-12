#!/usr/bin/env python
# coding: utf-8
"""
Shared pytest fixtures for robustness testing.

This module provides centralized fixtures to eliminate code duplication
across robustness test files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .test_utils import (
    ModelLoader,
    ResponseParser,
    S3SessionManager,
    TestValidation,
    create_agent_team_factory,
    create_single_agent_factory,
)

# Skip all robustness tests if pytest is not available
# (should never happen since we're in conftest.py, but keep for consistency)
pytest_available = True

# Configuration path
ROBUSTNESS_CONFIG_PATH = Path(__file__).parent / "robustness_config.yaml"


@pytest.fixture(scope="session")
def model_loader():
    """
    Shared model loader from config.

    Loads model configuration once per test session and caches the model instance.
    Falls back to centralized .modelconf / env var configuration when
    robustness_config.yaml validation fails (e.g. missing API key for Ollama).
    """
    try:
        loader = ModelLoader.from_config(ROBUSTNESS_CONFIG_PATH)
        return loader
    except FileNotFoundError:
        pytest.skip("robustness_config.yaml not found")
    except ImportError as e:
        pytest.skip(f"Required dependency not available: {e}")
    except ValueError:
        # Validation failed (e.g. no API key when using Ollama) --
        # fall through to centralized model_config.
        return None


@pytest.fixture(scope="session")
def model(model_loader):
    """
    Load LLM model instance (session-scoped, cached).

    Uses ModelLoader when available, otherwise falls back to the
    centralized .modelconf configuration.
    """
    if model_loader is not None:
        return model_loader.load_model()

    # Fallback: use centralized model config (.modelconf / env vars)
    from cs_copilot.model_config import load_model_from_config

    return load_model_from_config()


@pytest.fixture
def agent_team_factory(model):
    """
    Create agent teams with memory disabled.

    Returns a factory function that creates fresh agent teams for each test.
    Teams have memory disabled by default to ensure test isolation.

    Usage:
        def test_something(agent_team_factory):
            team = agent_team_factory()
            result = team.run("test prompt")
    """
    return create_agent_team_factory(model)


@pytest.fixture
def single_agent_factory(model):
    """
    Create the single-agent baseline (multi-agent-vs-single-agent ablation).

    Returns a factory that builds a fresh flat agent (all toolkits, no team) on
    the same session-scoped model as ``agent_team_factory``, exposing the same
    ``.run`` / ``.get_session_state`` interface.

    Usage:
        def test_something(agent_team_factory, single_agent_factory):
            team = agent_team_factory()
            single = single_agent_factory()
    """
    return create_single_agent_factory(model)


@pytest.fixture
def s3_session_manager():
    """
    Manage S3 session isolation with automatic cleanup.

    Creates a session manager that ensures S3 prefix is restored
    after each test, even if the test fails.

    Usage:
        def test_something(s3_session_manager):
            with s3_session_manager.create_isolated_session("test", 0, 0) as session_id:
                # Test code here
                pass
            # Automatic cleanup guaranteed
    """
    manager = S3SessionManager()
    yield manager
    # Ensure cleanup even if test failed
    manager.restore()


@pytest.fixture
def prompt_generator():
    """
    Create prompt variation generator.

    Returns a PromptVariationGenerator instance for creating
    semantically similar prompt variations.
    """
    from .prompt_variations import PromptVariationGenerator

    return PromptVariationGenerator()


@pytest.fixture
def comparator():
    """
    Create output comparator.

    Returns an OutputComparator instance for comparing outputs
    across prompt variations.
    """
    from .comparators import OutputComparator

    return OutputComparator()


@pytest.fixture
def metrics_calculator():
    """
    Create metrics calculator.

    Returns a RobustnessMetrics instance for calculating
    robustness scores.
    """
    from .metrics import RobustnessMetrics

    return RobustnessMetrics()


@pytest.fixture
def response_parser():
    """
    Response parsing utilities.

    Returns the ResponseParser class for extracting information
    from agent responses.
    """
    return ResponseParser


@pytest.fixture
def test_validator():
    """
    Test validation utilities.

    Returns the TestValidation class for common validation checks.
    """
    return TestValidation


# Session-scoped fixture to track test execution
@pytest.fixture(scope="session")
def robustness_test_session():
    """
    Track robustness test session metadata.

    Provides session-level information like test run ID, start time, etc.
    """
    import uuid
    from datetime import datetime

    session_data = {
        "session_id": str(uuid.uuid4())[:8],
        "start_time": datetime.now(),
        "tests_run": 0,
        "tests_passed": 0,
        "tests_failed": 0,
    }

    yield session_data

    # Log session summary
    session_data["end_time"] = datetime.now()
    duration = session_data["end_time"] - session_data["start_time"]

    print("\n" + "=" * 60)
    print("Robustness Test Session Summary")
    print("=" * 60)
    print(f"Session ID: {session_data['session_id']}")
    print(f"Duration: {duration}")
    print(f"Tests Run: {session_data['tests_run']}")
    print(f"Tests Passed: {session_data['tests_passed']}")
    print(f"Tests Failed: {session_data['tests_failed']}")
    print("=" * 60)


# Auto-use fixture to update session statistics
@pytest.fixture(autouse=True)
def track_test_result(request, robustness_test_session):
    """Automatically track test results in session."""
    robustness_test_session["tests_run"] += 1

    yield

    if hasattr(request.node, "rep_call"):
        if request.node.rep_call.passed:
            robustness_test_session["tests_passed"] += 1
        elif request.node.rep_call.failed:
            robustness_test_session["tests_failed"] += 1


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to track test outcomes."""
    outcome = yield
    rep = outcome.get_result()

    # Store test result on the item for tracking
    setattr(item, f"rep_{rep.when}", rep)
