#!/usr/bin/env python
"""
Robustness Testing Runner for Cs_copilot Agentic Operations.

This script provides a configurable robustness testing framework that:
- Runs selected tests based on a YAML configuration file
- Tests robustness of agent operations to prompt variations
- Generates detailed comparison metrics and reports
- Supports S3 session isolation for reproducible testing

Usage:
    uv run python tests/robustness/robustness_minimal_example.py
    uv run python tests/robustness/robustness_minimal_example.py --config custom_config.yaml
    uv run python tests/robustness/robustness_minimal_example.py --test chembl_download --n-variations 3
"""

import argparse
import copy
import hashlib
import json
import os
import re
import signal
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv  # noqa: E402

from cs_copilot.utils.logging import get_logger  # noqa: E402

# Import shared test utilities
sys.path.insert(0, str(Path(__file__).parent))
from config_schema import ConfigValidator  # noqa: E402
from reliability import (  # noqa: E402
    ReliabilityRunRecord,
    build_environment_manifest,
    evaluate_run,
    normalize_agno_output,
    save_reliability_bundle,
    save_system_comparison,
)
from test_utils import ResponseParser, S3SessionManager  # noqa: E402
from tool_tracker import ToolSequenceComparator  # noqa: E402

logger = get_logger(__name__)
load_dotenv()


class ReliabilityTimeoutError(TimeoutError):
    """Raised when one benchmark execution exceeds its configured wall time."""


class FixtureLoadError(RuntimeError):
    """Raised when a required frozen benchmark fixture cannot be loaded."""


class PrerequisiteError(RuntimeError):
    """Raised when a live benchmark input is unavailable."""


@contextmanager
def run_timeout(seconds: int):
    """Interrupt a run after ``seconds`` on POSIX main-thread executions."""
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise_timeout(signum, frame):  # noqa: ARG001
        raise ReliabilityTimeoutError(f"Run exceeded the {seconds}-second timeout")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


@dataclass
class TestConfig:
    """Configuration for a single test."""

    name: str
    enabled: bool
    prompt_key: str
    description: str = ""
    depends_on: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    custom_prompt: Optional[str] = None
    prompt_variants: List[str] = field(default_factory=list)
    validator: str = "execution_only"
    tier: str = "both"
    fixture: Dict[str, Any] = field(default_factory=dict)
    required_files: List[Dict[str, Any]] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RobustnessConfig:
    """Full robustness testing configuration."""

    n_variations: int = 5
    debug_mode: bool = False
    output_dir: str = "reports"
    save_artifacts: bool = True
    s3_session_isolation: bool = True
    repetitions: int = 1
    reliability_enabled: bool = False
    tier: str = "both"
    timeout_seconds: int = 0
    reliability_min_success_rate: float = 0.8
    pricing: Dict[str, float] = field(default_factory=dict)
    inference_settings: Dict[str, Any] = field(default_factory=dict)
    config_path: Optional[Path] = None

    # System under test: "team" (multi-agent) or "single_agent" (flat baseline).
    # Driven by the --system CLI flag; both arms use the same model/tasks/metrics.
    system: str = "team"

    # Model settings
    model_provider: str = "deepseek"
    model_id: str = "deepseek-chat"
    api_key_env: str = "DEEPSEEK_API_KEY"

    # Metrics settings
    weights: Dict[str, float] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)
    pass_threshold: float = 0.75

    # Tests to run
    tests: Dict[str, TestConfig] = field(default_factory=dict)

    # Reporting
    generate_markdown: bool = True
    generate_json: bool = True
    include_run_details: bool = True
    include_recommendations: bool = True


def load_config(config_path: Path) -> RobustnessConfig:
    """
    Load and validate configuration from YAML file.

    Performs comprehensive validation including:
    - Schema validation (all required fields present)
    - Type validation (correct data types)
    - Range validation (values within acceptable ranges)
    - Dependency validation (no circular dependencies)

    Args:
        config_path: Path to robustness_config.yaml

    Returns:
        Validated RobustnessConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    # Validate configuration before loading
    try:
        data = ConfigValidator.load_and_validate(config_path)
        logger.info("Configuration validation passed ✓")
    except ValueError as e:
        logger.error(f"Configuration validation failed:\n{e}")
        raise

    general = data.get("general", {})
    model = data.get("model", {})
    metrics = data.get("metrics", {})
    reporting = data.get("reporting", {})

    # Parse tests
    tests = {}
    for test_name, test_data in data.get("tests", {}).items():
        if test_data:
            tests[test_name] = TestConfig(
                name=test_name,
                enabled=test_data.get("enabled", False),
                prompt_key=test_data.get("prompt_key", test_name),
                description=test_data.get("description", ""),
                depends_on=test_data.get("depends_on", []),
                params=test_data.get("params", {}),
                prompt_variants=test_data.get("prompt_variants", []),
                validator=test_data.get("validator", "execution_only"),
                tier=test_data.get("tier", "both"),
                fixture=test_data.get("fixture", {}),
                required_files=test_data.get("required_files", []),
                steps=test_data.get("steps", []),
            )

    # Parse custom tests
    custom_tests = data.get("custom_tests") or {}
    for test_name, test_data in custom_tests.items():
        if test_data and test_data.get("enabled", False):
            tests[test_name] = TestConfig(
                name=test_name,
                enabled=True,
                prompt_key="",
                description=test_data.get("description", ""),
                custom_prompt=test_data.get("prompt", ""),
                validator=test_data.get("validator", "execution_only"),
                tier=test_data.get("tier", "both"),
                fixture=test_data.get("fixture", {}),
                required_files=test_data.get("required_files", []),
            )

    return RobustnessConfig(
        n_variations=general.get("n_variations", 5),
        debug_mode=general.get("debug_mode", False),
        output_dir=general.get("output_dir", "reports"),
        save_artifacts=general.get("save_artifacts", True),
        s3_session_isolation=general.get("s3_session_isolation", True),
        repetitions=general.get("repetitions", 1),
        reliability_enabled=general.get("reliability_enabled", False),
        tier=general.get("tier", "both"),
        timeout_seconds=general.get("timeout_seconds", 0),
        reliability_min_success_rate=general.get("reliability_min_success_rate", 0.8),
        pricing=model.get("pricing", {}),
        inference_settings=model.get("inference_settings", {}),
        config_path=config_path,
        model_provider=model.get("provider", "deepseek"),
        model_id=model.get("model_id", "deepseek-chat"),
        api_key_env=model.get("api_key_env", "DEEPSEEK_API_KEY"),
        weights=metrics.get("weights", {}),
        thresholds=metrics.get("thresholds", {}),
        pass_threshold=metrics.get("pass_threshold", 0.75),
        tests=tests,
        generate_markdown=reporting.get("generate_markdown", True),
        generate_json=reporting.get("generate_json", True),
        include_run_details=reporting.get("include_run_details", True),
        include_recommendations=reporting.get("include_recommendations", True),
    )


class RobustnessRunner:
    """Run robustness tests based on configuration."""

    def __init__(self, config: RobustnessConfig):
        """Initialize runner with configuration."""
        self.config = config
        self.test_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.system = getattr(config, "system", "team")
        self.results: Dict[str, Dict] = {}

        # Setup output directory (per-arm so team vs single_agent don't collide)
        self.output_dir = (
            Path(__file__).parent / config.output_dir / f"{self.test_run_id}_{self.system}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components lazily
        self._prompt_generator = None
        self._comparator = None
        self._metrics_calculator = None
        self._model = None
        self._s3_config = None
        self.reliability_records: List[Dict[str, Any]] = []

        # Use shared S3SessionManager for safe session isolation
        self._s3_session_manager = S3SessionManager()

    @property
    def prompt_generator(self):
        """Lazy load prompt variation generator."""
        if self._prompt_generator is None:
            from prompt_variations import PromptVariationGenerator

            self._prompt_generator = PromptVariationGenerator()
        return self._prompt_generator

    @property
    def comparator(self):
        """Lazy load output comparator."""
        if self._comparator is None:
            from comparators import OutputComparator

            self._comparator = OutputComparator()
        return self._comparator

    @property
    def metrics_calculator(self):
        """Lazy load metrics calculator."""
        if self._metrics_calculator is None:
            from metrics import RobustnessMetrics

            self._metrics_calculator = RobustnessMetrics(
                weights=self.config.weights if self.config.weights else None,
                thresholds=self.config.thresholds if self.config.thresholds else None,
            )
        return self._metrics_calculator

    def _get_model(self):
        """Get LLM model based on configuration."""
        if self._model is not None:
            return self._model

        if self.config.model_provider == "ollama":
            from agno.models.ollama import Ollama

            host = os.environ.get("OLLAMA_HOST")
            self._model = Ollama(
                id=self.config.model_id,
                host=host,
                **self.config.inference_settings,
            )
        else:
            api_key = os.environ.get(self.config.api_key_env)
            if not api_key:
                from getpass import getpass

                api_key = getpass(f"{self.config.api_key_env}: ")

            if self.config.model_provider == "deepseek":
                from agno.models.deepseek import DeepSeek

                self._model = DeepSeek(
                    id=self.config.model_id,
                    api_key=api_key,
                    **self.config.inference_settings,
                )
            elif self.config.model_provider == "openai":
                from agno.models.openai import OpenAIChat

                self._model = OpenAIChat(
                    id=self.config.model_id,
                    api_key=api_key,
                    **self.config.inference_settings,
                )
            elif self.config.model_provider == "anthropic":
                from agno.models.anthropic import Claude

                self._model = Claude(
                    id=self.config.model_id,
                    api_key=api_key,
                    **self.config.inference_settings,
                )
            elif self.config.model_provider == "openrouter":
                from agno.models.openrouter import OpenRouter

                self._model = OpenRouter(
                    id=self.config.model_id,
                    api_key=api_key,
                    **self.config.inference_settings,
                )
                if self.config.model_id.lower().startswith("deepseek/"):
                    self._model.supports_native_structured_outputs = False
            else:
                raise ValueError(f"Unknown model provider: {self.config.model_provider}")

        return self._model

    def _setup_s3(self):
        """Setup S3 configuration and check availability."""
        from cs_copilot.storage import get_s3_config, is_s3_enabled

        if not is_s3_enabled():
            if self.config.s3_session_isolation:
                raise RuntimeError(
                    "S3/MinIO must be enabled for robustness testing with session isolation. "
                    "Set USE_S3=true and provide endpoint, bucket, and credentials."
                )
            logger.warning("S3 not enabled - files will be stored locally")
            return None

        self._s3_config = get_s3_config()
        logger.info(f"S3 enabled - Bucket: {self._s3_config.bucket_name}")
        return self._s3_config

    def _set_s3_prefix(self, prefix: str):
        """
        Set S3 prefix for session isolation (deprecated - use S3SessionManager context manager).

        This method is kept for backward compatibility but should not be used directly.
        The run_test method now uses S3SessionManager.create_isolated_session() context manager.
        """
        from cs_copilot.storage.client import S3 as S3Client

        S3Client.prefix = prefix

    def _restore_s3_prefix(self):
        """
        Restore original S3 prefix (deprecated - use S3SessionManager).

        This method is kept for backward compatibility. S3SessionManager now handles
        restoration automatically in finally blocks via context managers.
        """
        self._s3_session_manager.restore()

    def _get_prompts(self, test_config: TestConfig) -> List[str]:
        """Get prompt variations for a test."""
        if test_config.prompt_variants:
            return test_config.prompt_variants[: self.config.n_variations]

        if test_config.custom_prompt:
            # For custom prompts, just use the single prompt
            return [test_config.custom_prompt]

        # Get variations from prompt generator
        variations = self.prompt_generator.get_variations(
            test_config.prompt_key, n=self.config.n_variations
        )

        # Handle interpolation and latent_exploration with molecule parameters
        if test_config.params:
            augmented_variations = []
            for var in variations:
                augmented = var
                if "molecule_a" in test_config.params and "molecule_b" in test_config.params:
                    augmented = (
                        f"{var} Molecule A: {test_config.params['molecule_a']}, "
                        f"Molecule B: {test_config.params['molecule_b']}"
                    )
                elif "seed_molecule" in test_config.params:
                    augmented = f"{var} Seed molecule: {test_config.params['seed_molecule']}"
                augmented_variations.append(augmented)
            return augmented_variations

        return variations

    def _extract_files_from_response(self, response_text: str) -> Set[str]:
        """Extract file paths from agent response text (wrapper for ResponseParser)."""
        return ResponseParser.extract_files(response_text)

    def _extract_smiles_from_response(self, response: str) -> List[str]:
        """Extract SMILES strings from agent response."""
        smiles = []

        # Pattern 1: Backtick enclosed
        backtick_pattern = r"`([A-Za-z0-9@+\-\[\]\(\)=#$]+)`"
        smiles.extend(re.findall(backtick_pattern, response))

        # Pattern 2: Lines starting with SMILES-like strings
        for line in response.split("\n"):
            line = line.strip()
            if line and not line.startswith(("#", "-", "*", ">")):
                if re.match(
                    r"^[A-Za-z0-9@+\-\[\]\(\)=#$]+$", line.split()[0] if line.split() else ""
                ):
                    smiles.append(line.split()[0])

        # Remove duplicates while preserving order
        seen = set()
        unique_smiles = []
        for s in smiles:
            if s not in seen and len(s) > 2:
                seen.add(s)
                unique_smiles.append(s)

        return unique_smiles

    def _collect_state_files(self, session_state: Dict[str, Any]) -> Dict[str, str]:
        """Collect nested artifact pointers from structured session state."""
        from cs_copilot.storage import S3

        files: Dict[str, str] = {}
        artifact_suffixes = (
            ".csv",
            ".csv.gz",
            ".parquet",
            ".json",
            ".html",
            ".md",
            ".txt",
            ".png",
            ".svg",
            ".pdf",
            ".pkl",
            ".pkl.gz",
            ".sdf",
            ".fasta",
        )

        def visit(value: Any, path: str, depth: int = 0) -> None:
            if depth > 8:
                return
            if isinstance(value, dict):
                for key, item in value.items():
                    visit(item, f"{path}.{key}" if path else str(key), depth + 1)
                return
            if isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    visit(item, f"{path}[{index}]", depth + 1)
                return
            if not isinstance(value, str) or not value:
                return
            value_lower = value.lower().split("?", 1)[0]
            key_lower = path.lower()
            looks_like_pointer = (
                value.startswith("s3://")
                or value_lower.endswith(artifact_suffixes)
                or key_lower.endswith(("_path", ".path", "_uri", ".uri"))
            )
            if not looks_like_pointer or value.startswith(("http://", "https://")):
                return
            if value.startswith("s3://") or Path(value).is_absolute() or not self._s3_config:
                files[f"state:{path}"] = value
            else:
                files[f"state:{path}"] = S3.path(value)

        visit(session_state, "")
        return files

    def _build_system(self):
        """Build the system under test for the current arm.

        Both arms use the same model instance and keep memory disabled, so the
        only difference is the agentic structure: the multi-agent ``team`` vs the
        ``single_agent`` flat baseline. Both expose ``.run(prompt, stream=False)``
        and ``.get_session_state()``, so the rest of the runner is arm-agnostic.
        """
        model = self._get_model()
        if self.system == "single_agent":
            from cs_copilot.agents import get_cs_copilot_single_agent

            return get_cs_copilot_single_agent(
                model=model,
                debug_mode=self.config.debug_mode,
            )

        from cs_copilot.agents import get_cs_copilot_agent_team

        team = get_cs_copilot_agent_team(
            model=model,
            debug_mode=self.config.debug_mode,
            show_members_responses=False,
            enable_memory=False,  # Disable memory for session isolation
        )
        # Agno otherwise omits specialist RunOutputs from the coordinator result,
        # which would undercount member tokens and hide domain-tool failures.
        team.store_member_responses = True
        return team

    @staticmethod
    def _merge_state(target: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Deep-merge fixture state while retaining agent-required defaults."""
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                RobustnessRunner._merge_state(target[key], value)
            else:
                target[key] = value

    @staticmethod
    def _json_safe_state(value: Any, *, depth: int = 0) -> Any:
        """Serialize pointer-based state while redacting credential-like keys."""
        if depth > 20:
            return str(value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                key_text = str(key)
                if re.search(
                    r"(api[_-]?key|authorization|credential|password|secret|access[_-]?token)",
                    key_text,
                    re.IGNORECASE,
                ):
                    result[key_text] = "[REDACTED]"
                else:
                    result[key_text] = RobustnessRunner._json_safe_state(
                        item,
                        depth=depth + 1,
                    )
            return result
        if isinstance(value, (list, tuple, set)):
            return [RobustnessRunner._json_safe_state(item, depth=depth + 1) for item in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return str(value)

    def _load_fixture_state(self, fixture: Dict[str, Any]) -> Dict[str, Any]:
        """Load and verify an optional frozen session-state fixture."""
        if not fixture:
            return {}

        required = bool(fixture.get("required", False))
        raw_path = fixture.get("session_state_path")
        if not raw_path:
            if required:
                raise FixtureLoadError("Required fixture has no session_state_path")
            return {}

        expanded_path = os.path.expandvars(str(raw_path))
        if "$" in expanded_path:
            raise FixtureLoadError(
                f"Fixture path contains an unresolved environment variable: {raw_path}"
            )
        fixture_path = Path(expanded_path).expanduser()
        if not fixture_path.is_absolute():
            config_dir = self.config.config_path.parent if self.config.config_path else Path.cwd()
            fixture_path = config_dir / fixture_path
        if not fixture_path.is_file():
            message = f"Fixture file does not exist: {fixture_path}"
            if required:
                raise FixtureLoadError(message)
            logger.warning(message)
            return {}

        payload = fixture_path.read_bytes()
        expected_hash = os.path.expandvars(str(fixture.get("sha256") or "")).strip()
        if "$" in expected_hash:
            raise FixtureLoadError("Fixture SHA-256 contains an unresolved environment variable")
        actual_hash = hashlib.sha256(payload).hexdigest()
        if expected_hash and actual_hash.lower() != expected_hash.lower():
            raise FixtureLoadError(
                f"Fixture SHA-256 mismatch for {fixture_path}: "
                f"expected {expected_hash}, got {actual_hash}"
            )

        try:
            loaded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise FixtureLoadError(f"Fixture is not valid JSON: {fixture_path}") from exc
        if not isinstance(loaded, dict):
            raise FixtureLoadError(f"Fixture must contain a JSON object: {fixture_path}")
        state = loaded.get("session_state", loaded)
        if not isinstance(state, dict):
            raise FixtureLoadError("Fixture session_state must be a JSON object")
        return state

    def _apply_fixture(self, agent: Any, fixture: Dict[str, Any]) -> None:
        fixture_state = self._load_fixture_state(fixture)
        self._apply_fixture_state(agent, fixture_state)

    def _apply_fixture_state(self, agent: Any, fixture_state: Dict[str, Any]) -> None:
        if not fixture_state:
            return

        states: List[Dict[str, Any]] = []
        root_state = getattr(agent, "session_state", None)
        if isinstance(root_state, dict):
            states.append(root_state)
        for member in getattr(agent, "members", None) or []:
            member_state = getattr(member, "session_state", None)
            if isinstance(member_state, dict) and all(
                member_state is not state for state in states
            ):
                states.append(member_state)
        if not states:
            raise FixtureLoadError("System under test does not expose mutable session state")
        for state in states:
            self._merge_state(state, fixture_state)

    def _validate_required_files(self, requirements: List[Dict[str, Any]]) -> None:
        """Fail before agent construction when a required benchmark input is absent."""
        for requirement in requirements:
            name = str(requirement.get("name") or "required input")
            env_name = str(requirement.get("env") or "").strip()
            raw_path = os.environ.get(env_name, "").strip() if env_name else ""
            if not raw_path:
                raw_path = str(
                    requirement.get("path") or requirement.get("default_path") or ""
                ).strip()
            if not raw_path:
                source = f" environment variable {env_name}" if env_name else ""
                raise PrerequisiteError(f"{name} has no configured path.{source}")

            expanded_path = os.path.expandvars(raw_path)
            if "$" in expanded_path:
                raise PrerequisiteError(
                    f"{name} path contains an unresolved environment variable: {raw_path}"
                )
            required_path = Path(expanded_path).expanduser()
            if not required_path.is_absolute():
                config_dir = (
                    self.config.config_path.parent if self.config.config_path else Path.cwd()
                )
                required_path = config_dir / required_path
            if not required_path.is_file():
                override = f" Set {env_name} to override this path." if env_name else ""
                raise PrerequisiteError(
                    f"Required benchmark input is missing: {name} ({required_path}).{override}"
                )
            logger.info("Benchmark prerequisite available: %s (%s)", name, required_path)

    @staticmethod
    def _snapshot_session_state(agent: Any) -> Dict[str, Any]:
        """Capture in-memory state without requiring a persistent Agno session."""
        session_state = getattr(agent, "session_state", None)
        if not isinstance(session_state, dict):
            member_states = [
                getattr(member, "session_state", None)
                for member in (getattr(agent, "members", None) or [])
            ]
            session_state = next(
                (state for state in member_states if isinstance(state, dict)),
                None,
            )

        if not isinstance(session_state, dict):
            getter = getattr(agent, "get_session_state", None)
            if callable(getter):
                try:
                    loaded_state = getter()
                    session_state = loaded_state if isinstance(loaded_state, dict) else {}
                except Exception as exc:
                    logger.warning(
                        "Could not load persisted session state; retaining completed run: %s",
                        exc,
                    )
                    session_state = {}
            else:
                session_state = {}

        try:
            return copy.deepcopy(session_state)
        except Exception:
            return dict(session_state)

    def _failure_output(
        self,
        *,
        prompt: str,
        test_name: str,
        run_id: int,
        session_id: str,
        status: str,
        error: str,
        validator_name: str,
        prompt_variant: int,
        repetition: int,
        stage_name: Optional[str],
        tier: str,
        started_at: datetime,
        started_timer: float,
    ) -> Dict[str, Any]:
        finished_at = datetime.now(timezone.utc)
        output: Dict[str, Any] = {
            "run_id": run_id,
            "prompt": prompt,
            "session_id": session_id,
            "response": "",
            "response_truncated": "",
            "session_state_keys": [],
            "session_state": {},
            "generated_files": {},
            "s3_files": {},
            "smiles_generated": [],
            "n_molecules": 0,
            "status": status,
            "error": error,
            "system_under_test": self.system,
            "test_name": test_name,
            "stage_name": stage_name,
            "prompt_variant": prompt_variant,
            "repetition": repetition,
            "tier": tier,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "wall_time_seconds": max(0.0, time.perf_counter() - started_timer),
            "timestamp": finished_at.isoformat(),
            "telemetry": normalize_agno_output(None, pricing=self.config.pricing),
        }
        output["validation"] = evaluate_run(validator_name, output)
        return output

    def _run_single_variation(
        self,
        prompt: str,
        test_name: str,
        run_id: int,
        s3_prefix: Optional[str] = None,
        *,
        agent: Any = None,
        fixture: Optional[Dict[str, Any]] = None,
        required_files: Optional[List[Dict[str, Any]]] = None,
        validator_name: str = "execution_only",
        prompt_variant: int = 0,
        repetition: int = 0,
        stage_name: Optional[str] = None,
        tier: str = "both",
    ) -> Dict:
        """
        Run agent with a single prompt variation.

        Note: s3_prefix parameter is deprecated. S3 session isolation is now
        handled by the context manager in run_test().
        """
        from cs_copilot.storage import S3

        # S3 prefix is now handled by context manager in run_test()
        # No need to set it here

        session_id = (
            f"robustness_{self.test_run_id}_{test_name}_run{run_id}_" f"{uuid.uuid4().hex[:8]}"
        )
        started_at = datetime.now(timezone.utc)
        started_timer = time.perf_counter()

        logger.info(f"Running {test_name} variation {run_id + 1}")
        logger.debug(f"Session ID: {session_id}")
        logger.debug(f"Prompt: {prompt[:100]}...")

        try:
            self._validate_required_files(required_files or [])

            # Build the system under test (multi-agent team or single-agent
            # baseline); memory disabled for isolation, same model for both arms.
            if agent is None:
                fixture_state = self._load_fixture_state(fixture or {})
                agent = self._build_system()
                self._apply_fixture_state(agent, fixture_state)

            # Run the agent
            started_at = datetime.now(timezone.utc)
            started_timer = time.perf_counter()
            with run_timeout(self.config.timeout_seconds):
                result = agent.run(prompt, stream=False)

            # Capture the completed model output before any optional state
            # inspection. Memory-disabled Agno systems have no persisted session,
            # but their in-memory ``session_state`` remains authoritative.
            telemetry = normalize_agno_output(result, pricing=self.config.pricing)
            response_text = str(result.content) if result.content else ""
            session_state_snapshot = self._snapshot_session_state(agent)
            session_state = session_state_snapshot

            # Collect generated files
            generated_files = {}
            s3_files = {}

            # From response text
            files_from_response = self._extract_files_from_response(response_text)
            for filename in files_from_response:
                s3_url = S3.path(filename) if self._s3_config else filename
                generated_files[f"response:{filename}"] = s3_url
                s3_files[f"response:{filename}"] = s3_url

            # From nested, pointer-based session state.
            state_files = self._collect_state_files(session_state)
            generated_files.update(state_files)
            s3_files.update(
                {key: value for key, value in state_files.items() if value.startswith("s3://")}
            )

            # Extract SMILES if applicable
            smiles_generated = self._extract_smiles_from_response(response_text)

            finished_at = datetime.now(timezone.utc)
            output = {
                "run_id": run_id,
                "prompt": prompt,
                "session_id": session_id,
                "response": response_text,
                "response_object": result,  # Store for tool sequence extraction
                "response_truncated": (
                    response_text[:1000] if len(response_text) > 1000 else response_text
                ),
                "session_state_keys": list(session_state_snapshot.keys()),
                "session_state": session_state_snapshot,
                "generated_files": generated_files,
                "s3_files": s3_files,
                "smiles_generated": smiles_generated,
                "n_molecules": len(smiles_generated),
                "s3_prefix": s3_prefix,
                "timestamp": finished_at.isoformat(),
                "status": "success",
                "system_under_test": self.system,
                "test_name": test_name,
                "stage_name": stage_name,
                "prompt_variant": prompt_variant,
                "repetition": repetition,
                "tier": tier,
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "wall_time_seconds": max(0.0, time.perf_counter() - started_timer),
                "telemetry": telemetry,
            }
            output["validation"] = evaluate_run(validator_name, output)
            return output

        except KeyboardInterrupt:
            logger.warning(f"Run {run_id + 1} interrupted")
            return self._failure_output(
                prompt=prompt,
                test_name=test_name,
                run_id=run_id,
                session_id=session_id,
                status="interrupted",
                error="Run interrupted",
                validator_name=validator_name,
                prompt_variant=prompt_variant,
                repetition=repetition,
                stage_name=stage_name,
                tier=tier,
                started_at=started_at,
                started_timer=started_timer,
            )

        except ReliabilityTimeoutError as e:
            logger.error(f"Run {run_id + 1} timed out: {e}")
            return self._failure_output(
                prompt=prompt,
                test_name=test_name,
                run_id=run_id,
                session_id=session_id,
                status="timeout",
                error=str(e),
                validator_name=validator_name,
                prompt_variant=prompt_variant,
                repetition=repetition,
                stage_name=stage_name,
                tier=tier,
                started_at=started_at,
                started_timer=started_timer,
            )

        except Exception as e:
            logger.error(f"Run {run_id + 1} failed: {e}")
            status = (
                "fixture_error"
                if isinstance(e, FixtureLoadError)
                else "prerequisite_error" if isinstance(e, PrerequisiteError) else "failed"
            )
            return self._failure_output(
                prompt=prompt,
                test_name=test_name,
                run_id=run_id,
                session_id=session_id,
                status=status,
                error=str(e),
                validator_name=validator_name,
                prompt_variant=prompt_variant,
                repetition=repetition,
                stage_name=stage_name,
                tier=tier,
                started_at=started_at,
                started_timer=started_timer,
            )

    def _compare_outputs(self, outputs: List[Dict], test_name: str) -> Dict:
        """Compare outputs from multiple runs."""
        comparison_results = {}

        # Filter successful runs
        successful_outputs = [o for o in outputs if o.get("status") == "success"]

        if len(successful_outputs) < 2:
            logger.warning(f"Not enough successful runs to compare for {test_name}")
            return {"error": "Insufficient successful runs for comparison"}

        # Compare text responses
        texts = [o["response"] for o in successful_outputs if o.get("response")]
        if len(texts) >= 2:
            comparison_results["text"] = self.comparator.compare_text_outputs(texts)

        # Compare generated molecule counts (for autoencoder tests)
        if any(o.get("smiles_generated") for o in successful_outputs):
            import numpy as np

            n_mols = [o.get("n_molecules", 0) for o in successful_outputs]
            mean_mols = np.mean(n_mols) if n_mols else 0
            comparison_results["data"] = {
                "count_cv": np.std(n_mols) / mean_mols if mean_mols > 0 else 1.0,
                "count_mean": mean_mols,
                "count_std": np.std(n_mols),
                "row_jaccard": 1.0 - (np.std(n_mols) / mean_mols if mean_mols > 0 else 0),
                "column_match": 1.0,
                "value_stability": np.std(n_mols) / mean_mols if mean_mols > 0 else 0,
            }

        # Process consistency
        completion_rate = len(successful_outputs) / len(outputs) if outputs else 0

        # Extract tool sequences and calculate similarity
        tool_sequences = []
        for output in successful_outputs:
            # Try to get agent response object if available
            response_obj = output.get("response_object") or output.get("agent_response")
            if response_obj:
                seq = ToolSequenceComparator.extract_tool_sequence(response_obj)
                tool_sequences.append(seq)
            else:
                # Fallback: try to extract from session state
                session_state = output.get("session_state", {})
                seq = ToolSequenceComparator.extract_tool_sequence(session_state)
                tool_sequences.append(seq)

        # Calculate tool sequence similarity
        tool_similarity = ToolSequenceComparator.compare_sequences(tool_sequences)

        comparison_results["process"] = {
            "completion_rate": completion_rate,
            "tool_sequence_similarity": tool_similarity,
        }

        # Log tool sequence info for debugging
        if tool_sequences:
            logger.debug(f"Tool sequences: {tool_sequences}")
            logger.debug(f"Tool sequence similarity: {tool_similarity:.3f}")

        return comparison_results

    def _save_artifacts(self, test_name: str, outputs: List[Dict], comparison: Dict, score: float):
        """Save test artifacts for later analysis."""
        if not self.config.save_artifacts:
            return

        artifacts_dir = self.output_dir / test_name
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Save each run's details
        for output in outputs:
            run_dir = artifacts_dir / f"run_{output['run_id']}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Save prompt
            (run_dir / "prompt.txt").write_text(output.get("prompt", ""))

            # Save response
            response_path = run_dir / "response.txt"
            response_path.write_text(output.get("response", ""))
            output["response_path"] = str(response_path.relative_to(self.output_dir))

            # Capture a reloadable state boundary for building reviewed frozen fixtures.
            state_payload = json.dumps(
                {"session_state": self._json_safe_state(output.get("session_state") or {})},
                indent=2,
                sort_keys=True,
                default=str,
            ).encode()
            state_path = run_dir / "session_state.json"
            state_path.write_bytes(state_payload)
            output["session_state_fixture_path"] = str(state_path.relative_to(self.output_dir))
            output["session_state_fixture_sha256"] = hashlib.sha256(state_payload).hexdigest()

            # Save run metadata
            metadata = {
                k: v
                for k, v in output.items()
                if k not in ["prompt", "response", "response_object", "session_state"]
            }
            (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

        # Save comparison results
        (artifacts_dir / "comparison.json").write_text(
            json.dumps(comparison, indent=2, default=str)
        )

        # Save score
        (artifacts_dir / "score.txt").write_text(f"{score:.4f}")

        logger.info(f"Artifacts saved to {artifacts_dir}")

    def _to_reliability_record(self, output: Dict[str, Any]) -> Dict[str, Any]:
        telemetry = output.get("telemetry")
        telemetry = telemetry if isinstance(telemetry, dict) else {}
        validation = output.get("validation")
        validation = validation if isinstance(validation, dict) else {}
        models = telemetry.get("models") or []
        first_model = models[0] if models and isinstance(models[0], dict) else {}

        record = ReliabilityRunRecord(
            benchmark_run_id=self.test_run_id,
            case_name=str(output.get("stage_name") or output.get("test_name") or "unknown"),
            run_id=str(output.get("run_id")),
            session_id=str(output.get("session_id") or ""),
            system_under_test=self.system,
            tier=str(output.get("tier") or "both"),
            prompt_variant=int(output.get("prompt_variant") or 0),
            repetition=int(output.get("repetition") or 0),
            prompt=str(output.get("prompt") or ""),
            response_path=output.get("response_path"),
            execution_status=str(output.get("status") or "unknown"),
            task_success=bool(validation.get("task_success")),
            started_at=str(output.get("started_at") or output.get("timestamp") or ""),
            finished_at=str(output.get("finished_at") or output.get("timestamp") or ""),
            wall_time_seconds=float(output.get("wall_time_seconds") or 0),
            model_provider=first_model.get("model_provider") or self.config.model_provider,
            model_id=first_model.get("model_id") or self.config.model_id,
            input_tokens=int(telemetry.get("input_tokens") or 0),
            output_tokens=int(telemetry.get("output_tokens") or 0),
            total_tokens=int(telemetry.get("total_tokens") or 0),
            reasoning_tokens=int(telemetry.get("reasoning_tokens") or 0),
            cache_read_tokens=int(telemetry.get("cache_read_tokens") or 0),
            cache_write_tokens=int(telemetry.get("cache_write_tokens") or 0),
            llm_duration_seconds=telemetry.get("llm_duration_seconds"),
            estimated_cost=telemetry.get("estimated_cost"),
            tool_call_count=int(telemetry.get("tool_call_count") or 0),
            failed_tool_call_count=int(telemetry.get("failed_tool_call_count") or 0),
            tool_calls=telemetry.get("tool_calls") or [],
            validations=validation.get("checks") or [],
            failure_categories=validation.get("failure_categories") or [],
            generated_files=output.get("generated_files") or {},
            scientific_outcome=validation.get("scientific_outcome") or {},
            error=output.get("error"),
        )
        return record.to_dict()

    @contextmanager
    def _isolated_session(self, *, prompt_idx: int, repetition: int):
        if self._s3_config and self.config.s3_session_isolation:
            with self._s3_session_manager.create_isolated_session(
                test_run_id=self.test_run_id,
                prompt_idx=prompt_idx,
                variation_idx=repetition,
            ) as session_id:
                logger.debug(f"Created isolated S3 session: {session_id}")
                yield session_id
        else:
            yield None

    def _run_independent_test(
        self,
        test_config: TestConfig,
        prompts: List[str],
    ) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        run_id = 0
        for prompt_idx, prompt in enumerate(prompts):
            for repetition in range(self.config.repetitions):
                with self._isolated_session(
                    prompt_idx=prompt_idx,
                    repetition=repetition,
                ):
                    output = self._run_single_variation(
                        prompt=prompt,
                        test_name=test_config.name,
                        run_id=run_id,
                        fixture=test_config.fixture,
                        required_files=test_config.required_files,
                        validator_name=test_config.validator,
                        prompt_variant=prompt_idx,
                        repetition=repetition,
                        tier=test_config.tier,
                    )
                outputs.append(output)
                run_id += 1
        return outputs

    def _run_chain_test(self, test_config: TestConfig) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        run_id = 0
        for repetition in range(self.config.repetitions):
            with self._isolated_session(prompt_idx=0, repetition=repetition):
                chain_session_id = (
                    f"robustness_{self.test_run_id}_{test_config.name}_chain{repetition}_"
                    f"{uuid.uuid4().hex[:8]}"
                )
                agent = None
                preparation_error = None
                try:
                    self._validate_required_files(test_config.required_files)
                    fixture_state = self._load_fixture_state(test_config.fixture)
                    agent = self._build_system()
                    self._apply_fixture_state(agent, fixture_state)
                except Exception as exc:
                    preparation_error = exc

                chain_blocked = False
                for step_idx, step in enumerate(test_config.steps):
                    prompt = str(step["prompt"])
                    stage_name = str(step.get("name") or f"{test_config.name}_step_{step_idx + 1}")
                    validator_name = str(step.get("validator") or test_config.validator)

                    if preparation_error is not None or chain_blocked:
                        started_at = datetime.now(timezone.utc)
                        started_timer = time.perf_counter()
                        status = (
                            "fixture_error"
                            if isinstance(preparation_error, FixtureLoadError)
                            else (
                                (
                                    "prerequisite_error"
                                    if isinstance(preparation_error, PrerequisiteError)
                                    else "failed"
                                )
                                if preparation_error is not None
                                else "blocked"
                            )
                        )
                        error = (
                            str(preparation_error)
                            if preparation_error is not None
                            else "A preceding stage failed; dependent stage was not executed"
                        )
                        output = self._failure_output(
                            prompt=prompt,
                            test_name=test_config.name,
                            run_id=run_id,
                            session_id=chain_session_id,
                            status=status,
                            error=error,
                            validator_name=validator_name,
                            prompt_variant=step_idx,
                            repetition=repetition,
                            stage_name=stage_name,
                            tier=test_config.tier,
                            started_at=started_at,
                            started_timer=started_timer,
                        )
                    else:
                        output = self._run_single_variation(
                            prompt=prompt,
                            test_name=test_config.name,
                            run_id=run_id,
                            agent=agent,
                            validator_name=validator_name,
                            prompt_variant=step_idx,
                            repetition=repetition,
                            stage_name=stage_name,
                            tier=test_config.tier,
                        )
                        output["session_id"] = chain_session_id
                        chain_blocked = output.get("status") != "success"

                    outputs.append(output)
                    run_id += 1
        return outputs

    def run_test(self, test_config: TestConfig) -> Dict:
        """Run a robustness/reliability test with repetition and isolation."""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running test: {test_config.name}")
        logger.info(f"Description: {test_config.description}")
        logger.info(f"{'=' * 60}\n")

        prompts = [] if test_config.steps else self._get_prompts(test_config)
        expected_runs = (
            len(test_config.steps) * self.config.repetitions
            if test_config.steps
            else len(prompts) * self.config.repetitions
        )
        logger.info(
            f"Running {expected_runs} executions " f"({self.config.repetitions} repetition(s))"
        )

        self._setup_s3()

        try:
            outputs = (
                self._run_chain_test(test_config)
                if test_config.steps
                else self._run_independent_test(test_config, prompts)
            )
        finally:
            logger.debug("Ensuring S3 prefix restoration...")
            self._restore_s3_prefix()

        for index, output in enumerate(outputs):
            status = "✅" if output.get("validation", {}).get("task_success") else "❌"
            logger.info(f"  Run {index + 1}/{len(outputs)}: {status}")

        # Preserve completed executions before optional comparison/scoring work.
        # Missing visualization or embedding dependencies must not erase costly
        # live model outputs and telemetry.
        reliability_records = [self._to_reliability_record(output) for output in outputs]
        self.reliability_records.extend(reliability_records)

        try:
            comparison = self._compare_outputs(outputs, test_config.name)
        except Exception as exc:
            logger.warning(
                "Optional output comparison failed for %s; retaining run records: %s",
                test_config.name,
                exc,
            )
            comparison = {"error": f"Output comparison unavailable: {exc}"}

        # Calculate robustness score
        score = self.metrics_calculator.calculate_robustness_score(comparison)

        # Save artifacts
        self._save_artifacts(test_config.name, outputs, comparison, score)

        successful_tasks = sum(record["task_success"] for record in reliability_records)
        task_success_rate = (
            successful_tasks / len(reliability_records) if reliability_records else 0
        )
        reliability_mode = (
            self.config.reliability_enabled
            or test_config.validator != "execution_only"
            or bool(test_config.steps)
        )

        # Prepare result
        result = {
            "test_name": test_config.name,
            "description": test_config.description,
            "n_variations": len(prompts),
            "n_runs": len(outputs),
            "repetitions": self.config.repetitions,
            "successful_runs": sum(1 for o in outputs if o.get("status") == "success"),
            "successful_tasks": successful_tasks,
            "task_success_rate": task_success_rate,
            "robustness_score": score,
            "rating": self.metrics_calculator.get_rating(score),
            "passed": (
                task_success_rate >= self.config.reliability_min_success_rate
                if reliability_mode
                else score >= self.config.pass_threshold
            ),
            "comparison": comparison,
            "outputs": outputs if self.config.include_run_details else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(f"\nTest '{test_config.name}' completed:")
        logger.info(f"  Score: {score:.3f}")
        logger.info(f"  Rating: {result['rating']}")
        logger.info(f"  Objective success rate: {task_success_rate:.1%}")
        logger.info(f"  Passed: {'✅' if result['passed'] else '❌'}")

        return result

    def run_all_tests(self) -> Dict:
        """Run all enabled tests."""
        logger.info(f"\n{'#' * 60}")
        logger.info("Starting Robustness Test Suite")
        logger.info(f"Test Run ID: {self.test_run_id}")
        logger.info(f"{'#' * 60}\n")

        # Get enabled tests in the requested live/frozen tier.
        enabled_tests = [
            test_config
            for test_config in self.config.tests.values()
            if test_config.enabled
            and (
                self.config.tier == "both"
                or test_config.tier == "both"
                or test_config.tier == self.config.tier
            )
        ]

        if not enabled_tests:
            message = f"No enabled tests match tier '{self.config.tier}'"
            logger.warning(message)
            return {
                "error": message,
                "total_tests": 0,
                "passed": 0,
                "failed": 1,
                "pass_rate": 0,
                "average_robustness_score": 0,
                "overall_rating": "N/A",
                "results": {},
            }

        logger.info(f"Enabled tests: {[t.name for t in enabled_tests]}")

        # Run each test
        results = {}
        for test_config in enabled_tests:
            try:
                result = self.run_test(test_config)
                results[test_config.name] = result
                self.results[test_config.name] = result
            except Exception as e:
                logger.error(f"Test '{test_config.name}' failed with error: {e}")
                results[test_config.name] = {
                    "test_name": test_config.name,
                    "status": "error",
                    "error": str(e),
                }

        # Generate summary
        summary = self._generate_summary(results)
        if self.config.reliability_enabled:
            config_path = self.config.config_path or Path(__file__)
            environment_manifest = build_environment_manifest(
                project_root=project_root,
                model_provider=self.config.model_provider,
                model_id=self.config.model_id,
                config_path=config_path,
                pricing=self.config.pricing,
                inference_settings=self.config.inference_settings,
            )
            reliability_summary = save_reliability_bundle(
                self.output_dir / "reliability",
                self.reliability_records,
                environment_manifest=environment_manifest,
            )
            summary["reliability"] = reliability_summary
        self._save_reports(summary)

        return summary

    def _generate_summary(self, results: Dict) -> Dict:
        """Generate test suite summary."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r.get("passed", False))
        failed_tests = total_tests - passed_tests

        scores = [r.get("robustness_score", 0) for r in results.values() if "robustness_score" in r]
        avg_score = sum(scores) / len(scores) if scores else 0

        summary = {
            "test_run_id": self.test_run_id,
            "system_under_test": self.system,
            "timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "average_robustness_score": avg_score,
            "overall_rating": self.metrics_calculator.get_rating(avg_score),
            "pass_threshold": self.config.pass_threshold,
            "results": results,
        }

        return summary

    def _save_reports(self, summary: Dict):
        """Save summary reports."""
        # Save JSON summary
        if self.config.generate_json:
            json_path = self.output_dir / "summary.json"
            with open(json_path, "w") as f:
                # Session state can contain tuple keys (for example cached
                # descriptor/model pairs). ``default=str`` only handles values;
                # JSON rejects non-primitive dictionary keys before consulting it.
                json.dump(self._json_safe_state(summary), f, indent=2)
            logger.info(f"JSON summary saved to {json_path}")

        # Generate and save markdown report
        if self.config.generate_markdown:
            report = self._generate_markdown_report(summary)
            md_path = self.output_dir / "report.md"
            md_path.write_text(report)
            logger.info(f"Markdown report saved to {md_path}")

    def _generate_markdown_report(self, summary: Dict) -> str:
        """Generate comprehensive markdown report."""
        report = f"""# Robustness Test Report

**Test Run ID:** {summary['test_run_id']}
**Date:** {summary['timestamp']}

## Summary

| Metric | Value |
|--------|-------|
| Total Tests | {summary['total_tests']} |
| Passed | {summary['passed']} |
| Failed | {summary['failed']} |
| Pass Rate | {summary['pass_rate']:.1%} |
| Average Score | {summary['average_robustness_score']:.3f} |
| Overall Rating | {summary['overall_rating']} |
| Pass Threshold | {summary['pass_threshold']:.2f} |

## Test Results

"""
        for test_name, result in summary.get("results", {}).items():
            status = "✅ PASSED" if result.get("passed", False) else "❌ FAILED"
            score = result.get("robustness_score", 0)
            rating = result.get("rating", "N/A")

            report += f"""### {test_name}

- **Status:** {status}
- **Score:** {score:.3f}
- **Rating:** {rating}
- **Description:** {result.get('description', 'N/A')}
- **Variations:** {result.get('n_variations', 'N/A')}
- **Executions:** {result.get('n_runs', 'N/A')}
- **Successful Runs:** {result.get('successful_runs', 'N/A')}
- **Objective Task Success:** {result.get('task_success_rate', 0):.1%}

"""
            # Include comparison details
            comparison = result.get("comparison", {})
            if comparison and "text" in comparison:
                text_metrics = comparison["text"]
                report += f"""#### Text Comparison
- Semantic Similarity: {text_metrics.get('semantic_similarity', 0):.3f}
- Entity Overlap: {text_metrics.get('entity_overlap', 0):.3f}
- Numeric Consistency: {text_metrics.get('numeric_consistency', 0):.3f}

"""

        # Recommendations
        if self.config.include_recommendations:
            report += """## Recommendations

"""
            avg_score = summary["average_robustness_score"]
            if avg_score >= 0.9:
                report += (
                    "✅ **Excellent robustness.** The system handles prompt variations very well.\n"
                )
            elif avg_score >= 0.8:
                report += (
                    "✅ **Good robustness.** Minor inconsistencies but acceptable for production.\n"
                )
            elif avg_score >= 0.7:
                report += (
                    "⚠️ **Acceptable robustness** but room for improvement. Monitor closely.\n"
                )
            else:
                report += "❌ **Concerning robustness.** Significant inconsistencies detected. Review agent prompts and tool implementations.\n"

        report += f"""
---
*Generated by Cs_copilot Robustness Testing Framework*
*Output directory: {self.output_dir}*
"""
        return report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run robustness tests for Cs_copilot agentic operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  uv run python tests/robustness/robustness_minimal_example.py

  # Run with custom config
  uv run python tests/robustness/robustness_minimal_example.py --config my_config.yaml

  # Run specific test with custom variations
  uv run python tests/robustness/robustness_minimal_example.py --test chembl_download --n-variations 3

  # Run multiple tests
  uv run python tests/robustness/robustness_minimal_example.py --test chembl_download --test autoencoder_sampling
        """,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "robustness_config.yaml",
        help="Path to configuration YAML file",
    )

    parser.add_argument(
        "--test",
        action="append",
        dest="tests",
        help="Specific test(s) to run (can be used multiple times)",
    )

    parser.add_argument(
        "--n-variations",
        type=int,
        help="Number of prompt variations to use (overrides config)",
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        help="Independent repetitions per prompt or workflow chain (overrides config)",
    )

    parser.add_argument(
        "--tier",
        choices=["live", "frozen", "both"],
        help="Run live, frozen, or both benchmark tiers (overrides config)",
    )

    parser.add_argument(
        "--timeout-seconds",
        type=int,
        help="Wall-time limit for each agent execution; zero disables it",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )

    parser.add_argument(
        "--system",
        choices=["team", "single_agent", "both"],
        default="team",
        help=(
            "System under test: 'team' (multi-agent, default), 'single_agent' "
            "(flat baseline), or 'both' to run each arm and write a comparison. "
            "Both arms share the same model, tasks, and metrics."
        ),
    )

    parser.add_argument(
        "--arm-order",
        choices=["team-first", "single-agent-first"],
        default="team-first",
        help=(
            "Execution order when --system both is selected. Alternate this across "
            "independent benchmark batches to reduce temporal service bias."
        ),
    )

    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List available tests and exit",
    )

    parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="List available prompt categories and exit",
    )

    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow tracking for test runs (logs all metrics, parameters, and artifacts)",
    )

    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="robustness_testing",
        help="MLflow experiment name (default: robustness_testing)",
    )

    return parser.parse_args()


def _make_runner(config: RobustnessConfig, args) -> "RobustnessRunner":
    """Build a runner for one arm, MLflow-enhanced when requested."""
    if args.mlflow:
        try:
            from mlflow_runner import MLflowRobustnessRunner

            logger.info(f"Creating MLflow-enhanced runner (experiment: {args.mlflow_experiment})")
            return MLflowRobustnessRunner(
                config, experiment_name=args.mlflow_experiment, enable_mlflow=True
            )
        except ImportError as e:
            logger.warning(f"MLflow dependencies not available: {e}. Using standard runner.")
    return RobustnessRunner(config)


def compare_systems(
    summaries: Dict[str, Dict],
    records_by_arm: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
) -> Path:
    """Write paired reliability and secondary robustness comparison artifacts."""
    comparison_md = save_system_comparison(
        output_dir,
        records_by_arm,
        robustness_summaries=summaries,
    )
    logger.info(f"Comparison written to {comparison_md}")
    return comparison_md


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    if not args.config.exists():
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    # Handle list commands
    if args.list_tests:
        print("\nAvailable tests:")
        for name, test in config.tests.items():
            status = "✓ enabled" if test.enabled else "✗ disabled"
            print(f"  {name}: {status}")
            print(f"    {test.description}")
        sys.exit(0)

    if args.list_prompts:
        from prompt_variations import PromptVariationGenerator

        generator = PromptVariationGenerator()
        print("\nAvailable prompt categories:")
        for name in generator.list_available_prompts():
            print(f"  - {name}")
        sys.exit(0)

    # Apply command line overrides
    if args.n_variations:
        config.n_variations = args.n_variations

    if args.repetitions is not None:
        if args.repetitions < 1:
            raise ValueError("--repetitions must be positive")
        config.repetitions = args.repetitions

    if args.tier:
        config.tier = args.tier

    if args.timeout_seconds is not None:
        if args.timeout_seconds < 0:
            raise ValueError("--timeout-seconds must be zero or positive")
        config.timeout_seconds = args.timeout_seconds

    if args.debug:
        config.debug_mode = True

    if args.tests:
        # Enable only specified tests
        for test_name in config.tests:
            config.tests[test_name].enabled = test_name in args.tests

    # One or both arms of the multi-agent-vs-single-agent comparison.
    if args.system == "both":
        arms = (
            ["team", "single_agent"] if args.arm_order == "team-first" else ["single_agent", "team"]
        )
    else:
        arms = [args.system]

    summaries: Dict[str, Dict] = {}
    records_by_arm: Dict[str, List[Dict[str, Any]]] = {}
    shared_model = None
    first_run_id: Optional[str] = None
    try:
        for arm in arms:
            config.system = arm
            runner = _make_runner(config, args)
            if first_run_id is None:
                first_run_id = runner.test_run_id
            # Hold the model constant across arms: reuse the first arm's model
            # instance so the only variable is the agentic structure.
            if shared_model is not None:
                runner._model = shared_model

            logger.info(f"\n{'#' * 60}\nSYSTEM UNDER TEST: {arm}\n{'#' * 60}")
            summary = runner.run_all_tests()
            summaries[arm] = summary
            records_by_arm[arm] = list(runner.reliability_records)
            shared_model = runner._get_model()

            # Print per-arm summary
            print(f"\n{'=' * 60}")
            print(f"ROBUSTNESS SUITE COMPLETED — system: {arm}")
            print(f"{'=' * 60}")
            print(f"Total Tests: {summary.get('total_tests', 0)}")
            print(f"Passed: {summary.get('passed', 0)}")
            print(f"Failed: {summary.get('failed', 0)}")
            print(f"Average Score: {summary.get('average_robustness_score', 0):.3f}")
            print(f"Overall Rating: {summary.get('overall_rating', 'N/A')}")
            print(f"Reports saved to: {runner.output_dir}")
            print(f"{'=' * 60}")

        # Cross-arm comparison artifact (only meaningful for --system both)
        if len(summaries) > 1:
            comparison_dir = (
                Path(__file__).parent / config.output_dir / f"{first_run_id}_comparison"
            )
            comparison_md = compare_systems(summaries, records_by_arm, comparison_dir)
            print(f"\nMulti-agent vs single-agent comparison written to: {comparison_md}")

        # Exit non-zero if any arm reported failing tests.
        if any(summary.get("failed", 0) > 0 for summary in summaries.values()):
            sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("\nTest suite interrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        raise


if __name__ == "__main__":
    main()
