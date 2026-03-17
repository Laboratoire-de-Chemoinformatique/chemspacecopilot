"""Core MLflow tracking functionality."""

import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional

from .config import MLflowConfig
from .utils import Timer, sanitize_run_name

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Main MLflow tracking interface for Cs_copilot.

    This class provides context managers and methods for tracking
    sessions, agent executions, and tool calls in MLflow.
    """

    def __init__(self, config: Optional[MLflowConfig] = None):
        """Initialize MLflow tracker.

        Args:
            config: MLflow configuration. If None, loads from environment.
        """
        self.config = config or MLflowConfig.from_env()
        self._mlflow = None
        self._initialized = False
        self._active_run_stack: list = []

        if self.config.is_enabled():
            self._initialize_mlflow()

    def _initialize_mlflow(self):
        """Initialize MLflow client and set tracking URI."""
        try:
            import mlflow

            self._mlflow = mlflow
            self._mlflow.set_tracking_uri(self.config.tracking_uri)
            self._initialized = True
            logger.info(f"MLflow tracking initialized: {self.config.tracking_uri}")
        except ImportError:
            logger.warning("MLflow not installed. Tracking disabled.")
            self.config.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")
            self.config.enabled = False

    def is_enabled(self) -> bool:
        """Check if tracking is enabled and initialized."""
        return self.config.is_enabled() and self._initialized

    @contextmanager
    def track_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        interface: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        """Context manager for tracking a user session.

        Args:
            session_id: Unique session identifier
            user_id: User identifier (optional)
            interface: Interface name (e.g., 'chainlit', 'streamlit')
            experiment_name: Override default experiment name

        Yields:
            MLflow run object (or None if tracking disabled)
        """
        if not self.is_enabled():
            yield None
            return

        exp_name = experiment_name or self.config.experiment_name
        run_name = sanitize_run_name(f"session_{session_id[:8]}")

        # Set or create experiment
        try:
            self._mlflow.set_experiment(exp_name)
        except Exception as e:
            logger.error(f"Failed to set experiment {exp_name}: {e}")
            yield None
            return

        # Start session-level run
        timer = Timer()
        with timer:
            with self._mlflow.start_run(run_name=run_name) as run:
                self._active_run_stack.append(run.info.run_id)

                # Log session parameters
                params = {"session_id": session_id}
                if user_id:
                    params["user_id"] = user_id
                if interface:
                    params["interface"] = interface

                self._safe_log_params(params)

                # Log session tags
                tags = {"session_type": "user_session"}
                if interface:
                    tags["interface"] = interface
                self._safe_log_tags(tags)

                try:
                    yield run
                finally:
                    # Log session metrics
                    if timer.duration:
                        self._safe_log_metrics({"session_duration_seconds": timer.get_duration()})
                    self._active_run_stack.pop()

    @contextmanager
    def track_agent_run(
        self,
        agent_name: str,
        prompt: str,
        agent_type: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        """Context manager for tracking an agent execution.

        Args:
            agent_name: Name of the agent
            prompt: User prompt/query
            agent_type: Type of agent (optional)
            run_name: Override default run name

        Yields:
            MLflow run object (or None if tracking disabled)
        """
        if not self.is_enabled():
            yield None
            return

        run_name = run_name or sanitize_run_name(f"agent_{agent_name}")

        timer = Timer()
        with timer:
            with self._mlflow.start_run(run_name=run_name, nested=True) as run:
                self._active_run_stack.append(run.info.run_id)

                # Log agent parameters
                params = {
                    "agent_name": agent_name,
                    "prompt_preview": prompt[:200] if prompt else "",
                }
                if agent_type:
                    params["agent_type"] = agent_type

                self._safe_log_params(params)

                # Log agent tags
                tags = {"agent_name": agent_name}
                if agent_type:
                    tags["agent_type"] = agent_type
                self._safe_log_tags(tags)

                try:
                    yield run
                finally:
                    # Log execution metrics
                    metrics = {"execution_time_seconds": timer.get_duration()}
                    self._safe_log_metrics(metrics)
                    self._active_run_stack.pop()

    @contextmanager
    def track_tool_call(
        self, tool_name: str, args: Optional[Dict[str, Any]] = None, run_name: Optional[str] = None
    ):
        """Context manager for tracking a tool call.

        Args:
            tool_name: Name of the tool
            args: Tool arguments (optional)
            run_name: Override default run name

        Yields:
            MLflow run object (or None if tracking disabled)
        """
        if not self.is_enabled():
            yield None
            return

        run_name = run_name or sanitize_run_name(f"tool_{tool_name}")

        timer = Timer()
        with timer:
            with self._mlflow.start_run(run_name=run_name, nested=True) as run:
                self._active_run_stack.append(run.info.run_id)

                # Log tool parameters
                params = {"tool_name": tool_name}
                if args:
                    # Log only serializable args
                    for key, value in args.items():
                        try:
                            params[f"arg_{key}"] = str(value)[:200]  # Truncate long values
                        except Exception:
                            continue

                self._safe_log_params(params)

                # Log tool tags
                tags = {"tool_name": tool_name}
                self._safe_log_tags(tags)

                try:
                    yield run
                finally:
                    # Log execution metrics
                    metrics = {"execution_time_seconds": timer.get_duration()}
                    self._safe_log_metrics(metrics)
                    self._active_run_stack.pop()

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to current active run.

        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number
        """
        if not self.is_enabled() or not self._active_run_stack:
            return

        self._safe_log_metrics(metrics, step)

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to current active run.

        Args:
            params: Dictionary of parameter names to values
        """
        if not self.is_enabled() or not self._active_run_stack:
            return

        self._safe_log_params(params)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact to current active run.

        Args:
            local_path: Path to local file
            artifact_path: Path within artifact directory
        """
        if not self.is_enabled() or not self._active_run_stack:
            return

        try:
            self._mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            logger.error(f"Failed to log artifact {local_path}: {e}")

    def log_text(self, text: str, artifact_file: str):
        """Log text content as an artifact.

        Args:
            text: Text content
            artifact_file: Artifact filename
        """
        if not self.is_enabled() or not self._active_run_stack:
            return

        try:
            self._mlflow.log_text(text, artifact_file)
        except Exception as e:
            logger.error(f"Failed to log text artifact {artifact_file}: {e}")

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str):
        """Log dictionary as JSON artifact.

        Args:
            dictionary: Dictionary to log
            artifact_file: Artifact filename (should end with .json)
        """
        if not self.is_enabled() or not self._active_run_stack:
            return

        try:
            self._mlflow.log_dict(dictionary, artifact_file)
        except Exception as e:
            logger.error(f"Failed to log dict artifact {artifact_file}: {e}")

    def register_prompt_version(
        self,
        name: str,
        template: Any,
        commit_message: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Register or reuse a prompt in the MLflow Prompt Registry.

        Args:
            name: Prompt name in registry
            template: Prompt template text or structured prompt
            commit_message: Optional commit message for versioning
            tags: Optional tags to apply to the prompt

        Returns:
            Prompt object or None if registration failed or disabled.
        """
        if not self.is_enabled() or not self._mlflow:
            return None

        genai = getattr(self._mlflow, "genai", None)
        if genai is None:
            logger.warning(
                "MLflow genai prompt registry not available. Skipping prompt registration."
            )
            return None

        try:
            existing = None
            try:
                existing = genai.load_prompt(name, allow_missing=True)
            except TypeError:
                try:
                    existing = genai.load_prompt(name)
                except Exception:
                    existing = None

            if existing is not None:
                existing_template = getattr(existing, "template", None)
                if existing_template == template:
                    return existing

            return genai.register_prompt(
                name=name, template=template, commit_message=commit_message, tags=tags
            )
        except Exception as e:
            logger.warning(f"Failed to register prompt {name}: {e}")
            return None

    def _safe_log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Safely log metrics, catching and logging errors."""
        try:
            self._mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics {metrics}: {e}")

    def _safe_log_params(self, params: Dict[str, Any]):
        """Safely log parameters, catching and logging errors."""
        try:
            # Convert all values to strings and truncate if needed
            safe_params = {}
            for key, value in params.items():
                str_value = str(value)
                # MLflow has a 500 character limit for param values
                safe_params[key] = str_value[:500] if len(str_value) > 500 else str_value
            self._mlflow.log_params(safe_params)
        except Exception as e:
            logger.error(f"Failed to log params {params}: {e}")

    def _safe_log_tags(self, tags: Dict[str, str]):
        """Safely log tags, catching and logging errors."""
        try:
            self._mlflow.set_tags(tags)
        except Exception as e:
            logger.error(f"Failed to log tags {tags}: {e}")


# Global tracker instance
_global_tracker: Optional[MLflowTracker] = None


def get_tracker(config: Optional[MLflowConfig] = None) -> MLflowTracker:
    """Get or create global MLflow tracker instance.

    Args:
        config: Optional configuration. If None, uses existing tracker or creates new one.

    Returns:
        MLflowTracker instance
    """
    global _global_tracker

    if _global_tracker is None:
        _global_tracker = MLflowTracker(config)
    elif config is not None:
        # Update config if provided
        _global_tracker = MLflowTracker(config)

    return _global_tracker


def reset_tracker():
    """Reset global tracker instance. Useful for testing."""
    global _global_tracker
    _global_tracker = None
