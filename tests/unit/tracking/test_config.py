"""Unit tests for MLflow configuration."""

from cs_copilot.tracking.config import MLflowConfig


def test_default_config():
    """Test default configuration values."""
    config = MLflowConfig()
    assert config.enabled is True
    assert config.tracking_uri == "file:///tmp/mlflow"
    assert config.experiment_name == "production_sessions"
    assert config.track_costs is True
    assert config.cost_per_1k_prompt_tokens == 0.00027
    assert config.cost_per_1k_completion_tokens == 0.0011
    assert config.offline_mode is False


def test_config_from_env(monkeypatch):
    """Test configuration loading from environment variables."""
    monkeypatch.setenv("MLFLOW_TRACKING_ENABLED", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test_experiment")
    monkeypatch.setenv("MLFLOW_TRACK_COSTS", "false")
    monkeypatch.setenv("COST_PER_1K_TOKENS_PROMPT", "0.001")
    monkeypatch.setenv("COST_PER_1K_TOKENS_COMPLETION", "0.002")
    monkeypatch.setenv("MLFLOW_OFFLINE_MODE", "false")

    config = MLflowConfig.from_env()

    assert config.enabled is True
    assert config.tracking_uri == "http://localhost:5000"
    assert config.experiment_name == "test_experiment"
    assert config.track_costs is False
    assert config.cost_per_1k_prompt_tokens == 0.001
    assert config.cost_per_1k_completion_tokens == 0.002
    assert config.offline_mode is False


def test_config_disabled(monkeypatch):
    """Test configuration with tracking disabled."""
    monkeypatch.setenv("MLFLOW_TRACKING_ENABLED", "false")

    config = MLflowConfig.from_env()

    assert config.enabled is False
    assert config.is_enabled() is False


def test_config_offline_mode(monkeypatch):
    """Test configuration with offline mode enabled."""
    monkeypatch.setenv("MLFLOW_TRACKING_ENABLED", "true")
    monkeypatch.setenv("MLFLOW_OFFLINE_MODE", "true")

    config = MLflowConfig.from_env()

    assert config.enabled is True
    assert config.offline_mode is True
    assert config.is_enabled() is False  # Should be disabled in offline mode


def test_config_boolean_parsing(monkeypatch):
    """Test boolean value parsing from environment variables."""
    # Test various true values
    for value in ["true", "True", "TRUE", "1", "yes", "Yes", "on", "On"]:
        monkeypatch.setenv("MLFLOW_TRACKING_ENABLED", value)
        config = MLflowConfig.from_env()
        assert config.enabled is True, f"Failed for value: {value}"

    # Test various false values
    for value in ["false", "False", "FALSE", "0", "no", "No", "off", "Off"]:
        monkeypatch.setenv("MLFLOW_TRACKING_ENABLED", value)
        config = MLflowConfig.from_env()
        assert config.enabled is False, f"Failed for value: {value}"


def test_config_from_yaml(tmp_path):
    """Test configuration loading from YAML file."""
    yaml_content = """
enabled: true
tracking_uri: "sqlite:///test.db"
experiment_name: "yaml_experiment"
track_costs: true
cost_per_1k_prompt_tokens: 0.005
cost_per_1k_completion_tokens: 0.01
offline_mode: false
"""
    yaml_file = tmp_path / "mlflow_config.yaml"
    yaml_file.write_text(yaml_content)

    config = MLflowConfig.from_env(yaml_file)

    assert config.enabled is True
    assert config.tracking_uri == "sqlite:///test.db"
    assert config.experiment_name == "yaml_experiment"
    assert config.track_costs is True
    assert config.cost_per_1k_prompt_tokens == 0.005
    assert config.cost_per_1k_completion_tokens == 0.01


def test_config_env_overrides_yaml(tmp_path, monkeypatch):
    """Test that environment variables override YAML configuration."""
    yaml_content = """
tracking_uri: "sqlite:///test.db"
experiment_name: "yaml_experiment"
"""
    yaml_file = tmp_path / "mlflow_config.yaml"
    yaml_file.write_text(yaml_content)

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env-override:5000")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "env_experiment")

    config = MLflowConfig.from_env(yaml_file)

    assert config.tracking_uri == "http://env-override:5000"
    assert config.experiment_name == "env_experiment"
