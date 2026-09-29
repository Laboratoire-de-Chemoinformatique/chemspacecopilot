"""The runtime-neutral LLM port used by toolkits and MCP facades."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cs_copilot.execution.llm import (
    LLMUnavailableError,
    llm_policy_of,
    normalize_llm_policy,
    require_model,
    resolve_model,
)
from cs_copilot.mcp.context import MCPAgentContext
from cs_copilot.mcp.llm import broker


def test_policy_vocabulary_is_shared_with_the_mcp_broker():
    assert broker.normalize_llm_policy is normalize_llm_policy
    assert normalize_llm_policy(None) == "external"
    assert normalize_llm_policy(" Agno_Model ") == "agno-model"
    with pytest.raises(ValueError, match="Unsupported LLM policy"):
        normalize_llm_policy("sometimes")


def test_model_resolution_follows_the_declared_policy():
    model = object()
    assert resolve_model(None) is None
    assert resolve_model(SimpleNamespace(model=model)) is model
    assert llm_policy_of(SimpleNamespace(model=model)) is None
    assert resolve_model(SimpleNamespace(model=model, llm_policy="agno-model")) is model
    assert resolve_model(SimpleNamespace(model=model, llm_policy="disabled")) is None
    assert resolve_model(MCPAgentContext()) is None
    assert llm_policy_of(SimpleNamespace(model=model, llm_policy=object())) is None


def test_require_model_raises_a_runtime_neutral_error():
    with pytest.raises(LLMUnavailableError) as missing:
        require_model(MCPAgentContext(), "LLM design")
    assert str(missing.value).startswith("LLM design requires an agent with a model")
    assert isinstance(missing.value, RuntimeError)
    assert missing.value.purpose == "LLM design"

    with pytest.raises(LLMUnavailableError, match="disabled"):
        require_model(SimpleNamespace(model=object(), llm_policy="disabled"), "LLM design")


@pytest.mark.parametrize(
    ("module", "cls", "error_cls", "fallback"),
    [
        (
            "cs_copilot.tools.chemistry.molecular_designer_toolkit",
            "MolecularDesignerToolkit",
            "MolecularDesignerError",
            "autoencoder",
        ),
        (
            "cs_copilot.tools.chemistry.peptide_designer_toolkit",
            "PeptideDesignerToolkit",
            "PeptideDesignerError",
            "wae",
        ),
    ],
)
def test_toolkit_llm_engine_errors_do_not_mention_a_runtime(module, cls, error_cls, fallback):
    import importlib

    toolkit_module = importlib.import_module(module)
    toolkit = getattr(toolkit_module, cls).__new__(getattr(toolkit_module, cls))

    with pytest.raises(getattr(toolkit_module, error_cls)) as raised:
        toolkit._get_engine("llm", MCPAgentContext())

    message = str(raised.value)
    assert "requires an agent with a model" in message
    assert f"engine='{fallback}'" in message
    for runtime_detail in ("MCP", "MCPAgentContext", "agno_team_run"):
        assert runtime_detail not in message
