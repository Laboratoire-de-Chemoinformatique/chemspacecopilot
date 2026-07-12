#!/usr/bin/env python
# coding: utf-8
"""Unit tests for the single-agent baseline factory (multi-agent-vs-single-agent ablation).

The single agent must hold the UNION of the team specialists' toolkits + merged
session_state, be registered, but stay OUT of the runtime team (which is what makes
the comparison "single vs multi"). All checks are offline: domain toolkits are
patched to lightweight named dummies so no model backend loads.
"""

import inspect

from agno.agent import Agent
from agno.models.base import Model

from cs_copilot.agents import factories as factory_module
from cs_copilot.agents import list_available_agent_types, teams
from cs_copilot.agents.instructions import SINGLE_AGENT_INSTRUCTIONS

# Toolkit classes the single agent unions (one instance each). Ordering in the
# factory guarantees the molecular/autoencoder design tools win Agno's first-wins
# name dedupe over the peptide toolkit's shared method names.
_TOOLKIT_NAMES = (
    "AutoencoderToolkit",
    "ChemblToolkit",
    "ChemicalSimilarityToolkit",
    "GTMToolkit",
    "MolecularDesignerToolkit",
    "PeptideDesignerToolkit",
    "PointerPandasTools",
    "SessionMemoryToolkit",
    "SkillToolkit",
    "SynPlannerToolkit",
)
_EXPECTED_CALLABLES = {
    "save_gtm_landscape_plot",
    "save_gtm_plot",
    "save_rich_report",
    "save_markdown_report",
}
# Member factories whose session_state the single agent must cover (the 7 team
# members that actually declare session_state; SynPlanner/Peptide declare none).
_STATEFUL_MEMBER_FACTORIES = (
    "ChEMBLDownloaderFactory",
    "GTMAgentFactory",
    "ChemoinformaticianFactory",
    "ReportGeneratorFactory",
    "MolecularDesignerFactory",
)


class _ConstructionModel(Model):
    """Minimal Agno model for construction-only tests (never invoked)."""

    def invoke(self, *args, **kwargs):
        raise NotImplementedError

    async def ainvoke(self, *args, **kwargs):
        raise NotImplementedError

    def invoke_stream(self, *args, **kwargs):
        raise NotImplementedError
        yield

    async def ainvoke_stream(self, *args, **kwargs):
        raise NotImplementedError
        yield

    def _parse_provider_response(self, response, **kwargs):
        raise NotImplementedError

    def _parse_provider_response_delta(self, response):
        raise NotImplementedError


def _named_dummy(name: str):
    """A no-op class whose ``__name__`` matches the real toolkit it replaces."""
    return type(name, (), {"__init__": lambda self, *args, **kwargs: None})


def _patch_named_dummy_toolkits(monkeypatch):
    for name in _TOOLKIT_NAMES:
        monkeypatch.setattr(factory_module, name, _named_dummy(name))


def _toolkit_and_callable_names(config):
    toolkits = [t.__class__.__name__ for t in config.tools if not inspect.isfunction(t)]
    callables = [t.__name__ for t in config.tools if inspect.isfunction(t)]
    return toolkits, callables


def test_single_agent_is_registered():
    assert "single_agent" in list_available_agent_types()


def test_single_agent_excluded_from_runtime_team(monkeypatch):
    """The team stays 7 members; the flat baseline must not join it."""

    def fake_create_agent(agent_type, model, **kwargs):
        return Agent(
            name=f"{agent_type}_agent",
            model=model,
            session_state=kwargs.get("session_state"),
            telemetry=False,
        )

    monkeypatch.setattr(teams, "create_agent", fake_create_agent)
    monkeypatch.setattr(teams, "analyze_resources", lambda: {"cpu": "test"})
    model = _ConstructionModel(id="test-model", provider="test")

    team = teams.get_cs_copilot_agent_team(model, enable_memory=False, enable_mlflow_tracking=False)
    member_names = {member.name for member in team.members}

    assert len(team.members) == 7
    assert "single_agent_agent" not in member_names


def test_single_agent_unions_specialist_toolkits(monkeypatch):
    _patch_named_dummy_toolkits(monkeypatch)
    config = factory_module.SingleAgentFactory().get_agent_config()
    toolkits, callables = _toolkit_and_callable_names(config)

    # Every specialist toolkit present exactly once; robustness toolkit excluded.
    assert set(toolkits) == set(_TOOLKIT_NAMES)
    assert len(toolkits) == len(_TOOLKIT_NAMES)
    assert "RobustnessAnalysisToolkit" not in toolkits
    assert set(callables) == _EXPECTED_CALLABLES

    # Ordering invariant: molecular/autoencoder precede peptide so, under Agno's
    # first-wins dedupe, the small-molecule design tools survive the shared names
    # (validate_design_candidates, decode_latent, list_design_engines, ...).
    assert toolkits.index("MolecularDesignerToolkit") < toolkits.index("PeptideDesignerToolkit")
    assert toolkits.index("AutoencoderToolkit") < toolkits.index("PeptideDesignerToolkit")


def test_single_agent_session_state_covers_members(monkeypatch):
    """Drift guard: the flat agent's session_state must be a superset of each
    stateful team member's session_state top-level keys."""
    _patch_named_dummy_toolkits(monkeypatch)
    single_state = factory_module.SingleAgentFactory().get_agent_config().session_state

    missing = {}
    for factory_name in _STATEFUL_MEMBER_FACTORIES:
        member_state = getattr(factory_module, factory_name)().get_agent_config().session_state
        absent = [key for key in member_state if key not in single_state]
        if absent:
            missing[factory_name] = absent

    assert not missing, f"single_agent session_state missing member keys: {missing}"
    # Nested defaults preserved by the merge.
    assert single_state["gtm_cache"]["metadata"]["optimization_strategy"] is None
    assert single_state["report_outputs"]["report_paths"] == {}
    assert single_state["data_file_paths"]["clean_dataset_path"] is None


def test_single_agent_instructions_dedup_and_neutralize_handoffs():
    # Shared policy blocks collapse to one occurrence: no duplicate lines.
    assert len(SINGLE_AGENT_INSTRUCTIONS) == len(set(SINGLE_AGENT_INSTRUCTIONS))

    text = "\n".join(SINGLE_AGENT_INSTRUCTIONS).lower()
    # Preamble neutralizes the team-only handoff lines.
    assert "no team" in text
    assert "do that work yourself" in text
    # Union spans all seven specialist roles (each references its own skill).
    for skill_slug in (
        "chembl-target-retrieval",
        "gtm-density-landscape",
        "molecular-design",
        "peptide-design",
        "report-generation",
        "retrosynthesis",
    ):
        assert (
            skill_slug in text
        ), f"single-agent instructions missing {skill_slug!r} role knowledge"
