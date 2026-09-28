"""Tests for bounded handoffs and role-specific tool policies."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cs_copilot.agents.context import ContextBudget, ContextBuilder
from cs_copilot.agents.contracts import (
    ROLE_POLICIES,
    ExecutionBudget,
    HandoffEnvelope,
    get_role_policy,
    record_handoff,
    validate_role_tools,
)
from cs_copilot.agents.factories import AgentConfig, AgentCreationError, BaseAgentFactory
from cs_copilot.workflows.runtime import HandoffEnvelope as RuntimeHandoffEnvelope


def test_agents_reexport_the_canonical_runtime_handoff_contract():
    assert HandoffEnvelope is RuntimeHandoffEnvelope

    envelope = HandoffEnvelope.create(
        run_id="run-001",
        workflow_slug="chembl-to-gtm-report",
        task_id="report",
        sender_role="gtm_agent",
        receiver_role="report_generator",
        objective="Write the evidence-backed report.",
        required_capabilities=("report_save_rich",),
        expected_output_artifacts=("html_report_path",),
        expected_output_schema={"type": "report"},
        context_summary="GTM artifacts are registered and validated.",
        budget=ExecutionBudget(max_tokens=1200, max_tool_calls=2),
    )

    payload = envelope.to_dict()
    assert payload["workflow_slug"] == "chembl-to-gtm-report"
    assert payload["required_capabilities"] == ["report_save_rich"]
    assert payload["expected_output_artifacts"] == ["html_report_path"]
    assert payload["budget"]["max_tool_calls"] == 2
    assert HandoffEnvelope.from_dict(payload) == envelope


def test_handoff_rejects_private_reasoning_or_full_history():
    payload = {
        "run_id": "run-001",
        "workflow_slug": "demo",
        "task_id": "task-001",
        "sender_role": "sender",
        "receiver_role": "receiver",
        "objective": "Do the task",
        "trace_id": "trace",
        "span_id": "span",
        "history": ["entire chat"],
    }
    with pytest.raises(ValueError, match="forbidden private/history"):
        HandoffEnvelope.from_mapping(payload)


def test_record_handoff_uses_runtime_duck_type_without_reformatting():
    recorded = []
    runtime = SimpleNamespace(record_handoff=recorded.append)
    envelope = HandoffEnvelope.create(
        run_id="run-001",
        workflow_slug="demo",
        task_id="task-001",
        sender_role="sender",
        receiver_role="receiver",
        objective="Do the task",
    )

    assert record_handoff(runtime, envelope) is True
    assert recorded == [envelope]
    assert record_handoff(None, envelope) is False


def test_context_builder_reports_truncation_and_limits_recent_messages():
    envelope = HandoffEnvelope.create(
        run_id="run-001",
        workflow_slug="demo",
        task_id="task-001",
        sender_role="sender",
        receiver_role="receiver",
        objective="Analyze registered artifacts",
    )
    budget = ContextBudget(
        system_policy=12,
        procedure=70,
        run_summary=10,
        tool_schemas=10,
        recent_messages=16,
    )
    context = ContextBuilder(budget).build(
        envelope,
        system_policy="policy " * 50,
        procedure="procedure " * 50,
        run_summary="summary " * 50,
        tool_schemas="schema " * 50,
        recent_messages=[f"message-{index}" for index in range(20)],
    )

    assert set(context.truncated) == set(budget.as_dict())
    assert all(
        context.token_counts[name] <= allocation for name, allocation in budget.as_dict().items()
    )
    assert "message-0" not in context.sections["recent_messages"]
    assert "message-19" in context.sections["recent_messages"]


def test_every_production_role_has_a_canonical_profile_and_enforced_allowlist():
    expected_profiles = {
        "coordinator": "standard",
        "chembl_downloader": "chembl-retrieval",
        "gtm_agent": "gtm-analysis",
        "chemoinformatician": "chemoinformatics",
        "report_generator": "reporting",
        "molecular_designer": "molecular-design",
        "peptide_designer": "peptide-design",
        "synplanner": "retrosynthesis",
        "robustness_evaluation": "robustness",
        "single_agent": "standard",
    }
    assert {role: policy.profile for role, policy in ROLE_POLICIES.items()} == expected_profiles

    allowed_type = type("ChemblToolkit", (), {})
    denied_type = type("SynPlannerToolkit", (), {})
    policy = get_role_policy("chembl_downloader")
    validate_role_tools(policy, [allowed_type()])
    with pytest.raises(ValueError, match="SynPlannerToolkit"):
        validate_role_tools(policy, [denied_type()])

    def dangerous_write():
        return None

    with pytest.raises(ValueError, match="dangerous_write"):
        validate_role_tools(policy, [dangerous_write])

    coordinator_policy = get_role_policy("coordinator")
    validate_role_tools(
        coordinator_policy,
        [
            type("SessionMemoryToolkit", (), {})(),
            type("SkillToolkit", (), {})(),
        ],
    )
    with pytest.raises(ValueError, match="ChemblToolkit"):
        validate_role_tools(coordinator_policy, [allowed_type()])


def test_factory_cannot_construct_a_role_with_a_disallowed_toolkit():
    denied_type = type("SynPlannerToolkit", (), {})

    class MisconfiguredChemblFactory(BaseAgentFactory):
        agent_type = "chembl_downloader"

        def get_agent_config(self):
            return AgentConfig(
                name="misconfigured",
                description="must fail before agent construction",
                tools=[denied_type()],
            )

    with pytest.raises(AgentCreationError, match="SynPlannerToolkit"):
        MisconfiguredChemblFactory().create_agent(
            object(),
            enable_mlflow_tracking=False,
        )
