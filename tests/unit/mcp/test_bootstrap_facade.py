"""Tests for MCP bootstrap recommendations."""

from __future__ import annotations

from cs_copilot.mcp.facades.bootstrap import mcp_bootstrap_facade


def test_bootstrap_recommends_chembl_preflight_for_chembl_request():
    result = mcp_bootstrap_facade().bootstrap(
        "Fetch ChEMBL human EGFR binding data, full name epidermal growth "
        "factor receptor, any mechanism"
    )

    assert result["recommended_prompt"] == "cs_copilot_mcp_workflow"
    assert result["recommended_workflow"] == "chembl-target-retrieval"
    assert "chembl-target-retrieval" in result["relevant_skills"]
    assert result["preflight_tools"] == ["chembl_prepare_retrieval"]
    assert result["status"] == "ok"
    assert result["next_action"]["type"] == "fetch_context"
    assert "prompt:cs_copilot_mcp_workflow" in result["next_action"]["ids"]
    assert result["action_plan"][1] == {
        "type": "call_tool",
        "phase": "workflow_start",
        "tool": "workflow_start_run",
        "arguments": {
            "workflow_slug": "chembl-target-retrieval",
            "workflow_inputs": {
                "retrieval_request": (
                    "Fetch ChEMBL human EGFR binding data, full name epidermal "
                    "growth factor receptor, any mechanism"
                )
            },
        },
        "guard": (
            "For a new run, persist the supplied root request input. "
            "When resuming an active run without revalidating inputs, "
            "call with workflow_slug only so pinned constraints, budget, "
            "and workflow inputs are reused."
        ),
    }
    assert result["action_plan"][2]["arguments"] == {"status": "planning"}
    assert result["action_plan"][3]["arguments"] == {"status": "running"}
    assert result["action_plan"][4]["type"] == "call_tool"
    assert result["action_plan"][4]["phase"] == "preflight"
    assert result["action_plan"][4]["tool"] == "chembl_prepare_retrieval"
    assert "policy_results" not in result
    assert (
        "target_specificity"
        in result["clarification_contract"]["do_not_infer_missing_requirements"]
    )
    assert (
        "chembl_fetch_compounds"
        in result["clarification_contract"]["blocked_tools_until_clarified"]
    )


def test_bootstrap_recommends_gtm_workflow_for_activity_landscape():
    result = mcp_bootstrap_facade().bootstrap(
        "Create a GTM activity landscape for the current clean dataset"
    )

    assert result["recommended_workflow"] == "gtm-activity-landscape"
    assert "gtm-activity-landscape" in result["relevant_skills"]
    assert "chemspace_plan_analysis" in result["preflight_tools"]
    assert "gtm_optimization" not in result["preflight_tools"]
    assert "gtm_optimization" in result["recommended_next_tools"]
    assert "workflow:gtm-activity-landscape" in result["fetch_ids"]
    assert result["action_plan"][0]["type"] == "fetch_context"
    assert result["action_plan"][1]["tool"] == "workflow_start_run"
    assert result["action_plan"][4]["tool"] == "chemspace_plan_analysis"


def test_bootstrap_recommends_gtm_density_workflow_for_density_landscape():
    result = mcp_bootstrap_facade().bootstrap(
        "Show the GTM density map for the current clean dataset"
    )

    assert result["recommended_workflow"] == "gtm-density-landscape"
    assert "gtm-density-landscape" in result["relevant_skills"]
    assert "gtm-activity-landscape" not in result["relevant_skills"]
    assert "chemspace_plan_analysis" in result["preflight_tools"]
    assert "gtm_save_density_plot" in result["recommended_next_tools"]
    assert "workflow:gtm-density-landscape" in result["fetch_ids"]


def test_bootstrap_combined_density_activity_fetches_both_gtm_skills():
    result = mcp_bootstrap_facade().bootstrap(
        "Show the density map and create an activity landscape"
    )

    assert result["recommended_workflow"] == "gtm-density-landscape"
    assert result["relevant_skills"] == [
        "gtm-density-landscape",
        "gtm-activity-landscape",
    ]
    assert "skill:gtm-density-landscape" in result["fetch_ids"]
    assert "skill:gtm-activity-landscape" in result["fetch_ids"]


def test_bootstrap_recommends_peptide_skill_for_peptide_design():
    result = mcp_bootstrap_facade().bootstrap(
        "Design antimicrobial peptide candidates and rank them"
    )

    assert result["recommended_workflow"] is None
    assert "peptide-design" in result["relevant_skills"]
    assert "skill:peptide-design" in result["fetch_ids"]


def test_bootstrap_routes_broad_request_to_preflight_not_bootstrap_questions():
    result = mcp_bootstrap_facade().bootstrap("help me analyze compounds")

    assert result["status"] == "ok"
    assert result["bootstrap_questions"] == []
    assert "chemspace_plan_analysis" in result["preflight_tools"]
    assert result["next_action"]["type"] == "fetch_context"
    assert result["next_action"]["ids"] == ["prompt:cs_copilot_mcp_workflow"]
    assert result["action_plan"][1]["tool"] == "chemspace_plan_analysis"
    assert result["clarification_contract"]["source"] == "preflight_tools"


def test_bootstrap_questions_are_limited_to_bootstrap_blockers():
    result = mcp_bootstrap_facade().bootstrap("")

    assert result["status"] == "needs_clarification"
    assert result["next_action"]["type"] == "ask_clarification"
    assert result["next_action"]["phase"] == "bootstrap"
    assert result["bootstrap_questions"] == ["What cs_copilot task should be planned?"]


def test_bootstrap_explicit_workflow_slug_wins():
    result = mcp_bootstrap_facade().bootstrap(
        "use this dataset",
        workflow_slug="dataset-normalization",
    )

    assert result["recommended_workflow"] == "dataset-normalization"
    assert result["recommended_workflow_id"] == "workflow:dataset-normalization"
    assert "Explicit workflow_slug" in " ".join(result["rationale"])


def test_bootstrap_does_not_fabricate_file_backed_workflow_inputs():
    result = mcp_bootstrap_facade().bootstrap(
        "normalize the current CSV dataset",
        workflow_slug="dataset-normalization",
    )

    start = next(
        action for action in result["action_plan"] if action.get("tool") == "workflow_start_run"
    )
    assert start["arguments"] == {"workflow_slug": "dataset-normalization"}


def test_bootstrap_scopes_pilot_calls_to_catalog_tasks_in_dependency_order():
    result = mcp_bootstrap_facade().bootstrap(
        "Retrieve EGFR activity data, build GTM landscapes, and write a report",
        workflow_slug="chembl-to-gtm-report",
    )

    plan = result["action_plan"]
    start_index = next(
        index for index, action in enumerate(plan) if action.get("tool") == "workflow_start_run"
    )
    lifecycle_indices = [
        index
        for index, action in enumerate(plan)
        if action.get("tool") == "workflow_transition_run"
    ]
    task_actions = [action for action in plan if action["type"] == "execute_workflow_task"]

    assert start_index < min(lifecycle_indices)
    assert [plan[index]["arguments"]["status"] for index in lifecycle_indices] == [
        "planning",
        "running",
    ]
    assert [task["task_id"] for task in task_actions] == [
        "chembl-preflight",
        "chembl-retrieval",
        "gtm-preflight",
        "gtm-model",
        "gtm-landscapes",
        "report",
    ]
    assert task_actions[2]["depends_on"] == ["chembl-retrieval"]
    assert task_actions[2]["role"] == "gtm_agent"
    assert task_actions[2]["profile"] == "gtm-analysis"
    assert task_actions[2]["required_tools"] == ["chemspace_plan_analysis"]
    assert "gtm_load_model_only" in task_actions[2]["tool_allowlist"]

    planned_domain_calls = {
        step["tool"]
        for task in task_actions
        for step in task["steps"]
        if step["phase"] in {"preflight", "task_execution"}
    }
    assert planned_domain_calls == set(result["required_tools"]) | set(result["preflight_tools"])
    assert not any(action["type"] == "call_tools_after_preflight" for action in plan)


def test_bootstrap_task_handoffs_are_structured_bounded_and_first():
    result = mcp_bootstrap_facade().bootstrap(
        "Retrieve EGFR activity data, build GTM landscapes, and write a report",
        workflow_slug="chembl-to-gtm-report",
    )

    tasks = [
        action for action in result["action_plan"] if action["type"] == "execute_workflow_task"
    ]
    for task in tasks:
        handoff = task["steps"][0]
        arguments = handoff["arguments"]
        assert handoff["tool"] == "workflow_record_handoff"
        assert handoff["phase"] == "task_handoff"
        assert arguments["task_id"] == task["task_id"]
        assert arguments["sender_role"] in {"supervisor", "coordinator"}
        assert arguments["receiver_role"] == task["role"]
        assert arguments["required_capabilities"] == task["required_tools"]
        assert arguments["input_artifact_contracts"] == task["input_artifacts"]
        assert arguments["expected_output_artifacts"] == task["output_artifacts"]
        assert set(arguments["expected_output_schema"]["properties"]) == set(
            task["output_artifacts"]
        )
        assert arguments["expected_output_schema"]["type"] == "object"
        assert arguments["expected_output_schema"]["additionalProperties"] is False
        assert arguments["acceptance_criteria"] == task["acceptance_criteria"]
        assert arguments["objective"]
        assert arguments["context_summary"]
        assert 0 < arguments["budget"]["max_tokens"] <= 8_000
        assert 0 < arguments["budget"]["max_tool_calls"] <= 24
        assert 0 < arguments["budget"]["timeout_seconds"] <= 900
        assert {"run_id", "workflow_slug", "trace_id", "span_id"}.isdisjoint(arguments)

        running_index = next(
            index
            for index, step in enumerate(task["steps"])
            if step.get("tool") == "workflow_transition_task"
            and step["arguments"]["status"] == "running"
        )
        assert running_index == 1


def test_bootstrap_places_each_preflight_after_its_task_running_transition():
    result = mcp_bootstrap_facade().bootstrap(
        "Retrieve EGFR activity data, build GTM landscapes, and write a report",
        workflow_slug="chembl-to-gtm-report",
    )

    tasks = {
        action["task_id"]: action
        for action in result["action_plan"]
        if action["type"] == "execute_workflow_task"
    }
    for task_id, preflight_tool in (
        ("chembl-preflight", "chembl_prepare_retrieval"),
        ("gtm-preflight", "chemspace_plan_analysis"),
    ):
        steps = tasks[task_id]["steps"]
        running_index = next(
            index
            for index, step in enumerate(steps)
            if step.get("tool") == "workflow_transition_task"
            and step["arguments"]["status"] == "running"
        )
        preflight_index = next(
            index for index, step in enumerate(steps) if step.get("tool") == preflight_tool
        )
        assert running_index < preflight_index
