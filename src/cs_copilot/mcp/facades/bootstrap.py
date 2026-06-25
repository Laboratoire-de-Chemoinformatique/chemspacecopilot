"""Bootstrap recommendations for external MCP clients."""

from __future__ import annotations

import functools
from typing import Any

from cs_copilot.routing import RoutingResult, match_request

_MCP_PROMPT = "cs_copilot_mcp_workflow"


class MCPBootstrapFacade:
    """Return first-step orchestration guidance without mutating session state."""

    def bootstrap(
        self,
        user_request: str,
        workflow_slug: str | None = None,
    ) -> dict[str, Any]:
        """Recommend MCP prompts, workflow contracts, skills, and next actions.

        Bootstrap is intentionally an organization step.  Domain-specific
        clarification questions belong to the read-only preflight tools because
        those tools can receive richer session context than bootstrap has.

        Workflow / skill / preflight selection is delegated to
        :func:`cs_copilot.routing.match_request`, the single source of truth
        shared with the Agno team's routing prose.
        """

        request = " ".join(str(user_request or "").split())
        routing: RoutingResult = match_request(request, workflow_slug=workflow_slug)
        workflow = routing.workflow
        workflow_error = routing.workflow_error

        preflight_tools = list(routing.preflight_tools)
        skill_slugs = list(routing.skills)
        required_tools = list(workflow.required_tools) if workflow else []
        optional_tools = list(workflow.optional_tools) if workflow else []
        recommended_next_tools = _dedupe([*required_tools, *optional_tools])

        questions = _dedupe(
            [
                *(["Which workflow should be used?"] if workflow_error else []),
                *(["What cs_copilot task should be planned?"] if not request else []),
            ]
        )
        status = "needs_clarification" if questions else "ok"

        fetch_ids = _dedupe(
            [
                f"prompt:{_MCP_PROMPT}",
                *(["workflow:" + workflow.slug] if workflow else []),
                *[f"skill:{slug}" for slug in skill_slugs],
            ]
        )
        action_plan = _action_plan(
            status=status,
            questions=questions,
            fetch_ids=fetch_ids,
            preflight_tools=preflight_tools,
            required_tools=required_tools,
        )
        payload: dict[str, Any] = {
            "status": status,
            "recommended_prompt": _MCP_PROMPT,
            "recommended_prompt_id": f"prompt:{_MCP_PROMPT}",
            "recommended_workflow": workflow.slug if workflow else None,
            "recommended_workflow_id": f"workflow:{workflow.slug}" if workflow else None,
            "domain_prompt": workflow.recommended_prompt if workflow else None,
            "domain_prompt_id": (
                f"prompt:{workflow.recommended_prompt}"
                if workflow and workflow.recommended_prompt
                else None
            ),
            "relevant_skills": skill_slugs,
            "relevant_skill_ids": [f"skill:{slug}" for slug in skill_slugs],
            "preflight_tools": preflight_tools,
            "recommended_next_tools": recommended_next_tools,
            "required_tools": required_tools,
            "optional_tools": optional_tools,
            "fetch_ids": fetch_ids,
            "next_action": action_plan[0],
            "action_plan": action_plan,
            "bootstrap_questions": questions,
            "clarification_contract": _clarification_contract(
                bootstrap_questions=questions,
                preflight_tools=preflight_tools,
                blocked_tools=[*required_tools, *optional_tools],
            ),
            "rationale": _rationale(workflow, skill_slugs, workflow_slug, workflow_error),
        }
        if workflow_error:
            payload["warning"] = workflow_error
        return payload


def _action_plan(
    *,
    status: str,
    questions: list[str],
    fetch_ids: list[str],
    preflight_tools: list[str],
    required_tools: list[str],
) -> list[dict[str, Any]]:
    if status == "needs_clarification":
        return [
            {
                "type": "ask_clarification",
                "phase": "bootstrap",
                "questions": questions,
            }
        ]

    actions: list[dict[str, Any]] = []
    if fetch_ids:
        actions.append({"type": "fetch_context", "ids": fetch_ids})
    if preflight_tools:
        actions.extend(
            {
                "type": "call_tool",
                "phase": "preflight",
                "tool": tool,
                "guard": "If needs_clarification=true, ask the returned questions before write tools.",
            }
            for tool in preflight_tools
        )
    if required_tools:
        actions.append(
            {
                "type": "call_tools_after_preflight",
                "tools": required_tools,
            }
        )
    return actions or [{"type": "answer_directly"}]


def _clarification_contract(
    *,
    bootstrap_questions: list[str],
    preflight_tools: list[str],
    blocked_tools: list[str],
) -> dict[str, Any]:
    source = "bootstrap" if bootstrap_questions else "preflight_tools"
    return {
        "source": source if preflight_tools or bootstrap_questions else "none",
        "ask_user_when": (
            "A preflight tool returns needs_clarification=true, or bootstrap "
            "status is needs_clarification."
        ),
        "do_not_infer_missing_requirements": [
            "target_specificity",
            "target_confirmation",
            "organism",
            "assay_types",
            "mechanism",
            "analysis_intent",
            "data_source",
        ],
        "blocked_tools_until_clarified": _dedupe(blocked_tools),
    }


def _rationale(
    workflow: Any,
    skill_slugs: list[str],
    workflow_slug: str | None,
    workflow_error: str | None,
) -> list[str]:
    notes = [f"Use prompt:{_MCP_PROMPT} for MCP-native orchestration."]
    if workflow_slug and workflow:
        notes.append(f"Explicit workflow_slug selected workflow:{workflow.slug}.")
    elif workflow:
        notes.append(f"Matched request to workflow:{workflow.slug}.")
    elif workflow_error:
        notes.append("The requested workflow_slug did not match a known workflow.")
    if skill_slugs:
        notes.append("Fetch relevant skill procedures before mutating tools.")
    if workflow and workflow.preflight_tools:
        notes.append("Run read-only preflight tools before write tools.")
    return notes


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


@functools.lru_cache(maxsize=1)
def mcp_bootstrap_facade() -> MCPBootstrapFacade:
    return MCPBootstrapFacade()
