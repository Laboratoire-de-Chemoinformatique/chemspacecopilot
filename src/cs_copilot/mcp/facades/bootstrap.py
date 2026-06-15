"""Bootstrap recommendations for external MCP clients."""

from __future__ import annotations

import functools
from typing import Any

_MCP_PROMPT = "cs_copilot_mcp_workflow"


class MCPBootstrapFacade:
    """Return first-step orchestration guidance without mutating session state."""

    def bootstrap(
        self,
        user_request: str,
        workflow_slug: str | None = None,
    ) -> dict[str, Any]:
        """Recommend MCP prompts, workflow contracts, skills, and next actions."""

        request = " ".join(str(user_request or "").split())
        workflow, workflow_error = _select_workflow(request, workflow_slug)
        skill_slugs = _select_skills(request, workflow.slug if workflow else None)

        chembl_decision = None
        chemical_space_decision = None
        if _looks_like_chembl_request(request):
            from cs_copilot.workflows import prepare_chembl_retrieval

            chembl_decision = prepare_chembl_retrieval(request)
        if _looks_like_chemical_space_request(request):
            from cs_copilot.workflows import plan_chemical_space_analysis

            chemical_space_decision = plan_chemical_space_analysis(request)

        preflight_tools = _dedupe(
            [
                *(workflow.preflight_tools if workflow else ()),
                *(
                    _policy_preflight_tool(
                        chembl_decision,
                        tool_name="chembl_prepare_retrieval",
                    )
                ),
                *(
                    _policy_preflight_tool(
                        chemical_space_decision,
                        tool_name="chemspace_plan_analysis",
                    )
                ),
            ]
        )
        recommended_next_tools = _dedupe(
            [
                *_policy_recommended_tools(chembl_decision),
                *_policy_recommended_tools(chemical_space_decision),
            ]
        )
        required_tools = list(workflow.required_tools) if workflow else []
        optional_tools = list(workflow.optional_tools) if workflow else []

        questions = _dedupe(
            [
                *(chembl_decision or {}).get("clarifying_questions", []),
                *(chemical_space_decision or {}).get("clarifying_questions", []),
                *(["Which workflow should be used?"] if workflow_error else []),
                *(["What cs_copilot task should be planned?"] if not request else []),
            ]
        )
        status = "needs_clarification" if questions else "ok"
        next_action = _next_action(
            status=status,
            questions=questions,
            workflow=workflow,
            skill_slugs=skill_slugs,
            preflight_tools=preflight_tools,
        )

        fetch_ids = _dedupe(
            [
                f"prompt:{_MCP_PROMPT}",
                *(["workflow:" + workflow.slug] if workflow else []),
                *[f"skill:{slug}" for slug in skill_slugs],
            ]
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
            "next_action": next_action,
            "rationale": _rationale(workflow, skill_slugs, workflow_slug, workflow_error),
        }
        policy_results: dict[str, Any] = {}
        if chembl_decision is not None:
            policy_results["chembl_prepare_retrieval"] = chembl_decision
        if chemical_space_decision is not None:
            policy_results["chemspace_plan_analysis"] = chemical_space_decision
        if policy_results:
            payload["policy_results"] = policy_results
        if workflow_error:
            payload["warning"] = workflow_error
        return payload


def _select_workflow(request: str, workflow_slug: str | None):
    from cs_copilot.workflows import get_workflow, search_workflows

    if workflow_slug:
        try:
            return get_workflow(workflow_slug), None
        except KeyError as exc:
            return None, str(exc)

    lower = request.lower()
    explicit_slug = _workflow_slug_from_terms(lower)
    if explicit_slug:
        return get_workflow(explicit_slug), None

    results = search_workflows(request, limit=1) if request else []
    if results and _search_result_is_useful(request, results[0].slug):
        return results[0], None
    return None, None


def _workflow_slug_from_terms(lower: str) -> str | None:
    if _contains_any(lower, ("robustness", "prompt variation", "test run")):
        return "robustness-report"
    if _contains_any(lower, ("normalize", "normalise", "standardize dataset", "uploaded")):
        return "dataset-normalization"
    if _contains_any(lower, ("retrosynthesis", "synthesis", "synthetic route")):
        return "retrosynthesis-for-candidates"
    if _contains_any(lower, ("chembl", "bioactivity", "assay", "activity data")):
        if _contains_any(lower, ("gtm", "map", "landscape", "report")):
            return "chembl-to-gtm-report"
        return "chembl-target-retrieval"
    if _contains_any(lower, ("activity landscape", "gtm", "density map", "project")):
        return "gtm-activity-landscape"
    if _contains_any(lower, ("candidate", "analog", "analogue", "design")) and _contains_any(
        lower, ("gtm", "project", "map")
    ):
        return "candidate-design-to-gtm"
    return None


def _select_skills(request: str, workflow_slug: str | None) -> list[str]:
    from cs_copilot.skills import get_skill, search_skills

    lower = request.lower()
    slugs: list[str] = []
    if workflow_slug:
        try:
            get_skill(workflow_slug)
        except KeyError:
            pass
        else:
            slugs.append(workflow_slug)

    if _contains_any(lower, ("peptide", "amino acid", "amp", "antimicrobial")):
        slugs.append("peptide-design")
    elif _contains_any(lower, ("design", "generate", "analog", "analogue", "candidate")):
        slugs.append("molecular-design")

    if _contains_any(lower, ("report", "summary")):
        slugs.append("report-generation")
    if _contains_any(lower, ("retrosynthesis", "synthesis", "synthetic route")):
        slugs.append("retrosynthesis-planning")

    if not slugs and request:
        slugs.extend(spec.slug for spec in search_skills(request, limit=2))
    return _dedupe(slugs)


def _looks_like_chembl_request(request: str) -> bool:
    lower = request.lower()
    return bool(request) and _contains_any(
        lower,
        ("chembl", "bioactivity", "assay", "activity data", "fetch", "retrieve", "download"),
    )


def _looks_like_chemical_space_request(request: str) -> bool:
    lower = request.lower()
    return bool(request) and _contains_any(
        lower,
        (
            "chemical space",
            "compound",
            "compounds",
            "gtm",
            "map",
            "landscape",
            "density",
            "project",
            "scaffold",
            "sar",
            "report",
            "chembl",
        ),
    )


def _policy_preflight_tool(decision: dict[str, Any] | None, *, tool_name: str) -> list[str]:
    if not decision:
        return []
    return [tool_name]


def _policy_recommended_tools(decision: dict[str, Any] | None) -> list[str]:
    if not decision:
        return []
    tools = [str(tool) for tool in decision.get("recommended_next_tools", []) if tool]
    tool = decision.get("recommended_next_tool")
    if tool:
        tools.append(str(tool))
    return tools


def _next_action(
    *,
    status: str,
    questions: list[str],
    workflow: Any,
    skill_slugs: list[str],
    preflight_tools: list[str],
) -> dict[str, Any]:
    if status == "needs_clarification":
        return {
            "type": "ask_clarification",
            "questions": questions,
        }
    if preflight_tools:
        return {
            "type": "call_tool",
            "tool": preflight_tools[0],
        }
    fetch_ids = _dedupe(
        [
            *(["workflow:" + workflow.slug] if workflow else []),
            *[f"skill:{slug}" for slug in skill_slugs],
        ]
    )
    return {
        "type": "fetch_context" if fetch_ids else "answer_directly",
        "ids": fetch_ids,
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
    return notes


def _search_result_is_useful(request: str, slug: str) -> bool:
    lower = request.lower()
    if slug == "candidate-design-to-gtm" and not _contains_any(
        lower, ("gtm", "project", "projection", "map")
    ):
        return False
    terms = set(lower.replace("-", " ").split())
    slug_terms = set(slug.replace("-", " ").split())
    return bool(terms & slug_terms)


def _contains_any(value: str, terms: tuple[str, ...]) -> bool:
    return any(term in value for term in terms)


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


@functools.lru_cache(maxsize=1)
def mcp_bootstrap_facade() -> MCPBootstrapFacade:
    return MCPBootstrapFacade()
