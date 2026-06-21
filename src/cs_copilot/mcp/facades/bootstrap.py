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
        """Recommend MCP prompts, workflow contracts, skills, and next actions.

        Bootstrap is intentionally an organization step.  Domain-specific
        clarification questions belong to the read-only preflight tools because
        those tools can receive richer session context than bootstrap has.
        """

        request = " ".join(str(user_request or "").split())
        workflow, workflow_error = _select_workflow(request, workflow_slug)

        preflight_tools = _dedupe(
            [
                *(workflow.preflight_tools if workflow else ()),
                *_request_preflight_tools(request),
            ]
        )
        skill_slugs = _select_skills(
            request,
            workflow.slug if workflow else None,
            allow_fallback_search=not preflight_tools,
        )
        required_tools = list(workflow.required_tools) if workflow else []
        optional_tools = list(workflow.optional_tools) if workflow else []
        recommended_next_tools = _recommended_execution_tools(workflow)

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


def _select_skills(
    request: str,
    workflow_slug: str | None,
    *,
    allow_fallback_search: bool = True,
) -> list[str]:
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

    if allow_fallback_search and not slugs and request:
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
        ),
    )


def _request_preflight_tools(request: str) -> list[str]:
    tools: list[str] = []
    if _looks_like_chembl_request(request):
        tools.append("chembl_prepare_retrieval")
    if _looks_like_chemical_space_request(request):
        tools.append("chemspace_plan_analysis")
    return tools


def _recommended_execution_tools(workflow: Any) -> list[str]:
    if not workflow:
        return []
    return _dedupe([*workflow.required_tools, *workflow.optional_tools])


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
