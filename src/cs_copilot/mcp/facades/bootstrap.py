"""Bootstrap recommendations for external MCP clients."""

from __future__ import annotations

from typing import Any

from cs_copilot.routing import RoutingResult, match_request

_MCP_PROMPT = "cs_copilot_mcp_workflow"
_DEFAULT_HANDOFF_MAX_TOKENS = 8_000
_DEFAULT_HANDOFF_MAX_TOOL_CALLS = 24
_DEFAULT_HANDOFF_TIMEOUT_SECONDS = 900


class MCPBootstrapFacade:
    """Return first-step orchestration guidance without mutating session state."""

    def bootstrap(
        self,
        user_request: str,
        workflow_slug: str | None = None,
    ) -> dict[str, Any]:
        """Recommend MCP prompts, workflow contracts, skills, and next actions.

        Bootstrap is intentionally an organization step.  Domain-specific
        clarification questions belong to the scientific preflight tools because
        those tools can receive richer session context than bootstrap has.  Some
        preflights also persist their resulting task plan as a session artifact.

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
            workflow=workflow,
            user_request=request,
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
    workflow: Any,
    user_request: str,
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
    if workflow is not None:
        start_arguments: dict[str, Any] = {"workflow_slug": workflow.slug}
        request_inputs = {
            contract.name: user_request
            for contract in workflow.input_artifacts
            if contract.required and contract.kind == "request"
        }
        if request_inputs:
            start_arguments["workflow_inputs"] = request_inputs
        actions.extend(
            [
                {
                    "type": "call_tool",
                    "phase": "workflow_start",
                    "tool": "workflow_start_run",
                    "arguments": start_arguments,
                    "guard": (
                        "For a new run, persist the supplied root request input. "
                        "When resuming an active run without revalidating inputs, "
                        "call with workflow_slug only so pinned constraints, budget, "
                        "and workflow inputs are reused."
                    ),
                },
                {
                    "type": "call_tool",
                    "phase": "workflow_lifecycle",
                    "tool": "workflow_transition_run",
                    "arguments": {"status": "planning"},
                    "guard": (
                        "Call when the started run is submitted or input_required; "
                        "skip when an idempotently reused run is already planning or running."
                    ),
                },
                {
                    "type": "call_tool",
                    "phase": "workflow_lifecycle",
                    "tool": "workflow_transition_run",
                    "arguments": {"status": "running"},
                    "guard": (
                        "Call when the run is planning; skip when an idempotently "
                        "reused run is already running."
                    ),
                },
            ]
        )
        if workflow.tasks:
            actions.extend(_task_actions(workflow))
            actions.append(
                {
                    "type": "call_tool",
                    "phase": "workflow_completion",
                    "tool": "workflow_complete_run",
                    "guard": (
                        "Call only after every required task output is registered "
                        "and its acceptance criteria are satisfied."
                    ),
                }
            )
            return actions

    # Workflows without a declared task DAG retain their catalog-level plan.
    # Once a workflow declares tasks, every domain call is emitted inside its
    # task action above and this legacy branch is deliberately skipped.
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


def _task_actions(workflow: Any) -> list[dict[str, Any]]:
    preflight_tools = frozenset(workflow.preflight_tools)
    optional_tools = list(workflow.optional_tools)
    actions: list[dict[str, Any]] = []
    for task in _topological_tasks(workflow.tasks):
        required_tools = list(task.required_tools)
        tool_allowlist = _effective_tool_allowlist(
            task=task,
            required_tools=required_tools,
            optional_tools=optional_tools,
        )
        steps: list[dict[str, Any]] = [
            _handoff_action(
                workflow=workflow,
                task=task,
                required_tools=required_tools,
                tool_allowlist=tool_allowlist,
            ),
            {
                "type": "call_tool",
                "phase": "task_lifecycle",
                "tool": "workflow_transition_task",
                "arguments": {"task_id": task.task_id, "status": "running"},
                "guard": (
                    "The workflow run must be running and every declared "
                    "dependency must be completed."
                ),
            },
        ]
        for tool in required_tools:
            phase = "preflight" if tool in preflight_tools else "task_execution"
            step: dict[str, Any] = {
                "type": "call_tool",
                "phase": phase,
                "tool": tool,
                "task_id": task.task_id,
            }
            if phase == "preflight":
                step["guard"] = (
                    "The declared task must be running. If needs_clarification=true, "
                    "ask the returned questions before write tools."
                )
            steps.append(step)
        steps.append(
            {
                "type": "call_tool",
                "phase": "task_lifecycle",
                "tool": "workflow_transition_task",
                "arguments": {"task_id": task.task_id, "status": "completed"},
                "guard": (
                    "Complete only after declared output artifacts are registered "
                    "and acceptance criteria are satisfied."
                ),
            }
        )
        actions.append(
            {
                "type": "execute_workflow_task",
                "phase": "task",
                "task_id": task.task_id,
                "role": task.role,
                "profile": task.profile,
                "depends_on": list(task.depends_on),
                "tool_allowlist": tool_allowlist,
                "required_tools": required_tools,
                "input_artifacts": list(task.input_artifacts),
                "output_artifacts": list(task.output_artifacts),
                "acceptance_criteria": list(task.acceptance_criteria),
                "steps": steps,
            }
        )
    return actions


def _handoff_action(
    *,
    workflow: Any,
    task: Any,
    required_tools: list[str],
    tool_allowlist: list[str],
) -> dict[str, Any]:
    criteria = list(task.acceptance_criteria)
    objective = f"Execute catalog task {task.task_id!r} for workflow {workflow.slug!r}."
    if criteria:
        objective = f"{objective} Acceptance target: {criteria[0]}"
    dependencies = list(task.depends_on)
    sender_role = _handoff_sender_role(task.role)
    max_tool_calls = min(
        _DEFAULT_HANDOFF_MAX_TOOL_CALLS,
        max(1, len(tool_allowlist) + 2),
    )
    dependency_summary = ", ".join(dependencies) if dependencies else "none; this is a root task"
    return {
        "type": "call_tool",
        "phase": "task_handoff",
        "tool": "workflow_record_handoff",
        "arguments": {
            "task_id": task.task_id,
            "sender_role": sender_role,
            "receiver_role": task.role,
            "objective": objective,
            "constraints": [
                "Use only the task tool allowlist and registered run artifacts.",
                "Do not include conversation history or private reasoning in the handoff.",
            ],
            "required_capabilities": required_tools,
            "input_artifact_contracts": list(task.input_artifacts),
            "expected_output_artifacts": list(task.output_artifacts),
            "expected_output_schema": _expected_output_schema(workflow, task),
            "acceptance_criteria": criteria,
            "context_summary": (
                f"Catalog task {task.task_id!r}; declared dependencies: "
                f"{dependency_summary}. Resolve inputs from registered artifacts."
            ),
            "budget": {
                "max_tokens": _DEFAULT_HANDOFF_MAX_TOKENS,
                "max_tool_calls": max_tool_calls,
                "timeout_seconds": _DEFAULT_HANDOFF_TIMEOUT_SECONDS,
            },
        },
    }


def _handoff_sender_role(receiver_role: str) -> str:
    """Keep the coordinator distinct from the role receiving the handoff."""

    return "coordinator" if receiver_role == "supervisor" else "supervisor"


def _expected_output_schema(workflow: Any, task: Any) -> dict[str, Any]:
    contracts = {
        contract.name: contract
        for contract in workflow.output_artifacts
        if contract.name in task.output_artifacts
    }
    properties = {
        name: {
            "type": "string",
            "description": f"Registered artifact ID for the {name!r} contract.",
            "x-artifact-kind": contracts[name].kind,
        }
        for name in task.output_artifacts
    }
    required = [
        name for name in task.output_artifacts if name in contracts and contracts[name].required
    ]
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _effective_tool_allowlist(
    *,
    task: Any,
    required_tools: list[str],
    optional_tools: list[str],
) -> list[str]:
    """Combine task requirements with role/profile-compatible optional tools."""

    from cs_copilot.mcp.tools_registry import all_specs

    specs = {spec.mcp_name: spec for spec in all_specs()}
    allowed = list(required_tools)
    for tool_name in optional_tools:
        spec = specs.get(tool_name)
        if spec is not None and task.role in spec.roles and task.profile in spec.profiles:
            allowed.append(tool_name)
    return _dedupe(allowed)


def _topological_tasks(tasks: Any) -> list[Any]:
    """Return stable dependency order for a registry-validated task DAG."""

    ordered: list[Any] = []
    completed: set[str] = set()
    remaining = list(tasks)
    while remaining:
        ready = [task for task in remaining if set(task.depends_on).issubset(completed)]
        if not ready:  # pragma: no cover - rejected while loading the catalog
            raise ValueError("Workflow task DAG has no dependency-ready task")
        for task in ready:
            ordered.append(task)
            completed.add(task.task_id)
            remaining.remove(task)
    return ordered


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
        notes.append(
            "Run declared preflight tools before downstream tools; some preflights "
            "persist plan artifacts."
        )
    return notes


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def mcp_bootstrap_facade() -> MCPBootstrapFacade:
    return MCPBootstrapFacade()
