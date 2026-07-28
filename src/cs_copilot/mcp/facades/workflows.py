"""Workflow policy and catalog facades for MCP tool registration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, List, Optional

_REFERENCE_ONLY_INPUT_KIND_TOKENS = frozenset(
    {
        "dataset",
        "descriptor",
        "file",
        "model",
        "path",
        "report",
        "table",
        "visualization",
    }
)


class WorkflowPolicyFacade:
    """Workflow preflight helpers for external MCP reasoners."""

    def prepare_chembl_retrieval(
        self,
        target: str | None = None,
        target_type: str | None = None,
        organism: str | None = None,
        assay_types: list[str] | None = None,
        mechanism: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Validate the ChEMBL retrieval dimensions you decided with the user.

        You are the reasoning engine: extract the target (gene symbol, protein
        name, ChEMBL id, or organism-level target), organism, assay types, and
        mechanism from the request and pass them here. This gate only checks
        completeness and returns clarifying questions for anything missing — do
        not infer fields just to make it pass.
        """
        from cs_copilot.workflows import prepare_chembl_retrieval

        return prepare_chembl_retrieval(
            target=target,
            target_type=target_type,
            organism=organism,
            assay_types=assay_types,
            mechanism=mechanism,
            notes=notes,
        )

    def plan_chemical_space_analysis(
        self,
        analysis_intents: list[str] | None = None,
        dataset_source: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Validate a chemical-space analysis plan you classified for the user.

        Pass the analysis intents (e.g. chembl_retrieval, gtm_build,
        activity_landscape, report_generation) and the dataset source
        (session_clean_dataset, explicit_path, uploaded_dataset, or
        chembl_retrieval). This gate checks both are present and maps the intents
        to recommended execution tools.
        """
        from cs_copilot.workflows import plan_chemical_space_analysis

        return plan_chemical_space_analysis(
            analysis_intents=analysis_intents,
            dataset_source=dataset_source,
            notes=notes,
        )


class WorkflowCatalogFacade:
    """Direct MCP access to reusable workflow contracts."""

    def list(self, include_content: bool = False) -> List[dict[str, Any]]:
        """List reusable cs_copilot workflow contracts."""
        from cs_copilot.workflows import list_workflows

        return [spec.as_dict(include_content=include_content) for spec in list_workflows()]

    def search(
        self,
        query: str,
        limit: int = 10,
        include_content: bool = False,
    ) -> List[dict[str, Any]]:
        """Search reusable cs_copilot workflow contracts."""
        from cs_copilot.workflows import search_workflows

        return [
            spec.as_dict(include_content=include_content)
            for spec in search_workflows(query, limit=limit)
        ]

    def fetch(self, slug: str, include_content: bool = True) -> dict[str, Any]:
        """Fetch one reusable cs_copilot workflow contract by slug."""
        from cs_copilot.workflows import get_workflow

        return get_workflow(slug).as_dict(include_content=include_content)


class WorkflowRuntimeFacade:
    """MCP-safe lifecycle and artifact operations for the active v2 run."""

    def start_run(
        self,
        workflow_slug: str,
        constraints: dict[str, Any] | None = None,
        budget: dict[str, Any] | None = None,
        workflow_inputs: dict[str, Any] | None = None,
        agent: Optional[Any] = None,
        session_state: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Start or reuse a catalog workflow run for the active MCP profile.

        Omit constraints, budget, and workflow_inputs to retain the immutable
        values already bound to an active run. Supplying an empty mapping is an
        explicit request for an empty value and therefore conflicts with a
        non-empty active value. ``workflow_inputs`` maps a pinned input contract
        name to either a JSON value or ``{"artifact_id": "..."}``.
        """

        from cs_copilot.mcp.profiles import (
            get_profile,
            validate_pinned_workflow_profile,
            validate_workflow_profile,
        )
        from cs_copilot.storage import sanitize_workflow_slug
        from cs_copilot.workflows import RunContext, RunStatus, get_workflow

        state = self._session_state(agent, session_state)
        profile_name = state.get("mcp_profile")
        if not profile_name:
            raise ValueError("session_state['mcp_profile'] is required to start a workflow")
        profile = get_profile(str(profile_name))
        requested_slug = sanitize_workflow_slug(workflow_slug)

        runtime = self._optional_runtime(agent, state)
        run = runtime.run if runtime is not None else None
        terminal = {
            RunStatus.COMPLETED,
            RunStatus.PARTIAL,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
        if run is not None and run.workflow_slug == requested_slug and run.status not in terminal:
            from cs_copilot.mcp.context import restore_active_task_scope

            validate_pinned_workflow_profile(profile, run.workflow_contract)
            requested_constraints = (
                dict(run.constraints) if constraints is None else dict(constraints)
            )
            requested_budget = dict(run.budget) if budget is None else dict(budget)
            if run.constraints != requested_constraints or run.budget != requested_budget:
                raise ValueError(
                    f"Active workflow run {run.run_id!r} uses different constraints or "
                    "budget; finish it before starting a run with changed contract inputs."
                )
            self._apply_workflow_inputs(
                runtime,
                workflow_inputs,
                reused=True,
            )
            materialized = self._materialize_catalog_tasks(runtime)
            runtime.bind_session_state(state)
            restore_active_task_scope(state, runtime.run)
            if agent is not None:
                agent.run_context = runtime
            return self._start_run_payload(
                runtime,
                profile=profile.name,
                reused=True,
                materialized_task_ids=materialized,
            )

        validate_workflow_profile(profile, requested_slug)
        workflow = get_workflow(requested_slug)
        requested_constraints = dict(constraints) if constraints is not None else {}
        requested_budget = dict(budget) if budget is not None else {}
        self._validate_workflow_inputs(workflow.as_dict(), workflow_inputs)
        if (
            run is not None
            and run.status not in terminal
            and not self._is_untouched_placeholder(run)
        ):
            raise ValueError(
                f"Cannot start workflow {workflow.slug!r} while nonterminal catalog run "
                f"{run.run_id!r} ({run.workflow_slug!r}, status={run.status.value!r}) "
                "is active. Complete, fail, or cancel the active run first."
            )

        session_id = run.session_id if run is not None else None
        runtime = RunContext.create(
            workflow.slug,
            session_state=state,
            session_id=session_id,
            constraints=requested_constraints,
            budget=requested_budget,
        )
        self._apply_workflow_inputs(
            runtime,
            workflow_inputs,
            reused=False,
        )
        materialized = self._materialize_catalog_tasks(runtime)
        from cs_copilot.mcp.context import clear_active_task_scope

        clear_active_task_scope(state)
        if agent is not None:
            agent.run_context = runtime
        return self._start_run_payload(
            runtime,
            profile=profile.name,
            reused=False,
            materialized_task_ids=materialized,
        )

    def get_run(
        self,
        agent: Optional[Any] = None,
        session_state: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Return the active run manifest snapshot."""

        return self._runtime(agent, session_state).manifest_payload()

    def abandon_tool_invocation(
        self,
        span_id: str,
        reason: str,
        confirm_not_running: bool = False,
        agent: Optional[Any] = None,
        session_state: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Reconcile one durable tool span only after confirming its worker stopped.

        This is a supervisor-only crash-recovery operation. It releases the
        lifecycle guard for a domain invocation whose durable ``started`` event
        has no terminal event. The caller must first establish out of band that
        no process or worker is still executing the span.
        """

        if confirm_not_running is not True:
            raise ValueError(
                "confirm_not_running=true is required after verifying that no "
                "process or worker is still executing this span"
            )
        from cs_copilot.mcp.manifests import is_tool_span_active

        if is_tool_span_active(span_id):
            raise ValueError(f"MCP invocation {span_id!r} is still active in this server process")
        return (
            self._runtime(agent, session_state)
            .abandon_tool_invocation(span_id, reason=reason)
            .to_dict()
        )

    def add_task(
        self,
        task_id: str,
        role: str,
        profile: str,
        step: str,
        input_artifact_ids: list[str] | None = None,
        agent: Optional[Any] = None,
        session_state: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Add one role/profile-scoped task to the active run."""

        from cs_copilot.workflows import TaskRecord

        task = TaskRecord(
            task_id=task_id,
            role=role,
            profile=profile,
            step=step,
            input_artifact_ids=list(input_artifact_ids or ()),
        )
        return self._runtime(agent, session_state).add_task(task).to_dict()

    def transition_run(
        self,
        status: str,
        reason: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        error_retryable: bool = False,
        agent: Optional[Any] = None,
        session_state: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Apply one validated run lifecycle transition."""

        from cs_copilot.mcp.context import (
            clear_active_task_scope,
            restore_active_task_scope,
        )
        from cs_copilot.workflows import RunStatus

        error = self._error(error_code, error_message, error_retryable)
        state = self._session_state(agent, session_state)
        runtime = self._runtime(agent, state)
        run = runtime.transition_run(
            status,
            reason=reason,
            error=error,
        )
        if run.status is RunStatus.RUNNING:
            restore_active_task_scope(state, run)
        elif run.status in {
            RunStatus.INPUT_REQUIRED,
            RunStatus.COMPLETED,
            RunStatus.PARTIAL,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }:
            clear_active_task_scope(state)
        return run.to_dict()

    def transition_task(
        self,
        task_id: str,
        status: str,
        error_code: str | None = None,
        error_message: str | None = None,
        error_retryable: bool = False,
        agent: Optional[Any] = None,
        session_state: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Apply one validated task lifecycle transition."""

        from cs_copilot.mcp.context import (
            bind_active_task_scope,
            clear_active_task_scope,
        )
        from cs_copilot.workflows import TaskStatus

        error = self._error(error_code, error_message, error_retryable)
        state = self._session_state(agent, session_state)
        runtime = self._runtime(agent, state)
        target = TaskStatus(status)
        existing = runtime.run.tasks.get(task_id)
        if (
            target is TaskStatus.RUNNING
            and existing is not None
            and existing.status is TaskStatus.RUNNING
            and error is None
        ):
            # Reactivating an already-running task is a process-local scope
            # selection, not a second durable lifecycle transition.
            runtime.verify_task_inputs(task_id)
            task = existing
        else:
            task = runtime.transition_task(task_id, target, error=error)
        if task.status.value == "running":
            bind_active_task_scope(state, task, run=runtime.run)
        elif task.status.value in {
            "input_required",
            "completed",
            "failed",
            "cancelled",
            "skipped",
        }:
            clear_active_task_scope(state, task_id=task.task_id)
        return task.to_dict()

    def record_handoff(
        self,
        task_id: str,
        sender_role: str,
        receiver_role: str,
        objective: str,
        constraints: list[str] | None = None,
        required_capabilities: list[str] | None = None,
        acceptance_criteria: list[str] | None = None,
        input_artifact_contracts: list[str] | None = None,
        input_artifact_ids: list[str] | None = None,
        expected_output_artifacts: list[str] | None = None,
        expected_output_schema: dict[str, Any] | None = None,
        context_summary: str | None = None,
        budget: dict[str, Any] | None = None,
        parent_span_id: str | None = None,
        agent: Optional[Any] = None,
        session_state: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Validate and persist a minimal structured role handoff."""

        from cs_copilot.workflows import HandoffEnvelope

        runtime = self._runtime(agent, session_state)
        run = runtime.run
        if run is None:  # pragma: no cover - guarded by RunContext
            raise RuntimeError("workflow run is not initialized")
        envelope = HandoffEnvelope.create(
            run_id=run.run_id,
            workflow_slug=run.workflow_slug,
            task_id=task_id,
            sender_role=sender_role,
            receiver_role=receiver_role,
            objective=objective,
            constraints=constraints or (),
            required_capabilities=required_capabilities or (),
            acceptance_criteria=acceptance_criteria or (),
            input_artifact_ids=input_artifact_ids or (),
            expected_output_artifacts=expected_output_artifacts or (),
            expected_output_schema=expected_output_schema or {},
            context_summary=context_summary,
            budget=budget or {},
            trace_id=run.trace_id,
            parent_span_id=parent_span_id,
        )
        return runtime.record_handoff(
            envelope,
            input_artifact_contracts=input_artifact_contracts or (),
        ).to_dict()

    def register_artifact(
        self,
        path: str,
        artifact_type: str,
        mime_type: str,
        artifact_id: str | None = None,
        producer_task_id: str | None = None,
        producer_tool: str | None = None,
        provenance: dict[str, Any] | None = None,
        trust: str = "external",
        agent: Optional[Any] = None,
        session_state: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Checksum and register one file contained by the active run root."""

        state = self._session_state(agent, session_state)
        runtime = self._runtime(agent, state)
        pending = runtime.pending_tool_invocations(domain_only=True)
        if pending:
            raise ValueError(
                "artifacts cannot be registered through the control plane while "
                f"domain tool calls are in flight: {', '.join(pending)}"
            )
        if trust not in {"external", "untrusted"}:
            raise ValueError(
                "workflow_register_artifact accepts only external or untrusted "
                "content; internal trust is assigned exclusively by server-side "
                "tool result registration"
            )
        if producer_task_id is not None or producer_tool is not None:
            raise ValueError(
                "workflow_register_artifact cannot assert producer identity; "
                "producer task/tool provenance is assigned by automatic tool "
                "result registration"
            )
        manual_provenance = {
            **dict(provenance or {}),
            "registration": "workflow_register_artifact",
        }
        return runtime.register_artifact(
            path,
            artifact_type=artifact_type,
            mime_type=mime_type,
            artifact_id=artifact_id,
            producer_task_id=None,
            active_task_id=None,
            producer_tool="workflow_register_artifact",
            provenance=manual_provenance,
            trust=trust,
        ).to_dict()

    def verify_artifact(
        self,
        artifact_id: str,
        agent: Optional[Any] = None,
        session_state: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Recompute and verify a registered artifact checksum and size."""

        return self._runtime(agent, session_state).verify_artifact(artifact_id).to_dict()

    def complete_run(
        self,
        required_artifact_types: list[str] | None = None,
        required_task_ids: list[str] | None = None,
        agent: Optional[Any] = None,
        session_state: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Complete the active run or mark it partial when contracts are unmet."""

        return (
            self._runtime(agent, session_state)
            .complete(
                required_artifact_types=required_artifact_types,
                required_task_ids=required_task_ids,
            )
            .to_dict()
        )

    @staticmethod
    def _runtime(agent: Optional[Any], session_state: Optional[dict[str, Any]]):
        from cs_copilot.workflows import RunContext

        runtime = getattr(agent, "run_context", None) if agent is not None else None
        if runtime is not None:
            return runtime
        state = (
            session_state if session_state is not None else getattr(agent, "session_state", None)
        )
        if not isinstance(state, dict):
            raise ValueError("The active MCP workflow run context is missing.")
        runtime = RunContext.from_session_state(state)
        if agent is not None:
            agent.run_context = runtime
        return runtime

    @staticmethod
    def _session_state(
        agent: Optional[Any],
        session_state: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        state = (
            session_state if session_state is not None else getattr(agent, "session_state", None)
        )
        if not isinstance(state, dict):
            raise ValueError("The active MCP session state is missing.")
        return state

    @staticmethod
    def _optional_runtime(agent: Optional[Any], state: dict[str, Any]):
        from cs_copilot.storage import OUTPUT_CONTEXT_KEY
        from cs_copilot.workflows import RunContext

        runtime = getattr(agent, "run_context", None) if agent is not None else None
        if runtime is not None:
            return runtime
        if OUTPUT_CONTEXT_KEY not in state:
            return None
        runtime = RunContext.from_session_state(state)
        if agent is not None:
            agent.run_context = runtime
        return runtime

    @staticmethod
    def _is_untouched_placeholder(run: Any) -> bool:
        return (
            run.workflow_slug == "mcp-session"
            and run.status.value == "submitted"
            and not run.constraints
            and not run.budget
            and not run.tasks
            and not run.artifacts
            and not run.handoffs
        )

    @staticmethod
    def _workflow_input_contracts(contract: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        return {
            str(item["name"]): item
            for item in contract.get("input_artifacts", ())
            if isinstance(item, Mapping) and item.get("name")
        }

    @classmethod
    def _validate_workflow_inputs(
        cls,
        workflow_contract: Mapping[str, Any],
        workflow_inputs: Mapping[str, Any] | None,
    ) -> None:
        if workflow_inputs is None:
            return
        if not isinstance(workflow_inputs, Mapping):
            raise TypeError("workflow_inputs must be a mapping")
        contracts = cls._workflow_input_contracts(workflow_contract)
        unknown = sorted(str(name) for name in workflow_inputs if str(name) not in contracts)
        if unknown:
            raise ValueError(
                "workflow_inputs contains undeclared input artifact contracts: "
                + ", ".join(unknown)
            )
        for name, value in workflow_inputs.items():
            contract = contracts[str(name)]
            if cls._artifact_id_reference(value) is not None:
                continue
            kind = str(contract.get("kind") or "").lower()
            if any(token in kind for token in _REFERENCE_ONLY_INPUT_KIND_TOKENS):
                raise ValueError(
                    f"workflow input {name!r} has file-backed kind {kind!r}; "
                    "register the file as a run artifact and supply "
                    "{'artifact_id': '<id>'}"
                )
            cls._inline_input_bytes(workflow_contract, contract, value)

    @classmethod
    def _apply_workflow_inputs(
        cls,
        runtime: Any,
        workflow_inputs: Mapping[str, Any] | None,
        *,
        reused: bool,
    ) -> None:
        if workflow_inputs is None:
            return
        run = runtime.run
        if run is None:  # pragma: no cover - guarded by RunContext
            raise RuntimeError("workflow run is not initialized")
        cls._validate_workflow_inputs(run.workflow_contract, workflow_inputs)
        requested_names = {str(name) for name in workflow_inputs}
        bound_names = set(run.workflow_inputs)
        omitted_bindings = sorted(bound_names - requested_names)
        if reused and omitted_bindings:
            raise ValueError(
                f"Active workflow run {run.run_id!r} already has pinned workflow "
                "inputs omitted by the explicit workflow_inputs mapping: "
                + ", ".join(omitted_bindings)
            )

        additions = requested_names - bound_names
        if (
            reused
            and additions
            and (run.handoffs or any(task.status.value != "pending" for task in run.tasks.values()))
        ):
            raise ValueError(
                f"Active workflow run {run.run_id!r} has started execution; "
                "missing workflow inputs can no longer be added."
            )

        contracts = cls._workflow_input_contracts(run.workflow_contract)
        for name, value in workflow_inputs.items():
            artifact_type = str(name)
            contract = contracts[artifact_type]
            referenced_id = cls._artifact_id_reference(value)
            bound_id = run.workflow_inputs.get(artifact_type)
            if referenced_id is not None:
                try:
                    referenced = run.artifacts[referenced_id]
                except KeyError as exc:
                    raise ValueError(
                        f"workflow input {artifact_type!r} references unknown run "
                        f"artifact {referenced_id!r}"
                    ) from exc
                if referenced.artifact_type != artifact_type:
                    raise ValueError(
                        f"workflow input {artifact_type!r} references artifact "
                        f"{referenced_id!r} of type {referenced.artifact_type!r}"
                    )
                if bound_id != referenced_id:
                    raise ValueError(
                        f"workflow input {artifact_type!r} is bound to "
                        f"{bound_id!r}, not {referenced_id!r}"
                    )
                continue

            encoded = cls._inline_input_bytes(run.workflow_contract, contract, value)
            if bound_id is not None:
                artifact = run.artifacts.get(bound_id)
                expected = hashlib.sha256(encoded).hexdigest()
                if artifact is None or artifact.sha256 != expected:
                    raise ValueError(
                        f"workflow input {artifact_type!r} differs from the "
                        f"pinned value in active run {run.run_id!r}"
                    )
                continue

            from cs_copilot.storage import S3, sanitize_workflow_slug

            safe_name = sanitize_workflow_slug(artifact_type)
            relative_path = f"inputs/{safe_name}.json"
            with S3.open(runtime.layout.artifact_rel_path(relative_path), "wb") as handle:
                handle.write(encoded)
            runtime.register_artifact(
                relative_path,
                artifact_type=artifact_type,
                mime_type="application/json",
                artifact_id=f"workflow-input-{safe_name}",
                producer_tool="workflow_start_run",
                provenance={
                    "source": "inline_workflow_input",
                    "workflow_contract_sha256": run.workflow_contract.get("contract_sha256"),
                    "input_contract": dict(contract),
                },
                trust="external",
            )

    @staticmethod
    def _artifact_id_reference(value: Any) -> str | None:
        if not isinstance(value, Mapping) or set(value) != {"artifact_id"}:
            return None
        from cs_copilot.storage import validate_identifier

        return validate_identifier(value["artifact_id"], field="workflow input artifact_id")

    @staticmethod
    def _inline_input_bytes(
        workflow_contract: Mapping[str, Any],
        input_contract: Mapping[str, Any],
        value: Any,
    ) -> bytes:
        payload = {
            "schema_version": 1,
            "workflow_slug": workflow_contract.get("slug"),
            "workflow_version": workflow_contract.get("version"),
            "workflow_contract_sha256": workflow_contract.get("contract_sha256"),
            "input_contract": dict(input_contract),
            "value": value,
        }
        try:
            serialized = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError("inline workflow input values must be finite JSON values") from exc
        return f"{serialized}\n".encode("utf-8")

    @staticmethod
    def _materialize_catalog_tasks(runtime: Any) -> list[str]:
        from cs_copilot.workflows import TaskRecord

        run = runtime.run
        if run is None:  # pragma: no cover - guarded by RunContext
            raise RuntimeError("workflow run is not initialized")
        materialized: list[str] = []
        for contract in run.workflow_contract.get("tasks", ()):
            if not isinstance(contract, Mapping):
                raise ValueError("pinned workflow task contract must be a mapping")
            task_id = str(contract["task_id"])
            if task_id in runtime.run.tasks:
                continue
            criteria = tuple(str(item) for item in contract.get("acceptance_criteria", ()))
            step = criteria[0] if criteria else f"Execute catalog task {task_id}."
            runtime.add_task(
                TaskRecord(
                    task_id=task_id,
                    role=str(contract["role"]),
                    profile=str(contract["profile"]),
                    step=step,
                )
            )
            materialized.append(task_id)
        return materialized

    @staticmethod
    def _start_run_payload(
        runtime: Any,
        *,
        profile: str,
        reused: bool,
        materialized_task_ids: list[str],
    ) -> dict[str, Any]:
        payload = runtime.manifest_payload()
        payload["profile"] = profile
        payload["reused"] = reused
        payload["materialized_task_ids"] = list(materialized_task_ids)
        payload["output_context"] = runtime.bind_session_state(None)
        return payload

    @staticmethod
    def _error(
        code: str | None,
        message: str | None,
        retryable: bool,
    ):
        if code is None and message is None:
            return None
        if not code or not message:
            raise ValueError("error_code and error_message must be provided together")
        from cs_copilot.workflows import ToolError

        return ToolError(code=code, message=message, retryable=retryable)


def workflow_policy_facade() -> WorkflowPolicyFacade:
    return WorkflowPolicyFacade()


def workflow_catalog_facade() -> WorkflowCatalogFacade:
    return WorkflowCatalogFacade()


def workflow_runtime_facade() -> WorkflowRuntimeFacade:
    return WorkflowRuntimeFacade()
