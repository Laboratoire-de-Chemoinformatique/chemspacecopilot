"""Workflow policy and catalog facades for MCP tool registration."""

from __future__ import annotations

import functools
from typing import Any, List


class WorkflowPolicyFacade:
    """Read-only workflow preflight helpers for external MCP reasoners."""

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


@functools.lru_cache(maxsize=1)
def workflow_policy_facade() -> WorkflowPolicyFacade:
    return WorkflowPolicyFacade()


@functools.lru_cache(maxsize=1)
def workflow_catalog_facade() -> WorkflowCatalogFacade:
    return WorkflowCatalogFacade()
