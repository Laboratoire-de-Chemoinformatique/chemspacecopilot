"""Workflow policy and catalog facades for MCP tool registration."""

from __future__ import annotations

import functools
from typing import Any, List


class WorkflowPolicyFacade:
    """Read-only workflow preflight helpers for external MCP reasoners."""

    def prepare_chembl_retrieval(
        self,
        user_request: str,
        session_summary: str | None = None,
    ) -> dict[str, Any]:
        """Preflight a ChEMBL request before calling retrieval tools."""
        from cs_copilot.workflows import prepare_chembl_retrieval

        return prepare_chembl_retrieval(
            user_request=user_request,
            session_summary=session_summary,
        )

    def plan_chemical_space_analysis(
        self,
        user_request: str,
        session_summary: str | None = None,
    ) -> dict[str, Any]:
        """Preflight a chemical-space analysis request before mutating tools."""
        from cs_copilot.workflows import plan_chemical_space_analysis

        return plan_chemical_space_analysis(
            user_request=user_request,
            session_summary=session_summary,
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
