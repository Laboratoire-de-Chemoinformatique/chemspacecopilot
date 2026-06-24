"""MCP-specific GTM facade helpers."""

from __future__ import annotations

from typing import Any, List, Optional

from cs_copilot.tools.chemography import gtm_operations


class GTMMCPFacade:
    """Small MCP-only wrappers around GTM operations not exposed by GTMToolkit."""

    def save_density_plot(
        self,
        dataset_file: str,
        gtm_model_file: Optional[str] = None,
        mark_nodes: Optional[List[int]] = None,
        use_default: bool = False,
        descriptor_type: Optional[str] = None,
        agent: Any | None = None,
    ) -> str:
        """Generate and save a GTM density landscape with projected compound points."""

        resolved_model = gtm_operations.resolve_gtm_model_path(
            gtm_model_file,
            agent=agent,
            use_default=use_default,
        )
        return gtm_operations.save_gtm_plot(
            dataset_file,
            resolved_model,
            mark_nodes=mark_nodes,
            descriptor_type=descriptor_type,
            agent=agent,
        )
