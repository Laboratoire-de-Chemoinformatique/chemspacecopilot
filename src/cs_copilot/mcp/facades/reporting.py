"""Report-export facade for MCP tool registration."""

from __future__ import annotations

import functools


class ReportExportFacade:
    """Adapter exposing report-export module functions as methods."""

    def __init__(self) -> None:
        from cs_copilot.tools.io.report_export import save_markdown_report, save_rich_report

        self.save_markdown = save_markdown_report
        self.save_rich = save_rich_report


@functools.lru_cache(maxsize=1)
def report_facade() -> ReportExportFacade:
    return ReportExportFacade()
