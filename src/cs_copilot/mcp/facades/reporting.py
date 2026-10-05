"""Report-export facade for MCP tool registration."""

from __future__ import annotations


class ReportExportFacade:
    """Adapter exposing report-export module functions as methods."""

    def __init__(self) -> None:
        from cs_copilot.tools.io.report_export import save_markdown_report, save_rich_report
        from cs_copilot.tools.io.reporting_evidence import get_report_evidence

        self.get_evidence = get_report_evidence
        self.save_markdown = save_markdown_report
        self.save_rich = save_rich_report


def report_facade() -> ReportExportFacade:
    return ReportExportFacade()
