"""Robustness-analysis MCP tool specs."""

from __future__ import annotations

from typing import List

from ..tool_adapter import ToolSpec
from .common import factory

_ROBUSTNESS = factory("cs_copilot.tools.analysis.robustness_toolkit:RobustnessAnalysisToolkit")

_METHODS = [
    ("load_test_results", "Load the raw results of a robustness test run.", True),
    ("load_test_summary_csv", "Load the per-prompt summary CSV of a robustness test run.", True),
    ("list_available_test_runs", "List robustness test runs available under the data root.", True),
    ("analyze_score_distribution", "Summarise score distribution for a robustness run.", True),
    ("identify_failing_prompts", "List failing prompts above a score threshold.", True),
    ("compare_test_runs", "Compare two robustness test runs side by side.", True),
    ("analyze_temporal_trends", "Summarise robustness score trends across runs.", True),
    ("generate_insights", "Generate textual insights about a robustness run.", True),
    ("export_analysis_report", "Persist a robustness analysis report to storage.", False),
]

SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name=f"robustness_{name}",
        toolkit_factory=_ROBUSTNESS,
        method=name,
        summary=summary,
        read_only=read_only,
    )
    for name, summary, read_only in _METHODS
]
