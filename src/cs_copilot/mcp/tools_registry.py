"""Explicit registry of cs_copilot toolkit methods exposed as MCP tools.

The public entry points in this module are intentionally stable:
``iter_specs()`` and ``all_specs()`` remain the compatibility surface used by
the server, tests, and ChatGPT-compatible catalog rendering. Individual spec
groups and toolkit facades live in smaller modules under
``cs_copilot.mcp.tool_specs`` and ``cs_copilot.mcp.facades``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List

from .tool_adapter import ToolSpec
from .tool_specs import (
    chembl,
    chemistry,
    design,
    gtm,
    llm,
    pandas,
    reporting,
    robustness,
    session,
    skills,
    synplanner,
    workflow,
)


def _with_group(specs: Iterable[ToolSpec], group: str) -> Iterable[ToolSpec]:
    for spec in specs:
        yield replace(spec, group=spec.group or group)


def iter_specs() -> Iterable[ToolSpec]:
    """Yield every :class:`ToolSpec` exposed by the MCP server."""

    yield from _with_group(chembl.SPECS, "chembl")
    yield from _with_group(gtm.SPECS, "gtm")
    yield from _with_group(chemistry.SPECS, "chem")
    yield from _with_group(session.SPECS, "session")
    yield from _with_group(reporting.SPECS, "report")
    yield from _with_group(workflow.SPECS, "workflow")
    yield from _with_group(llm.SPECS, "llm")
    yield from _with_group(robustness.SPECS, "robustness")
    yield from _with_group(skills.SPECS, "skills")
    yield from _with_group(pandas.SPECS, "pandas")
    yield from _with_group(design.MOLECULAR_SPECS, "molecular_design")
    yield from _with_group(design.PEPTIDE_SPECS, "peptide_design")
    yield from _with_group(synplanner.SPECS, "synplanner")


def all_specs() -> List[ToolSpec]:
    """Return every :class:`ToolSpec` as a list."""

    return list(iter_specs())
