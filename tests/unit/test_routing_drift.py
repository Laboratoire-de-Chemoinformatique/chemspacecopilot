"""Drift guards for ``cs_copilot.routing``.

These keep the generated team prose in lockstep with the catalog and enforce the
module's import-safety invariant (it must stay free of the Agno team / MCP stack
so it can be shared by both the deterministic bootstrap and the Agno team).
"""

from __future__ import annotations

import ast
from pathlib import Path

from cs_copilot.routing import render_routing_rules, routing_domains

ROUTING_MODULE = Path(__file__).resolve().parents[2] / "src" / "cs_copilot" / "routing.py"
FORBIDDEN_PREFIXES = (
    "agno",
    "cs_copilot.agents",
    "cs_copilot.mcp",
    "cs_copilot.model_config",
    "chainlit_app",
)


def test_prose_covers_every_routing_domain():
    prose = render_routing_rules()
    for domain in routing_domains():
        assert domain["agent"] in prose, f"agent {domain['agent']} missing from routing prose"
        for keyword in domain["keywords"]:
            assert keyword in prose, (
                f"keyword {keyword!r} for {domain['anchor_skill']} missing from routing prose "
                "(catalog keywords drifted from the generated team description)"
            )


def _iter_imported_modules(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            yield node.module or ""


def test_routing_has_no_forbidden_imports():
    tree = ast.parse(ROUTING_MODULE.read_text(encoding="utf-8"), filename=str(ROUTING_MODULE))
    for name in _iter_imported_modules(tree):
        for forbidden in FORBIDDEN_PREFIXES:
            assert not name.startswith(forbidden), f"routing.py imports forbidden module {name!r}"
