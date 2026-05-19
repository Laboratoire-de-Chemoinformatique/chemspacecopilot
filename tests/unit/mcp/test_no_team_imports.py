"""Guard test: ``cs_copilot.mcp`` must not import the Agno team / factories.

The MCP server must never execute the Agno multi-agent system in its default
mode. To enforce that statically, we AST-walk every module under
``src/cs_copilot/mcp`` and reject any import of forbidden symbols.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "src" / "cs_copilot" / "mcp"
FORBIDDEN_PREFIXES = (
    "cs_copilot.agents.teams",
    "cs_copilot.agents.factories",
    "cs_copilot.agents.registry",
    "cs_copilot.model_config",
    "chainlit_app",
)


def _iter_imports(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            yield module


@pytest.mark.parametrize("path", sorted(PACKAGE_ROOT.rglob("*.py")))
def test_no_forbidden_imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for name in _iter_imports(tree):
        for forbidden in FORBIDDEN_PREFIXES:
            assert not name.startswith(
                forbidden
            ), f"{path.relative_to(PACKAGE_ROOT)} imports forbidden module {name!r}"
