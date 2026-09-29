"""Runtime import isolation, checked in a fresh interpreter.

The AST guards in ``tests/unit/mcp/test_no_team_imports.py`` only see import
statements. Package ``__init__`` side effects are invisible to them: an
eager ``cs_copilot/agents/__init__.py`` once turned the MCP prompt registry's
harmless ``from cs_copilot.agents import instructions`` into a full Agno team
import. These tests run representative entry points in a clean subprocess and
inspect ``sys.modules`` afterwards.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

TEAM_RUNTIME_MODULES = (
    "cs_copilot.agents.teams",
    "cs_copilot.agents.factories",
    "cs_copilot.agents.registry",
    "cs_copilot.agents.single_agent",
    "cs_copilot.agents.delegation",
    "cs_copilot.model_config",
    "chainlit_app",
)
_MARKER = "__CS_COPILOT_MODULES__"


def _loaded_modules(code: str, tmp_path: Path) -> set[str]:
    env = {
        **os.environ,
        "USE_S3": "false",
        "AGNO_TELEMETRY": "false",
        "SESSION_ID": "import-isolation",
    }
    script = (
        textwrap.dedent(code)
        + f"\nimport json as _json, sys as _sys\nprint({_MARKER!r} + _json.dumps(sorted(_sys.modules)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    line = next(line for line in result.stdout.splitlines() if line.startswith(_MARKER))
    return set(json.loads(line[len(_MARKER) :]))


def _loaded_from(modules: set[str], prefixes: tuple[str, ...]) -> list[str]:
    return sorted(
        module
        for module in modules
        if any(module == prefix or module.startswith(prefix + ".") for prefix in prefixes)
    )


def test_agents_instructions_import_stays_lightweight(tmp_path):
    modules = _loaded_modules(
        """
        import cs_copilot.agents
        from cs_copilot.agents import instructions

        assert instructions.AGENT_TEAM_INSTRUCTIONS
        """,
        tmp_path,
    )

    assert not _loaded_from(modules, TEAM_RUNTIME_MODULES + ("agno",))


def test_mcp_package_surfaces_do_not_load_the_agno_team(tmp_path):
    modules = _loaded_modules(
        """
        import importlib
        import pkgutil

        import cs_copilot.mcp as package

        for info in pkgutil.walk_packages(package.__path__, prefix=package.__name__ + "."):
            importlib.import_module(info.name)

        from cs_copilot.mcp import prompts_registry, tools_registry
        from cs_copilot.mcp.facades.bootstrap import mcp_bootstrap_facade

        for spec in prompts_registry.all_specs():
            if not spec.arguments:
                assert spec.render()
        assert tools_registry.all_specs()
        assert mcp_bootstrap_facade().bootstrap(
            "Retrieve EGFR inhibitors from ChEMBL, build a GTM map, and write a report",
            workflow_slug="chembl-to-gtm-report",
        )
        """,
        tmp_path,
    )

    assert not _loaded_from(modules, TEAM_RUNTIME_MODULES)


@pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="requires the mcp extra")
def test_mcp_server_build_does_not_load_the_agno_team(tmp_path):
    modules = _loaded_modules(
        """
        from cs_copilot.mcp.session import BootstrapConfig, apply_session_id, bootstrap

        apply_session_id("import-isolation")
        ctx = bootstrap(BootstrapConfig(session_id="import-isolation"))

        from cs_copilot.mcp.server import build_server

        assert build_server(ctx) is not None
        assert build_server(ctx, enable_agno_team_tool=True) is not None
        """,
        tmp_path,
    )

    assert not _loaded_from(modules, TEAM_RUNTIME_MODULES)
