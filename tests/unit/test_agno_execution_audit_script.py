"""scripts/agno_execution_audit.py summarizes observe-mode audits per tool."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agno_execution_audit.py"


def _load():
    spec = importlib.util.spec_from_file_location("agno_execution_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _call(tool, audit, status="success"):
    return SimpleNamespace(
        event_type="tool_call_recorded",
        payload={"tool_name": tool, "status": status, "execution_audit": audit},
    )


def test_summarize_counts_enforce_conflicts_per_tool():
    module = _load()
    events = [
        _call("gtm_save_density_plot", {"writes": [{"verdict": "create_new", "path": "a.png"}]}),
        _call(
            "gtm_save_density_plot",
            {"writes": [{"verdict": "protected_artifact", "path": "a.png"}]},
            status="error",
        ),
        _call("pandas_run_operation", {"would_deny_writes": "outside the run", "writes": []}),
        _call("skill_list", {"writes": []}),
        SimpleNamespace(event_type="tool_progress", payload={}),
    ]

    report = module.summarize(events)

    assert report["gtm_save_density_plot"]["calls"] == 2
    assert report["gtm_save_density_plot"]["errors"] == 1
    assert report["gtm_save_density_plot"]["write_verdicts"] == {
        "create_new": 1,
        "protected_artifact": 1,
    }
    assert report["gtm_save_density_plot"]["enforce_conflicts"] == 1
    assert report["pandas_run_operation"]["enforce_conflicts"] == 1
    assert report["skill_list"]["enforce_conflicts"] == 0
