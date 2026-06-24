"""Tests for MCP-specific GTM facade helpers."""

from __future__ import annotations

from types import SimpleNamespace

from cs_copilot.mcp.facades.gtm import GTMMCPFacade


def test_save_density_plot_resolves_session_model_and_calls_plot_writer(monkeypatch):
    calls = {}
    agent = SimpleNamespace(session_state={"_current_gtm_model_path": "session-model.pkl.gz"})

    def fake_resolve(gtm_model_file, *, agent=None, use_default=False, generate_framesets=False):
        calls["resolve"] = {
            "gtm_model_file": gtm_model_file,
            "agent": agent,
            "use_default": use_default,
            "generate_framesets": generate_framesets,
        }
        return "resolved-model.pkl.gz"

    def fake_save(dataset_file, gtm_model_file, *, mark_nodes=None, descriptor_type=None, agent=None):
        calls["save"] = {
            "dataset_file": dataset_file,
            "gtm_model_file": gtm_model_file,
            "mark_nodes": mark_nodes,
            "descriptor_type": descriptor_type,
            "agent": agent,
        }
        return "GTM plot saved to S3: `density.html` and `density.png`"

    monkeypatch.setattr(
        "cs_copilot.mcp.facades.gtm.gtm_operations.resolve_gtm_model_path",
        fake_resolve,
    )
    monkeypatch.setattr(
        "cs_copilot.mcp.facades.gtm.gtm_operations.save_gtm_plot",
        fake_save,
    )

    result = GTMMCPFacade().save_density_plot(
        "clean.csv",
        mark_nodes=[1, 2],
        descriptor_type="morgan",
        agent=agent,
    )

    assert result == "GTM plot saved to S3: `density.html` and `density.png`"
    assert calls["resolve"] == {
        "gtm_model_file": None,
        "agent": agent,
        "use_default": False,
        "generate_framesets": False,
    }
    assert calls["save"] == {
        "dataset_file": "clean.csv",
        "gtm_model_file": "resolved-model.pkl.gz",
        "mark_nodes": [1, 2],
        "descriptor_type": "morgan",
        "agent": agent,
    }
