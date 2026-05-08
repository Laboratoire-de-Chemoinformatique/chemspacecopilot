#!/usr/bin/env python
# coding: utf-8
"""Tests for the Peptide Designer public facade."""

import cs_copilot.tools as tools
import cs_copilot.tools.chemistry as chemistry
from cs_copilot.agents.registry import list_available_agent_types
from cs_copilot.tools.chemistry.peptide_designer_toolkit import (
    PeptideDesignerError,
    PeptideDesignerToolkit,
)


def test_peptide_designer_public_registry_name_replaces_peptide_wae():
    """The public agent type should be Peptide Designer, not Peptide WAE."""
    agent_types = list_available_agent_types()

    assert "peptide_designer" in agent_types
    assert "peptide_wae" not in agent_types


def test_peptide_designer_toolkit_public_name(monkeypatch, tmp_path):
    """The public toolkit name should be peptide_designer."""
    monkeypatch.setattr(PeptideDesignerToolkit, "_ensure_model_exists", lambda self: None)
    monkeypatch.setattr(PeptideDesignerToolkit, "_load_model", lambda self: None)

    toolkit = PeptideDesignerToolkit(model_path=str(tmp_path), device="cpu")

    assert toolkit.name == "peptide_designer"
    assert "sample_peptides" in toolkit.functions


def test_peptide_designer_public_exports_replace_peptide_wae():
    """Only the Peptide Designer class should be exported from public tool packages."""
    assert tools.PeptideDesignerToolkit is PeptideDesignerToolkit
    assert chemistry.PeptideDesignerToolkit is PeptideDesignerToolkit
    assert issubclass(PeptideDesignerError, Exception)
    assert not hasattr(tools, "PeptideWAEToolkit")
    assert not hasattr(chemistry, "PeptideWAEToolkit")
