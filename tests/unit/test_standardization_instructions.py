#!/usr/bin/env python
# coding: utf-8
"""Tests for agent instructions around standardized dataset artifacts."""

from cs_copilot.agents.prompts import (
    AGENT_TEAM_INSTRUCTIONS,
    CHEMBL_INSTRUCTIONS,
    CHEMOINFORMATICIAN_INSTRUCTIONS,
    GTM_AGENT_INSTRUCTIONS,
    REPORT_GENERATOR_INSTRUCTIONS,
)


def _joined(instructions: list[str]) -> str:
    return "\n".join(instructions)


def test_agents_reference_clean_dataset_artifact_contract():
    team = _joined(AGENT_TEAM_INSTRUCTIONS)
    chembl = _joined(CHEMBL_INSTRUCTIONS)
    chemoinformatician = _joined(CHEMOINFORMATICIAN_INSTRUCTIONS)
    gtm = _joined(GTM_AGENT_INSTRUCTIONS)
    report = _joined(REPORT_GENERATOR_INSTRUCTIONS)

    assert "clean_dataset_path" in team
    assert "raw_dataset_path" in team
    assert "descriptor_parquet_path" in team
    assert "dataset_path is only a backward-compatible clean-data alias" in team

    assert "raw_dataset_path for provenance and clean_dataset_path" in chembl
    assert "descriptor_parquet_path" in chembl
    assert "standardization report" in chembl.lower()

    assert "clean_dataset_path" in chemoinformatician
    assert "final_activity_mapping" in chemoinformatician
    assert "descriptor_parquet_path" in chemoinformatician

    assert "clean_dataset_path" in gtm
    assert "legacy clean-data alias" in gtm

    assert "raw dataset path" in report
    assert "clean dataset path" in report
    assert "descriptor Parquet path" in report
