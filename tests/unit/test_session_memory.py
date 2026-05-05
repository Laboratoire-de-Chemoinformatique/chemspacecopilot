#!/usr/bin/env python
# coding: utf-8
"""Tests for shared structured session working memory."""

import pandas as pd

from cs_copilot.tools.io.session_memory import (
    SessionMemoryToolkit,
    list_loadable_session_data,
    load_candidate_set_artifact,
    register_compounds_from_candidates,
    register_generated_candidate_set,
    register_session_object,
    resolve_candidate_set,
    resolve_loadable_session_data,
    resolve_session_reference,
    select_session_object,
)


def test_register_compounds_updates_current_and_summary():
    state = {}

    ids = register_compounds_from_candidates(
        state,
        [{"smiles": "CCO", "valid": True}, {"smiles": "CCN", "valid": True}],
        source_agent="molecular_designer_agent",
        source_tool="design_molecules",
        label_prefix="Candidate",
    )

    assert ids == ["cmp_001", "cmp_002"]
    assert state["session_objects"]["current"]["compound"] == "cmp_001"
    assert "cmp_001" in state["session_memory_summary"]
    assert state["session_objects"]["compounds"]["cmp_002"]["smiles"] == "CCN"
    assert "properties" not in state["session_objects"]["compounds"]["cmp_002"]


def test_resolve_current_and_numbered_references():
    state = {}
    register_compounds_from_candidates(
        state,
        ["CCO", "CCN"],
        source_agent="test",
        source_tool="sample_molecules",
        label_prefix="Sample",
    )
    select_session_object(state, "cmp_002")

    current = resolve_session_reference(state, "that compound", "compound")
    numbered = resolve_session_reference(state, "compound 1", "compound")

    assert current["status"] == "resolved"
    assert current["object"]["smiles"] == "CCN"
    assert numbered["object"]["smiles"] == "CCO"


def test_register_map_zone_and_node_records():
    state = {}
    map_id = register_session_object(
        state,
        "map",
        {"map_type": "gtm", "dataset_path": "dataset.csv"},
        label="EGFR GTM map",
    )
    zone_id = register_session_object(
        state,
        "zone",
        {"map_id": map_id, "zone_type": "active", "node_ids": [10, 11]},
        label="Active zone",
    )
    node_id = register_session_object(
        state,
        "node",
        {"map_id": map_id, "zone_id": zone_id, "node_index": 10, "x": 1, "y": 2},
    )

    assert map_id == "map_001"
    assert zone_id == "zone_001"
    assert node_id == "node_map_001_10"
    assert state["session_objects"]["current"]["zone"] == zone_id


def test_session_memory_toolkit_resolves_and_selects():
    state = {}
    register_compounds_from_candidates(
        state,
        ["CCO"],
        source_agent="test",
        source_tool="sample_molecules",
        label_prefix="Sample",
    )
    toolkit = SessionMemoryToolkit()

    resolved = toolkit.resolve_session_reference("CCO", "compound", session_state=state)
    selected = toolkit.select_session_object("cmp_001", session_state=state)
    listed = toolkit.list_session_objects("compound", session_state=state)

    assert resolved["status"] == "resolved"
    assert selected["status"] == "selected"
    assert listed[0]["id"] == "cmp_001"


def test_list_loadable_session_data_lists_dataframes_and_csv_paths():
    state = {
        "analysis_input": pd.DataFrame({"smi": ["CCO"], "activity_final": [7.0]}),
        "landscape_files": {"landscape_data_csv": "/tmp/landscape.csv"},
        "_gtm_prepared_dataset_cache": {"private": "/tmp/private.csv"},
    }

    loadable = list_loadable_session_data(state)
    keys = {entry["session_key"] for entry in loadable}

    assert "analysis_input" in keys
    assert "landscape_files.landscape_data_csv" in keys
    assert "_gtm_prepared_dataset_cache.private" not in keys


def test_resolve_loadable_session_data_prefers_primary_path_in_container():
    state = {
        "analysis_outputs": {
            "supplementary_data": ["/tmp/supplementary.csv"],
            "primary_data_csv": "/tmp/primary.csv",
        }
    }

    resolved = resolve_loadable_session_data(state, "analysis_outputs")

    assert resolved["session_key"] == "analysis_outputs.primary_data_csv"
    assert resolved["path"] == "/tmp/primary.csv"


def test_generated_candidate_set_resolves_top_candidates_over_dataset_compounds(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    state = {}
    candidates = [
        {"smiles": "CCO", "score": 0.9, "valid": True, "rationale": "Verbose rationale"},
        {"smiles": "CCN", "score": 0.8, "valid": True},
        {"smiles": "CCC", "score": 0.7, "valid": True},
    ]
    register_session_object(
        state,
        "compound",
        {"smiles": "CHEMBL-SEED"},
        label="ChEMBL seed compound",
        source_tool="fetch_compounds",
        set_current=False,
    )
    candidate_ids = register_compounds_from_candidates(
        state,
        candidates,
        source_agent="molecular_designer_agent",
        source_tool="design_molecules",
        label_prefix="Generated candidate",
        provenance={
            "origin_type": "generated",
            "origin_agent": "molecular_designer",
            "generation_engine": "llm",
        },
    )
    candidate_set_id = register_generated_candidate_set(
        state,
        candidate_ids,
        source_agent="molecular_designer_agent",
        source_tool="design_molecules",
        origin_agent="molecular_designer",
        generation_engine="llm",
        generation_mode="design",
        session_key="designed_molecules",
        label="LLM generated candidates",
        goal="Design examples",
        count_attempted=3,
        candidates=candidates,
    )

    resolved = resolve_candidate_set(state, "top 2 candidates")
    artifact = load_candidate_set_artifact(state, candidate_set_id)

    assert candidate_set_id == "cset_001"
    assert state["session_objects"]["current"]["candidate_set"] == "cset_001"
    assert state["session_objects"]["current"]["generated_compounds"] == "cset_001"
    assert state["designed_molecules"]["candidate_set_id"] == "cset_001"
    assert state["designed_molecules"]["artifact_path"].endswith("candidate_sets/cset_001.json")
    assert state["designed_molecules"]["preview"] == [
        {"smiles": "CCO", "valid": True, "score": 0.9},
        {"smiles": "CCN", "valid": True, "score": 0.8},
        {"smiles": "CCC", "valid": True, "score": 0.7},
    ]
    assert [compound["smiles"] for compound in resolved["compounds"]] == ["CCO", "CCN"]
    assert resolved["compounds"][0]["origin_agent"] == "molecular_designer"
    assert resolved["compounds"][0]["generation_engine"] == "llm"
    assert state["session_objects"]["compounds"]["cmp_002"]["candidate_set_id"] == "cset_001"
    assert state["session_objects"]["candidate_sets"]["cset_001"]["artifact_format"] == "json"
    assert artifact["status"] == "loaded"
    assert [candidate["smiles"] for candidate in artifact["candidates"]] == ["CCO", "CCN", "CCC"]


def test_session_memory_toolkit_resolves_candidate_set():
    state = {}
    candidate_ids = register_compounds_from_candidates(
        state,
        ["CCO", "CCN"],
        source_agent="test",
        source_tool="sample_molecules",
        label_prefix="Sample",
        provenance={
            "origin_type": "generated",
            "origin_agent": "autoencoder_toolkit",
            "generation_engine": "autoencoder",
        },
    )
    register_generated_candidate_set(
        state,
        candidate_ids,
        source_agent="test",
        source_tool="sample_molecules",
        origin_agent="autoencoder_toolkit",
        generation_engine="autoencoder",
        generation_mode="sample",
        session_key="sampled",
        label="Autoencoder samples",
    )

    resolved = SessionMemoryToolkit().resolve_candidate_set(
        "autoencoder candidates",
        top_n=1,
        session_state=state,
    )

    assert resolved["status"] == "resolved"
    assert resolved["candidate_set"]["id"] == "cset_001"
    assert resolved["compounds"][0]["smiles"] == "CCO"
