#!/usr/bin/env python
# coding: utf-8
"""Tests for shared structured session working memory."""

from cs_copilot.tools.io.session_memory import (
    SessionMemoryToolkit,
    register_compounds_from_candidates,
    register_session_object,
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
