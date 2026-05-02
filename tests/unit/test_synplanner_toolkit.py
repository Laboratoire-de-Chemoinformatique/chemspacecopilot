"""Unit tests for the SynPlanner integration toolkit."""

from __future__ import annotations

import sys
import types

import pytest

from cs_copilot.tools.chemistry.synplanner_toolkit import SynPlannerToolkit


@pytest.fixture(autouse=True)
def fake_synplanner(monkeypatch):
    """Provide a lightweight stub of the external SynPlanner package."""

    module = types.ModuleType("synplanner")

    class FakeSynPlanner:
        def __init__(self, prefer_gpu: bool = False):
            self.prefer_gpu = prefer_gpu
            self.loaded = False
            self.invocations = []

        def load(self):
            self.loaded = True

        def plan(self, smiles: str, top_k: int = 3):
            self.invocations.append((smiles, top_k))
            return [
                {
                    "score": 0.42,
                    "steps": [
                        {
                            "description": "Break ester to acid + alcohol",
                            "reactants": ["acid chloride", "salicylic acid"],
                            "products": ["aspirin"],
                            "reagents": ["pyridine"],
                        },
                        {
                            "summary": "Assemble salicylic acid",
                            "precursors": ["phenol"],
                            "targets": ["salicylic acid"],
                        },
                    ],
                }
            ]

    def name_to_smiles(name: str) -> str:
        lookup = {
            "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        }
        key = name.lower()
        if key not in lookup:
            raise KeyError(name)
        return lookup[key]

    def canonicalize_smiles(smiles: str) -> str:
        return f"canonical({smiles})"

    module.SynPlanner = FakeSynPlanner
    module.name_to_smiles = name_to_smiles
    module.canonicalize_smiles = canonicalize_smiles

    monkeypatch.setitem(sys.modules, "synplanner", module)

    yield module

    sys.modules.pop("synplanner", None)


@pytest.fixture
def fake_pubchem(monkeypatch):
    module = types.ModuleType("pubchempy")

    class FakeCompound:
        def __init__(self, smiles: str):
            self.canonical_smiles = smiles
            self.isomeric_smiles = smiles
            self.connectivity_smiles = smiles

    queries = []

    def get_compounds(value: str, namespace: str = "name"):
        queries.append((namespace, value))
        if namespace == "smiles":
            return [FakeCompound("CCO")]  # Valid SMILES for ethanol
        if namespace == "name":
            return [FakeCompound("CC(=O)OC1=CC=CC=C1C(=O)O")]  # Aspirin SMILES
        return []

    module.get_compounds = get_compounds
    module.queries = queries

    monkeypatch.setitem(sys.modules, "pubchempy", module)

    yield module

    sys.modules.pop("pubchempy", None)


@pytest.fixture
def empty_pubchem(monkeypatch):
    module = types.ModuleType("pubchempy")
    module.queries = []

    def get_compounds(value: str, namespace: str = "name"):
        module.queries.append((namespace, value))
        return []

    module.get_compounds = get_compounds

    monkeypatch.setitem(sys.modules, "pubchempy", module)

    yield module

    sys.modules.pop("pubchempy", None)


def test_identify_input_accepts_smiles():
    toolkit = SynPlannerToolkit()

    result = toolkit.identify_input("CCO")

    assert result["smiles"] == "CCO"
    assert result["source"] == "smiles"


def test_identify_input_rejects_empty():
    toolkit = SynPlannerToolkit()

    from cs_copilot.tools.chemistry.synplanner_toolkit import SynPlannerError

    with pytest.raises(SynPlannerError):
        toolkit.identify_input("")

    with pytest.raises(SynPlannerError):
        toolkit.identify_input("   ")


def test_convert_name_to_smiles_prefers_pubchem(fake_synplanner, fake_pubchem):
    toolkit = SynPlannerToolkit()

    smiles = toolkit.convert_name_to_smiles("aspirin", llm_smiles_guess="CCO")

    # PubChem returns a SMILES that gets canonicalized by RDKit
    assert smiles is not None
    assert len(smiles) > 0
    # Verify PubChem was queried with the SMILES namespace
    assert any(ns == "smiles" for ns, _ in fake_pubchem.queries)


def test_convert_name_to_smiles_uses_pubchem_name_lookup(fake_synplanner, fake_pubchem):
    toolkit = SynPlannerToolkit()

    smiles = toolkit.convert_name_to_smiles("aspirin")

    assert smiles is not None
    assert len(smiles) > 0
    # Verify PubChem was queried with the name namespace
    assert any(ns == "name" for ns, _ in fake_pubchem.queries)


def test_convert_name_to_smiles_raises_when_no_resolution(fake_synplanner, empty_pubchem):
    """When PubChem returns nothing and no LLM guess, should raise."""
    toolkit = SynPlannerToolkit()

    from cs_copilot.tools.chemistry.synplanner_toolkit import SynPlannerError

    with pytest.raises(SynPlannerError, match="Could not resolve"):
        toolkit.convert_name_to_smiles("totally_unknown_molecule_xyz")


def test_convert_name_to_smiles_with_llm_guess_asks_confirmation(fake_synplanner, empty_pubchem):
    """When PubChem fails but LLM guess is provided, should raise UserConfirmationRequiredError."""
    toolkit = SynPlannerToolkit()

    from cs_copilot.tools.chemistry.synplanner_toolkit import UserConfirmationRequiredError

    with pytest.raises(UserConfirmationRequiredError) as exc_info:
        toolkit.convert_name_to_smiles("mystery", llm_smiles_guess="CCN")

    assert exc_info.value.smiles is not None
    assert exc_info.value.molecule_name == "mystery"


def test_toolkit_registration():
    """Verify all expected tools are registered."""
    toolkit = SynPlannerToolkit()

    # Check that the toolkit has the expected tool methods
    assert hasattr(toolkit, "identify_input")
    assert hasattr(toolkit, "convert_name_to_smiles")
    assert hasattr(toolkit, "plan_synthesis")
    assert hasattr(toolkit, "describe_plan")
    assert hasattr(toolkit, "get_route_visualizations")


def test_toolkit_name():
    toolkit = SynPlannerToolkit()
    assert toolkit.name == "synplanner"


def test_identify_input_with_valid_smiles():
    toolkit = SynPlannerToolkit()

    # Test with a valid SMILES string
    result = toolkit.identify_input("c1ccccc1")  # benzene
    assert result["source"] == "smiles"
    assert result["smiles"] is not None


def test_ensure_sequence_static_method():
    assert SynPlannerToolkit._ensure_sequence(None) == []
    assert SynPlannerToolkit._ensure_sequence("single") == ["single"]
    assert SynPlannerToolkit._ensure_sequence(["a", "b"]) == ["a", "b"]
    assert SynPlannerToolkit._ensure_sequence(("x", "y")) == ["x", "y"]


def test_normalise_steps():
    toolkit = SynPlannerToolkit()

    steps = [
        {
            "description": "Step A",
            "reactants": ["R1"],
            "products": ["P1"],
            "reagents": ["Rg1"],
        },
        {
            "summary": "Step B",
            "precursors": ["Pre1"],
            "targets": ["T1"],
            "conditions": None,
        },
    ]

    normalised = toolkit._normalise_steps(steps)
    assert len(normalised) == 2
    assert normalised[0].index == 1
    assert normalised[0].description == "Step A"
    assert normalised[0].reactants == ["R1"]
    assert normalised[0].products == ["P1"]
    assert normalised[0].reagents == ["Rg1"]
    assert normalised[1].index == 2
    assert normalised[1].description == "Step B"
    assert normalised[1].reactants == ["Pre1"]
    assert normalised[1].products == ["T1"]
    assert normalised[1].reagents == []


def test_normalise_routes_empty():
    toolkit = SynPlannerToolkit()

    assert toolkit._normalise_routes(None) == []
    assert toolkit._normalise_routes([]) == []
    assert toolkit._normalise_routes("not a list") == []


class _FakeTree:
    def __init__(self, winning_nodes):
        self.winning_nodes = winning_nodes
        self.curr_iteration = 12
        self.curr_time = 1.2345
        self.curr_tree_size = 7
        self.nodes_depth = {1: 0, 2: 1, 3: 2}

    def __len__(self):
        return self.curr_tree_size - 1

    def route_score(self, node_id):
        return 0.7

    def route_to_node(self, node_id):
        return []


def _patch_planning_dependencies(monkeypatch, toolkit):
    monkeypatch.setattr(toolkit, "_load_synplanner_components", lambda: None)
    monkeypatch.setattr(toolkit, "_generate_route_visualizations", lambda *args, **kwargs: [])
    monkeypatch.setattr(toolkit, "get_basic_descriptors", lambda smiles: {})


def test_build_search_profiles_use_synplanner_documented_baseline():
    toolkit = SynPlannerToolkit()

    profiles = toolkit._build_search_profiles()
    standard = profiles[0]
    broader = next(profile for profile in profiles if profile.name == "broader_expansion")
    exploratory = next(profile for profile in profiles if profile.name == "exploratory_uct")

    assert standard.name == "standard"
    assert standard.tree_config["max_iterations"] == 100
    assert standard.tree_config["max_tree_size"] == 10000
    assert standard.tree_config["max_time"] == 120
    assert standard.tree_config["max_depth"] == 9
    assert standard.tree_config["search_strategy"] == "expansion_first"
    assert standard.tree_config["ucb_type"] == "uct"
    assert standard.tree_config["c_ucb"] == 0.1
    assert standard.tree_config["min_mol_size"] == 6
    assert standard.tree_config["epsilon"] == 0.0
    assert standard.tree_config["silent"] is True
    assert standard.policy_config["top_rules"] == 50
    assert standard.policy_config["rule_prob_threshold"] == 0.0
    assert broader.policy_config["top_rules"] == 100
    assert broader.policy_config["rule_prob_threshold"] == 0.0
    assert exploratory.tree_config["c_ucb"] == 0.5
    assert exploratory.tree_config["epsilon"] == 0.1


def test_plan_synthesis_stops_when_standard_profile_finds_route(monkeypatch):
    toolkit = SynPlannerToolkit()
    _patch_planning_dependencies(monkeypatch, toolkit)
    calls = []

    def fake_create_tree(smiles, profile=None):
        calls.append(profile.name)
        return _FakeTree([42])

    monkeypatch.setattr(toolkit, "_create_and_search_tree", fake_create_tree)

    plan = toolkit.plan_synthesis("CCO")

    assert calls == ["standard"]
    assert len(plan["routes"]) == 1
    assert plan["successful_attempt"] == "standard"
    assert plan["llm_fallback_allowed"] is False
    assert len(plan["attempts"]) == 1
    assert plan["attempts"][0]["route_count"] == 1
    assert "raw" not in plan


def test_plan_synthesis_retries_until_later_profile_finds_route(monkeypatch):
    toolkit = SynPlannerToolkit()
    _patch_planning_dependencies(monkeypatch, toolkit)
    calls = []

    def fake_create_tree(smiles, profile=None):
        calls.append(profile.name)
        winning_nodes = [42] if profile.name == "broader_expansion" else []
        return _FakeTree(winning_nodes)

    monkeypatch.setattr(toolkit, "_create_and_search_tree", fake_create_tree)

    plan = toolkit.plan_synthesis("CCO")

    assert calls == ["standard", "longer_search", "deeper_search", "broader_expansion"]
    assert len(plan["routes"]) == 1
    assert plan["successful_attempt"] == "broader_expansion"
    assert plan["llm_fallback_allowed"] is False
    assert [attempt["route_count"] for attempt in plan["attempts"]] == [0, 0, 0, 1]
    assert plan["attempts"][-1]["parameters"]["policy"]["top_rules"] == 100


def test_plan_synthesis_all_profiles_fail_allows_llm_fallback(monkeypatch):
    toolkit = SynPlannerToolkit()
    _patch_planning_dependencies(monkeypatch, toolkit)
    calls = []

    def fake_create_tree(smiles, profile=None):
        calls.append(profile.name)
        return _FakeTree([])

    monkeypatch.setattr(toolkit, "_create_and_search_tree", fake_create_tree)

    plan = toolkit.plan_synthesis("CCO")

    assert plan["routes"] == []
    assert plan["successful_attempt"] is None
    assert plan["llm_fallback_allowed"] is True
    assert calls == [attempt["profile"] for attempt in plan["attempts"]]
    assert calls[-1] == "evaluation_first"
    assert all("tree" not in attempt for attempt in plan["attempts"])


def test_plan_synthesis_stores_report_ready_session_state(monkeypatch):
    toolkit = SynPlannerToolkit()
    _patch_planning_dependencies(monkeypatch, toolkit)
    agent = types.SimpleNamespace(session_state={})
    session_state = {}

    monkeypatch.setattr(
        toolkit, "_create_and_search_tree", lambda smiles, profile=None: _FakeTree([42])
    )

    plan = toolkit.plan_synthesis("CCO", agent=agent, session_state=session_state)

    report_plan = agent.session_state["synplanner_plan"]
    assert session_state["synplanner_plan"] == report_plan
    assert plan["synthesis_report_data"] == report_plan
    assert report_plan["query"] == "CCO"
    assert report_plan["smiles"] == "CCO"
    assert len(report_plan["routes"]) == 1
    assert report_plan["successful_attempt"] == "standard"
    assert report_plan["llm_fallback_allowed"] is False
    assert report_plan["visualizations"] == []
    assert "raw" not in report_plan
    assert "tree" not in report_plan


def test_plan_synthesis_stores_report_state_without_agent(monkeypatch):
    toolkit = SynPlannerToolkit()
    _patch_planning_dependencies(monkeypatch, toolkit)
    session_state = {}

    monkeypatch.setattr(
        toolkit, "_create_and_search_tree", lambda smiles, profile=None: _FakeTree([42])
    )

    plan = toolkit.plan_synthesis("CCO", session_state=session_state)

    report_plan = session_state["synplanner_plan"]
    assert plan["synthesis_report_data"] == report_plan
    assert report_plan["smiles"] == "CCO"
    assert len(report_plan["routes"]) == 1


def test_plan_synthesis_stores_no_route_report_state(monkeypatch):
    toolkit = SynPlannerToolkit()
    _patch_planning_dependencies(monkeypatch, toolkit)
    agent = types.SimpleNamespace(session_state={})
    session_state = {}

    monkeypatch.setattr(
        toolkit, "_create_and_search_tree", lambda smiles, profile=None: _FakeTree([])
    )

    plan = toolkit.plan_synthesis("CCO", agent=agent, session_state=session_state)

    report_plan = agent.session_state["synplanner_plan"]
    assert session_state["synplanner_plan"] == report_plan
    assert plan["synthesis_report_data"] == report_plan
    assert report_plan["routes"] == []
    assert report_plan["successful_attempt"] is None
    assert report_plan["llm_fallback_allowed"] is True
    assert report_plan["attempts"]
    assert all("tree" not in attempt for attempt in report_plan["attempts"])


def test_plan_synthesis_keeps_large_visualization_payloads_out_of_report_state(monkeypatch):
    toolkit = SynPlannerToolkit()
    agent = types.SimpleNamespace(session_state={})
    session_state = {}
    large_visualization = {
        "node_id": 42,
        "score": 0.7,
        "svg": "<svg>large payload</svg>",
        "svg_data_url": "data:image/svg+xml;base64,large-payload",
        "png_path": "s3://bucket/sessions/test/synplanner_route_42.png",
        "svg_path": "s3://bucket/sessions/test/synplanner_route_42.svg",
    }

    monkeypatch.setattr(toolkit, "_load_synplanner_components", lambda: None)
    monkeypatch.setattr(toolkit, "get_basic_descriptors", lambda smiles: {})
    monkeypatch.setattr(
        toolkit, "_create_and_search_tree", lambda smiles, profile=None: _FakeTree([42])
    )
    monkeypatch.setattr(
        toolkit,
        "_generate_route_visualizations",
        lambda *args, **kwargs: [large_visualization],
    )

    plan = toolkit.plan_synthesis("CCO", agent=agent, session_state=session_state)

    report_viz = agent.session_state["synplanner_plan"]["visualizations"][0]
    assert session_state["synplanner_plan"] == agent.session_state["synplanner_plan"]
    assert plan["synthesis_report_data"] == agent.session_state["synplanner_plan"]
    assert report_viz["png_path"] == large_visualization["png_path"]
    assert report_viz["svg_path"] == large_visualization["svg_path"]
    assert "svg" not in report_viz
    assert "svg_data_url" not in report_viz


def test_get_route_visualizations_updates_report_ready_session_state(monkeypatch):
    toolkit = SynPlannerToolkit()
    _patch_planning_dependencies(monkeypatch, toolkit)
    agent = types.SimpleNamespace(session_state={})
    session_state = {}

    monkeypatch.setattr(
        toolkit, "_create_and_search_tree", lambda smiles, profile=None: _FakeTree([42])
    )
    toolkit.plan_synthesis("CCO", agent=agent, session_state=session_state)

    toolkit._last_plan["visualizations"] = [
        {
            "node_id": 42,
            "score": 0.7,
            "svg": "<svg>large payload</svg>",
            "svg_data_url": "data:image/svg+xml;base64,large-payload",
            "png_path": "s3://bucket/sessions/test/synplanner_route_42.png",
            "svg_path": "s3://bucket/sessions/test/synplanner_route_42.svg",
        }
    ]

    result = toolkit.get_route_visualizations("CCO", agent=agent, session_state=session_state)

    assert result["num_routes"] == 1
    report_plan = agent.session_state["synplanner_plan"]
    assert session_state["synplanner_plan"] == report_plan
    assert result["synthesis_report_data"] == report_plan
    assert report_plan["visualization_available"] is True
    assert report_plan["num_visualizations"] == 1
    assert report_plan["visualizations"][0]["png_path"].endswith("synplanner_route_42.png")
    assert "svg" not in report_plan["visualizations"][0]
    assert "svg_data_url" not in report_plan["visualizations"][0]
