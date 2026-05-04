"""Unit tests for GTM sampling helpers exposed through the toolkit."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

tools_package_name = "cs_copilot.tools"
if tools_package_name not in sys.modules:
    tools_stub = ModuleType(tools_package_name)
    tools_stub.__path__ = [str(SRC_PATH / "cs_copilot" / "tools")]
    sys.modules[tools_package_name] = tools_stub

from cs_copilot.tools.chemography import gtm_operations  # noqa: E402
from cs_copilot.tools.chemography.gtm import GTMToolkit  # noqa: E402


def make_toolkit(**overrides) -> GTMToolkit:
    """Create a GTMToolkit instance with mocked GTMData tables."""

    toolkit = GTMToolkit()
    gtm_data = SimpleNamespace(
        source_mols=None,
        activity_landscapes={},
        landscape_artifacts={},
        node_lookup_by_coords=None,
        node_lookup_by_node=None,
        source=None,
    )
    for key, value in overrides.items():
        setattr(gtm_data, key, value)
    toolkit._gtm_data = gtm_data
    return toolkit


def extract_node_indices(table_str: str) -> list[int]:
    """Parse the node_index column from the stringified DataFrame."""

    lines = table_str.splitlines()[1:]
    return [int(line.split()[0]) for line in lines]


def test_sample_dense_nodes_prioritizes_filtered_density():
    source_mols = pd.DataFrame(
        {
            "node_index": [1, 2, 3],
            "smi": ["mol-1", "mol-2", "mol-3"],
            "x": [0, 1, 2],
            "y": [0, 1, 2],
        }
    )
    density_table = pd.DataFrame(
        {
            "nodes": [1, 2, 3],
            "filtered_density": [0.2, 0.9, 0.8],
            "density": [0.3, 0.1, 0.7],
        }
    )
    toolkit = make_toolkit(source_mols=source_mols, source=density_table)

    result = toolkit.sample_dense_nodes(top_n=2)

    assert extract_node_indices(result) == [2, 3]


def test_sample_activity_landscape_nodes_infers_probability_column():
    source_mols = pd.DataFrame(
        {
            "node_index": [10, 20, 30],
            "smi": ["n10", "n20", "n30"],
            "x": [0, 0, 0],
            "y": [0, 0, 0],
        }
    )
    activity_table = pd.DataFrame(
        {
            "nodes": [10, 20, 30],
            "potency_prob": [0.9, 0.6, 0.2],
            "selectivity_prob": [0.1, 0.9, 0.4],
        }
    )
    toolkit = make_toolkit(
        source_mols=source_mols,
        activity_landscapes={"classification": activity_table},
    )

    result = toolkit.sample_activity_landscape_nodes(
        top_n=2,
        landscape_type="classification",
    )

    assert extract_node_indices(result) == [10, 20]


def test_sample_top_activity_molecules_uses_row_level_activity_column():
    source_mols = pd.DataFrame(
        {
            "node_index": [1, 2, 3],
            "smi": ["low", "high", "mid"],
            "activity_final": [5.1, 8.4, 6.7],
        }
    )
    toolkit = make_toolkit(source_mols=source_mols)

    result = toolkit.sample_top_activity_molecules(
        activity_column="activity_final",
        top_n=2,
        return_format="smiles",
    )

    assert result == ["high", "mid"]


def test_activity_landscape_sampler_rejects_molecule_activity_column():
    source_mols = pd.DataFrame(
        {
            "node_index": [10],
            "smi": ["n10"],
        }
    )
    landscape = pd.DataFrame(
        {
            "nodes": [10],
            "filtered_reg_density": [7.5],
        }
    )
    toolkit = make_toolkit(
        source_mols=source_mols,
        activity_landscapes={"regression": landscape},
    )

    try:
        toolkit.sample_activity_landscape_nodes(metric_column="activity_final")
    except ValueError as exc:
        assert "molecule-level activity column" in str(exc)
        assert "sample_top_activity_molecules" in str(exc)
    else:
        raise AssertionError("Expected molecule-level activity hint")


def test_sample_by_coordinates_uses_lookup_table():
    source_mols = pd.DataFrame(
        {
            "node_index": [5, 6],
            "smi": ["node-5", "node-6"],
            "x": [0, 2],
            "y": [0, 2],
        }
    )
    lookup = pd.DataFrame(
        {"nodes": [5, 6]},
        index=pd.MultiIndex.from_tuples([(0, 0), (2, 2)], names=["x", "y"]),
    )
    toolkit = make_toolkit(
        source_mols=source_mols,
        node_lookup_by_coords=lookup,
    )

    result = toolkit.sample_by_coordinates([(2, 2)])

    assert extract_node_indices(result) == [6]


def test_sample_nodes_returns_smiles_list():
    source_mols = pd.DataFrame(
        {
            "node_index": [1, 2],
            "smi": ["mol-1", "mol-2"],
        }
    )
    toolkit = make_toolkit(source_mols=source_mols)

    result = toolkit.sample_nodes([1, 2], return_format="smiles")

    assert result == ["mol-1", "mol-2"]


def test_sample_nodes_preserves_activity_evidence_in_dataframe_and_memory():
    source_mols = pd.DataFrame(
        {
            "node_index": [1, 2],
            "smi": ["mol-1", "mol-2"],
            "molecule_chembl_id": ["CHEMBL1", "CHEMBL2"],
            "activity_final": [8.5, 6.1],
            "pchembl_value": [8.5, 6.1],
        }
    )
    toolkit = make_toolkit(source_mols=source_mols)
    session_state = {}

    result = toolkit.sample_nodes([1], return_format="dataframe", session_state=session_state)

    assert result.loc[0, "activity_final"] == 8.5
    zone = session_state["session_objects"]["zones"]["zone_001"]
    assert zone["sample_preview"][0]["activity_final"] == 8.5


def test_gtm_source_mols_builder_keeps_scalar_metadata_and_drops_heavy_columns():
    coords = pd.DataFrame({"node_index": [1], "x": [0.5], "y": [0.5]})
    molecules = pd.DataFrame(
        {
            "smi": ["CCO"],
            "project_label": ["lead-series-a"],
            "batch_score": [8.5],
            "morgan_fingerprint": [[0, 1, 0]],
            "image": ["data:image/png;base64,..."],
        }
    )

    source_mols = gtm_operations._build_source_mols(coords, molecules)

    assert source_mols.loc[0, "batch_score"] == 8.5
    assert source_mols.loc[0, "project_label"] == "lead-series-a"
    assert "morgan_fingerprint" not in source_mols.columns
    assert "image" not in source_mols.columns


def test_sample_dense_nodes_dataframe_return():
    source_mols = pd.DataFrame(
        {
            "node_index": [1, 2, 3],
            "smi": ["mol-1", "mol-2", "mol-3"],
            "x": [0, 1, 2],
            "y": [0, 1, 2],
        }
    )
    density_table = pd.DataFrame(
        {
            "nodes": [1, 2, 3],
            "filtered_density": [0.2, 0.9, 0.8],
        }
    )
    toolkit = make_toolkit(source_mols=source_mols, source=density_table)

    result = toolkit.sample_dense_nodes(top_n=1, return_format="dataframe")

    assert isinstance(result, pd.DataFrame)
    assert list(result["node_index"]) == [2]


def test_sample_activity_landscape_nodes_empty_smiles_return():
    source_mols = pd.DataFrame(
        {
            "node_index": [101, 102],
            "smi": ["inactive-1", "inactive-2"],
        }
    )
    activity_table = pd.DataFrame(
        {
            "nodes": [101, 102],
            "filtered_reg_density": [0.1, 0.2],
        }
    )
    toolkit = make_toolkit(
        source_mols=source_mols,
        activity_landscapes={"regression": activity_table},
    )

    result = toolkit.sample_activity_landscape_nodes(
        min_value=0.9,
        metric_column="filtered_reg_density",
        return_format="smiles",
    )

    assert result == []


def test_sample_active_nodes_is_not_registered_as_agent_tool():
    toolkit = GTMToolkit()

    assert "sample_active_nodes" not in toolkit.functions
    assert "sample_activity_landscape_nodes" in toolkit.functions
    assert "sample_top_activity_molecules" in toolkit.functions
