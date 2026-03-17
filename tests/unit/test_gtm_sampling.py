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

from cs_copilot.tools.chemography.gtm import GTMToolkit  # noqa: E402


def make_toolkit(**overrides) -> GTMToolkit:
    """Create a GTMToolkit instance with mocked GTMData tables."""

    toolkit = GTMToolkit()
    gtm_data = SimpleNamespace(
        source_mols=None,
        activity_table=None,
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


def test_sample_active_nodes_infers_probability_column():
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
    toolkit = make_toolkit(source_mols=source_mols, activity_table=activity_table)

    result = toolkit.sample_active_nodes(top_n=2)

    assert extract_node_indices(result) == [10, 20]


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


def test_sample_active_nodes_empty_smiles_return():
    source_mols = pd.DataFrame(
        {
            "node_index": [101, 102],
            "smi": ["inactive-1", "inactive-2"],
        }
    )
    activity_table = pd.DataFrame(
        {
            "nodes": [101, 102],
            "potency": [0.1, 0.2],
        }
    )
    toolkit = make_toolkit(source_mols=source_mols, activity_table=activity_table)

    result = toolkit.sample_active_nodes(
        min_value=0.9, activity_column="potency", return_format="smiles"
    )

    assert result == []
