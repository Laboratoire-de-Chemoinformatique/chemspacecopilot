"""Drift guard: every Agno team tool maps to an execution contract.

In-process tool calls are recorded under the identity and contract of their
MCP twin whenever one exists. A toolkit function without a twin falls back to
a conservative synthesized contract; that set is pinned here so that adding
a tool without deciding its contract is a visible change.
"""

from __future__ import annotations

import inspect
from typing import Any, Iterator

import pytest

from cs_copilot.agents import factories as factory_module
from cs_copilot.agents.execution_contracts import (
    binding_target,
    contract_index,
    function_key,
    resolve_contract,
)
from cs_copilot.mcp.tools_registry import all_specs
from cs_copilot.tools import SessionMemoryToolkit, SkillToolkit
from cs_copilot.tools.chemistry.autoencoder_toolkit import AutoencoderToolkit
from cs_copilot.tools.chemistry.peptide_designer_toolkit import PeptideDesignerToolkit

DEFAULT_CONTRACT_TOOLS = {
    "agno.AutoencoderToolkit.decode_latent",
    "agno.AutoencoderToolkit.encode_smiles",
    "agno.AutoencoderToolkit.explore_latent_neighborhood",
    "agno.AutoencoderToolkit.get_latent_dimension",
    "agno.AutoencoderToolkit.get_model_info",
    "agno.AutoencoderToolkit.interpolate_molecules",
    "agno.AutoencoderToolkit.reconstruct_smiles",
    "agno.AutoencoderToolkit.sample_molecules",
    "agno.AutoencoderToolkit.validate_model_loaded",
    "agno.ChemblToolkit.get_capabilities",
    "agno.ChemblToolkit.ping",
    "agno.ChemicalSimilarityToolkit.batch_validate_smiles",
    "agno.ChemicalSimilarityToolkit.filter_valid_smiles",
    "agno.ChemicalSimilarityToolkit.generate_fingerprint",
    "agno.ChemicalSimilarityToolkit.get_basic_descriptors",
    "agno.ChemicalSimilarityToolkit.get_lipinski_descriptors",
    "agno.ChemicalSimilarityToolkit.get_molecular_formula",
    "agno.ChemicalSimilarityToolkit.get_molecular_weight",
    "agno.ChemicalSimilarityToolkit.get_smiles_statistics",
    "agno.ChemicalSimilarityToolkit.canonicalize_smiles",
    "agno.ChemicalSimilarityToolkit.validate_smiles",
    "agno.PeptideDesignerToolkit.analyze_peptide_candidates",
    "agno.PeptideDesignerToolkit.sample_peptides_from_landscape",
}


@pytest.fixture(scope="module")
def agno_tools() -> list[tuple[str, Any, Any, str]]:
    patches = []
    from unittest.mock import patch

    for cls in (AutoencoderToolkit, PeptideDesignerToolkit):
        for name in ("_ensure_model_exists", "_load_model"):
            if hasattr(cls, name):
                patcher = patch.object(cls, name, lambda *args, **kwargs: None)
                patcher.start()
                patches.append(patcher)
    try:
        return list(_iter_team_tools())
    finally:
        for patcher in patches:
            patcher.stop()


def _iter_team_tools() -> Iterator[tuple[str, Any, Any, str]]:
    for cls in vars(factory_module).values():
        if not (
            inspect.isclass(cls)
            and issubclass(cls, factory_module.BaseAgentFactory)
            and getattr(cls, "agent_type", None)
        ):
            continue
        for tool in cls().get_agent_config().tools:
            yield from _tool_functions(cls.agent_type, tool)
    for tool in (SessionMemoryToolkit(), SkillToolkit()):
        yield from _tool_functions("coordinator", tool)


def _tool_functions(role: str, tool: Any) -> Iterator[tuple[str, Any, Any, str]]:
    functions = getattr(tool, "functions", None)
    if isinstance(functions, dict):
        for name, function in functions.items():
            yield role, tool, function.entrypoint, name
    else:
        yield role, None, tool, tool.__name__


def test_every_catalog_binding_resolves_to_an_exposed_agno_tool(agno_tools):
    exposed = {function_key(function) for _, _, function, _ in agno_tools}
    index = contract_index()
    assert index, "no MCP spec declares an Agno binding"
    for spec in all_specs():
        for binding in spec.agno_bindings:
            key = function_key(binding_target(binding))
            assert index[key].mcp_name == spec.mcp_name
            assert key in exposed, f"{spec.mcp_name}: {binding} is not an Agno team tool"


def test_every_agno_tool_resolves_to_a_contract(agno_tools):
    defaults = set()
    for role, owner, function, name in agno_tools:
        contract, source = resolve_contract(function, owner=owner)
        assert contract.mcp_name, (role, name)
        if source == "default":
            defaults.add(contract.mcp_name)
    assert defaults == DEFAULT_CONTRACT_TOOLS


def test_bound_agno_tools_use_their_mcp_twin_identity(agno_tools):
    by_function = {
        getattr(function, "__name__", name): resolve_contract(function, owner=owner)[0].mcp_name
        for _, owner, function, name in agno_tools
    }
    assert by_function["fetch_compounds"] == "chembl_fetch_compounds"
    assert by_function["plan_analysis"] == "chemspace_plan_analysis"
    assert by_function["save_gtm_plot"] == "gtm_save_density_plot"
    assert by_function["save_rich_report"] == "report_save_rich"
    assert by_function["create_pandas_dataframe"] == "pandas_create_dataframe"
