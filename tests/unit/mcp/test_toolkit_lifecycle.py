"""MCP toolkit lifecycle and declarative network-contract tests."""

from __future__ import annotations

from cs_copilot.mcp.context import MCPAgentContext
from cs_copilot.mcp.server import _register_tools
from cs_copilot.mcp.tool_adapter import ToolSpec
from cs_copilot.mcp.tools_registry import all_specs


class _Toolkit:
    def first(self) -> None:
        pass

    def second(self) -> None:
        pass


class _Server:
    def __init__(self) -> None:
        self.tools: list[tuple[object, dict[str, object]]] = []

    def add_tool(self, tool: object, **kwargs: object) -> None:
        self.tools.append((tool, kwargs))


class _Annotations:
    def __init__(self, **kwargs: object) -> None:
        self.values = kwargs


def test_registration_shares_instances_within_server_and_isolates_servers(monkeypatch):
    built: list[_Toolkit] = []
    adapted: list[tuple[str, _Toolkit, MCPAgentContext]] = []

    def toolkit_factory() -> _Toolkit:
        instance = _Toolkit()
        built.append(instance)
        return instance

    specs = (
        ToolSpec(
            mcp_name="test_first",
            toolkit_factory=toolkit_factory,
            method="first",
            summary="First test tool.",
            read_only=True,
        ),
        ToolSpec(
            mcp_name="test_second",
            toolkit_factory=toolkit_factory,
            method="second",
            summary="Second test tool.",
            read_only=True,
        ),
    )

    def fake_iter_specs(profile=None):
        del profile
        return iter(specs)

    def fake_build_tool(spec, instance, ctx):
        adapted.append((spec.mcp_name, instance, ctx))

        def tool() -> None:
            pass

        return tool

    monkeypatch.setattr("cs_copilot.mcp.tools_registry.iter_specs", fake_iter_specs)
    monkeypatch.setattr("cs_copilot.mcp.tool_adapter.build_tool", fake_build_tool)

    first_ctx = MCPAgentContext()
    second_ctx = MCPAgentContext()
    _register_tools(_Server(), first_ctx, _Annotations, profile="standard")
    _register_tools(_Server(), second_ctx, _Annotations, profile="standard")

    assert len(built) == 2
    assert adapted[0][1] is built[0]
    assert adapted[1][1] is built[0]
    assert adapted[2][1] is built[1]
    assert adapted[3][1] is built[1]
    assert built[0] is not built[1]
    assert all(entry[2] is first_ctx for entry in adapted[:2])
    assert all(entry[2] is second_ctx for entry in adapted[2:])


def test_lazy_import_factory_returns_fresh_instances():
    from cs_copilot.mcp.tool_specs.common import factory

    toolkit_factory = factory("cs_copilot.mcp.facades.skills:SkillFacade")

    assert toolkit_factory() is not toolkit_factory()


def test_network_capable_tools_declare_their_contract():
    network_tools = {spec.mcp_name for spec in all_specs() if spec.requires_network}

    assert network_tools == {
        "chembl_fetch_compounds",
        "gtm_create_activity_landscapes",
        "gtm_load_and_prep_data",
        "gtm_load_density_matrix",
        "gtm_load_model_only",
        "gtm_optimization",
        "gtm_project_data",
        "gtm_save_density_plot",
        "gtm_save_landscape_plot",
        "mol_design_molecules",
        "mol_generate_analogs",
        "mol_interpolate_molecules",
        "mol_register_design_candidates",
        "peptide_decode_latent",
        "peptide_design_interpolation",
        "peptide_design_peptides",
        "peptide_encode_peptides",
        "peptide_explore_latent_neighborhood",
        "peptide_generate_analogs",
        "peptide_get_latent_dimension",
        "peptide_get_model_info",
        "peptide_interpolate_peptides",
        "peptide_reconstruct_sequence",
        "peptide_sample_peptides",
        "peptide_validate_model_loaded",
        "synplanner_convert_name_to_smiles",
        "synplanner_get_route_visualizations",
        "synplanner_identify_input",
        "synplanner_plan_synthesis",
    }
