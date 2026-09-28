"""GTM MCP tool specs."""

from __future__ import annotations

from typing import List

from ..tool_adapter import ToolSpec
from .common import factory

_GTM = factory("cs_copilot.tools.chemography.gtm:GTMToolkit")
_GTM_MCP = factory("cs_copilot.mcp.facades.gtm:GTMMCPFacade")
_READ_ARTIFACT_FIELDS = {
    "gtm_optimization": ("df_csv_path",),
    "gtm_load_model_only": ("gtm_file",),
    "gtm_load_density_matrix": ("dataset_file", "gtm_file"),
    "gtm_load_and_prep_data": ("dataset", "gtm_model"),
    "gtm_create_activity_landscapes": ("dataset", "gtm_model"),
    "gtm_load_activity_landscape_csv": ("landscape_csv",),
    "gtm_save_landscape_plot": (
        "landscape_file",
        "overlay_dataset_file",
        "gtm_model_file",
    ),
    "gtm_project_data": ("dataset_file", "gtm_model_file"),
    "gtm_train_on_latent_space": ("latent_vectors_csv",),
    "gtm_load_latent_data": ("latent_vectors_csv", "sequences_csv"),
    "gtm_create_peptide_activity_landscapes": (
        "dbaasp_path",
        "latent_vectors_csv",
    ),
}
_TRUSTED_PICKLE_FIELDS = {
    "gtm_load_model_only": ("gtm_file",),
    "gtm_load_density_matrix": ("gtm_file",),
    "gtm_load_and_prep_data": ("gtm_model",),
    "gtm_create_activity_landscapes": ("gtm_model",),
    "gtm_save_landscape_plot": ("gtm_model_file",),
    "gtm_project_data": ("gtm_model_file",),
}

_METHODS = [
    (
        "gtm_optimization",
        "gtm_optimization",
        "Build a GTM model from a dataset (optimisation pass).",
        False,
        True,
    ),
    (
        "gtm_save_model_and_data",
        "save_gtm_and_data",
        "Persist a fitted GTM model and the projected source dataset.",
        False,
        False,
    ),
    (
        "gtm_load_model_only",
        "load_gtm_model_only",
        "Load a previously saved GTM model into the session.",
        False,
        True,
    ),
    (
        "gtm_load_density_matrix",
        "load_gtm_get_density_matrix",
        "Load a GTM model and return its node density / responsibility matrix.",
        False,
        True,
    ),
    (
        "gtm_load_and_prep_data",
        "load_and_prep_data",
        "Project a dataset onto a loaded GTM model and prepare lookup tables.",
        False,
        True,
    ),
    (
        "gtm_analyze_scaffolds_in_nodes",
        "analyze_scaffolds_in_nodes",
        "Summarise scaffolds residing in the given GTM node ids.",
        True,
        False,
    ),
    (
        "gtm_check_source_datasets_in_nodes",
        "check_source_datasets_in_nodes",
        "Report which source datasets contribute to the given GTM node ids.",
        True,
        False,
    ),
    (
        "gtm_node_id_from_coords",
        "node_id_from_coords",
        "Return the GTM node id closest to a (x, y) latent coordinate.",
        True,
        False,
    ),
    (
        "gtm_get_density_summary",
        "get_density_summary",
        "Return the top-N densest GTM nodes.",
        True,
        False,
    ),
    (
        "gtm_get_activity_landscape_summary",
        "get_activity_landscape_summary",
        "Summarise an activity landscape view built from the loaded GTM map.",
        True,
        False,
    ),
    (
        "gtm_get_node_lookup_summary",
        "get_node_lookup_summary",
        "Return a compact lookup table for the loaded GTM map.",
        True,
        False,
    ),
    (
        "gtm_sample_nodes",
        "sample_nodes",
        "Sample molecules located inside the given GTM nodes.",
        False,
        False,
    ),
    (
        "gtm_sample_dense_nodes",
        "sample_dense_nodes",
        "Sample molecules from the densest GTM nodes.",
        False,
        False,
    ),
    (
        "gtm_sample_activity_landscape_nodes",
        "sample_activity_landscape_nodes",
        "Sample molecules from activity-landscape regions of interest.",
        False,
        False,
    ),
    (
        "gtm_sample_top_activity_molecules",
        "sample_top_activity_molecules",
        "Sample top-activity molecules on the loaded GTM map.",
        False,
        False,
    ),
    (
        "gtm_sample_by_coordinates",
        "sample_by_coordinates",
        "Sample molecules near the supplied (x, y) latent coordinates.",
        False,
        False,
    ),
    (
        "gtm_create_activity_landscapes",
        "create_activity_landscapes",
        "Build activity landscape views from the loaded GTM map and dataset.",
        False,
        True,
    ),
    (
        "gtm_load_activity_landscape_csv",
        "load_activity_landscape_csv",
        "Load a previously saved activity-landscape CSV back into the session.",
        False,
        False,
    ),
    (
        "gtm_save_landscape_plot",
        "save_gtm_landscape_plot",
        "Save a static GTM activity / density landscape plot.",
        False,
        True,
    ),
    (
        "gtm_project_data",
        "project_data_on_gtm",
        "Project a new dataset onto a loaded GTM model.",
        False,
        True,
    ),
    (
        "gtm_train_on_latent_space",
        "train_gtm_on_latent_space",
        "Train a GTM model on autoencoder latent vectors.",
        False,
        False,
    ),
    (
        "gtm_load_latent_data",
        "load_latent_data_on_gtm",
        "Load latent-space data and project it onto a loaded GTM model.",
        False,
        False,
    ),
    (
        "gtm_create_peptide_activity_landscapes",
        "create_peptide_activity_landscapes",
        "Build peptide-specific activity landscape views.",
        False,
        False,
    ),
]

SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name=mcp_name,
        toolkit_factory=_GTM,
        method=method,
        summary=summary,
        read_only=read_only,
        requires_network=requires_network,
        read_artifact_fields=_READ_ARTIFACT_FIELDS.get(mcp_name, ()),
        trusted_pickle_fields=_TRUSTED_PICKLE_FIELDS.get(mcp_name, ()),
    )
    for mcp_name, method, summary, read_only, requires_network in _METHODS
]

SPECS.append(
    ToolSpec(
        mcp_name="gtm_save_density_plot",
        toolkit_factory=_GTM_MCP,
        method="save_density_plot",
        summary=(
            "Generate and save a GTM density landscape plot with projected compound points "
            "from a dataset and the current or explicit GTM model."
        ),
        read_only=False,
        requires_network=True,
        read_artifact_fields=("dataset_file", "gtm_model_file"),
        trusted_pickle_fields=("gtm_model_file",),
    )
)
