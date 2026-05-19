"""Explicit registry of ChemSpace toolkit methods exposed as MCP tools.

The registry is intentionally curated rather than introspected: it lets us
choose names, hide internal helpers, and document overrides like the ChEMBL
LLM-as-judge gating without surfacing those knobs to the MCP client.

Toolkit instances are created lazily by their factories on first use so that
heavy module imports (torch, RDKit caches, ChEMBL DB drivers) are paid only
when the corresponding tool is actually called.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Iterable, List

from .tool_adapter import ToolSpec


def _factory(import_path: str) -> Callable[[], Any]:
    """Return a memoised factory that lazily imports and instantiates a toolkit."""

    @functools.lru_cache(maxsize=1)
    def _build() -> Any:
        module_name, _, class_name = import_path.rpartition(":")
        if not module_name or not class_name:
            raise ValueError(f"Invalid factory path: {import_path!r}")
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        return cls()

    return _build


_CHEMBL = _factory("cs_copilot.tools.databases.chembl:ChemblToolkit")
_GTM = _factory("cs_copilot.tools.chemography.gtm:GTMToolkit")
_SIMILARITY = _factory("cs_copilot.tools.chemistry.similarity_toolkit:ChemicalSimilarityToolkit")
_SESSION_MEMORY = _factory("cs_copilot.tools.io.session_memory:SessionMemoryToolkit")
_ROBUSTNESS = _factory("cs_copilot.tools.analysis.robustness_toolkit:RobustnessAnalysisToolkit")


# Report export is a pair of module-level functions, not a toolkit instance.
# The facade below adapts them so the adapter can treat them uniformly.
class _ReportExportFacade:
    """Tiny adapter that exposes ``report_export`` module functions as methods."""

    def __init__(self) -> None:
        from cs_copilot.tools.io.report_export import save_markdown_report, save_rich_report

        self.save_markdown = save_markdown_report
        self.save_rich = save_rich_report


@functools.lru_cache(maxsize=1)
def _report_facade() -> _ReportExportFacade:
    return _ReportExportFacade()


# ChEMBL ---------------------------------------------------------------------

_CHEMBL_SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name="chembl_fetch_compounds",
        toolkit_factory=_CHEMBL,
        method="fetch_compounds",
        summary=(
            "Fetch ChEMBL bioactivity data for one or more keyword targets. "
            "In MCP mode the in-process LLM-as-judge filtering is disabled; "
            "use the chembl_retrieval_judge / chembl_metadata_judge prompts "
            "if you want to perform equivalent filtering with this client."
        ),
        forces={"enable_retrieval_judge": False, "enable_metadata_judge": False},
    ),
    ToolSpec(
        mcp_name="chembl_describe_dataset",
        toolkit_factory=_CHEMBL,
        method="describe_dataset",
        summary="Return a structural summary of a previously fetched ChEMBL dataset by path.",
    ),
    ToolSpec(
        mcp_name="chembl_convert_to_chembl_query",
        toolkit_factory=_CHEMBL,
        method="convert_to_chembl_query",
        summary=(
            "Rewrite a free-form natural language query into the canonical "
            "ChEMBL keyword form accepted by chembl_fetch_compounds."
        ),
    ),
]


# GTM ------------------------------------------------------------------------

# (mcp_name, method_name, summary) — explicit names avoid both `gtm_gtm_*`
# doubles and bare names like `save_gtm_and_data`.
_GTM_METHODS = [
    (
        "gtm_optimization",
        "gtm_optimization",
        "Build a GTM model from a dataset (optimisation pass).",
    ),
    (
        "gtm_save_model_and_data",
        "save_gtm_and_data",
        "Persist a fitted GTM model and the projected source dataset.",
    ),
    (
        "gtm_load_model_only",
        "load_gtm_model_only",
        "Load a previously saved GTM model into the session.",
    ),
    (
        "gtm_load_density_matrix",
        "load_gtm_get_density_matrix",
        "Load a GTM model and return its node density / responsibility matrix.",
    ),
    (
        "gtm_load_and_prep_data",
        "load_and_prep_data",
        "Project a dataset onto a loaded GTM model and prepare lookup tables.",
    ),
    (
        "gtm_analyze_scaffolds_in_nodes",
        "analyze_scaffolds_in_nodes",
        "Summarise scaffolds residing in the given GTM node ids.",
    ),
    (
        "gtm_check_source_datasets_in_nodes",
        "check_source_datasets_in_nodes",
        "Report which source datasets contribute to the given GTM node ids.",
    ),
    (
        "gtm_node_id_from_coords",
        "node_id_from_coords",
        "Return the GTM node id closest to a (x, y) latent coordinate.",
    ),
    ("gtm_get_density_summary", "get_density_summary", "Return the top-N densest GTM nodes."),
    (
        "gtm_get_activity_landscape_summary",
        "get_activity_landscape_summary",
        "Summarise an activity landscape view built from the loaded GTM map.",
    ),
    (
        "gtm_get_node_lookup_summary",
        "get_node_lookup_summary",
        "Return a compact lookup table for the loaded GTM map.",
    ),
    ("gtm_sample_nodes", "sample_nodes", "Sample molecules located inside the given GTM nodes."),
    (
        "gtm_sample_dense_nodes",
        "sample_dense_nodes",
        "Sample molecules from the densest GTM nodes.",
    ),
    (
        "gtm_sample_activity_landscape_nodes",
        "sample_activity_landscape_nodes",
        "Sample molecules from activity-landscape regions of interest.",
    ),
    (
        "gtm_sample_top_activity_molecules",
        "sample_top_activity_molecules",
        "Sample top-activity molecules on the loaded GTM map.",
    ),
    (
        "gtm_sample_by_coordinates",
        "sample_by_coordinates",
        "Sample molecules near the supplied (x, y) latent coordinates.",
    ),
    (
        "gtm_create_activity_landscapes",
        "create_activity_landscapes",
        "Build activity landscape views from the loaded GTM map and dataset.",
    ),
    (
        "gtm_load_activity_landscape_csv",
        "load_activity_landscape_csv",
        "Load a previously saved activity-landscape CSV back into the session.",
    ),
    (
        "gtm_save_landscape_plot",
        "save_gtm_landscape_plot",
        "Save a static GTM activity / density landscape plot.",
    ),
    (
        "gtm_project_data",
        "project_data_on_gtm",
        "Project a new dataset onto a loaded GTM model.",
    ),
    (
        "gtm_train_on_latent_space",
        "train_gtm_on_latent_space",
        "Train a GTM model on autoencoder latent vectors.",
    ),
    (
        "gtm_load_latent_data",
        "load_latent_data_on_gtm",
        "Load latent-space data and project it onto a loaded GTM model.",
    ),
    (
        "gtm_create_peptide_activity_landscapes",
        "create_peptide_activity_landscapes",
        "Build peptide-specific activity landscape views.",
    ),
]


_GTM_SPECS: List[ToolSpec] = [
    ToolSpec(mcp_name=mcp_name, toolkit_factory=_GTM, method=method, summary=summary)
    for mcp_name, method, summary in _GTM_METHODS
]


# Chemical similarity --------------------------------------------------------

_SIMILARITY_METHODS = [
    ("calculate_tanimoto_similarity", "Tanimoto similarity for one or more SMILES pairs."),
    ("calculate_dice_similarity", "Dice similarity for one or more SMILES pairs."),
    ("calculate_tversky_similarity", "Tversky similarity for one or more SMILES pairs."),
    ("calculate_cosine_similarity", "Cosine similarity for one or more SMILES pairs."),
    (
        "calculate_euclidean_distance",
        "Euclidean distance between fingerprint vectors of SMILES pairs.",
    ),
    (
        "calculate_all_similarities",
        "Compute Tanimoto / Dice / Tversky / cosine in a single call.",
    ),
    (
        "find_most_similar",
        "Find the most similar molecules to a query SMILES from a candidate list.",
    ),
]


_SIMILARITY_SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name=f"chem_{name}",
        toolkit_factory=_SIMILARITY,
        method=name,
        summary=summary,
    )
    for name, summary in _SIMILARITY_METHODS
]


# Session memory ------------------------------------------------------------

_SESSION_METHODS = [
    ("list_session_objects", "List structured objects stored in the active session."),
    ("list_loadable_session_data", "List session-resident datasets that can be reloaded."),
    ("get_session_object", "Return a session object by id."),
    ("select_session_object", "Mark a session object as the current one for its role."),
    ("resolve_session_reference", "Resolve a free-form reference to a session object id."),
    ("resolve_candidate_set", "Resolve a candidate-set reference to a stored object."),
    ("load_candidate_set_artifact", "Materialise a candidate-set artifact path."),
    (
        "materialize_candidate_set_dataset",
        "Materialise the dataset rows of a candidate set.",
    ),
    ("summarize_session_memory", "Return a compact textual summary of session memory."),
]


_SESSION_SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name=f"session_{name}",
        toolkit_factory=_SESSION_MEMORY,
        method=name,
        summary=summary,
    )
    for name, summary in _SESSION_METHODS
]


# Reports --------------------------------------------------------------------

_REPORT_SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name="report_save_markdown",
        toolkit_factory=_report_facade,
        method="save_markdown",
        summary="Save a markdown report into the session-scoped storage layout.",
    ),
    ToolSpec(
        mcp_name="report_save_rich",
        toolkit_factory=_report_facade,
        method="save_rich",
        summary="Save an image-rich (HTML/PDF/Markdown) report into the session layout.",
    ),
]


# Robustness analysis -------------------------------------------------------

_ROBUSTNESS_METHODS = [
    ("load_test_results", "Load the raw results of a robustness test run."),
    ("load_test_summary_csv", "Load the per-prompt summary CSV of a robustness test run."),
    ("list_available_test_runs", "List robustness test runs available under the data root."),
    ("analyze_score_distribution", "Summarise score distribution for a robustness run."),
    ("identify_failing_prompts", "List failing prompts above a score threshold."),
    ("compare_test_runs", "Compare two robustness test runs side by side."),
    ("analyze_temporal_trends", "Summarise robustness score trends across runs."),
    ("generate_insights", "Generate textual insights about a robustness run."),
    ("export_analysis_report", "Persist a robustness analysis report to storage."),
]


_ROBUSTNESS_SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name=f"robustness_{name}",
        toolkit_factory=_ROBUSTNESS,
        method=name,
        summary=summary,
    )
    for name, summary in _ROBUSTNESS_METHODS
]


def iter_specs() -> Iterable[ToolSpec]:
    """Yield every :class:`ToolSpec` exposed by the MCP server."""

    yield from _CHEMBL_SPECS
    yield from _GTM_SPECS
    yield from _SIMILARITY_SPECS
    yield from _SESSION_SPECS
    yield from _REPORT_SPECS
    yield from _ROBUSTNESS_SPECS


def all_specs() -> List[ToolSpec]:
    """Return every :class:`ToolSpec` as a list (handy for tests)."""

    return list(iter_specs())
