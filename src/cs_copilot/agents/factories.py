#!/usr/bin/env python
# coding: utf-8
"""
Agent factory classes for creating specialized cs_copilot agents.
Contains the base factory class and all specialized factory implementations.
"""

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agno.agent import Agent
from agno.models.base import Model  # Agno v2 base class

from cs_copilot.tools import (
    AutoencoderToolkit,
    ChemblToolkit,
    ChemicalSimilarityToolkit,
    GTMToolkit,
    MolecularDesignerToolkit,
    PeptideDesignerToolkit,
    PointerPandasTools,
    SessionMemoryToolkit,
    SkillToolkit,
    SynPlannerToolkit,
    # SessionToolkit,
    save_gtm_landscape_plot,
    save_gtm_plot,
    save_markdown_report,
    save_rich_report,
)
from cs_copilot.tools.analysis import RobustnessAnalysisToolkit

from .descriptions import (
    CHEMBL_DESCRIPTION,
    CHEMOINFORMATICIAN_DESCRIPTION,
    GTM_AGENT_DESCRIPTION,
    MOLECULAR_DESIGNER_DESCRIPTION,
    PEPTIDE_DESIGNER_DESCRIPTION,
    REPORT_GENERATOR_DESCRIPTION,
    ROBUSTNESS_EVALUATION_DESCRIPTION,
    SINGLE_AGENT_DESCRIPTION,
    SYNPLANNER_DESCRIPTION,
)
from .instructions import (
    CHEMBL_INSTRUCTIONS,
    CHEMOINFORMATICIAN_INSTRUCTIONS,  # Comprehensive chemoinformatics analysis
    GTM_AGENT_INSTRUCTIONS,  # Unified GTM agent (all GTM operations)
    MOLECULAR_DESIGNER_INSTRUCTIONS,
    PEPTIDE_DESIGNER_INSTRUCTIONS,  # Peptide Designer for amino acid sequence generation
    REPORT_GENERATOR_INSTRUCTIONS,  # Universal presentation layer
    ROBUSTNESS_EVALUATION_INSTRUCTIONS,
    SINGLE_AGENT_INSTRUCTIONS,  # Single-agent baseline (union of specialist knowledge)
    SYNPLANNER_INSTRUCTIONS,
)


@dataclass
class AgentConfig:
    """Configuration for creating an agent."""

    name: str
    description: str
    tools: List[Any] = field(default_factory=list)
    instructions: List[str] = field(default_factory=list)
    session_state: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate the agent configuration."""
        if not self.name:
            raise ValueError("Agent name cannot be empty")
        if not self.description:
            raise ValueError("Agent description cannot be empty")
        if not isinstance(self.tools, list):
            raise TypeError("Tools must be a list")
        if not isinstance(self.instructions, list):
            raise TypeError("Instructions must be a list")


class AgentCreationError(Exception):
    """Exception raised when agent creation fails."""

    pass


def _merge_session_state_defaults(target: Dict[str, Any], defaults: Dict[str, Any]) -> None:
    """Merge agent default state into shared state without replacing existing values."""
    for key, value in (defaults or {}).items():
        if key not in target:
            target[key] = deepcopy(value)
            continue
        if isinstance(target[key], dict) and isinstance(value, dict):
            _merge_session_state_defaults(target[key], value)


class BaseAgentFactory(ABC):
    """Base class for creating agents with common configuration and error handling."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def get_agent_config(self) -> AgentConfig:
        """Return the configuration for this agent type."""
        pass

    def create_agent(
        self,
        model: Model,
        markdown: bool = True,
        debug_mode: bool = False,
        enable_mlflow_tracking: bool = True,
        **kwargs,
    ) -> Agent:
        """Create an agent with error handling and validation.

        Args:
            model: Model to use for the agent
            markdown: Whether to enable markdown formatting
            debug_mode: Whether to enable debug mode
            enable_mlflow_tracking: Whether to enable MLflow tracking for this agent
            **kwargs: Additional keyword arguments for agent creation

        Returns:
            Created agent instance
        """
        try:
            config = self.get_agent_config()
            config.validate()
            provided_session_state = kwargs.pop("session_state", None)

            # Log agent creation
            self.logger.info(f"Creating agent: {config.name}")

            # Create agent with common parameters
            agent_kwargs = {
                "model": model,
                "name": config.name,
                "description": config.description,
                "tools": config.tools,
                "markdown": markdown,
                "debug_mode": debug_mode,
                "enable_agentic_state": True,
                "add_session_state_to_context": True,
            }

            # Add optional parameters if they exist
            if config.instructions:
                agent_kwargs["instructions"] = config.instructions
            if provided_session_state is not None:
                if config.session_state:
                    _merge_session_state_defaults(provided_session_state, config.session_state)
                agent_kwargs["session_state"] = provided_session_state
            elif config.session_state:
                agent_kwargs["session_state"] = config.session_state

            # Add any additional kwargs passed in
            agent_kwargs.update(kwargs)

            agent = Agent(**agent_kwargs)

            # Wrap agent methods with MLflow tracking if enabled
            if enable_mlflow_tracking:
                agent = self._wrap_agent_with_tracking(agent, config)

            self.logger.info(f"Successfully created agent: {config.name}")
            return agent

        except Exception as e:
            self.logger.error(
                f"Failed to create agent {config.name if 'config' in locals() else 'unknown'}: {str(e)}"
            )
            raise AgentCreationError(f"Failed to create agent: {str(e)}") from e

    def _wrap_agent_with_tracking(self, agent: Agent, config: AgentConfig) -> Agent:
        """Wrap agent execution methods with MLflow tracking.

        Args:
            agent: Agent instance to wrap
            config: Agent configuration

        Returns:
            Agent with wrapped methods
        """
        try:
            from cs_copilot.tracking import get_tracker
            from cs_copilot.tracking.utils import build_prompt_signature

            tracker = get_tracker()

            if not tracker.is_enabled():
                return agent

            # Get the agent type from the factory
            agent_type = getattr(self.__class__, "agent_type", None)

            def build_prompt_template() -> Optional[str]:
                sections = []
                if config.description:
                    sections.append(str(config.description).strip())
                if config.instructions:
                    normalized = [
                        str(item).strip() for item in config.instructions if item is not None
                    ]
                    instructions_text = "\n".join(normalized).strip()
                    if instructions_text:
                        sections.append(instructions_text)
                template = "\n\n".join([section for section in sections if section])
                return template.strip() if template else None

            def build_prompt_name() -> str:
                base_name = agent_type or config.name
                safe_name = str(base_name).replace(" ", "_").lower()
                return f"cs_copilot.{safe_name}"

            prompt_template = build_prompt_template()
            prompt_signature = build_prompt_signature(prompt_template)
            prompt_registry_name = build_prompt_name()

            def register_prompt_in_registry():
                if not prompt_template:
                    return
                commit_message = None
                if prompt_signature:
                    commit_message = f"cs_copilot auto update ({prompt_signature.version})"
                prompt_obj = tracker.register_prompt_version(
                    name=prompt_registry_name,
                    template=prompt_template,
                    commit_message=commit_message,
                    tags={
                        "agent_name": agent.name,
                        "agent_type": agent_type or "unknown",
                        "component": "cs_copilot",
                    },
                )
                if prompt_obj:
                    version = getattr(prompt_obj, "version", None)
                    tracker.log_params(
                        {
                            "prompt_registry_name": prompt_registry_name,
                            "prompt_registry_version": str(version) if version is not None else "",
                            "prompt_registry_uri": (
                                f"prompts:/{prompt_registry_name}/{version}"
                                if version is not None
                                else ""
                            ),
                        }
                    )

            # Wrap run() method
            original_run = agent.run

            def tracked_run(*args, **kwargs):
                # Extract prompt from args
                prompt = args[0] if args else kwargs.get("message", "")

                with tracker.track_agent_run(
                    agent_name=agent.name, prompt=str(prompt), agent_type=agent_type
                ):
                    # Log agent configuration
                    tracker.log_params(
                        {
                            "agent_name": agent.name,
                            "agent_type": agent_type or "unknown",
                            "num_tools": len(config.tools),
                            "tools": ",".join([t.__class__.__name__ for t in config.tools]),
                        }
                    )
                    register_prompt_in_registry()

                    result = original_run(*args, **kwargs)

                    # Log result metrics if available
                    if hasattr(result, "content") and result.content:
                        from cs_copilot.tracking.utils import count_tokens

                        tracker.log_metrics(
                            {"output_tokens_estimate": float(count_tokens(result.content))}
                        )

                    return result

            agent.run = tracked_run

            # Wrap arun() method (async version)
            original_arun = agent.arun

            async def tracked_arun(*args, **kwargs):
                # Extract prompt from args
                prompt = args[0] if args else kwargs.get("message", "")

                with tracker.track_agent_run(
                    agent_name=agent.name, prompt=str(prompt), agent_type=agent_type
                ):
                    # Log agent configuration
                    tracker.log_params(
                        {
                            "agent_name": agent.name,
                            "agent_type": agent_type or "unknown",
                            "num_tools": len(config.tools),
                            "tools": ",".join([t.__class__.__name__ for t in config.tools]),
                        }
                    )
                    register_prompt_in_registry()

                    result = await original_arun(*args, **kwargs)

                    # Log result metrics if available
                    if hasattr(result, "content") and result.content:
                        from cs_copilot.tracking.utils import count_tokens

                        tracker.log_metrics(
                            {"output_tokens_estimate": float(count_tokens(result.content))}
                        )

                    return result

            agent.arun = tracked_arun

            self.logger.debug(f"MLflow tracking enabled for agent: {agent.name}")

        except ImportError:
            self.logger.warning(
                "MLflow tracking module not available. Agent will run without tracking."
            )
        except Exception as e:
            self.logger.warning(f"Failed to enable MLflow tracking for agent: {e}")

        return agent


class ChEMBLDownloaderFactory(BaseAgentFactory):
    """Factory for creating ChemBL downloader agents."""

    agent_type = "chembl_downloader"

    def get_agent_config(self) -> AgentConfig:
        return AgentConfig(
            name="chembl_agent",
            description=CHEMBL_DESCRIPTION,
            tools=[
                ChemblToolkit(),
                PointerPandasTools(),
                SkillToolkit(),
                # SessionToolkit(),
            ],
            instructions=CHEMBL_INSTRUCTIONS,
            session_state={
                "data_file_paths": {
                    "dataset_path": None,  # Backward-compatible alias for clean_dataset_path.
                    "raw_dataset_path": None,
                    "clean_dataset_path": None,
                    "filtered_dataset_path": None,
                    "descriptor_parquet_path": None,
                    "standardization_report_path": None,
                }
            },
        )


class ChemoinformaticianFactory(BaseAgentFactory):
    """Factory for creating comprehensive chemoinformatics analysis agents.

    This agent is a versatile chemoinformatician capable of:
    - **Chemotype Analysis**: Scaffold extraction, chemotype profiling, structural diversity
    - **Clustering**: Molecular clustering using various methods (k-means, hierarchical, DBSCAN)
    - **SAR Analysis**: Structure-Activity Relationship analysis, activity cliffs, matched molecular pairs
    - **Similarity Analysis**: Molecular similarity, diversity metrics, nearest neighbor searches

    GTM-Integrated Design:
    - Primary use case: Downstream analysis after GTM agents (nodes as clusters)
    - Also works with ANY data source: t-SNE clusters, user CSVs, ChEMBL families
    - Standardized input: DataFrame with 'smiles' + optional 'cluster_id' + optional 'activity'
    - Produces structured data output (DataFrames, dicts) - NO report generation
    - Report generation handled by separate ReportGeneratorAgent

    Tools:
    - ChemicalSimilarityToolkit: Fingerprints, similarity metrics, scaffold extraction
    - PointerPandasTools: DataFrame operations with S3 support
    - GTMToolkit: Access to GTM data (source_mols, node projections)
    """

    agent_type = "chemoinformatician"

    def get_agent_config(self) -> AgentConfig:
        return AgentConfig(
            name="chemoinformatician_agent",
            description=CHEMOINFORMATICIAN_DESCRIPTION,
            tools=[
                ChemicalSimilarityToolkit(),
                PointerPandasTools(),
                GTMToolkit(),  # Enable GTM data access for downstream analysis
                SkillToolkit(),
                # Future: QSARToolkit, ClusteringToolkit, DescriptorToolkit
            ],
            instructions=CHEMOINFORMATICIAN_INSTRUCTIONS,
            session_state={
                # Normalized input data for analysis
                "analysis_input": None,  # DataFrame with standardized columns (smiles, cluster_id?, activity?)
                # Chemotype/Scaffold Analysis
                "chemotype_analysis": {
                    "scaffolds_per_cluster": None,
                    "similarity_matrix": None,
                    "summary_stats": None,
                    "metadata": {},
                    "output_paths": {
                        "scaffolds_csv": None,
                        "similarity_csv": None,
                    },
                },
                # Clustering Analysis
                "clustering_results": {
                    "cluster_assignments": None,  # DataFrame with cluster_id column
                    "cluster_metrics": None,  # Silhouette, Davies-Bouldin, etc.
                    "cluster_centroids": None,
                    "method": None,  # 'gtm', 'kmeans', 'dbscan', 'hierarchical', etc.
                },
                # SAR Analysis
                "sar_analysis": {
                    "activity_cliffs": None,  # Detected activity cliffs
                    "mmps": None,  # Matched molecular pairs
                    "series_analysis": None,  # Chemical series breakdown
                    "potency_trends": None,
                },
                # Similarity/Diversity
                "similarity_analysis": {
                    "similarity_matrix": None,
                    "diversity_metrics": None,
                    "nearest_neighbors": None,
                },
                # General data paths
                "analysis_outputs": {
                    "primary_data_csv": None,
                    "supplementary_data": [],
                },
            },
        )


class MolecularDesignerFactory(BaseAgentFactory):
    """Factory for creating small-molecule design agents.

    Supports two modes:
    - **Engine-driven design**: Use autoencoder or LLM engines behind a common facade
    - **Standalone autoencoder**: Encode/decode SMILES, sample latent space, interpolate, explore neighborhoods
    - **GTM-guided**: Combine GTM maps with generative engines for targeted molecular design
      from specific map regions (by density, activity, or coordinates)

    Enhanced with GTM cache awareness to avoid redundant GTM loading when working with GTM Agent
    in the same session.
    """

    agent_type = "molecular_designer"

    def get_agent_config(self) -> AgentConfig:
        autoencoder_toolkit = AutoencoderToolkit()
        return AgentConfig(
            name="molecular_designer_agent",
            description=MOLECULAR_DESIGNER_DESCRIPTION,
            tools=[
                MolecularDesignerToolkit(autoencoder_toolkit=autoencoder_toolkit),
                autoencoder_toolkit,
                GTMToolkit(),
                ChemicalSimilarityToolkit(),
                PointerPandasTools(),
                SkillToolkit(),
            ],
            instructions=MOLECULAR_DESIGNER_INSTRUCTIONS,
            session_state={
                "data_file_paths": {
                    "dataset_path": None,  # Backward-compatible alias for clean_dataset_path.
                    "raw_dataset_path": None,
                    "clean_dataset_path": None,
                    "filtered_dataset_path": None,
                    "descriptor_parquet_path": None,
                    "standardization_report_path": None,
                },
            },
        )


class GTMAgentFactory(BaseAgentFactory):
    """Factory for creating unified GTM agents (consolidates optimization, loading, density, activity, projection).

    This factory creates a single agent that handles all GTM-related operations via mode-based dispatch:
    - optimize: Build and optimize new GTM maps
    - load: Load existing GTM models from S3/local/HuggingFace
    - density: Analyze compound distributions and neighborhood preservation
    - activity: Create activity landscapes for SAR analysis
    - project: Project external datasets onto existing GTM maps

    Features smart caching to avoid redundant GTM loading across operations.
    """

    agent_type = "gtm_agent"

    def get_agent_config(self) -> AgentConfig:
        return AgentConfig(
            name="gtm_agent",
            description=GTM_AGENT_DESCRIPTION,
            tools=[
                GTMToolkit(),
                PointerPandasTools(),
                SessionMemoryToolkit(),
                save_gtm_landscape_plot,
                save_gtm_plot,
                SkillToolkit(),
            ],
            instructions=GTM_AGENT_INSTRUCTIONS,
            session_state={
                "gtm_cache": {
                    "model": None,
                    "dataset": None,
                    "metadata": {
                        "optimization_strategy": None,
                    },
                },
                "gtm_file_paths": {
                    "gtm_path": None,
                    "dataset_path": None,
                    "gtm_plot_path": None,
                },
                "analysis_results": {
                    "density_csv": None,
                    "activity_csv": None,
                    "projection_csv": None,
                    "plots": [],
                },
                "landscape_files": {  # Backward compatibility
                    "landscape_data_csv": None,
                    "landscape_plot": None,
                },
            },
        )


class ReportGeneratorFactory(BaseAgentFactory):
    """Factory for creating report generation agents.

    This agent handles ALL report generation and visualization across different analysis types:
    - Chemotype analysis reports
    - GTM density reports
    - GTM activity/SAR reports
    - Molecular designer generation reports
    - Combined/custom reports

    **Separation of Concerns**: Analysis agents produce structured data, Report Generator handles presentation.

    This architecture enables:
    - Consistent formatting across all report types
    - Reusable visualization patterns
    - Easy updates to report styles (change in one place)
    - Clean separation: data processing vs visualization/formatting
    """

    agent_type = "report_generator"

    def get_agent_config(self) -> AgentConfig:
        return AgentConfig(
            name="report_generator_agent",
            description=REPORT_GENERATOR_DESCRIPTION,
            tools=[
                PointerPandasTools(),
                save_gtm_landscape_plot,  # For saved GTM landscape tables
                save_gtm_plot,  # For GTM-specific visualizations
                save_rich_report,  # Persists image-rich HTML/PDF reports
                save_markdown_report,  # Persists the final markdown report
                SkillToolkit(),
                # Plotting libraries (matplotlib, seaborn) available via Python environment
            ],
            instructions=REPORT_GENERATOR_INSTRUCTIONS,
            session_state={
                "report_outputs": {
                    "report_path": None,
                    "report_paths": {},
                    "plots": [],
                    "report_type": None,
                },
            },
        )


class RobustnessEvaluationFactory(BaseAgentFactory):
    """Factory for creating robustness test evaluation agents."""

    agent_type = "robustness_evaluation"

    def get_agent_config(self) -> AgentConfig:
        return AgentConfig(
            name="robustness_evaluator_agent",
            description=ROBUSTNESS_EVALUATION_DESCRIPTION,
            tools=[
                PointerPandasTools(),
                RobustnessAnalysisToolkit(),
                SkillToolkit(),
            ],
            instructions=ROBUSTNESS_EVALUATION_INSTRUCTIONS,
            session_state={
                "loaded_results": {},
                "analysis_outputs": {
                    "summary_report": None,
                    "comparison_report": None,
                    "recommendations": None,
                },
            },
        )


class SynPlannerFactory(BaseAgentFactory):
    """Factory for creating retrosynthetic planning agents powered by SynPlanner.

    This agent wraps the official SynPlanner package to perform retrosynthetic
    analysis on target molecules.  It accepts SMILES strings or molecule names,
    resolves them to canonical SMILES (via PubChem / RDKit), runs the MCTS-based
    retrosynthesis search, and returns structured route descriptions with
    optional SVG/PNG visualizations.
    """

    agent_type = "synplanner"

    def get_agent_config(self) -> AgentConfig:
        return AgentConfig(
            name="synplanner_agent",
            description=SYNPLANNER_DESCRIPTION,
            tools=[
                SynPlannerToolkit(),
                SkillToolkit(),
            ],
            instructions=SYNPLANNER_INSTRUCTIONS,
        )


class PeptideDesignerFactory(BaseAgentFactory):
    """Factory for creating peptide design agents.

    This agent exposes a Peptide Designer facade over multiple peptide design
    engines. The default WAE engine encodes, decodes, samples, and interpolates
    amino acid sequences; the LLM engine proposes sequence candidates from
    natural-language objectives. The WAE model can generate any peptides;
    activity landscape data comes from DBAASP (antimicrobial peptides specifically).

    Key capabilities:
    - **Encoding**: Convert peptide sequences to 100-dimensional latent vectors
    - **Decoding**: Generate peptide sequences from latent vectors
    - **Sampling**: Generate novel peptides from Gaussian prior
    - **Interpolation**: Smooth transitions between peptides in latent space
    - **Neighborhood exploration**: Generate peptide analogs
    - **GTM integration**: Train GTMs on latent space, create activity landscapes
    - **Activity landscapes**: Use DBAASP data (specific to antimicrobial peptides)

    Input format: Space-separated single-letter amino acid codes
    Example: "M L L L L L A L A L L A L L L A L L L"
    """

    agent_type = "peptide_designer"

    def get_agent_config(self) -> AgentConfig:
        return AgentConfig(
            name="peptide_designer_agent",
            description=PEPTIDE_DESIGNER_DESCRIPTION,
            tools=[
                PeptideDesignerToolkit(),
                GTMToolkit(),
                PointerPandasTools(),
                save_gtm_landscape_plot,
                save_gtm_plot,
                SkillToolkit(),
            ],
            instructions=PEPTIDE_DESIGNER_INSTRUCTIONS,
        )


class SingleAgentFactory(BaseAgentFactory):
    """Factory for the single-agent baseline used in the multi-agent ablation.

    One flat Agno ``Agent`` that holds the UNION of the seven team specialists'
    toolkits (ChEMBL, GTM, chemoinformatics, molecular + peptide design,
    retrosynthesis, reporting) with no coordinator, no routing, and no
    per-specialist context isolation. It is deliberately NOT added to the runtime
    team (``teams.py`` stays 7 members); it is constructed separately for the
    robustness comparison so the only variable vs the team is the agentic
    structure (same model, same tools, same tasks).

    Tool ordering matters: Agno registers toolkit methods by name and keeps the
    first-registered on a name clash (later duplicates are silently dropped). The
    molecular / autoencoder design toolkits are listed BEFORE the peptide toolkit
    so the ~9 shared design method names (``validate_design_candidates``,
    ``decode_latent``, ``list_design_engines``, ...) resolve to the small-molecule
    implementations. The current comparison task set contains no peptide-design
    prompts, so this preserves full capability parity for every measured task.
    """

    agent_type = "single_agent"

    def get_agent_config(self) -> AgentConfig:
        autoencoder_toolkit = AutoencoderToolkit()
        return AgentConfig(
            name="single_agent",
            description=SINGLE_AGENT_DESCRIPTION,
            tools=[
                # Data + space
                ChemblToolkit(),
                GTMToolkit(),
                ChemicalSimilarityToolkit(),
                # Design engines: molecular/autoencoder first so they win the
                # name-dedupe against the peptide toolkit's shared method names.
                MolecularDesignerToolkit(autoencoder_toolkit=autoencoder_toolkit),
                autoencoder_toolkit,
                PeptideDesignerToolkit(),
                SynPlannerToolkit(),
                # Shared infrastructure (one instance each).
                SessionMemoryToolkit(),
                PointerPandasTools(),
                SkillToolkit(),
                # Plot / report callables.
                save_gtm_landscape_plot,
                save_gtm_plot,
                save_rich_report,
                save_markdown_report,
            ],
            instructions=SINGLE_AGENT_INSTRUCTIONS,
            # Union of the team members' session_state defaults so the flat agent
            # carries the same artifact/analysis slots the team accumulates. Keys
            # mirror ChEMBL/MolecularDesigner (data_file_paths), Chemoinformatician
            # (analysis_input, chemotype_analysis, clustering_results, sar_analysis,
            # similarity_analysis, analysis_outputs), GTM (gtm_cache, gtm_file_paths,
            # analysis_results, landscape_files), and Report (report_outputs).
            # A drift guard in tests/unit/test_single_agent_factory.py checks these
            # stay in sync with the member factories.
            session_state={
                "data_file_paths": {
                    "dataset_path": None,  # Backward-compatible alias for clean_dataset_path.
                    "raw_dataset_path": None,
                    "clean_dataset_path": None,
                    "filtered_dataset_path": None,
                    "descriptor_parquet_path": None,
                    "standardization_report_path": None,
                },
                "analysis_input": None,
                "chemotype_analysis": {
                    "scaffolds_per_cluster": None,
                    "similarity_matrix": None,
                    "summary_stats": None,
                    "metadata": {},
                    "output_paths": {
                        "scaffolds_csv": None,
                        "similarity_csv": None,
                    },
                },
                "clustering_results": {
                    "cluster_assignments": None,
                    "cluster_metrics": None,
                    "cluster_centroids": None,
                    "method": None,
                },
                "sar_analysis": {
                    "activity_cliffs": None,
                    "mmps": None,
                    "series_analysis": None,
                    "potency_trends": None,
                },
                "similarity_analysis": {
                    "similarity_matrix": None,
                    "diversity_metrics": None,
                    "nearest_neighbors": None,
                },
                "gtm_cache": {
                    "model": None,
                    "dataset": None,
                    "metadata": {
                        "optimization_strategy": None,
                    },
                },
                "gtm_file_paths": {
                    "gtm_path": None,
                    "dataset_path": None,
                    "gtm_plot_path": None,
                },
                "analysis_results": {
                    "density_csv": None,
                    "activity_csv": None,
                    "projection_csv": None,
                    "plots": [],
                },
                "landscape_files": {
                    "landscape_data_csv": None,
                    "landscape_plot": None,
                },
                "report_outputs": {
                    "report_path": None,
                    "report_paths": {},
                    "plots": [],
                    "report_type": None,
                },
                "analysis_outputs": {
                    "primary_data_csv": None,
                    "supplementary_data": [],
                },
            },
        )
