#!/usr/bin/env python
# coding: utf-8
"""
Agent factory classes for creating specialized cs_copilot agents.
Contains the base factory class and all specialized factory implementations.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agno.agent import Agent
from agno.models.base import Model  # Agno v2 base class

from cs_copilot.tools import (
    AutoencoderToolkit,
    ChemblToolkit,
    ChemicalSimilarityToolkit,
    GTMToolkit,
    PeptideWAEToolkit,
    PointerPandasTools,
    SynPlannerToolkit,
    # SessionToolkit,
    save_gtm_plot,
)
from cs_copilot.tools.analysis import RobustnessAnalysisToolkit

from .prompts import (
    AUTOENCODER_INSTRUCTIONS,
    CHEMBL_INSTRUCTIONS,
    CHEMOINFORMATICIAN_INSTRUCTIONS,  # Comprehensive chemoinformatics analysis
    GTM_AGENT_INSTRUCTIONS,  # Unified GTM agent (all GTM operations)
    PEPTIDE_WAE_INSTRUCTIONS,  # Peptide WAE for amino acid sequence generation
    REPORT_GENERATOR_INSTRUCTIONS,  # Universal presentation layer
    ROBUSTNESS_EVALUATION_INSTRUCTIONS,
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
            if config.session_state:
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
            description="""
            You are a specialized agent for downloading and processing bioactivity data from the ChEMBL database.
            Your role is to query ChEMBL based on user requests (e.g., protein targets, compound types),
            retrieve relevant bioactivity data, validate data quality, and prepare structured datasets
            for downstream cheminformatics analysis.
            """,
            tools=[
                ChemblToolkit(),
                PointerPandasTools(),
                # SessionToolkit(),
            ],
            instructions=CHEMBL_INSTRUCTIONS,
            session_state={
                "data_file_paths": {
                    "dataset_path": None,
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
            description="""
            You are an expert chemoinformatician specialized in computational chemistry and molecular analysis.
            Primary use case: Downstream analysis after GTM operations (analyzing molecules in GTM nodes/clusters).

            **Core Competencies**:

            1. **Chemotype & Scaffold Analysis**:
               - Murcko scaffold decomposition and profiling
               - Scaffold frequency per cluster/node
               - Structural diversity metrics

            2. **Clustering & Chemical Space Analysis**:
               - Works with GTM nodes (primary), or any clustering method
               - Cluster characterization and comparison
               - Chemical space coverage analysis

            3. **SAR Analysis (Structure-Activity Relationships)**:
               - Activity cliff detection
               - Matched molecular pair (MMP) analysis
               - Potency distribution across clusters/scaffolds

            4. **Similarity & Diversity**:
               - Tanimoto/Dice similarity calculations
               - Diversity analysis (Shannon entropy, coverage)
               - Nearest neighbor searches

            **Input Format**:
            - Standardized DataFrame with 'smiles' column
            - Optional 'cluster_id' (from GTM node_index or other clustering)
            - Optional 'activity' (for SAR analysis)
            - Use `normalize_for_analysis` tool to standardize input from any source

            **Output**:
            - Structured data (DataFrames, dicts) saved to session_state
            - NO visualizations (handled by Report Generator)
            """,
            tools=[
                ChemicalSimilarityToolkit(),
                PointerPandasTools(),
                GTMToolkit(),  # Enable GTM data access for downstream analysis
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


class AutoencoderFactory(BaseAgentFactory):
    """Factory for creating autoencoder-based molecular generation agents.

    Supports two modes:
    - **Standalone**: Encode/decode SMILES, sample from latent space, interpolate, explore neighborhoods
    - **GTM-guided**: Combine GTM maps with autoencoders for targeted molecular generation from
      specific map regions (by density, activity, or coordinates)

    Enhanced with GTM cache awareness to avoid redundant GTM loading when working with GTM Agent
    in the same session.
    """

    agent_type = "autoencoder"
    aliases = ["autoencoder_gtm_sampling"]

    def get_agent_config(self) -> AgentConfig:
        return AgentConfig(
            name="autoencoder_agent",
            description="""
            You are a scientific assistant specialized in molecular generation and analysis using LSTM
            autoencoders. You operate in two modes:

            **Standalone mode**: Encode molecules to latent representations, generate novel structures
            by sampling from latent space, interpolate between molecules, and explore chemical space
            neighborhoods to understand structure-property relationships.

            **GTM-guided mode**: Combine Generative Topographic Mapping (GTM) with autoencoders for
            targeted molecular generation. Sample molecules from specific regions of GTM maps
            (by density, activity, or coordinates), encode them to latent space, and generate novel
            molecules by exploring neighborhoods around regions of interest.

            **Cache-Aware**: Automatically reuses GTM models cached by GTM Agent in session_state,
            eliminating redundant loading for multi-step workflows (e.g., GTM density → sampling).
            """,
            tools=[
                AutoencoderToolkit(),
                GTMToolkit(),
                ChemicalSimilarityToolkit(),
                PointerPandasTools(),
            ],
            instructions=AUTOENCODER_INSTRUCTIONS,
            session_state={
                "data_file_paths": {
                    "dataset_path": None,
                },
            },
        )


class GTMAgentFactory(BaseAgentFactory):
    """Factory for creating unified GTM agents (consolidates optimization, loading, density, activity, projection).

    This factory creates a single agent that handles all GTM-related operations via mode-based dispatch:
    - optimize: Build and optimize new GTM maps
    - load: Load existing GTM models from S3/local/HuggingFace
    - density: Analyze compound distributions and neighborhood preservation
    - activity: Create activity-density landscapes for SAR analysis
    - project: Project external datasets onto existing GTM maps

    Features smart caching to avoid redundant GTM loading across operations.
    """

    agent_type = "gtm_agent"

    def get_agent_config(self) -> AgentConfig:
        return AgentConfig(
            name="gtm_agent",
            description="""
            You are a unified scientific assistant for all GTM (Generative Topographic Mapping) operations.
            Your role is to handle building, loading, and analyzing GTM-based maps of chemical space.

            Capabilities:
            - **Optimize**: Build and optimize new GTM maps from chemical datasets
            - **Load**: Retrieve existing GTM models from storage (S3, local, HuggingFace)
            - **Density**: Analyze compound distributions and neighborhood preservation on GTM maps
            - **Activity**: Create activity-density landscapes for structure-activity relationship (SAR) exploration
            - **Project**: Map external datasets onto existing GTM maps for comparative analysis

            Key Features:
            - Smart caching: Automatically reuses loaded GTM models across operations within the same session
            - Mode-based dispatch: Detects operation type from user requests and executes appropriate workflow
            - Session state integration: Shares GTM data with other agents
            """,
            tools=[
                GTMToolkit(),
                PointerPandasTools(),
                save_gtm_plot,
            ],
            instructions=GTM_AGENT_INSTRUCTIONS,
            session_state={
                "gtm_cache": {
                    "model": None,
                    "dataset": None,
                    "metadata": {},
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
    - Autoencoder generation reports
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
            description="""
            You are a specialized agent for generating reports and visualizations from analysis results.
            Your role is to create well-formatted, comprehensive reports that present scientific findings
            in a clear, actionable manner.

            Capabilities:
            - **Multi-format reports**: Generate markdown, HTML, or text reports
            - **Visualization creation**: Produce publication-quality plots and charts
            - **Template-based formatting**: Consistent structure across different report types
            - **Flexible input handling**: Works with results from any analysis agent

            Report Types Supported:
            - Chemotype analysis: Scaffold distributions, similarity heatmaps, cluster comparisons
            - GTM density: Density overlays, neighborhood preservation, coverage analysis
            - GTM activity/SAR: Activity landscapes, potency hotspots, structure-activity insights
            - Autoencoder generation: Generated molecules, diversity metrics, similarity analyses
            - Combined reports: Multi-analysis integration with comparative visualizations

            Key Features:
            - **Analysis-agnostic**: Reads structured data from session_state (any analysis type)
            - **Consistent formatting**: Uniform markdown structure, color schemes, plot styles
            - **Embedded visualizations**: Inline plots in reports for easy consumption
            - **Actionable insights**: Highlights key findings and provides recommendations

            This separation enables analysis agents to focus on data processing while Report Generator
            handles all presentation concerns.
            """,
            tools=[
                PointerPandasTools(),
                save_gtm_plot,  # For GTM-specific visualizations
                # Plotting libraries (matplotlib, seaborn) available via Python environment
            ],
            instructions=REPORT_GENERATOR_INSTRUCTIONS,
            session_state={
                "report_outputs": {
                    "report_path": None,
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
            description="""
            You are a specialized agent for analyzing robustness test results. Your role is to load
            test results from S3 or local storage, analyze metrics and score distributions, identify
            patterns and issues in failing prompts, and generate actionable recommendations for
            improving system robustness across prompt variations.
            """,
            tools=[
                PointerPandasTools(),
                RobustnessAnalysisToolkit(),
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
            description=(
                "You are a retrosynthetic planning assistant powered by SynPlanner. "
                "Given a target molecule (as a SMILES string or common name), you "
                "identify the canonical structure, run the SynPlanner retrosynthesis "
                "engine, and present the best synthetic routes with step-by-step "
                "descriptions and visualizations."
            ),
            tools=[SynPlannerToolkit()],
            instructions=SYNPLANNER_INSTRUCTIONS,
        )


class PeptideWAEFactory(BaseAgentFactory):
    """Factory for creating peptide WAE-based sequence generation agents.

    This agent uses a Wasserstein Autoencoder (WAE) trained on peptide data
    to encode, decode, sample, and interpolate amino acid sequences. The WAE
    can generate any peptides; activity landscape data comes from DBAASP
    (antimicrobial peptides specifically).

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

    agent_type = "peptide_wae"

    def get_agent_config(self) -> AgentConfig:
        return AgentConfig(
            name="peptide_wae_agent",
            description="""
            You are a scientific assistant specialized in peptide sequence generation and analysis
            using Wasserstein Autoencoders (WAE). You work with amino acid sequences represented
            as space-separated single-letter codes (e.g., "M L L L L L A L A L L A L L L").

            **Core Capabilities**:
            - **Encode peptides**: Convert peptide sequences to 100-dimensional latent representations
            - **Decode latent vectors**: Generate peptide sequences from latent space
            - **Sample new peptides**: Generate novel peptides from Gaussian prior
            - **Interpolate**: Create smooth transitions between peptides in latent space
            - **Explore neighborhoods**: Generate peptide analogs with controlled diversity
            - **GTM on latent space**: Train Generative Topographic Maps on WAE latent vectors
            - **Activity landscapes**: Create per-organism antimicrobial activity landscapes from DBAASP data

            **Key Parameters**:
            - Max sequence length: 25 amino acids
            - Latent dimension: 100
            - Supported amino acids: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, U, V, W, Y, Z

            **Use Cases**:
            - Generate novel peptide candidates (any peptides)
            - Generate novel antimicrobial peptide candidates
            - Explore peptide chemical space around active sequences
            - Interpolate between peptides to understand structure-activity relationships
            - Test sequence reconstruction for model quality assessment
            - Build GTM maps of peptide latent space for visualization
            - Analyze antimicrobial activity patterns using DBAASP data on GTM landscapes
            - Sample peptides from specific GTM regions and decode to sequences

            **Note**: Activity landscapes use DBAASP data and are specific to antimicrobial peptides.
            """,
            tools=[
                PeptideWAEToolkit(),
                GTMToolkit(),
                PointerPandasTools(),
                save_gtm_plot,
            ],
            instructions=PEPTIDE_WAE_INSTRUCTIONS,
        )
