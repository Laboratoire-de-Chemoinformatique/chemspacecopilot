#!/usr/bin/env python
# coding: utf-8
"""
GTM-specific dimensionality reduction toolkit for chemical data analysis.

This module provides a high-level toolkit class that delegates to
gtm_operations.py for the actual implementations.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from agno.agent import Agent

from . import gtm_operations
from .dimensionality_reduction import BaseDRToolkit, DRToolkitError

logger = logging.getLogger(__name__)


# Public type used in method annotations; must be at module scope so
# runtime type evaluators (e.g., tool frameworks using get_type_hints)
# can resolve it without NameError.
SampleReturnFormat = Literal["text", "dataframe", "smiles", "sequences"]


class GTMError(DRToolkitError):
    """Exception raised for GTM-specific dimensionality reduction errors."""

    pass


class GTMToolkit(BaseDRToolkit):
    """
    GTM-specific dimensionality reduction toolkit for chemical data analysis.

    This class provides a high-level interface to GTM operations,
    delegating to gtm_operations.py for actual implementations.

    Load and prepare GTM data before running the analysis!!!
    """

    def __init__(self):
        """Initialize the GTMDimensionalityReductionToolkit."""
        super().__init__("gtm_dimensionality_reduction")
        # Register GTM-specific tools
        self.register(self.gtm_optimization)
        # self.register(self.calculate_map_ruggedness)
        self.register(self.save_gtm_and_data)
        self.register(self.load_dataframe_from_session)
        self.register(self.load_gtm_model_only)
        self.register(self.load_gtm_get_density_matrix)
        self.register(self.load_and_prep_data)
        self.register(self.analyze_scaffolds_in_nodes)
        self.register(self.check_source_datasets_in_nodes)
        self.register(self.node_id_from_coords)
        self.register(self.get_density_summary)
        self.register(self.get_activity_summary)
        self.register(self.get_node_lookup_summary)
        self.register(self.sample_nodes)
        self.register(self.sample_dense_nodes)
        self.register(self.sample_active_nodes)
        self.register(self.sample_by_coordinates)
        self.register(self.create_activity_landscapes)
        self.register(self.project_data_on_gtm)
        # Latent-space GTM tools (for peptide WAE integration)
        self.register(self.train_gtm_on_latent_space)
        self.register(self.load_latent_data_on_gtm)
        self.register(self.create_peptide_activity_landscapes)

        # Initialize data storage for chemotype analysis
        self._gtm_data = None

    def gtm_optimization(
        self,
        df_csv_path: str,
        dataset_name: str,
        gtm_name: str,
        smiles_column: str,
        agent: Agent,
    ) -> str:
        """
        Load a dataset of SMILES strings, optimize a Generative Topographic Mapping (GTM)
        model for entropy, store results in the agent's session state,
        and report the entropy score.

        Args:
            df_csv_path: Path to the CSV file containing the data table
            dataset_name: Key under which the cleaned DataFrame will be saved in agent.session_state
            gtm_name: Key under which the trained GTM model will be saved in agent.session_state
            smiles_column: Name of the column in the CSV that holds SMILES strings
            agent: The agent whose session_state dict will be updated

        Returns:
            Human-readable message reporting the best entropy score achieved

        Raises:
            FileNotFoundError: If df_csv_path does not point to an existing CSV file
            ValueError: If smiles_column is missing
        """
        return gtm_operations.optimize_gtm_model(
            df_csv_path, dataset_name, gtm_name, smiles_column, agent
        )

    def calculate_map_ruggedness(self, dataset_name: str, gtm_name: str, agent: Agent) -> str:
        """
        Compute the Topographic Ruggedness Index (TRI) for a GTM model.

        Args:
            dataset_name: Key under which the DataFrame is stored in agent.session_state
            gtm_name: Key under which the GTM model is stored in agent.session_state
            agent: Agent instance whose session_state holds both the dataset and GTM

        Returns:
            Human-readable message reporting the TRI value

        Raises:
            KeyError: If dataset_name or gtm_name is not present in agent.session_state
            ValueError: If inputs are invalid
        """
        return gtm_operations.calculate_gtm_ruggedness(dataset_name, gtm_name, agent)

    def save_gtm_and_data(self, dataset_name: str, gtm_name: str, agent: Agent) -> str:
        """
        Save a GTM model and its associated dataset from the agent's session state.

        Args:
            dataset_name: Key under which the DataFrame is stored in agent.session_state
            gtm_name: Key under which the GTM model is stored in agent.session_state
            agent: Agent instance whose session_state contains both the dataset and GTM

        Returns:
            Message containing the paths to the saved files

        Raises:
            KeyError: If dataset_name or gtm_name is not present in agent.session_state
            IOError: If saving either file fails
        """
        return gtm_operations.save_gtm_and_dataset(dataset_name, gtm_name, agent)

    def load_dataframe_from_session(
        self, dataframe_name: str, session_key: str, agent: Agent
    ) -> str:
        """
        Load a dataframe from the agent's session state into the pandas tools dataframes dictionary.

        Args:
            dataframe_name: Name to use for the dataframe in the pandas tools system
            session_key: Key under which the DataFrame is stored in agent.session_state
            agent: Agent instance whose session_state contains the dataframe

        Returns:
            Confirmation message with dataframe info

        Raises:
            KeyError: If session_key is not present in agent.session_state
            ValueError: If inputs are invalid
        """
        return gtm_operations.load_dataframe_from_session(dataframe_name, session_key, agent)

    def load_gtm_model_only(
        self,
        gtm_file: str | None = None,
        *,
        agent: Agent | None = None,
        use_default: bool = False,
        generate_framesets: bool = False,
    ) -> str:
        """Load only the GTM model and cache it for later projections.

        Priority: (1) session state model, (2) explicit path, (3) default model.
        If use_default=True, forces use of default model.

        Args:
            gtm_file: Optional explicit path to a GTM model file.
            agent: Optional Agent instance to check/store session state.
            use_default: If True, force use of default model even if session model exists.
            generate_framesets: When ``True`` and the model is downloaded from the
                default repository, generate cached frameset CSVs with smiles,
                activity tables, and lookup maps when the source data is
                available.
        """
        # First check if we already have the model in session state
        session_model = gtm_operations.get_session_gtm_model(agent)
        if session_model is not None and not use_default:
            logger.info("Using GTM model from session state")
            if self._gtm_data is None:
                self._gtm_data = gtm_operations.GTMData()
            self._gtm_data.gtm = session_model
            session_path = agent.session_state.get(
                gtm_operations.SESSION_GTM_MODEL_PATH_KEY, "session"
            )
            return f"gtm model from session state ({session_path}) has been loaded"

        # Resolve model path (will check session state path, then explicit, then default)
        resolved_model = gtm_operations.resolve_gtm_model_path(
            gtm_file, agent=agent, use_default=use_default, generate_framesets=generate_framesets
        )
        gtm_model = gtm_operations.load_gtm_model(resolved_model)

        # Store in session state if agent is available
        if agent is not None:
            gtm_operations.set_session_gtm_model(agent, gtm_model, resolved_model)

        if self._gtm_data is None:
            self._gtm_data = gtm_operations.GTMData()

        self._gtm_data.gtm = gtm_model
        return f"gtm model {resolved_model} has been loaded"

    def load_gtm_get_density_matrix(
        self,
        dataset_file: str,
        gtm_file: str | None = None,
        *,
        agent: Agent | None = None,
        use_default: bool = False,
    ) -> str:
        """
        Load GTM model and dataset, return density matrix information.

        Priority: (1) session state model, (2) explicit path, (3) default model.

        Args:
            dataset_file: Path to the CSV dataset file
            gtm_file: Path to the pickled GTM model file. If omitted, uses session state or default model.
            agent: Optional Agent instance to check/store session state.
            use_default: If True, force use of default model even if session model exists.

        Returns:
            Formatted string representation of the density DataFrame

        Raises:
            FileNotFoundError: If either file doesn't exist
            ValueError: If files are invalid or empty
        """
        resolved_model = gtm_operations.resolve_gtm_model_path(
            gtm_file, agent=agent, use_default=use_default
        )
        return gtm_operations.load_gtm_density_matrix(dataset_file, resolved_model)

    def load_and_prep_data(
        self,
        dataset: str,
        gtm_model: str | None = None,
        *,
        agent: Agent | None = None,
        use_default: bool = False,
    ) -> str:
        """
        Load GTM model and molecular data, compute coordinates, and prepare source_mols.

        Always uses the session state model if available. If missing or use_default=True,
        uses default model and stores it in session state. All datasets are projected onto
        the session state model.

        Args:
            dataset: Path to the dataset file
            gtm_model: Path to the GTM model file. If omitted, uses session state or default model.
            agent: Optional Agent instance to check/store session state.
            use_default: If True, force use of default model even if session model exists.

        Returns:
            Success message

        Raises:
            ValueError: If dataset or gtm_model paths are invalid
            FileNotFoundError: If files don't exist
        """
        # Check if we have a session model and use it if available (unless use_default=True)
        session_model = gtm_operations.get_session_gtm_model(agent)
        if session_model is not None and not use_default:
            logger.info("Using GTM model from session state for projection")
            resolved_model = agent.session_state.get(
                gtm_operations.SESSION_GTM_MODEL_PATH_KEY, "session"
            )
            # Use the session model directly instead of reloading
            gtm_for_projection = session_model
        else:
            # Resolve model path (will check session state path, then explicit, then default)
            resolved_model = gtm_operations.resolve_gtm_model_path(
                gtm_model, agent=agent, use_default=use_default
            )
            # Load the model
            gtm_for_projection = gtm_operations.load_gtm_model(resolved_model)
            # Store in session state if agent is available
            if agent is not None:
                gtm_operations.set_session_gtm_model(agent, gtm_for_projection, resolved_model)

        # Project dataset onto the model (using the resolved model path for data_load_and_prep)
        self._gtm_data = gtm_operations.load_and_prepare_gtm_data_with_model(
            dataset, resolved_model, gtm_for_projection
        )
        return (
            f"dataset {dataset} projected onto gtm model {resolved_model} and successfully loaded"
        )

    def analyze_scaffolds_in_nodes(self, list_of_nodes: List[int]) -> str:
        """
        Analyze molecular scaffolds in selected GTM nodes.

        Args:
            list_of_nodes: List of node indices to analyze

        Returns:
            String representation of scaffold frequency table

        Raises:
            ValueError: If list_of_nodes is empty or invalid
            AttributeError: If data hasn't been loaded yet
        """
        if self._gtm_data is None or self._gtm_data.source_mols is None:
            raise AttributeError("Data not loaded. Call load_and_prep_data() first.")

        return gtm_operations.analyze_scaffolds_in_nodes(self._gtm_data.source_mols, list_of_nodes)

    def check_source_datasets_in_nodes(self, list_of_nodes: List[int]) -> str:
        """
        Analyze data source distribution in selected GTM nodes.

        Args:
            list_of_nodes: List of node indices to analyze

        Returns:
            String representation of source frequency table

        Raises:
            ValueError: If list_of_nodes is empty or invalid
            AttributeError: If data hasn't been loaded yet
        """
        if self._gtm_data is None or self._gtm_data.source_mols is None:
            raise AttributeError("Data not loaded. Call load_and_prep_data() first.")

        return gtm_operations.check_source_datasets_in_nodes(
            self._gtm_data.source_mols, list_of_nodes
        )

    def node_id_from_coords(self, x: int | float, y: int | float) -> str:
        """
        Look up which GTM node corresponds to the given x,y coordinates.

        Coordinates are automatically converted to integers (floats like 7.5 are rounded to 8).
        This handles the case where coordinates come from visualization with 0.5 offset.

        Args:
            x: X coordinate (int or float, will be rounded to int)
            y: Y coordinate (int or float, will be rounded to int)

        Returns:
            String representation of the node ID

        Raises:
            AttributeError: If data hasn't been loaded yet
            ValueError: If coordinates are invalid or not found
        """
        # Convert float coordinates to integers (handles 0.5 offset from visualization)
        x_int = int(round(float(x)))
        y_int = int(round(float(y)))

        lookup_table = self._get_coordinate_lookup().copy()
        if isinstance(lookup_table.index, pd.MultiIndex):
            lookup_table = lookup_table.reset_index()

        return gtm_operations.get_node_id_from_coords(lookup_table, x_int, y_int)

    def get_density_summary(self, head: int = 10) -> str:
        """Return a formatted preview of the cached GTM density table."""

        if self._gtm_data is None or self._gtm_data.source is None:
            raise AttributeError("Data not loaded. Call load_and_prep_data() first.")

        head = max(head, 1)
        table = self._gtm_data.source.head(head).copy()
        # Ensure coordinates are integers (no 0.5 offset) for display
        if "x" in table.columns:
            table["x"] = table["x"].astype(int)
        if "y" in table.columns:
            table["y"] = table["y"].astype(int)
        return gtm_operations.df_as_str(table)

    def get_activity_summary(self, head: int = 10) -> str:
        """Return a formatted preview of the cached GTM activity table."""

        if self._gtm_data is None or self._gtm_data.activity_table is None:
            raise AttributeError("Data not loaded. Call load_and_prep_data() first.")

        head = max(head, 1)
        table = self._gtm_data.activity_table.head(head)
        return gtm_operations.df_as_str(table)

    def get_node_lookup_summary(self, head: int = 10) -> str:
        """Return a formatted preview of cached node coordinate lookup tables."""

        if (
            self._gtm_data is None
            or self._gtm_data.node_lookup_by_coords is None
            or self._gtm_data.node_lookup_by_node is None
        ):
            raise AttributeError("Data not loaded. Call load_and_prep_data() first.")

        head = max(head, 1)
        coords_table = self._gtm_data.node_lookup_by_coords.reset_index().head(head).copy()
        node_table = self._gtm_data.node_lookup_by_node.reset_index().head(head).copy()

        # Ensure coordinates are integers (no 0.5 offset) for display
        if "x" in coords_table.columns:
            coords_table["x"] = coords_table["x"].astype(int)
        if "y" in coords_table.columns:
            coords_table["y"] = coords_table["y"].astype(int)
        if "x" in node_table.columns:
            node_table["x"] = node_table["x"].astype(int)
        if "y" in node_table.columns:
            node_table["y"] = node_table["y"].astype(int)

        coords_str = gtm_operations.df_as_str(coords_table)
        node_str = gtm_operations.df_as_str(node_table)

        return (
            "Coordinate → Node lookup:\n"
            f"{coords_str}\n\n"
            "Node → Coordinate lookup:\n"
            f"{node_str}"
        )

    def _require_source_mols(self) -> None:
        if self._gtm_data is None or self._gtm_data.source_mols is None:
            raise AttributeError("Data not loaded. Call load_and_prep_data() first.")

    def _require_density_table(self) -> None:
        if self._gtm_data is None or self._gtm_data.source is None:
            raise AttributeError("Density table not loaded. Call load_and_prep_data() first.")

    def _require_activity_table(self) -> None:
        if self._gtm_data is None or self._gtm_data.activity_table is None:
            raise AttributeError("Activity table not loaded. Call load_and_prep_data() first.")

    @staticmethod
    def _find_sequence_column(df: pd.DataFrame) -> Optional[str]:
        """Find the SEQUENCE column in a DataFrame, or return None."""
        seq_col = gtm_operations.SEQUENCE_COLUMN
        if seq_col in df.columns:
            return seq_col
        for col in df.columns:
            if col.upper() == seq_col:
                return col
        return None

    def _get_coordinate_lookup(self):
        """
        Get coordinate lookup table, ensuring it uses integer coordinates (no 0.5 offset).

        Returns:
            DataFrame with integer coordinates for node lookups
        """
        if self._gtm_data is None:
            raise AttributeError("Data not loaded. Call load_and_prep_data() first.")

        # Prefer the explicitly created lookup table (always has integer coordinates)
        if self._gtm_data.node_lookup_by_coords is not None:
            return self._gtm_data.node_lookup_by_coords

        # Fallback to source table, but ensure coordinates are integers
        if self._gtm_data.source is not None:
            # Create a copy with integer coordinates to ensure consistency
            lookup = self._gtm_data.source[["x", "y", "nodes"]].copy()
            lookup["x"] = lookup["x"].astype(int)
            lookup["y"] = lookup["y"].astype(int)
            return lookup.set_index(["x", "y"])[["nodes"]]

        raise AttributeError(
            "Coordinate lookup table not available. Call load_and_prep_data() first."
        )

    def _format_sample_output(
        self,
        sampled: pd.DataFrame,
        return_format: SampleReturnFormat,
    ) -> Union[str, pd.DataFrame, List[str]]:
        """Normalize GTM sampling output so it can feed other toolkits."""

        # Replace float coordinates with integer coordinates from lookup table
        # This ensures displayed coordinates don't have the 0.5 offset
        if not sampled.empty and "node_index" in sampled.columns and self._gtm_data is not None:
            try:
                lookup_by_node = self._gtm_data.node_lookup_by_node
                if lookup_by_node is not None and not lookup_by_node.empty:
                    # Create a copy to avoid modifying the original
                    sampled = sampled.copy()
                    # Map node_index to integer coordinates
                    node_coords = lookup_by_node.reset_index()

                    # Replace x, y columns with integer coordinates from lookup table
                    if "x" in sampled.columns and "y" in sampled.columns:
                        # Merge with lookup table to get integer coordinates
                        sampled = sampled.merge(
                            node_coords[["nodes", "x", "y"]],
                            left_on="node_index",
                            right_on="nodes",
                            how="left",
                            suffixes=("_old", ""),
                        )
                        # Drop the old float coordinate columns and the merge key
                        if "x_old" in sampled.columns:
                            sampled = sampled.drop(columns=["x_old", "y_old", "nodes"])
                        # Ensure integer type
                        if "x" in sampled.columns:
                            sampled["x"] = sampled["x"].astype("Int64")  # Nullable integer type
                        if "y" in sampled.columns:
                            sampled["y"] = sampled["y"].astype("Int64")
            except Exception as e:
                # If lookup fails, log but don't fail - just use original coordinates
                logger.warning(f"Could not replace coordinates with integer values: {e}")

        if return_format == "dataframe":
            return sampled.reset_index(drop=True)

        if return_format == "sequences":
            seq_col = self._find_sequence_column(sampled)
            if seq_col is not None:
                return sampled[seq_col].tolist()
            return gtm_operations.df_as_str(sampled)

        if return_format == "smiles":
            try:
                smiles_column = gtm_operations.find_smiles_column(sampled)
                return sampled[smiles_column].tolist()
            except ValueError:
                # Fallback: try SEQUENCE column for peptide data
                seq_col = self._find_sequence_column(sampled)
                if seq_col is not None:
                    return sampled[seq_col].tolist()
                raise

        # Default human-readable table
        return gtm_operations.df_as_str(sampled)

    def _handle_empty_sample(
        self,
        return_format: SampleReturnFormat,
        message: str,
    ) -> Union[str, pd.DataFrame, List[str]]:
        """Return an empty payload matching the requested format."""

        if return_format == "dataframe":
            return pd.DataFrame()
        if return_format in ("smiles", "sequences"):
            return []
        return message

    def sample_nodes(
        self,
        node_ids: List[int],
        sample_size: int | None = None,
        random_state: int | None = None,
        return_format: SampleReturnFormat = "text",
    ) -> Union[str, pd.DataFrame, List[str]]:
        """Sample molecules assigned to the provided GTM node identifiers."""

        self._require_source_mols()

        sampled = gtm_operations.sample_molecules_from_nodes(
            self._gtm_data.source_mols,
            node_ids,
            sample_size=sample_size,
            random_state=random_state,
        )

        if sampled.empty:
            return self._handle_empty_sample(
                return_format,
                "No molecules found for the requested nodes.",
            )

        return self._format_sample_output(sampled, return_format)

    def sample_dense_nodes(
        self,
        top_n: int = 5,
        min_density: float | None = None,
        sample_size: int | None = None,
        random_state: int | None = None,
        use_filtered: bool = True,
        return_format: SampleReturnFormat = "text",
    ) -> Union[str, pd.DataFrame, List[str]]:
        """Sample molecules from the densest GTM nodes."""

        self._require_source_mols()
        self._require_density_table()

        nodes = gtm_operations.select_nodes_by_density(
            self._gtm_data.source,
            top_n=top_n,
            min_density=min_density,
            use_filtered=use_filtered,
        )

        if not nodes:
            return self._handle_empty_sample(
                return_format,
                "No nodes matched the density criteria.",
            )

        sampled = gtm_operations.sample_molecules_from_nodes(
            self._gtm_data.source_mols,
            nodes,
            sample_size=sample_size,
            random_state=random_state,
        )

        if sampled.empty:
            return self._handle_empty_sample(
                return_format,
                "No molecules found for the selected dense nodes.",
            )

        return self._format_sample_output(sampled, return_format)

    def sample_active_nodes(
        self,
        top_n: int = 5,
        min_value: float | None = None,
        activity_column: str | None = None,
        ascending: bool = False,
        sample_size: int | None = None,
        random_state: int | None = None,
        return_format: SampleReturnFormat = "text",
    ) -> Union[str, pd.DataFrame, List[str]]:
        """Sample molecules from nodes ranked by an activity metric."""

        self._require_source_mols()
        self._require_activity_table()

        nodes = gtm_operations.select_nodes_by_activity(
            self._gtm_data.activity_table,
            activity_column=activity_column,
            top_n=top_n,
            min_value=min_value,
            ascending=ascending,
        )

        if not nodes:
            return self._handle_empty_sample(
                return_format,
                "No nodes matched the activity criteria.",
            )

        sampled = gtm_operations.sample_molecules_from_nodes(
            self._gtm_data.source_mols,
            nodes,
            sample_size=sample_size,
            random_state=random_state,
        )

        if sampled.empty:
            return self._handle_empty_sample(
                return_format,
                "No molecules found for the selected activity nodes.",
            )

        return self._format_sample_output(sampled, return_format)

    def sample_by_coordinates(
        self,
        coordinates: Iterable[Tuple[int | float, int | float] | Sequence[int | float] | dict],
        sample_size: int | None = None,
        random_state: int | None = None,
        allow_missing: bool = False,
        return_format: SampleReturnFormat = "text",
    ) -> Union[str, pd.DataFrame, List[str]]:
        """
        Sample molecules located at the provided coordinate pairs.

        Coordinates are automatically converted to integers (floats like 7.5 are rounded to 8).
        This handles the case where coordinates come from visualization with 0.5 offset.
        """

        self._require_source_mols()
        lookup_table = self._get_coordinate_lookup()

        sampled = gtm_operations.sample_molecules_by_coordinates(
            self._gtm_data.source_mols,
            lookup_table,
            coordinates,
            sample_size=sample_size,
            random_state=random_state,
            allow_missing=allow_missing,
        )

        if sampled.empty:
            return self._handle_empty_sample(
                return_format,
                "No molecules found for the requested coordinates.",
            )

        return self._format_sample_output(sampled, return_format)

    def create_activity_landscapes(
        self,
        dataset: str,
        gtm_model: str | None = None,
        node_threshold: float = 0.1,
        chart_width: int = 600,
        chart_height: int = 600,
        *,
        agent: Agent | None = None,
        use_default: bool = False,
    ) -> str:
        """
        Create activity landscapes from GTM model and dataset.

        Priority: (1) session state model, (2) explicit path, (3) default model.

        Args:
            dataset: Path to the dataset file
            gtm_model: Path to the GTM model file. If omitted, uses session state or default model.
            node_threshold: Threshold below which nodes are excluded
            chart_width: Width of the output chart (pixels)
            chart_height: Height of the output chart (pixels)
            agent: Optional Agent instance to check/store session state.
            use_default: If True, force use of default model even if session model exists.

        Returns:
            Success message with file paths

        Raises:
            ValueError: If inputs are invalid
            FileNotFoundError: If dataset or model files don't exist
        """
        resolved_model = gtm_operations.resolve_gtm_model_path(
            gtm_model, agent=agent, use_default=use_default
        )
        return gtm_operations.create_activity_landscapes_tool(
            dataset, resolved_model, node_threshold, chart_width, chart_height
        )

    def project_data_on_gtm(self, dataset_file: str, gtm_model_file: str) -> str:
        """
        Preprocess a new dataset for projection onto an existing GTM map.

        Validates SMILES, filters invalid molecules, normalizes column names,
        and checks descriptor compatibility with the GTM model. Saves a
        preprocessed CSV that can be used directly with save_gtm_plot().

        Args:
            dataset_file: Path to CSV with a SMILES column ('smi', 'SMILES', etc.)
            gtm_model_file: Path to the GTM model file (.pkl.gz)

        Returns:
            Summary with path to preprocessed dataset CSV
        """
        return gtm_operations.project_data_on_gtm(dataset_file, gtm_model_file)

    # =========================================================================
    # Latent-Space GTM Tools (for Peptide WAE integration)
    # =========================================================================

    def train_gtm_on_latent_space(
        self,
        latent_vectors_csv: str,
        dataset_name: str,
        gtm_name: str,
        agent: Agent,
    ) -> str:
        """
        Train a GTM on pre-computed latent vectors (e.g. from Peptide WAE encoder).

        This tool builds a Generative Topographic Map directly on latent space vectors
        instead of molecular descriptors. Used for peptide WAE latent space analysis.

        The latent vectors CSV should contain numeric columns representing the latent
        dimensions (e.g. z_0, z_1, ..., z_99 for a 100-dim latent space), or the
        output of encode_peptides saved to CSV.

        Args:
            latent_vectors_csv: Path to CSV file containing latent vectors (one row per sample)
            dataset_name: Key under which the latent DataFrame will be saved in agent.session_state
            gtm_name: Key under which the trained GTM model will be saved in agent.session_state
            agent: The agent whose session_state dict will be updated

        Returns:
            Human-readable message reporting the best entropy score achieved
        """
        from cs_copilot.storage import S3

        logger.info(f"Training latent GTM from: {latent_vectors_csv}")

        # Load latent vectors
        with S3.open(latent_vectors_csv, "r") as f:
            df = pd.read_csv(f)

        # Extract numeric columns as latent vectors
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns found in latent vectors CSV")

        X = df[numeric_cols].values.astype(np.float64)
        logger.info(f"Loaded {X.shape[0]} latent vectors with {X.shape[1]} dimensions")

        # Train the GTM
        gtm_model, scaler, best_score = gtm_operations.train_latent_gtm(X)

        # Store in session state
        if agent.session_state is None:
            agent.session_state = {}
        agent.session_state[dataset_name] = df
        agent.session_state[gtm_name] = gtm_model
        agent.session_state[gtm_operations.SESSION_LATENT_GTM_MODEL_KEY] = gtm_model
        agent.session_state[gtm_operations.SESSION_LATENT_GTM_SCALER_KEY] = scaler

        # Also set as current GTM model for general access
        optimized_model_path = f"{gtm_name}_latent.pkl.gz"
        gtm_operations.set_session_gtm_model(agent, gtm_model, optimized_model_path)

        # Auto-populate GTMData so sampling works immediately after training
        seq_col = gtm_operations.SEQUENCE_COLUMN
        sequences = df[seq_col].tolist() if seq_col in df.columns else None
        self._gtm_data = gtm_operations.populate_gtm_data_from_latent_vectors(
            X, gtm_model, scaler, sequences=sequences, source_df=df
        )

        return (
            f"Latent GTM trained successfully on {X.shape[0]} vectors ({X.shape[1]} dims). "
            f"Entropy: {best_score:.4f}. Model stored as '{gtm_name}' in session state. "
            f"Sampling is now available ({len(self._gtm_data.source)} density nodes populated)."
        )

    def load_latent_data_on_gtm(
        self,
        latent_vectors_csv: str | None = None,
        sequences_csv: str | None = None,
        *,
        agent: Agent,
    ) -> str:
        """
        Load a dataset of latent vectors onto an existing latent GTM for sampling.

        Use this tool to project a different set of peptide latent vectors onto a
        previously trained latent GTM. This enables sampling from the new dataset
        using GTM sampling tools (sample_dense_nodes, sample_active_nodes, etc.).

        Prerequisites: A latent GTM must have been trained first via train_gtm_on_latent_space.

        Args:
            latent_vectors_csv: Path to CSV with latent vectors. If None, looks for
                               'latent_vectors' in session state.
            sequences_csv: Path to CSV with SEQUENCE column. If None, looks for sequences
                          in latent_vectors_csv or session state.
            agent: Agent instance to access session state.

        Returns:
            Success message with number of samples loaded.
        """
        from cs_copilot.storage import S3

        if agent.session_state is None:
            raise ValueError("Agent session_state is required")

        gtm_model = agent.session_state.get(gtm_operations.SESSION_LATENT_GTM_MODEL_KEY)
        scaler = agent.session_state.get(gtm_operations.SESSION_LATENT_GTM_SCALER_KEY)

        if gtm_model is None or scaler is None:
            raise ValueError(
                "No latent GTM model found in session state. "
                "Train a latent GTM first using train_gtm_on_latent_space."
            )

        # Load latent vectors
        if latent_vectors_csv:
            with S3.open(latent_vectors_csv, "r") as f:
                df = pd.read_csv(f)
        elif "latent_vectors" in agent.session_state:
            lv = agent.session_state["latent_vectors"]
            df = lv if isinstance(lv, pd.DataFrame) else pd.DataFrame(lv)
        else:
            raise ValueError(
                "No latent vectors provided. Pass latent_vectors_csv or store as "
                "'latent_vectors' in session state."
            )

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns found in latent vectors data")

        X = df[numeric_cols].values.astype(np.float64)

        # Get sequences
        seq_col = gtm_operations.SEQUENCE_COLUMN
        sequences = None
        if seq_col in df.columns:
            sequences = df[seq_col].tolist()
        elif sequences_csv:
            with S3.open(sequences_csv, "r") as f:
                seq_df = pd.read_csv(f)
            if seq_col in seq_df.columns:
                sequences = seq_df[seq_col].tolist()

        # Populate GTMData
        self._gtm_data = gtm_operations.populate_gtm_data_from_latent_vectors(
            X, gtm_model, scaler, sequences=sequences, source_df=df
        )

        return (
            f"Loaded {X.shape[0]} latent vectors onto latent GTM. "
            f"Density nodes: {len(self._gtm_data.source)}. "
            f"Sampling is now available."
        )

    def create_peptide_activity_landscapes(
        self,
        dbaasp_path: str | None,
        latent_vectors_csv: str | None = None,
        organism: str = "all",
        node_threshold: float = 0.1,
        chart_width: int = 600,
        chart_height: int = 600,
        *,
        agent: Agent | None = None,
    ) -> str:
        """
        Create antimicrobial activity landscapes for peptides on a WAE-based GTM.

        End-to-end workflow: loads DBAASP activity data, uses pre-encoded latent vectors
        (from session state or CSV), projects onto the latent GTM, and creates
        per-organism classification landscapes (active vs inactive).

        Prerequisites: A latent GTM must have been trained first via train_gtm_on_latent_space.
        The latent vectors for the DBAASP peptides must be available in session state
        (key: 'dbaasp_latent_vectors') or provided via latent_vectors_csv.

        Args:
            dbaasp_path: Path to DBAASP CSV file with SEQUENCE and organism activity columns.
                        If None, uses default path.
            latent_vectors_csv: Path to CSV with pre-encoded latent vectors for DBAASP sequences.
                              If None, looks for 'dbaasp_latent_vectors' in session state.
            organism: Organism name to create landscape for (e.g. 'E. coli', 'S. aureus').
                     Use "all" to process all organisms with sufficient data.
            node_threshold: Threshold below which nodes are excluded (default 0.1).
            chart_width: Width of the output chart (pixels).
            chart_height: Height of the output chart (pixels).
            agent: Agent instance to access session state for GTM model and latent vectors.

        Returns:
            Summary message with paths to saved landscape files.
        """
        from cs_copilot.storage import S3

        # Get latent GTM model and scaler from session state
        if agent is None or agent.session_state is None:
            raise ValueError("Agent with session_state required for peptide activity landscapes")

        gtm_model = agent.session_state.get(gtm_operations.SESSION_LATENT_GTM_MODEL_KEY)
        scaler = agent.session_state.get(gtm_operations.SESSION_LATENT_GTM_SCALER_KEY)

        if gtm_model is None or scaler is None:
            raise ValueError(
                "No latent GTM model found in session state. "
                "Train a latent GTM first using train_gtm_on_latent_space."
            )

        # Get latent vectors
        latent_vectors = None
        if latent_vectors_csv:
            with S3.open(latent_vectors_csv, "r") as f:
                lv_df = pd.read_csv(f)
            numeric_cols = lv_df.select_dtypes(include=[np.number]).columns.tolist()
            latent_vectors = lv_df[numeric_cols].values.astype(np.float64)
        elif "dbaasp_latent_vectors" in agent.session_state:
            latent_vectors = agent.session_state["dbaasp_latent_vectors"]
            if not isinstance(latent_vectors, np.ndarray):
                latent_vectors = np.array(latent_vectors)

        if latent_vectors is None:
            raise ValueError(
                "No latent vectors available. Either provide latent_vectors_csv "
                "or encode DBAASP sequences first and store as 'dbaasp_latent_vectors' in session state."
            )

        return gtm_operations.create_peptide_activity_landscapes_tool(
            dbaasp_path=dbaasp_path,
            latent_vectors=latent_vectors,
            gtm_model=gtm_model,
            scaler=scaler,
            organism=organism,
            node_threshold=node_threshold,
            chart_width=chart_width,
            chart_height=chart_height,
        )
