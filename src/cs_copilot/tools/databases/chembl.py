#!/usr/bin/env python
# coding: utf-8
"""
ChEMBL-specific database toolkit implementation.
"""

import logging
import time
from typing import Any, Dict, Optional, Sequence

import pandas as pd
from agno.agent import Agent

from cs_copilot.storage import S3
from cs_copilot.tools.chemistry.standardize import standardize_smiles_column
from cs_copilot.tools.io.utils import validate_positive_int

from .base import BaseDatabaseToolkit, DatabaseError, NotFound, RateLimited, ValidationError
from .types import DBConfig, PaginationMode, QueryParams, ResultPage

logger = logging.getLogger(__name__)


class ChemblToolkit(BaseDatabaseToolkit):
    """
    ChEMBL-specific database toolkit implementation.

    Provides access to ChEMBL database through their web API,
    handling ChEMBL-specific query formats, pagination, and data structures.
    """

    # ChEMBL-specific constants
    BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

    # Default field mappings for different ChEMBL resources
    ACTIVITY_FIELDS = [
        "activity_id",
        "assay_chembl_id",
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_type",
        "standard_value",
        "standard_units",
        "pchembl_value",
        "activity_comment",
        "data_validity_comment",
        "potential_duplicate",
    ]

    MOLECULE_FIELDS = ["molecule_chembl_id", "canonical_smiles", "molecule_structures"]

    ASSAY_FIELDS = [
        "assay_chembl_id",
        "description",
        "assay_type",
        "assay_organism",
        "target_chembl_id",
    ]

    def __init__(self, config: Optional[DBConfig] = None, **toolkit_kwargs):
        """
        Initialize ChEMBL toolkit.

        Args:
            config: Database configuration (optional, uses defaults if not provided)
            **toolkit_kwargs: Additional arguments for Agno Toolkit
        """
        if config is None:
            config = DBConfig(
                uri=self.BASE_URL,
                timeout_s=30.0,
                retries=3,
                page_size=100,
                rate_limit=10.0,  # ChEMBL rate limit
                supports_http_api=True,
                pagination_mode=PaginationMode.OFFSET_LIMIT,
            )

        # Set default instructions if not provided
        if "instructions" not in toolkit_kwargs:
            toolkit_kwargs["instructions"] = (
                "ChEMBL database toolkit for fetching compound bioactivity data, generating EDA reports, and analyzing chemical datasets."
            )

        super().__init__(
            config,
            name="chembl_toolkit",
            **toolkit_kwargs,
        )
        self._client = None  # Will be initialized lazily
        self._client_init_error = None  # Store initialization error if any
        self.register(self.fetch_compounds)
        self.register(self.describe_dataset)
        self.register(self.convert_to_chembl_query)

    def _ensure_client(self):
        """Lazy initialization of ChEMBL client."""
        if self._client is None and self._client_init_error is None:
            try:
                from chembl_webresource_client.new_client import new_client

                self._client = new_client
                logger.info("ChEMBL client initialized successfully")
            except Exception as e:
                self._client_init_error = e
                error_msg = f"Failed to initialize ChEMBL client: {e}. The ChEMBL API may be temporarily unavailable."
                logger.error(error_msg)
                raise DatabaseError(error_msg) from e
        elif self._client_init_error is not None:
            raise DatabaseError(f"ChEMBL client unavailable: {self._client_init_error}")
        return self._client

    def _get_resource_client(self, resource: str):
        client = self._ensure_client()
        try:
            return getattr(client, resource)
        except AttributeError as exc:
            raise ValidationError(f"Unsupported ChEMBL resource: {resource}") from exc

    def connect(self) -> None:
        """Connect to ChEMBL API (validate access)."""
        try:
            client = self._ensure_client()
            list(client.target.filter(target_chembl_id="CHEMBL1").only(["target_chembl_id"])[:1])
            self._connected = True
            logger.info("Connected to ChEMBL API successfully")
        except Exception as e:
            self._connected = False
            logger.error(f"Failed to connect to ChEMBL API: {e}")
            raise DatabaseError(f"ChEMBL connection failed: {e}") from e

    def ping(self) -> bool:
        """Test ChEMBL API connectivity."""
        try:
            client = self._ensure_client()
            return client.status is not None
        except Exception as e:
            logger.warning(f"ChEMBL ping failed: {e}")
            return False

    def query(self, params: QueryParams) -> ResultPage:
        """
        Execute a single-page query using the ChEMBL webresource client.

        The implementation mirrors the readable style from the official
        `demo_wrc.ipynb` notebook: pick a resource, apply filters, optionally
        restrict fields, and slice results with Pythonic indexing.
        """

        start_time = time.perf_counter()
        resource_name = params.extra_params.get("resource", "activity")
        limit = params.limit or self.config.page_size
        offset = params.offset or 0

        try:
            resource = self._get_resource_client(resource_name)

            # 1) Build the query using provided filters
            query = resource.filter(**(params.filters or {}))

            # 2) Keep only requested fields when provided
            if params.fields:
                query = query.only(list(params.fields))

            # 3) Apply offset via ``skip`` then pull a single page just like the
            #    notebook's ``activities[:20]`` pattern
            if offset:
                query = query.skip(offset)

            raw_records = list(query[:limit])
            mapped_records = [self.map_fields(record) for record in raw_records]

            has_more = len(raw_records) == limit
            next_offset = offset + limit if has_more else None

            query_time_ms = (time.perf_counter() - start_time) * 1000
            return ResultPage(
                records=mapped_records,
                total=None,
                next_offset=next_offset,
                has_more=has_more,
                query_time_ms=query_time_ms,
            )

        except Exception as exc:
            logger.error(f"ChEMBL query failed: {exc}")
            raise self.handle_error(exc) from exc

    def map_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map ChEMBL-specific field names to standard names.

        Args:
            record: Raw ChEMBL record

        Returns:
            Record with mapped field names
        """
        # ChEMBL records are generally well-structured, but we might want
        # to flatten some nested structures
        mapped = record.copy()

        # Handle nested molecule structures
        if "molecule_structures" in mapped and mapped["molecule_structures"]:
            structures = mapped["molecule_structures"]
            if isinstance(structures, dict) and "canonical_smiles" in structures:
                mapped["canonical_smiles"] = structures["canonical_smiles"]

        return mapped

    def handle_error(self, error: Exception) -> DatabaseError:
        """Map ChEMBL-specific errors to standard error types."""
        error_str = str(error).lower()

        if "timeout" in error_str or "timed out" in error_str:
            return RateLimited(f"ChEMBL API timeout: {error}")
        elif "not found" in error_str or "404" in error_str:
            return NotFound(f"ChEMBL resource not found: {error}")
        elif "rate limit" in error_str or "429" in error_str:
            return RateLimited(f"ChEMBL rate limit exceeded: {error}")
        elif "invalid" in error_str or "400" in error_str:
            return ValidationError(f"Invalid ChEMBL query: {error}")
        else:
            return DatabaseError(f"ChEMBL API error: {error}")

    def _normalize_assay_types(self, assay_types: Optional[Sequence[str]]) -> list:
        """Normalize human-readable assay types to ChEMBL codes."""

        if assay_types is None:
            return []

        if isinstance(assay_types, str):
            values = [assay_types]
        else:
            values = list(assay_types)

        mapping = {
            "binding": "B",
            "functional": "F",
            "admet": "A",
        }

        normalized = []
        for raw in values:
            if raw is None:
                continue
            item = str(raw).strip()
            if not item:
                continue

            lowered = item.lower()
            if lowered in mapping:
                normalized.append(mapping[lowered])
            elif item.upper() in mapping.values():
                normalized.append(item.upper())
            else:
                valid = ", ".join(sorted({*mapping.keys(), *mapping.values()}))
                raise ValueError(f"Invalid assay type '{item}'. Expected one of: {valid}.")

        # Remove duplicates while preserving order
        seen = set()
        ordered = []
        for code in normalized:
            if code not in seen:
                ordered.append(code)
                seen.add(code)

        return ordered

    def fetch_compounds(
        self,
        query: str = "bioactivity data",
        max_records: Optional[int] = None,
        organism: Optional[str] = None,
        assay_types: Optional[Sequence[str]] = None,
        mechanism: Optional[str] = None,
        agent: Optional[Agent] = None,
    ) -> str:
        """
        Fetch compound bioactivity data from ChEMBL database using multiple keywords.

        Args:
            query: Search term(s) for assay descriptions. Can be:
                - A single string (e.g., "kinase")
                - A comma-separated string (e.g., "kinase, BRAF, PPAR-alpha")
                - Multiple queries will be processed separately and merged
            max_records: Maximum number of records to return per keyword (None for all)
            organism: Optional organism filter (e.g., "Homo sapiens", "Influenza A")
            assay_types: Optional list of assay types to keep. MUST be a list, not a dict.
                Examples:
                - ["B", "F"] for binding and functional assays (default)
                - ["binding", "functional"] (human-friendly names)
                - [] to allow all assay types (empty list, not empty dict)
                - None to use default (binding + functional)
                Common mistake: Do NOT use {} (empty dict), use [] (empty list) instead.
            mechanism: Optional mechanism of action filter. When provided, only assays
                whose description contains this term (case-insensitive) are kept.
                Examples: "agonist", "antagonist", "modulator", "inverse agonist", None.

        Returns:
            Status message with information about fetched data

        Raises:
            ValueError: If query is empty or invalid
            DatabaseError: If ChEMBL API request fails
        """
        if not query or not isinstance(query, str):
            raise ValueError("query must be a non-empty string")

        if max_records is not None:
            validate_positive_int(max_records, "max_records")

        if mechanism is not None:
            if not isinstance(mechanism, str):
                raise ValueError("mechanism must be a string or None")
            mechanism = mechanism.strip() or None

        # Handle LLM confusion: convert empty dict {} to empty list []
        if isinstance(assay_types, dict):
            if len(assay_types) == 0:
                assay_types = []
            else:
                raise ValueError(
                    "assay_types must be a list/sequence, not a dict. "
                    "Use a list like ['B', 'F'] or [] for all types."
                )

        default_assay_types = ["B", "F"] if assay_types is None else assay_types
        assay_type_codes = self._normalize_assay_types(default_assay_types)

        # Parse multiple keywords from query string
        keywords = [kw.strip() for kw in query.split(",") if kw.strip()]
        if not keywords:
            raise ValueError("query must contain at least one non-empty keyword")

        logger.info(f"Fetching ChEMBL compounds for {len(keywords)} keyword(s): {keywords}")

        client = self._ensure_client()
        all_dataframes = []
        all_assay_ids = set()  # Track unique assays across all keywords

        try:
            # Process each keyword separately
            for keyword in keywords:
                logger.info(f"Processing keyword: '{keyword}'")

                # Step 1: Fetch assays for this keyword
                logger.debug(f"Fetching assays from ChEMBL for keyword: '{keyword}'")

                assay_filter_kwargs: Dict[str, Any] = {"description__icontains": keyword}
                if organism:
                    assay_filter_kwargs["target_organism__icontains"] = organism

                assays = list(client.assay.filter(**assay_filter_kwargs)[:1000])

                if mechanism:
                    mechanism_lower = mechanism.lower()
                    before = len(assays)
                    assays = [
                        a for a in assays if mechanism_lower in (a.get("description") or "").lower()
                    ]
                    logger.info(
                        f"Mechanism filter '{mechanism}' reduced assays from {before} "
                        f"to {len(assays)} for keyword '{keyword}'"
                    )

                assay_ids = [assay["assay_chembl_id"] for assay in assays]

                if not assay_ids:
                    logger.warning(f"No assays found for keyword: {keyword}")
                    continue

                logger.info(f"Found {len(assay_ids)} assays for keyword: '{keyword}'")
                all_assay_ids.update(assay_ids)

                # Step 2: Fetch activities for these assays
                logger.debug(f"Fetching activities from ChEMBL for keyword: '{keyword}'")
                activity_filter_kwargs: Dict[str, Any] = {"assay_chembl_id__in": assay_ids}

                if assay_type_codes:
                    activity_filter_kwargs["assay_type__in"] = assay_type_codes

                activity_query = client.activity.filter(**activity_filter_kwargs).only(
                    self.ACTIVITY_FIELDS
                )
                activity_limit = max_records or self.config.page_size
                activities = list(activity_query[:activity_limit])

                if not activities:
                    logger.warning(f"No activity data found for keyword: {keyword}")
                    continue

                logger.info(
                    f"Retrieved {len(activities)} activity records for keyword: '{keyword}'"
                )

                # Step 3: Convert to DataFrame and add SMILES
                df = self.to_dataframe(activities)
                df = self._merge_assay_data(df, assays)
                df["smi"] = df["canonical_smiles"]  # Use standard 'smi' column name
                all_dataframes.append(df)

            # Step 4: Merge all DataFrames and remove duplicates
            if not all_dataframes:
                return f"No data found for any of the keywords: {keywords}"

            logger.info(f"Merging {len(all_dataframes)} datasets and removing duplicates")
            merged_df = pd.concat(all_dataframes, ignore_index=True)

            # Remove duplicates based on activity_id (most specific identifier)
            # If activity_id is not available, use combination of molecule_chembl_id, assay_chembl_id, and standard_value
            initial_count = len(merged_df)
            if "activity_id" in merged_df.columns:
                merged_df = merged_df.drop_duplicates(subset=["activity_id"], keep="first")
            else:
                # Fallback: use combination of identifiers
                dedup_cols = ["molecule_chembl_id", "assay_chembl_id"]
                if "standard_value" in merged_df.columns:
                    dedup_cols.append("standard_value")
                merged_df = merged_df.drop_duplicates(subset=dedup_cols, keep="first")

            duplicates_removed = initial_count - len(merged_df)
            logger.info(
                f"Removed {duplicates_removed} duplicate records. Final dataset: {len(merged_df)} records"
            )

            # Step 5: Standardize SMILES
            pre_std_count = len(merged_df)
            merged_df = standardize_smiles_column(merged_df, "smi")
            merged_df = merged_df.dropna(subset=["smi"])
            std_dropped = pre_std_count - len(merged_df)
            if std_dropped:
                logger.info(f"Dropped {std_dropped} records with unstandardizable SMILES")

            # Step 6: Save to S3
            # Create filename from all keywords
            query_slug = "_".join(
                [kw.replace(" ", "_") for kw in keywords[:3]]
            )  # Limit to first 3 keywords for filename
            if len(keywords) > 3:
                query_slug += "_and_more"
            filename = self._save_chembl_data(merged_df, query_slug)

            # Store dataset path in session state for cross-agent access
            if agent is not None:
                if agent.session_state is None:
                    agent.session_state = {}
                if "data_file_paths" not in agent.session_state:
                    agent.session_state["data_file_paths"] = {}
                agent.session_state["data_file_paths"]["dataset_path"] = filename
                logger.info(f"Stored dataset_path in session_state: {filename}")

            total_assays = len(all_assay_ids)
            return self._format_success_message(
                merged_df,
                keywords,
                filename,
                total_assays,
                duplicates_removed,
                organism_filter=organism,
                assay_type_codes=assay_type_codes,
                mechanism_filter=mechanism,
            )

        except Exception as e:
            logger.error(f"Error fetching ChEMBL compounds: {e}")
            raise

    def _add_smiles_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add SMILES structures to activity DataFrame."""
        logger.debug("Fetching SMILES structures")

        # Get unique molecule ChEMBL IDs
        unique_molecule_ids = df["molecule_chembl_id"].unique().tolist()
        logger.info(f"Fetching SMILES for {len(unique_molecule_ids)} unique molecules")

        # Fetch molecules with SMILES data using the ChEMBL client
        molecule_query = (
            self._ensure_client()
            .molecule.filter(molecule_chembl_id__in=unique_molecule_ids)
            .only(self.MOLECULE_FIELDS)
        )
        molecules = list(molecule_query[: len(unique_molecule_ids)])

        # Build SMILES lookup dictionary
        smiles_dict = {}
        for mol in molecules:
            cid = mol.get("molecule_chembl_id")
            smi = mol.get("canonical_smiles")

            # Try nested structure if top-level SMILES not available
            if not smi:
                structs = mol.get("molecule_structures") or {}
                smi = structs.get("canonical_smiles")

            smiles_dict[cid] = smi

        # Add SMILES column to DataFrame
        df["smi"] = df["molecule_chembl_id"].map(smiles_dict)
        df = standardize_smiles_column(df, "smi")

        smiles_count = df["smi"].notna().sum()
        logger.info(f"Successfully mapped SMILES for {smiles_count}/{len(df)} records")

        return df

    def _merge_assay_data(
        self, activities_df: pd.DataFrame, assays: Sequence[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Attach assay metadata to activity data using assay_chembl_id."""
        if activities_df.empty or not assays:
            return activities_df

        assay_df = self.to_dataframe(assays)
        if assay_df.empty or "assay_chembl_id" not in assay_df.columns:
            logger.warning("Assay data missing assay_chembl_id; skipping assay merge")
            return activities_df

        return activities_df.merge(
            assay_df,
            on="assay_chembl_id",
            how="left",
            suffixes=("", "_assay"),
        )

    def _save_chembl_data(self, df: pd.DataFrame, query: str) -> str:
        """Save ChEMBL data to S3 and return filename."""
        filename = f"chembl_{query.replace(' ', '_')}.csv"

        try:
            with S3.open(filename, "w") as f:
                df.to_csv(f, index=False)
            logger.info(f"Saved ChEMBL data to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving ChEMBL data: {e}")
            raise

    def _format_success_message(
        self,
        df: pd.DataFrame,
        keywords: list,
        filename: str,
        total_assays: int = 0,
        duplicates_removed: int = 0,
        organism_filter: Optional[str] = None,
        assay_type_codes: Optional[Sequence[str]] = None,
        mechanism_filter: Optional[str] = None,
    ) -> str:
        """Format success message for ChEMBL data fetch."""
        sample_row = df.head(1).to_string(index=False) if not df.empty else "No data"
        columns_preview = ", ".join(df.columns[:6].tolist()) + (
            "..." if len(df.columns) > 6 else ""
        )

        keywords_str = ", ".join([f"'{kw}'" for kw in keywords])
        message = (
            f"✅ Fetched {len(df)} records from {len(keywords)} keyword(s): {keywords_str}\n"
            f"📄 Saved to S3: `{filename}`\n"
        )

        if total_assays > 0:
            message += f"🔬 Found {total_assays} unique assays across all keywords\n"

        assay_type_labels = {"B": "Binding", "F": "Functional", "A": "ADMET"}
        if assay_type_codes:
            readable_types = [assay_type_labels.get(code, code) for code in assay_type_codes]
            message += f"🧪 Assay types filtered: {', '.join(readable_types)}\n"

        if mechanism_filter:
            message += f"⚙️ Mechanism of action filter: {mechanism_filter}\n"

        if organism_filter:
            message += f"🧬 Target organism filter: {organism_filter}\n"

        if duplicates_removed > 0:
            message += f"🔄 Removed {duplicates_removed} duplicate records\n"

        message += (
            f"📊 Columns: {columns_preview}\n"
            f"🔍 Sample row:\n{sample_row}\n\n"
            f"💡 You can now use '{filename}' directly with pandas operations"
        )

        return message

    def describe_dataset(self, path_to_dataset: str) -> str:
        """
        Compute and return descriptive statistics for a tabular dataset.

        Args:
            path_to_dataset: S3 or local path to the CSV file containing the dataset

        Returns:
            String representation of the pandas DataFrame's descriptive statistics

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If path is empty or file is empty
            pandas.errors.ParserError: If the CSV file cannot be parsed
        """
        if not path_to_dataset:
            raise ValueError("path_to_dataset cannot be empty")

        logger.info(f"Describing dataset: {path_to_dataset}")

        try:
            with S3.open(path_to_dataset, "r") as f:
                df = pd.read_csv(f)

            if df.empty:
                raise ValueError(f"Dataset at '{path_to_dataset}' is empty")

            logger.info(f"Loaded dataset with shape {df.shape} for description")
            description = df.describe(include="all")
            return str(description)

        except Exception as e:
            logger.error(f"Error describing dataset {path_to_dataset}: {e}")
            raise

    def convert_to_chembl_query(self, natural_prompt: str) -> str:
        """
        Generate multiple keyword variations from a target description for ChEMBL assay description searches.

        Args:
            natural_prompt: Target description (should already have generic terms removed in previous steps)

        Returns:
            Formatted instruction for generating multiple ChEMBL search keywords

        Example:
            >>> convert_to_chembl_query("cyclin dependent kinase 2")
            Returns instruction to generate multiple keywords like: 'cdk2', 'kinase 2', 'cyclin dependent kinase 2'
        """
        if not natural_prompt or not isinstance(natural_prompt, str):
            raise ValueError("natural_prompt must be a non-empty string")

        return (
            f"You are preparing queries for ChEMBL's `assay_description__icontains` filter.\n"
            f"Given this target description: '{natural_prompt.strip()}', generate multiple keyword variations (typically 3-5) that are likely to appear in ChEMBL assay descriptions.\n"
            f"Include:\n"
            f"  - Abbreviations (e.g., 'cyclin dependent kinase 2' → 'cdk2')\n"
            f"  - Shortened forms (e.g., 'kinase 2')\n"
            f"  - Full names or common variations\n"
            f"Replace greek characters with their English names (e.g., 'α' → 'alpha', 'β' → 'beta').\n"
            f"Output a comma-separated list of keyword phrases suitable for searching assay descriptions.\n"
            f"Example: For 'cyclin dependent kinase 2', generate: 'cdk2, kinase 2, cyclin dependent kinase 2'"
        )
