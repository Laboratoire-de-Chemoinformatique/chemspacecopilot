#!/usr/bin/env python
# coding: utf-8
"""
ChEMBL-specific database toolkit implementation.
"""

import logging
import re
import time
from typing import Any, Dict, Optional, Sequence

import pandas as pd
from agno.agent import Agent

from cs_copilot.storage import S3
from cs_copilot.tools.chemistry.activity_schema import build_compound_memory_preview
from cs_copilot.tools.chemistry.clean_dataset import prepare_clean_dataset
from cs_copilot.tools.chemistry.standardize import standardize_smiles_column
from cs_copilot.tools.io.session_memory import (
    register_session_object,
    update_state_targets,
)

from .base import BaseDatabaseToolkit, DatabaseError, NotFound, RateLimited, ValidationError
from .chembl_fetcher import ChemblDataFetcher, RestChemblFetcher, SqlChemblFetcher
from .types import DBConfig, PaginationMode, QueryParams, ResultPage

# Any run of hyphens or whitespace counts as a single token separator.
_HYPHEN_OR_SPACE_RUN = re.compile(r"[-\s]+")

# Letter→digit boundary (e.g. "CDK2" → "CDK|2", "PDE4A" → "PDE|4A").
# Only splits where a letter is immediately followed by a digit; the
# digit and any trailing letters stay together as one unit.
_LETTER_DIGIT_BOUNDARY = re.compile(r"(?<=[a-zA-Z])(?=\d)")


def _build_punctuation_regex(keyword: str) -> str | None:
    """Build a regex that matches all hyphen/space variants of *keyword*.

    Returns ``None`` when the keyword cannot be split into multiple
    parts, signalling the caller to use plain substring matching.

    **Two splitting passes**:

    1. Split on explicit hyphens/spaces (``_HYPHEN_OR_SPACE_RUN``).
    2. Sub-split each resulting token at letter→digit boundaries
       (``_LETTER_DIGIT_BOUNDARY``), so ``"CDK2"`` → ``["CDK", "2"]``
       and ``"PDE4A"`` → ``["PDE", "4A"]``.

    This means single-token abbreviations like ``CDK2`` now produce a
    regex (``CDK[- ]?2``) that matches ``CDK2``, ``CDK-2``, and
    ``CDK 2``.

    **Symmetry guarantee**: inputs that tokenize identically across both
    passes produce the same regex.

    For *n* ≤ 3 final tokens the separator is ``[- ]?`` (optional),
    covering the concat form (e.g. ``"CDK2"`` from ``"CDK-2"``).
    For *n* ≥ 4 tokens the separator is ``[- ]`` (required).

    Uses ``[- ]`` (literal hyphen + space) rather than ``[-\\s]`` for
    maximum SQL dialect compatibility.

    Each token is ``re.escape``'d to handle metacharacters.
    """
    if not keyword:
        return None
    base = keyword.strip()
    if not base:
        return None

    # Pass 1: split on explicit hyphens/spaces.
    raw_tokens = [t for t in _HYPHEN_OR_SPACE_RUN.split(base) if t]
    if not raw_tokens:
        return None

    # Pass 2: sub-split each token at letter→digit boundaries.
    tokens: list[str] = []
    for t in raw_tokens:
        parts = _LETTER_DIGIT_BOUNDARY.split(t)
        tokens.extend(p for p in parts if p)

    if len(tokens) <= 1:
        return None
    escaped = [re.escape(t) for t in tokens]
    sep = "[- ]?" if len(tokens) <= 3 else "[- ]"
    return sep.join(escaped)


logger = logging.getLogger(__name__)


class ChemblToolkit(BaseDatabaseToolkit):
    """
    ChEMBL-specific database toolkit implementation.

    Supports multiple SQL backends (SQLite, PostgreSQL, MySQL) and the ChEMBL
    REST API.  The backend is selected automatically based on environment
    configuration.
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

    def __init__(
        self,
        config: Optional[DBConfig] = None,
        backend: str = "auto",
        **toolkit_kwargs,
    ):
        """
        Initialize ChEMBL toolkit.

        Args:
            config: Database configuration (optional, uses defaults if not provided)
            backend: Data source backend. One of:
                - "auto": Auto-detect from env vars (SQLite > PostgreSQL > MySQL > REST).
                  If an optional SQL driver is missing, fall back to the next candidate.
                - "rest": Force REST API (chembl_webresource_client)
                - "mysql": Force MySQL database
                - "postgresql": Force PostgreSQL database
                - "sqlite": Force SQLite database
            **toolkit_kwargs: Additional arguments for Agno Toolkit
        """
        if backend == "auto":
            resolved, fetcher = self._create_auto_fetcher()
        else:
            resolved = self._resolve_backend(backend)
            fetcher = self._create_fetcher(resolved)

        self._active_backend = resolved
        is_sql = resolved in ("mysql", "postgresql", "sqlite")

        if config is None:
            config = DBConfig(
                uri=self.BASE_URL,
                timeout_s=30.0,
                retries=3,
                page_size=100,
                rate_limit=10.0,
                supports_sql=is_sql,
                supports_http_api=True,
                pagination_mode=PaginationMode.OFFSET_LIMIT,
            )

        _BACKEND_LABELS = {
            "mysql": "Connected to a LOCAL MySQL ChEMBL database",
            "postgresql": "Connected to a LOCAL PostgreSQL ChEMBL database",
            "sqlite": "Connected to a LOCAL SQLite ChEMBL database",
        }
        if "instructions" not in toolkit_kwargs:
            if resolved in _BACKEND_LABELS:
                backend_label = (
                    f"Active backend: {_BACKEND_LABELS[resolved]}. "
                    "The ChEMBL REST API is also available as a fallback."
                )
            else:
                backend_label = "Active backend: Using the ChEMBL REST API."
            toolkit_kwargs["instructions"] = (
                "ChEMBL database toolkit for fetching compound bioactivity data, "
                "generating EDA reports, and analyzing chemical datasets. "
                f"{backend_label}"
            )

        super().__init__(
            config,
            name="chembl_toolkit",
            **toolkit_kwargs,
        )
        self._client = None  # Will be initialized lazily
        self._client_init_error = None  # Store initialization error if any
        self._fetcher = fetcher
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

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        """Resolve 'auto' to a concrete backend name."""
        import os

        if backend == "auto":
            if os.getenv("CHEMBL_SQLITE_PATH"):
                return "sqlite"
            if os.getenv("CHEMBL_PG_HOST"):
                return "postgresql"
            if os.getenv("CHEMBL_MYSQL_HOST"):
                return "mysql"
            return "rest"
        valid = ("rest", "mysql", "postgresql", "sqlite")
        if backend in valid:
            return backend
        raise ValueError(
            f"Unknown ChEMBL backend: {backend!r}. Use one of: {', '.join(valid)}, or 'auto'."
        )

    @staticmethod
    def _auto_backend_candidates() -> list[str]:
        """Return backend candidates for auto-detection in priority order."""
        import os

        candidates: list[str] = []
        if os.getenv("CHEMBL_SQLITE_PATH"):
            candidates.append("sqlite")
        if os.getenv("CHEMBL_PG_HOST"):
            candidates.append("postgresql")
        if os.getenv("CHEMBL_MYSQL_HOST"):
            candidates.append("mysql")
        candidates.append("rest")
        return candidates

    def _create_auto_fetcher(self) -> tuple[str, ChemblDataFetcher]:
        """Create the first usable backend for auto mode.

        Optional SQL backends are tried in priority order. Missing database
        driver packages do not abort startup; they trigger a fallback to the
        next candidate, ending with the REST API.
        """
        for candidate in self._auto_backend_candidates():
            try:
                return candidate, self._create_fetcher(candidate)
            except ImportError as exc:
                logger.warning(
                    "ChEMBL backend %s unavailable during auto-detection (%s). "
                    "Trying the next backend.",
                    candidate,
                    exc,
                )

        # The REST fetcher is expected to be always constructible, so reaching
        # this point would indicate an internal bug rather than configuration.
        raise RuntimeError("Failed to create any ChEMBL backend during auto-detection.")

    def _create_fetcher(self, backend: str) -> ChemblDataFetcher:
        """Create the data fetcher for an already-resolved backend."""
        if backend == "mysql":
            return SqlChemblFetcher.from_mysql_env()
        if backend == "postgresql":
            return SqlChemblFetcher.from_postgres_env()
        if backend == "sqlite":
            return SqlChemblFetcher.from_sqlite()
        return RestChemblFetcher(self._ensure_client)

    def _get_resource_client(self, resource: str):
        client = self._ensure_client()
        try:
            return getattr(client, resource)
        except AttributeError as exc:
            raise ValidationError(f"Unsupported ChEMBL resource: {resource}") from exc

    def connect(self) -> None:
        """Connect to ChEMBL data source (validate access)."""
        try:
            self._fetcher.connect()
            self._connected = True
            logger.info("Connected to ChEMBL successfully")
        except Exception as e:
            self._connected = False
            logger.error(f"Failed to connect to ChEMBL: {e}")
            raise DatabaseError(f"ChEMBL connection failed: {e}") from e

    def ping(self) -> bool:
        """Test ChEMBL data source connectivity."""
        try:
            return self._fetcher.ping()
        except Exception as e:
            logger.warning(f"ChEMBL ping failed: {e}")
            return False

    def get_capabilities(self) -> Dict[str, Any]:
        """Get information about database capabilities including the active backend."""
        caps = super().get_capabilities()
        caps["active_backend"] = self._active_backend
        return caps

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
        organism: Optional[str] = None,
        assay_types: Optional[Sequence[str]] = None,
        mechanism: Optional[str] = None,
        agent: Optional[Agent] = None,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Fetch compound bioactivity data from ChEMBL database using multiple keywords.

        Args:
            query: Search term(s) for assay descriptions. Can be:
                - A single string (e.g., "kinase")
                - A comma-separated string (e.g., "kinase, BRAF, PPAR-alpha")
                - Multiple queries will be processed separately and merged
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
            session_state: Shared session state auto-injected by Agno.

        Returns:
            Status message with information about fetched data

        Raises:
            ValueError: If query is empty or invalid
            DatabaseError: If ChEMBL API request fails
        """
        if not query or not isinstance(query, str):
            raise ValueError("query must be a non-empty string")

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
        raw_keywords = [kw.strip() for kw in query.split(",") if kw.strip()]
        if not raw_keywords:
            raise ValueError("query must contain at least one non-empty keyword")

        keywords: list[str] = raw_keywords

        logger.info(f"Fetching ChEMBL compounds for {len(keywords)} keyword(s): {keywords}")

        all_dataframes = []
        all_assay_ids = set()  # Track unique assays across all keywords

        try:
            # Process each keyword separately
            for keyword in keywords:
                logger.info(f"Processing keyword: '{keyword}'")

                # Step 1: Fetch assays for this keyword
                regex = _build_punctuation_regex(keyword)
                if regex:
                    logger.debug(f"Using regex for '{keyword}': {regex}")

                assays = self._fetcher.fetch_assays(keyword, organism, regex_pattern=regex)

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

                activities = self._fetcher.fetch_activities(
                    assay_ids, assay_type_codes, self.ACTIVITY_FIELDS
                )

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
                df["query_keywords"] = keyword
                all_dataframes.append(df)

            # Step 4: Merge all DataFrames and remove duplicates
            if not all_dataframes:
                return f"No data found for any of the keywords: {keywords}"

            logger.info(f"Merging {len(all_dataframes)} datasets and removing duplicates")
            merged_df = pd.concat(all_dataframes, ignore_index=True)

            # Determine dedup key
            if "activity_id" in merged_df.columns:
                dedup_key = ["activity_id"]
            else:
                dedup_key = ["molecule_chembl_id", "assay_chembl_id"]
                if "standard_value" in merged_df.columns:
                    dedup_key.append("standard_value")

            # Aggregate query_keywords for duplicate rows (multi-keyword only)
            if len(keywords) > 1:
                keywords_agg = (
                    merged_df.groupby(dedup_key, sort=False)["query_keywords"]
                    .apply(lambda s: "|".join(sorted(s.unique())))
                    .reset_index()
                )
                merged_df = merged_df.drop(columns=["query_keywords"]).merge(
                    keywords_agg, on=dedup_key, how="left"
                )

            # Remove duplicates
            initial_count = len(merged_df)
            merged_df = merged_df.drop_duplicates(subset=dedup_key, keep="first")

            duplicates_removed = initial_count - len(merged_df)
            logger.info(
                f"Removed {duplicates_removed} duplicate records. Final dataset: {len(merged_df)} records"
            )

            # Step 5: Prepare raw, clean, descriptor, and report artifacts.
            query_slug = "_".join(
                [kw.replace(" ", "_") for kw in keywords[:3]]
            )  # Limit to first 3 keywords for filename
            if len(keywords) > 3:
                query_slug += "_and_more"
            prepared = prepare_clean_dataset(
                merged_df,
                source_name=f"chembl_{query_slug}",
                smiles_column="smi",
                raw_filename=f"chembl_{query_slug}_raw.csv",
                clean_filename=f"chembl_{query_slug}_clean.csv",
                descriptor_filename=f"chembl_{query_slug}_descriptors.parquet",
                report_filename=f"chembl_{query_slug}_standardization_report.md",
                session_state=session_state or getattr(agent, "session_state", None),
            )
            total_assays = len(all_assay_ids)
            clean_df = prepared.clean_df
            activity_mapping = prepared.activity_mapping.to_dict()
            final_activity_mapping = prepared.final_activity_mapping.to_dict()

            # Store dataset path and compact memory objects for cross-agent access.
            for state in update_state_targets(agent, session_state):
                if "data_file_paths" not in state:
                    state["data_file_paths"] = {}
                state["data_file_paths"]["raw_dataset_path"] = prepared.raw_dataset_path
                state["data_file_paths"]["clean_dataset_path"] = prepared.clean_dataset_path
                state["data_file_paths"][
                    "descriptor_parquet_path"
                ] = prepared.descriptor_parquet_path
                state["data_file_paths"][
                    "standardization_report_path"
                ] = prepared.standardization_report_path
                state["data_file_paths"]["dataset_path"] = prepared.clean_dataset_path
                dataset_id = register_session_object(
                    state,
                    "dataset",
                    {
                        "dataset_path": prepared.clean_dataset_path,
                        "raw_dataset_path": prepared.raw_dataset_path,
                        "clean_dataset_path": prepared.clean_dataset_path,
                        "descriptor_parquet_path": prepared.descriptor_parquet_path,
                        "standardization_report_path": prepared.standardization_report_path,
                        "query_keywords": keywords,
                        "row_count": int(len(clean_df)),
                        "raw_row_count": int(len(merged_df)),
                        "unique_compounds": int(clean_df["smiles"].nunique()),
                        "assay_count": total_assays,
                        "organism_filter": organism,
                        "assay_type_codes": assay_type_codes,
                        "mechanism_filter": mechanism,
                        "activity_mapping": activity_mapping,
                        "final_activity_mapping": final_activity_mapping,
                        "activity_merge_policy": prepared.standardization_summary[
                            "activity_merge_policy"
                        ],
                        "stereochemistry_removed": prepared.standardization_summary[
                            "stereochemistry_removed"
                        ],
                        "standardization_summary": prepared.standardization_summary,
                    },
                    label=f"ChEMBL dataset: {', '.join(keywords[:3])}",
                    source_agent=getattr(agent, "name", None),
                    source_tool="fetch_compounds",
                    set_current=True,
                )
                for idx, compound in enumerate(
                    build_compound_memory_preview(clean_df, prepared.final_activity_mapping),
                    start=1,
                ):
                    register_session_object(
                        state,
                        "compound",
                        {**compound, "related": {"dataset_id": dataset_id}},
                        label=f"ChEMBL compound {idx}",
                        source_agent=getattr(agent, "name", None),
                        source_tool="fetch_compounds",
                        set_current=False,
                    )
                logger.info(
                    "Stored clean dataset_path in session_state: %s",
                    prepared.clean_dataset_path,
                )

            return self._format_success_message(
                clean_df,
                keywords,
                prepared.clean_dataset_path,
                total_assays,
                duplicates_removed,
                organism_filter=organism,
                assay_type_codes=assay_type_codes,
                mechanism_filter=mechanism,
                raw_dataset_path=prepared.raw_dataset_path,
                descriptor_parquet_path=prepared.descriptor_parquet_path,
                standardization_report_path=prepared.standardization_report_path,
                standardization_summary=prepared.standardization_summary,
            )

        except Exception as e:
            logger.error(f"Error fetching ChEMBL compounds: {e}")
            raise

    @staticmethod
    def _memory_compound_preview(df: pd.DataFrame, limit: int = 50) -> list[Dict[str, Any]]:
        """Return compact activity-bearing compound records for session memory."""
        return build_compound_memory_preview(df, limit=limit)

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
        """Save ChEMBL data and return the resolved storage path."""
        filename = f"chembl_{query.replace(' ', '_')}.csv"
        saved_path = S3.path(filename)

        try:
            with S3.open(filename, "w") as f:
                df.to_csv(f, index=False)
            logger.info(f"Saved ChEMBL data to {saved_path}")
            return saved_path
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
        raw_dataset_path: Optional[str] = None,
        descriptor_parquet_path: Optional[str] = None,
        standardization_report_path: Optional[str] = None,
        standardization_summary: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format success message for ChEMBL data fetch."""
        sample_row = df.head(1).to_string(index=False) if not df.empty else "No data"
        columns_preview = ", ".join(df.columns[:6].tolist()) + (
            "..." if len(df.columns) > 6 else ""
        )

        keywords_str = ", ".join([f"'{kw}'" for kw in keywords])
        save_label = "Saved to S3" if filename.startswith("s3://") else "Saved locally"
        message = (
            f"✅ Fetched and cleaned {len(df)} compound records from "
            f"{len(keywords)} keyword(s): {keywords_str}\n"
            f"📄 Clean dataset ({save_label}): `{filename}`\n"
        )
        if raw_dataset_path:
            message += f"📄 Raw dataset: `{raw_dataset_path}`\n"
        if descriptor_parquet_path:
            message += f"🧮 Descriptor Parquet: `{descriptor_parquet_path}`\n"
        if standardization_report_path:
            message += f"🧾 Standardization report: `{standardization_report_path}`\n"

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
            message += f"🔄 Removed {duplicates_removed} duplicate raw activity records\n"

        if standardization_summary:
            message += (
                "🧹 Standardization: "
                f"{standardization_summary.get('raw_rows')} raw rows → "
                f"{standardization_summary.get('rows_after_standardization')} valid rows → "
                f"{standardization_summary.get('clean_rows')} clean compounds; "
                f"{standardization_summary.get('invalid_smiles_rows')} invalid rows; "
                f"{standardization_summary.get('duplicate_rows_after_standardization')} "
                "post-standardization duplicate rows\n"
            )

        message += (
            f"📊 Columns: {columns_preview}\n"
            f"🔍 Sample row:\n{sample_row}\n\n"
            f"💡 Downstream agents will use the clean dataset path: `{filename}`"
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
        Generate multiple semantic keyword variations from a target description
        for ChEMBL assay description searches.

        The returned instruction directs the LLM to focus on **semantic**
        variants (abbreviations, synonyms, Greek letter replacement).
        Punctuation and hyphenation variants (``"epidermal growth factor
        receptor"`` vs ``"epidermal-growth-factor receptor"`` vs
        ``"epidermal growth factor-receptor"``) are automatically matched
        downstream by ``fetch_compounds`` via a regex built by
        :func:`_build_punctuation_regex` — the LLM should NOT spend
        its keyword budget on them.

        Args:
            natural_prompt: Target description (should already have generic
                terms removed in previous steps)

        Returns:
            Formatted instruction for generating ChEMBL search keywords
        """
        if not natural_prompt or not isinstance(natural_prompt, str):
            raise ValueError("natural_prompt must be a non-empty string")

        return (
            f"You are preparing queries for ChEMBL's `assay_description__icontains` filter.\n"
            f"Given this target description: '{natural_prompt.strip()}', generate multiple "
            f"SEMANTIC keyword variations (typically 2-4). Focus on:\n"
            f"  - Gene symbols / abbreviations (e.g., 'epidermal growth factor receptor' → 'egfr', "
            f"'phosphodiesterase 4A' → 'pde4a', 'B-Raf proto-oncogene' → 'braf', "
            f"'Janus kinase 2' → 'jak2').\n"
            f"  - Common full-name variants (e.g., 'phosphodiesterase 4A', "
            f"'epidermal growth factor receptor').\n"
            f"  - Literature synonyms (e.g., 'ERBB1' for EGFR; 'PRKCA' for "
            f"'protein kinase C alpha').\n"
            f"  - Greek character replacement (e.g., 'α' → 'alpha', 'β' → 'beta').\n"
            f"DO NOT generate punctuation / spacing variants yourself — the downstream "
            f"`fetch_compounds` tool automatically matches all hyphen/space combinations via "
            f"regex (e.g., 'epidermal growth factor receptor' automatically matches "
            f"'epidermal-growth factor receptor', 'epidermal growth-factor-receptor', etc.).\n"
            f"Output a comma-separated list of keyword phrases suitable for assay description "
            f"searches.\n"
            f"Example: For 'phosphodiesterase 4A', generate: 'pde4a, phosphodiesterase 4A'.\n"
            f"Example: For 'epidermal growth factor receptor', generate: "
            f"'egfr, epidermal growth factor receptor, erbb1'."
        )
