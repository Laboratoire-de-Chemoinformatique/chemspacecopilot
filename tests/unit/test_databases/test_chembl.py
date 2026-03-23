#!/usr/bin/env python
# coding: utf-8
"""
Tests for the ChEMBL database toolkit.
"""

from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from cs_copilot.tools.databases.base import DatabaseError, NotFound, RateLimited, ValidationError
from cs_copilot.tools.databases.chembl import ChemblToolkit
from cs_copilot.tools.databases.types import DBConfig, PaginationMode, QueryParams, ResultPage


class TestChemblToolkit:
    def test_init_default_config(self):
        """Test ChemblToolkit initialization with default config."""
        toolkit = ChemblToolkit()

        assert toolkit.config.uri == "https://www.ebi.ac.uk/chembl/api/data"
        assert toolkit.config.supports_http_api
        assert toolkit.config.rate_limit == 10.0

    def test_init_custom_config(self):
        """Test ChemblToolkit initialization with custom config."""
        config = DBConfig(uri="https://custom.chembl.api/data", timeout_s=60.0, page_size=50)
        toolkit = ChemblToolkit(config)

        assert toolkit.config.uri == "https://custom.chembl.api/data"
        assert toolkit.config.timeout_s == 60.0
        assert toolkit.config.page_size == 50
        assert toolkit.name == "chembl_toolkit"  # Check toolkit name

    @patch.object(ChemblToolkit, "_ensure_client")
    def test_connect_success(self, mock_ensure_client):
        """Test successful connection to ChEMBL API."""
        # Mock successful API call
        mock_client = Mock()
        mock_target = Mock()
        mock_target.__iter__ = Mock(return_value=iter([{"target_chembl_id": "CHEMBL1"}]))
        # Support slicing [:1]
        mock_target.__getitem__ = Mock(return_value=iter([{"target_chembl_id": "CHEMBL1"}]))
        mock_client.target.filter.return_value.only.return_value = mock_target
        mock_ensure_client.return_value = mock_client

        toolkit = ChemblToolkit()
        toolkit.connect()

        assert toolkit._connected
        mock_client.target.filter.assert_called_once()

    @patch.object(ChemblToolkit, "_ensure_client")
    def test_connect_failure(self, mock_ensure_client):
        """Test connection failure to ChEMBL API."""
        # Mock API failure
        mock_client = Mock()
        mock_client.target.filter.side_effect = Exception("API Error")
        mock_ensure_client.return_value = mock_client

        toolkit = ChemblToolkit()

        with pytest.raises(DatabaseError, match="ChEMBL connection failed"):
            toolkit.connect()

        assert not toolkit._connected

    @patch.object(ChemblToolkit, "_ensure_client")
    def test_ping_success(self, mock_ensure_client):
        """Test successful ping."""
        mock_client = Mock()
        mock_client.status = "OK"
        mock_ensure_client.return_value = mock_client

        toolkit = ChemblToolkit()
        result = toolkit.ping()

        assert result

    @patch.object(ChemblToolkit, "_ensure_client")
    def test_ping_failure(self, mock_ensure_client):
        """Test ping failure."""
        mock_client = Mock()
        mock_client.status = None
        mock_ensure_client.return_value = mock_client

        toolkit = ChemblToolkit()
        result = toolkit.ping()

        assert not result

    @patch.object(ChemblToolkit, "_ensure_client")
    def test_query_activity(self, mock_ensure_client):
        """Test querying activity resource."""
        # Mock ChEMBL API response
        mock_client = Mock()
        mock_activities = [
            {"activity_id": 1, "molecule_chembl_id": "CHEMBL1", "standard_value": 10.0},
            {"activity_id": 2, "molecule_chembl_id": "CHEMBL2", "standard_value": 20.0},
        ]

        # Mock the query chain: filter() -> only() -> skip() -> [slice]
        mock_query_after_skip = Mock()
        mock_query_after_skip.__getitem__ = Mock(return_value=mock_activities)
        mock_query_after_only = Mock()
        mock_query_after_only.skip = Mock(return_value=mock_query_after_skip)
        mock_query_after_filter = Mock()
        mock_query_after_filter.only = Mock(return_value=mock_query_after_only)
        mock_client.activity.filter = Mock(return_value=mock_query_after_filter)
        mock_ensure_client.return_value = mock_client

        toolkit = ChemblToolkit()

        params = QueryParams(
            filters={"assay_chembl_id__in": ["CHEMBL1", "CHEMBL2"]},
            fields=["activity_id", "molecule_chembl_id"],
            limit=100,
            offset=50,
            extra_params={"resource": "activity"},
        )

        result = toolkit.query(params)

        assert isinstance(result, ResultPage)
        assert len(result.records) == 2
        assert result.records[0]["activity_id"] == 1
        assert result.query_time_ms is not None
        # Verify skip was called with offset
        mock_query_after_only.skip.assert_called_once_with(50)

    @patch.object(ChemblToolkit, "_ensure_client")
    def test_query_default_resource(self, mock_ensure_client):
        """Test querying with default resource."""
        # Mock ChEMBL API response
        mock_client = Mock()
        mock_activities = [{"activity_id": 1, "molecule_chembl_id": "CHEMBL1"}]

        # Mock the query chain: filter() -> [slice] (no only() call since fields=None, no skip since offset=0)
        # Create a class that supports slicing to properly mock the ChEMBL query object
        class SliceableMock:
            def __init__(self, data):
                self.data = data

            def __getitem__(self, key):
                return self.data

        # Since params.fields is None, .only() won't be called, so filter() must return a sliceable mock
        mock_query_after_filter = SliceableMock(mock_activities)
        mock_client.activity.filter = Mock(return_value=mock_query_after_filter)
        mock_ensure_client.return_value = mock_client

        toolkit = ChemblToolkit()

        params = QueryParams(filters={"description__icontains": "kinase"}, limit=50)

        result = toolkit.query(params)

        assert isinstance(result, ResultPage)
        assert len(result.records) == 1
        # Default resource is "activity"
        mock_client.activity.filter.assert_called_once()

    @patch.object(ChemblToolkit, "_ensure_client")
    def test_query_molecule(self, mock_ensure_client):
        """Test querying molecule resource."""
        mock_client = Mock()
        mock_molecules = [{"molecule_chembl_id": "CHEMBL1", "canonical_smiles": "CCO"}]

        # Mock the query chain: filter() -> only() -> [slice] (no skip since offset=0)
        mock_query_after_only = Mock()
        mock_query_after_only.__getitem__ = Mock(return_value=mock_molecules)
        mock_query_after_filter = Mock()
        mock_query_after_filter.only = Mock(return_value=mock_query_after_only)
        mock_client.molecule.filter = Mock(return_value=mock_query_after_filter)
        mock_ensure_client.return_value = mock_client

        toolkit = ChemblToolkit()

        params = QueryParams(
            filters={"molecule_chembl_id__in": ["CHEMBL1"]},
            fields=toolkit.MOLECULE_FIELDS,
            limit=10,
            offset=0,
            extra_params={"resource": "molecule"},
        )

        result = toolkit.query(params)

        assert isinstance(result, ResultPage)
        assert len(result.records) == 1
        assert result.records[0]["molecule_chembl_id"] == "CHEMBL1"

    @patch.object(ChemblToolkit, "_ensure_client")
    def test_query_invalid_resource(self, mock_ensure_client):
        """Test querying with invalid resource."""
        mock_client = Mock()
        # Make getattr raise AttributeError for invalid resource
        # Use spec_set to prevent attribute access
        mock_client = Mock(spec=["activity", "molecule", "assay", "target"])
        mock_ensure_client.return_value = mock_client

        toolkit = ChemblToolkit()

        params = QueryParams(
            filters={},
            fields=[],
            limit=10,
            offset=0,
            extra_params={"resource": "invalid_resource"},
        )

        with pytest.raises(ValidationError, match="Unsupported ChEMBL resource"):
            toolkit.query(params)

    def test_map_fields_basic(self):
        """Test basic field mapping."""
        toolkit = ChemblToolkit()

        record = {"activity_id": 1, "molecule_chembl_id": "CHEMBL1", "standard_value": 10.0}

        mapped = toolkit.map_fields(record)

        # Should return record as-is for basic case
        assert mapped == record

    def test_map_fields_nested_structures(self):
        """Test field mapping with nested molecule structures."""
        toolkit = ChemblToolkit()

        record = {
            "molecule_chembl_id": "CHEMBL1",
            "molecule_structures": {"canonical_smiles": "CCO", "molecular_formula": "C2H6O"},
        }

        mapped = toolkit.map_fields(record)

        # Should extract canonical_smiles from nested structure
        assert mapped["canonical_smiles"] == "CCO"
        assert "molecule_structures" in mapped

    def test_handle_error_timeout(self):
        """Test error handling for timeouts."""
        toolkit = ChemblToolkit()

        error = Exception("Request timed out")
        mapped = toolkit.handle_error(error)

        assert isinstance(mapped, RateLimited)
        assert "ChEMBL API timeout" in str(mapped)

    def test_handle_error_not_found(self):
        """Test error handling for not found."""
        toolkit = ChemblToolkit()

        error = Exception("404 Not Found")
        mapped = toolkit.handle_error(error)

        assert isinstance(mapped, NotFound)
        assert "ChEMBL resource not found" in str(mapped)

    def test_handle_error_rate_limit(self):
        """Test error handling for rate limits."""
        toolkit = ChemblToolkit()

        error = Exception("429 Too Many Requests")
        mapped = toolkit.handle_error(error)

        assert isinstance(mapped, RateLimited)
        assert "ChEMBL rate limit exceeded" in str(mapped)

    def test_handle_error_validation(self):
        """Test error handling for validation errors."""
        toolkit = ChemblToolkit()

        error = Exception("400 Bad Request - Invalid parameter")
        mapped = toolkit.handle_error(error)

        assert isinstance(mapped, ValidationError)
        assert "Invalid ChEMBL query" in str(mapped)

    def test_handle_error_generic(self):
        """Test error handling for generic errors."""
        toolkit = ChemblToolkit()

        error = Exception("Something unexpected happened")
        mapped = toolkit.handle_error(error)

        assert isinstance(mapped, DatabaseError)
        assert "ChEMBL API error" in str(mapped)

    @patch.object(ChemblToolkit, "_ensure_client")
    @patch("cs_copilot.tools.databases.chembl.S3")
    def test_fetch_compounds_success(self, mock_s3, mock_ensure_client):
        """Test successful compound fetching."""
        # Mock ChEMBL client
        mock_client = Mock()
        # Mock assay search
        mock_assays = [{"assay_chembl_id": "CHEMBL1"}, {"assay_chembl_id": "CHEMBL2"}]
        mock_client.assay.filter.return_value = mock_assays

        # Mock activity search
        mock_activities = [
            {
                "activity_id": 1,
                "assay_chembl_id": "CHEMBL1",
                "molecule_chembl_id": "CHEMBL1",
                "standard_value": 10.0,
                "canonical_smiles": "CCO",
            },
            {
                "activity_id": 2,
                "assay_chembl_id": "CHEMBL2",
                "molecule_chembl_id": "CHEMBL2",
                "standard_value": 20.0,
                "canonical_smiles": "CCC",
            },
        ]
        mock_client.activity.filter.return_value.only.return_value = mock_activities
        mock_ensure_client.return_value = mock_client

        # Mock S3 operations - use MagicMock for context manager support
        mock_file = MagicMock()
        mock_s3.open.return_value.__enter__.return_value = mock_file
        mock_s3.open.return_value.__exit__.return_value = False

        toolkit = ChemblToolkit()
        result = toolkit.fetch_compounds("kinase", max_records=100)

        assert "✅ Fetched" in result
        assert "kinase" in result
        assert "chembl_kinase.csv" in result
        assert "Assay types filtered: Binding, Functional" in result

        # Verify API calls
        mock_client.assay.filter.assert_called_with(description__icontains="kinase")
        mock_client.activity.filter.assert_called_with(
            assay_chembl_id__in=["CHEMBL1", "CHEMBL2"], assay_type__in=["B", "F"]
        )
        # Note: molecule.filter is not called when canonical_smiles is already in activities data

    @patch.object(ChemblToolkit, "_ensure_client")
    def test_fetch_compounds_no_assays(self, mock_ensure_client):
        """Test compound fetching when no assays found."""
        # Mock ChEMBL client
        mock_client = Mock()
        # Mock empty assay search
        mock_client.assay.filter.return_value = []
        mock_ensure_client.return_value = mock_client

        toolkit = ChemblToolkit()
        result = toolkit.fetch_compounds("nonexistent_target")

        assert "No data found for any of the keywords" in result
        assert "nonexistent_target" in result

    @patch.object(ChemblToolkit, "_ensure_client")
    @patch("cs_copilot.tools.databases.chembl.S3")
    def test_fetch_compounds_with_filters(self, mock_s3, mock_ensure_client):
        """Test compound fetching with organism and assay-type filters."""

        mock_client = Mock()
        mock_assays = [{"assay_chembl_id": "CHEMBL1"}, {"assay_chembl_id": "CHEMBL2"}]
        mock_client.assay.filter.return_value = mock_assays

        mock_activities = [
            {
                "activity_id": 1,
                "assay_chembl_id": "CHEMBL1",
                "molecule_chembl_id": "CHEMBL1",
                "standard_value": 10.0,
                "canonical_smiles": "CCO",
                "assay_type": "B",
            },
            {
                "activity_id": 2,
                "assay_chembl_id": "CHEMBL2",
                "molecule_chembl_id": "CHEMBL2",
                "standard_value": 20.0,
                "canonical_smiles": "CCC",
                "assay_type": "F",
            },
        ]
        mock_client.activity.filter.return_value.only.return_value = mock_activities
        mock_ensure_client.return_value = mock_client

        mock_file = MagicMock()
        mock_s3.open.return_value.__enter__.return_value = mock_file
        mock_s3.open.return_value.__exit__.return_value = False

        toolkit = ChemblToolkit()
        result = toolkit.fetch_compounds(
            "virus",
            organism="HIV-1",
            assay_types=["binding", "functional"],
            max_records=50,
        )

        assert "Assay types filtered: Binding, Functional" in result
        assert "Target organism filter: HIV-1" in result

        mock_client.assay.filter.assert_called_with(
            description__icontains="virus", target_organism__icontains="HIV-1"
        )
        mock_client.activity.filter.assert_called_with(
            assay_chembl_id__in=["CHEMBL1", "CHEMBL2"], assay_type__in=["B", "F"]
        )

    def test_fetch_compounds_invalid_query(self):
        """Test compound fetching with invalid query."""
        toolkit = ChemblToolkit()

        with pytest.raises(ValueError, match="query must be a non-empty string"):
            toolkit.fetch_compounds("")

    def test_fetch_compounds_invalid_max_records(self):
        """Test compound fetching with invalid max_records."""
        toolkit = ChemblToolkit()

        with pytest.raises(ValueError, match="max_records must be a positive integer"):
            toolkit.fetch_compounds("kinase", max_records=-1)

    def test_fetch_compounds_invalid_assay_type(self):
        """Test validation for unsupported assay type labels."""

        toolkit = ChemblToolkit()

        with pytest.raises(ValueError, match="Invalid assay type 'phenotypic'"):
            toolkit.fetch_compounds("kinase", assay_types=["binding", "phenotypic"])

    @patch.object(ChemblToolkit, "_ensure_client")
    @patch("cs_copilot.tools.databases.chembl.S3")
    def test_fetch_compounds_empty_dict_coercion(self, mock_s3, mock_ensure_client):
        """Test that empty dict {} is coerced to empty list [] for assay_types."""
        # Mock ChEMBL client
        mock_client = Mock()
        mock_assays = [{"assay_chembl_id": "CHEMBL1"}]
        mock_client.assay.filter.return_value = mock_assays

        mock_activities = [
            {
                "activity_id": 1,
                "assay_chembl_id": "CHEMBL1",
                "molecule_chembl_id": "CHEMBL1",
                "standard_value": 10.0,
                "canonical_smiles": "CCO",
            }
        ]
        mock_client.activity.filter.return_value.only.return_value = mock_activities
        mock_ensure_client.return_value = mock_client

        # Mock S3 operations
        mock_file = MagicMock()
        mock_s3.open.return_value.__enter__.return_value = mock_file
        mock_s3.open.return_value.__exit__.return_value = False

        toolkit = ChemblToolkit()
        # This should not raise an error - empty dict should be coerced to empty list
        result = toolkit.fetch_compounds("kinase", assay_types={})

        # Verify the empty dict was treated as "all assay types"
        # (since empty list means "don't filter by assay type")
        assert isinstance(result, str)
        assert "kinase" in result.lower() or "success" in result.lower()

    def test_fetch_compounds_non_empty_dict_raises(self):
        """Test that non-empty dict for assay_types raises clear error."""
        toolkit = ChemblToolkit()

        with pytest.raises(ValueError, match="assay_types must be a list/sequence, not a dict"):
            toolkit.fetch_compounds("kinase", assay_types={"B": True, "F": False})

    @patch.object(ChemblToolkit, "_ensure_client")
    @patch("cs_copilot.tools.databases.chembl.S3")
    def test_fetch_compounds_multi_keyword_tracks_keywords(self, mock_s3, mock_ensure_client):
        """Test that query_keywords column tracks which keywords retrieved each row."""
        mock_client = Mock()

        # Two keywords: "cdk2" and "kinase"
        # activity_id=1 appears for both keywords (overlap)
        # activity_id=2 only from "cdk2", activity_id=3 only from "kinase"
        assays_cdk2 = [{"assay_chembl_id": "CHEMBL_A1"}]
        assays_kinase = [{"assay_chembl_id": "CHEMBL_A2"}]

        activities_cdk2 = [
            {
                "activity_id": 1,
                "assay_chembl_id": "CHEMBL_A1",
                "molecule_chembl_id": "MOL1",
                "standard_value": 10.0,
                "canonical_smiles": "CCO",
            },
            {
                "activity_id": 2,
                "assay_chembl_id": "CHEMBL_A1",
                "molecule_chembl_id": "MOL2",
                "standard_value": 20.0,
                "canonical_smiles": "CCC",
            },
        ]
        activities_kinase = [
            {
                "activity_id": 1,
                "assay_chembl_id": "CHEMBL_A2",
                "molecule_chembl_id": "MOL1",
                "standard_value": 10.0,
                "canonical_smiles": "CCO",
            },
            {
                "activity_id": 3,
                "assay_chembl_id": "CHEMBL_A2",
                "molecule_chembl_id": "MOL3",
                "standard_value": 30.0,
                "canonical_smiles": "CCCO",
            },
        ]

        # Return different assays per keyword call (use lists directly for list() compatibility)
        mock_client.assay.filter.side_effect = [assays_cdk2, assays_kinase]

        # Return different activities per keyword call
        act_filter_cdk2 = MagicMock()
        act_filter_cdk2.only.return_value = activities_cdk2
        act_filter_kinase = MagicMock()
        act_filter_kinase.only.return_value = activities_kinase

        mock_client.activity.filter.side_effect = [act_filter_cdk2, act_filter_kinase]
        mock_ensure_client.return_value = mock_client

        # Mock S3 operations
        mock_file = MagicMock()
        mock_s3.open.return_value.__enter__.return_value = mock_file
        mock_s3.open.return_value.__exit__.return_value = False

        toolkit = ChemblToolkit()
        # Patch _save_chembl_data to capture the DataFrame
        saved_dfs = []
        original_save = toolkit._save_chembl_data

        def capture_save(df, query):
            saved_dfs.append(df.copy())
            return original_save(df, query)

        toolkit._save_chembl_data = capture_save
        result = toolkit.fetch_compounds("cdk2, kinase")

        assert len(saved_dfs) == 1
        df = saved_dfs[0]
        assert "query_keywords" in df.columns

        # activity_id=1 appeared in both keywords
        row_overlap = df[df["activity_id"] == 1].iloc[0]
        assert row_overlap["query_keywords"] == "cdk2|kinase"

        # activity_id=2 only from "cdk2"
        row_cdk2 = df[df["activity_id"] == 2].iloc[0]
        assert row_cdk2["query_keywords"] == "cdk2"

        # activity_id=3 only from "kinase"
        row_kinase = df[df["activity_id"] == 3].iloc[0]
        assert row_kinase["query_keywords"] == "kinase"

    def test_convert_to_chembl_query_valid(self):
        """Test ChEMBL query conversion with valid input."""
        toolkit = ChemblToolkit()

        result = toolkit.convert_to_chembl_query("kinase 2 inhibitor activity")

        assert "ChEMBL's `assay_description__icontains` filter" in result
        assert "kinase 2 inhibitor activity" in result
        assert "kinase 2" in result  # Example output

    def test_convert_to_chembl_query_invalid(self):
        """Test ChEMBL query conversion with invalid input."""
        toolkit = ChemblToolkit()

        with pytest.raises(ValueError, match="natural_prompt must be a non-empty string"):
            toolkit.convert_to_chembl_query("")

        with pytest.raises(ValueError, match="natural_prompt must be a non-empty string"):
            toolkit.convert_to_chembl_query(None)

    @patch("cs_copilot.tools.databases.chembl.S3")
    def test_describe_dataset_success(self, mock_s3):
        """Test successful dataset description."""
        # Mock CSV file
        mock_csv_file = Mock()
        mock_s3.open.return_value.__enter__.return_value = mock_csv_file

        with patch("pandas.read_csv") as mock_read_csv:
            mock_df = pd.DataFrame(
                {"id": [1, 2, 3], "value": [10.0, 20.0, 30.0], "category": ["A", "B", "A"]}
            )
            mock_read_csv.return_value = mock_df

            toolkit = ChemblToolkit()
            result = toolkit.describe_dataset("test.csv")

            # Should contain statistical description
            assert isinstance(result, str)
            assert len(result) > 0

    @patch("cs_copilot.tools.databases.chembl.S3")
    def test_describe_dataset_empty_file(self, mock_s3):
        """Test dataset description with empty file."""
        mock_csv_file = Mock()
        mock_s3.open.return_value.__enter__.return_value = mock_csv_file

        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame()  # Empty DataFrame

            toolkit = ChemblToolkit()

            with pytest.raises(ValueError, match="Dataset at 'test.csv' is empty"):
                toolkit.describe_dataset("test.csv")

    def test_describe_dataset_invalid_path(self):
        """Test dataset description with invalid path."""
        toolkit = ChemblToolkit()

        with pytest.raises(ValueError, match="path_to_dataset cannot be empty"):
            toolkit.describe_dataset("")


class TestChemblConnectionManagement:
    """Test ChEMBL-specific connection management."""

    @patch.object(ChemblToolkit, "_ensure_client")
    def test_connect_validates_api_access(self, mock_ensure_client):
        """Test that connect() validates API access."""
        # Mock successful API validation
        mock_client = Mock()
        mock_target = Mock()
        mock_target.__iter__ = Mock(return_value=iter([{"target_chembl_id": "CHEMBL1"}]))
        # Support slicing [:1]
        mock_target.__getitem__ = Mock(return_value=iter([{"target_chembl_id": "CHEMBL1"}]))
        mock_client.target.filter.return_value.only.return_value = mock_target
        mock_ensure_client.return_value = mock_client

        toolkit = ChemblToolkit()
        toolkit.connect()

        assert toolkit._connected
        # Verify validation query was made
        mock_client.target.filter.assert_called_once()

    @patch.object(ChemblToolkit, "_ensure_client")
    def test_connect_handles_api_failure(self, mock_ensure_client):
        """Test that connect() handles API failures gracefully."""
        mock_client = Mock()
        mock_client.target.filter.side_effect = Exception("API unreachable")
        mock_ensure_client.return_value = mock_client

        toolkit = ChemblToolkit()

        with pytest.raises(DatabaseError, match="ChEMBL connection failed"):
            toolkit.connect()

        assert not toolkit._connected

    @patch.object(ChemblToolkit, "_ensure_client")
    def test_ping_checks_api_status(self, mock_ensure_client):
        """Test that ping() checks API status."""
        mock_client = Mock()
        mock_client.status = {"status": "OK"}
        mock_ensure_client.return_value = mock_client

        toolkit = ChemblToolkit()
        result = toolkit.ping()

        assert result

    @patch.object(ChemblToolkit, "_ensure_client")
    def test_ping_handles_api_unavailable(self, mock_ensure_client):
        """Test ping when API is unavailable."""
        mock_client = Mock()
        mock_client.status = None
        mock_ensure_client.return_value = mock_client

        toolkit = ChemblToolkit()
        result = toolkit.ping()

        assert not result

    @patch.object(ChemblToolkit, "_ensure_client")
    def test_reconnect_after_disconnect(self, mock_ensure_client):
        """Test reconnecting after disconnect."""
        # Mock successful connections
        mock_client = Mock()
        mock_target = Mock()
        mock_target.__iter__ = Mock(return_value=iter([{"target_chembl_id": "CHEMBL1"}]))
        # Support slicing [:1]
        mock_target.__getitem__ = Mock(return_value=iter([{"target_chembl_id": "CHEMBL1"}]))
        mock_client.target.filter.return_value.only.return_value = mock_target
        mock_ensure_client.return_value = mock_client

        toolkit = ChemblToolkit()

        # Initial connection
        toolkit.connect()
        assert toolkit._connected

        # Disconnect
        toolkit.close()
        assert not toolkit._connected

        # Reconnect
        toolkit.connect()
        assert toolkit._connected

        # Should have called connection twice
        assert mock_client.target.filter.call_count == 2

    @patch.object(ChemblToolkit, "_ensure_client")
    def test_context_manager_connection(self, mock_ensure_client):
        """Test using ChEMBL toolkit with context manager."""
        mock_client = Mock()
        mock_target = Mock()
        mock_target.__iter__ = Mock(return_value=iter([{"target_chembl_id": "CHEMBL1"}]))
        # Support slicing [:1]
        mock_target.__getitem__ = Mock(return_value=iter([{"target_chembl_id": "CHEMBL1"}]))
        mock_client.target.filter.return_value.only.return_value = mock_target
        mock_ensure_client.return_value = mock_client

        toolkit = ChemblToolkit()

        with toolkit:
            assert toolkit._connected

        assert not toolkit._connected

    def test_rate_limit_configuration(self):
        """Test that ChEMBL toolkit has correct rate limit configured."""
        toolkit = ChemblToolkit()

        assert toolkit.config.rate_limit == 10.0  # ChEMBL default

    def test_pagination_mode_configuration(self):
        """Test that ChEMBL toolkit uses correct pagination mode."""
        toolkit = ChemblToolkit()

        assert toolkit.config.pagination_mode == PaginationMode.OFFSET_LIMIT

    def test_custom_config_override(self):
        """Test that custom config can override defaults."""
        custom_config = DBConfig(
            uri="https://custom.api/chembl", timeout_s=120.0, rate_limit=5.0, page_size=50
        )

        toolkit = ChemblToolkit(custom_config)

        assert toolkit.config.uri == "https://custom.api/chembl"
        assert toolkit.config.timeout_s == 120.0
        assert toolkit.config.rate_limit == 5.0
        assert toolkit.config.page_size == 50


class TestChemblRateLimiting:
    """Test rate limiting behavior."""

    def test_rate_limit_error_detection(self):
        """Test that rate limit errors are properly detected."""
        toolkit = ChemblToolkit()

        rate_error = Exception("429 Too Many Requests")
        mapped = toolkit.handle_error(rate_error)

        assert isinstance(mapped, RateLimited)
        assert "rate limit" in str(mapped).lower()

    def test_timeout_error_detection(self):
        """Test that timeout errors are detected."""
        toolkit = ChemblToolkit()

        timeout_error = Exception("Request timed out after 30 seconds")
        mapped = toolkit.handle_error(timeout_error)

        assert isinstance(mapped, RateLimited)  # ChEMBL maps timeouts to RateLimited
        assert "timeout" in str(mapped).lower()

    @patch.object(ChemblToolkit, "_ensure_client")
    def test_query_execution_with_rate_limit_config(self, mock_ensure_client):
        """Test query execution respects rate limit configuration."""
        mock_client = Mock()
        mock_ensure_client.return_value = mock_client

        config = DBConfig(
            uri="https://www.ebi.ac.uk/chembl/api/data",
            rate_limit=5.0,  # 5 requests per second
            timeout_s=60.0,
        )
        toolkit = ChemblToolkit(config)

        assert toolkit.config.rate_limit == 5.0
        assert toolkit.config.timeout_s == 60.0


class TestChemblIntegration:
    """Integration tests that test the full toolkit functionality."""

    @pytest.mark.integration
    @patch.object(ChemblToolkit, "_ensure_client")
    @patch("cs_copilot.tools.databases.chembl.S3")
    def test_full_workflow(self, mock_s3, mock_ensure_client):
        """Test a complete workflow from query to dataset description."""
        # Mock the full ChEMBL API workflow
        mock_client = Mock()
        mock_assays = [{"assay_chembl_id": "CHEMBL1"}]
        mock_client.assay.filter.return_value = mock_assays

        mock_activities = [
            {
                "activity_id": 1,
                "assay_chembl_id": "CHEMBL1",
                "molecule_chembl_id": "CHEMBL1",
                "standard_value": 10.0,
                "canonical_smiles": "CCO",
            }
        ]
        mock_client.activity.filter.return_value.only.return_value = mock_activities
        mock_ensure_client.return_value = mock_client

        # Mock S3 operations - use MagicMock for context manager support
        mock_csv_file = MagicMock()
        mock_context1 = MagicMock()
        mock_context1.__enter__.return_value = mock_csv_file
        mock_context1.__exit__.return_value = False

        mock_context2 = MagicMock()
        mock_context2.__enter__.return_value = mock_csv_file
        mock_context2.__exit__.return_value = False

        mock_s3.open.side_effect = [mock_context1, mock_context2]

        # Mock pandas operations
        with patch("pandas.read_csv") as mock_read_csv:
            mock_df = pd.DataFrame(
                {
                    "activity_id": [1],
                    "molecule_chembl_id": ["CHEMBL1"],
                    "standard_value": [10.0],
                    "SMILES": ["CCO"],
                }
            )
            mock_read_csv.return_value = mock_df

            toolkit = ChemblToolkit()

            # Step 1: Fetch compounds
            fetch_result = toolkit.fetch_compounds("kinase", max_records=10)
            assert "✅ Fetched" in fetch_result

            # Step 2: Describe dataset
            desc_result = toolkit.describe_dataset("chembl_kinase.csv")
            assert isinstance(desc_result, str)
