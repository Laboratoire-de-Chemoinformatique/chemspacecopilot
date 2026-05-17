#!/usr/bin/env python
# coding: utf-8
"""
Tests for the ChEMBL database toolkit.
"""

import re
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from cs_copilot.tools.databases import chembl as chembl_module
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
        # Mock assay search - use plain list for proper list() iteration
        mock_assays = [{"assay_chembl_id": "CHEMBL1"}, {"assay_chembl_id": "CHEMBL2"}]
        mock_client.assay.filter.return_value = mock_assays

        # Mock activity search
        mock_activities = [
            {
                "activity_id": 1,
                "assay_chembl_id": "CHEMBL1",
                "molecule_chembl_id": "CHEMBL1",
                "standard_value": 10.0,
                "standard_units": "nM",
                "standard_type": "IC50",
                "canonical_smiles": "CCO",
            },
            {
                "activity_id": 2,
                "assay_chembl_id": "CHEMBL2",
                "molecule_chembl_id": "CHEMBL2",
                "standard_value": 20.0,
                "standard_units": "nM",
                "standard_type": "IC50",
                "canonical_smiles": "CCC",
            },
        ]
        mock_client.activity.filter.return_value.only.return_value = mock_activities
        mock_ensure_client.return_value = mock_client

        # Mock S3 operations - use MagicMock for context manager support
        mock_file = MagicMock()
        mock_s3.open.return_value.__enter__.return_value = mock_file
        mock_s3.open.return_value.__exit__.return_value = False
        mock_s3.path.side_effect = lambda rel: rel

        toolkit = ChemblToolkit()
        result = toolkit.fetch_compounds("kinase")

        assert "✅ Fetched" in result
        assert "kinase" in result
        assert "chembl_kinase_clean.csv" in result
        assert "chembl_kinase_raw.csv" in result
        assert "chembl_kinase_descriptors.parquet" in result
        assert "chembl_kinase_standardization_report.md" in result
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
        # Mock empty assay search - use plain list for proper list() iteration
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
        mock_s3.path.side_effect = lambda rel: rel

        toolkit = ChemblToolkit()
        result = toolkit.fetch_compounds(
            "virus",
            organism="HIV-1",
            assay_types=["binding", "functional"],
        )

        assert "Assay types filtered: Binding, Functional" in result
        assert "Target organism filter: HIV-1" in result

        mock_client.assay.filter.assert_called_with(
            description__icontains="virus", target_organism__icontains="HIV-1"
        )
        mock_client.activity.filter.assert_called_with(
            assay_chembl_id__in=["CHEMBL1", "CHEMBL2"], assay_type__in=["B", "F"]
        )

    @patch.object(ChemblToolkit, "_ensure_client")
    def test_fetch_compounds_uses_local_storage_by_default(
        self, mock_ensure_client, monkeypatch, tmp_path
    ):
        """Ambient AWS credentials should not switch ChEMBL dataset saves to S3."""
        from cs_copilot.storage import S3

        mock_client = Mock()
        mock_client.assay.filter.return_value = [{"assay_chembl_id": "CHEMBL1"}]
        mock_client.activity.filter.return_value.only.return_value = [
            {
                "activity_id": 1,
                "assay_chembl_id": "CHEMBL1",
                "molecule_chembl_id": "CHEMBL1",
                "standard_value": 10.0,
                "standard_units": "nM",
                "standard_type": "IC50",
                "canonical_smiles": "CCO",
            }
        ]
        mock_ensure_client.return_value = mock_client

        monkeypatch.chdir(tmp_path)
        for key in (
            "USE_S3",
            "S3_ENDPOINT_URL",
            "MINIO_ENDPOINT",
            "MINIO_ENDPOINT_URL",
            "MINIO_ACCESS_KEY",
            "MINIO_SECRET_KEY",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
        monkeypatch.setenv("ASSETS_BUCKET", "test-bucket")

        original_prefix = S3.prefix
        S3.prefix = "sessions/test-session"
        try:
            result = ChemblToolkit().fetch_compounds("kinase")
        finally:
            S3.prefix = original_prefix

        session_root = tmp_path / "data" / "sessions" / "test-session"
        expected_paths = {
            "clean": "workflows/*/01_chemical_space/datasets/clean/chembl_kinase_clean.csv",
            "raw": "workflows/*/01_chemical_space/datasets/raw/chembl_kinase_raw.csv",
            "descriptor": (
                "workflows/*/01_chemical_space/descriptors/" "chembl_kinase_descriptors.parquet"
            ),
            "report": (
                "workflows/*/01_chemical_space/standardization/"
                "chembl_kinase_standardization_report.md"
            ),
        }
        resolved_paths = {
            key: next(iter(session_root.glob(pattern))) for key, pattern in expected_paths.items()
        }
        assert all(path.exists() for path in resolved_paths.values())
        assert "Saved locally" in result
        assert str(resolved_paths["clean"].relative_to(tmp_path)) in result

    def test_fetch_compounds_invalid_query(self):
        """Test compound fetching with invalid query."""
        toolkit = ChemblToolkit()

        with pytest.raises(ValueError, match="query must be a non-empty string"):
            toolkit.fetch_compounds("")

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
        mock_s3.path.side_effect = lambda rel: rel

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

    def test_short_keyword_filter_rejects_protein_pref_name(self, monkeypatch):
        toolkit = ChemblToolkit()
        df = pd.DataFrame(
            [
                {
                    "activity_id": 1,
                    "assay_chembl_id": "CHEMBL_A1",
                    "query_keywords": "BRAF",
                    "target_pref_name": "Beta-secretase 1",
                    "target_type": "SINGLE PROTEIN",
                    "description": "BACE1 biochemical assay",
                    "smi": "CCO",
                },
                {
                    "activity_id": 2,
                    "assay_chembl_id": "CHEMBL_A2",
                    "query_keywords": "kinase",
                    "target_pref_name": "B-Raf proto-oncogene serine/threonine-protein kinase",
                    "target_type": "SINGLE PROTEIN",
                    "description": "BRAF biochemical assay",
                    "smi": "CCC",
                },
            ]
        )
        saved = []

        def fake_save(self, filtered_df, *, query_slug, session_state):
            saved.append(filtered_df.copy())
            return "filtered.csv"

        def fake_judge(self, judge_items, **kwargs):
            assert judge_items[0]["judge_basis"] == "target_pref_name"
            assert judge_items[0]["value"] == "Beta-secretase 1"
            return {
                item["item_id"]: chembl_module._ChemblJudgeDecision(
                    item_id=item["item_id"],
                    keep=False,
                    explanation="Unrelated protein target.",
                )
                for item in judge_items
            }

        def keep_metadata(self, judge_items, **kwargs):
            return {
                item["item_id"]: chembl_module._ChemblJudgeDecision(
                    item_id=item["item_id"],
                    keep=True,
                    explanation="Metadata matches.",
                )
                for item in judge_items
            }

        monkeypatch.setattr(ChemblToolkit, "_save_filtered_rows", fake_save)
        monkeypatch.setattr(ChemblToolkit, "_run_chembl_retrieval_judge", fake_judge)
        monkeypatch.setattr(ChemblToolkit, "_run_chembl_metadata_judge", keep_metadata)

        result = toolkit._filter_suspicious_short_keyword_rows(
            df,
            keywords=["BRAF", "kinase"],
            target_query="BRAF kinase",
            organism_filter=None,
            query_slug="braf_kinase",
            agent=Mock(model=object()),
            session_state={},
        )

        assert result.summary["suspicious_row_count"] == 1
        assert result.summary["filtered_row_count"] == 1
        assert result.filtered_rows_path == "filtered.csv"
        assert result.retained_df["activity_id"].tolist() == [2]
        assert saved[0]["filter_reason"].tolist() == ["llm_judge_rejected"]

    def test_short_keyword_filter_falls_back_to_description_when_pref_name_missing(
        self, monkeypatch
    ):
        toolkit = ChemblToolkit()
        df = pd.DataFrame(
            [
                {
                    "activity_id": 1,
                    "assay_chembl_id": "CHEMBL_A1",
                    "query_keywords": "CDK2",
                    "target_pref_name": None,
                    "target_type": "SINGLE PROTEIN",
                    "description": "Cyclin-dependent kinase 2 enzyme inhibition assay",
                    "smi": "CCO",
                }
            ]
        )

        def fake_judge(self, judge_items, **kwargs):
            assert judge_items[0]["judge_basis"] == "description"
            return {
                item["item_id"]: chembl_module._ChemblJudgeDecision(
                    item_id=item["item_id"],
                    keep=True,
                    explanation="Description names CDK2.",
                )
                for item in judge_items
            }

        monkeypatch.setattr(ChemblToolkit, "_run_chembl_retrieval_judge", fake_judge)

        result = toolkit._filter_suspicious_short_keyword_rows(
            df,
            keywords=["CDK2"],
            target_query="CDK2",
            organism_filter=None,
            query_slug="cdk2",
            agent=Mock(model=object()),
            session_state={},
        )

        assert result.summary["filtered_row_count"] == 0
        assert result.retained_df["activity_id"].tolist() == [1]

    def test_short_keyword_filter_uses_organism_basis(self, monkeypatch):
        toolkit = ChemblToolkit()
        df = pd.DataFrame(
            [
                {
                    "activity_id": 1,
                    "assay_chembl_id": "CHEMBL_A1",
                    "query_keywords": "HIV",
                    "target_type": "ORGANISM",
                    "target_organism": "Human immunodeficiency virus 1",
                    "assay_organism": "Homo sapiens",
                    "description": "HIV-1 antiviral assay",
                    "smi": "CCO",
                }
            ]
        )

        def fake_judge(self, judge_items, **kwargs):
            assert judge_items[0]["judge_scope"] == "organism"
            assert judge_items[0]["judge_basis"] == "target_organism"
            return {
                item["item_id"]: chembl_module._ChemblJudgeDecision(
                    item_id=item["item_id"],
                    keep=True,
                    explanation="Target organism matches HIV-1.",
                )
                for item in judge_items
            }

        def keep_metadata(self, judge_items, **kwargs):
            return {
                item["item_id"]: chembl_module._ChemblJudgeDecision(
                    item_id=item["item_id"],
                    keep=True,
                    explanation="Metadata matches.",
                )
                for item in judge_items
            }

        monkeypatch.setattr(ChemblToolkit, "_run_chembl_retrieval_judge", fake_judge)
        monkeypatch.setattr(ChemblToolkit, "_run_chembl_metadata_judge", keep_metadata)

        result = toolkit._filter_suspicious_short_keyword_rows(
            df,
            keywords=["HIV"],
            target_query="HIV",
            organism_filter="HIV-1",
            query_slug="hiv",
            agent=Mock(model=object()),
            session_state={},
        )

        assert result.summary["filtered_row_count"] == 0
        assert result.retained_df["activity_id"].tolist() == [1]

    def test_metadata_filter_rejects_wrong_protein_pref_name(self, monkeypatch):
        toolkit = ChemblToolkit()
        df = pd.DataFrame(
            [
                {
                    "activity_id": 1,
                    "assay_chembl_id": "CHEMBL_A1",
                    "query_keywords": "BRAF kinase",
                    "target_pref_name": "Beta-secretase 1",
                    "target_type": "SINGLE PROTEIN",
                    "target_organism": None,
                    "description": "BACE1 biochemical assay",
                    "smi": "CCO",
                },
                {
                    "activity_id": 2,
                    "assay_chembl_id": "CHEMBL_A2",
                    "query_keywords": "BRAF kinase",
                    "target_pref_name": "B-Raf proto-oncogene serine/threonine-protein kinase",
                    "target_type": "SINGLE PROTEIN",
                    "target_organism": None,
                    "description": "BRAF biochemical assay",
                    "smi": "CCC",
                },
            ]
        )
        saved = []

        def fake_save(self, filtered_df, *, query_slug, session_state):
            saved.append(filtered_df.copy())
            return "filtered.csv"

        def fake_metadata_judge(self, judge_items, **kwargs):
            decisions = {}
            for item in judge_items:
                assert item["judge_basis"] == "target_pref_name"
                keep = item["target_pref_name"].startswith("B-Raf")
                decisions[item["item_id"]] = chembl_module._ChemblJudgeDecision(
                    item_id=item["item_id"],
                    keep=keep,
                    explanation="Protein target matches BRAF." if keep else "Wrong protein target.",
                )
            return decisions

        monkeypatch.setattr(ChemblToolkit, "_save_filtered_rows", fake_save)
        monkeypatch.setattr(ChemblToolkit, "_run_chembl_metadata_judge", fake_metadata_judge)

        result = toolkit._filter_suspicious_short_keyword_rows(
            df,
            keywords=["BRAF kinase"],
            target_query="BRAF kinase",
            organism_filter=None,
            query_slug="braf_kinase",
            agent=Mock(model=object()),
            session_state={},
        )

        assert result.summary["suspicious_row_count"] == 0
        assert result.summary["metadata_judge_status"] == "completed"
        assert result.summary["metadata_judge_row_count"] == 2
        assert result.summary["metadata_filtered_row_count"] == 1
        assert result.summary["filtered_row_count"] == 1
        assert result.retained_df["activity_id"].tolist() == [2]
        assert saved[0]["filter_reason"].tolist() == ["metadata_llm_judge_rejected"]

    def test_metadata_filter_keeps_matching_protein_pref_name(self, monkeypatch):
        toolkit = ChemblToolkit()
        df = pd.DataFrame(
            [
                {
                    "activity_id": 1,
                    "assay_chembl_id": "CHEMBL_A1",
                    "query_keywords": "CDK2 kinase",
                    "target_pref_name": "Cyclin-dependent kinase 2",
                    "target_type": "SINGLE PROTEIN",
                    "target_organism": None,
                    "description": "CDK2 biochemical assay",
                    "smi": "CCO",
                }
            ]
        )

        def fake_metadata_judge(self, judge_items, **kwargs):
            assert judge_items[0]["fields_to_validate"] == ["target_pref_name"]
            return {
                item["item_id"]: chembl_module._ChemblJudgeDecision(
                    item_id=item["item_id"],
                    keep=True,
                    explanation="Protein target matches CDK2.",
                )
                for item in judge_items
            }

        monkeypatch.setattr(ChemblToolkit, "_run_chembl_metadata_judge", fake_metadata_judge)

        result = toolkit._filter_suspicious_short_keyword_rows(
            df,
            keywords=["CDK2 kinase"],
            target_query="CDK2 kinase",
            organism_filter=None,
            query_slug="cdk2_kinase",
            agent=Mock(model=object()),
            session_state={},
        )

        assert result.summary["metadata_judge_status"] == "completed"
        assert result.summary["filtered_row_count"] == 0
        assert result.retained_df["activity_id"].tolist() == [1]

    def test_metadata_filter_keeps_rows_when_pref_name_missing(self, monkeypatch):
        toolkit = ChemblToolkit()
        df = pd.DataFrame(
            [
                {
                    "activity_id": 1,
                    "assay_chembl_id": "CHEMBL_A1",
                    "query_keywords": "CDK2 kinase",
                    "target_pref_name": None,
                    "target_type": "SINGLE PROTEIN",
                    "target_organism": None,
                    "description": "CDK2 biochemical assay",
                    "smi": "CCO",
                }
            ]
        )

        def unexpected_metadata_judge(self, judge_items, **kwargs):
            raise AssertionError("Rows with empty judged metadata should not be judged.")

        monkeypatch.setattr(
            ChemblToolkit,
            "_run_chembl_metadata_judge",
            unexpected_metadata_judge,
        )

        result = toolkit._filter_suspicious_short_keyword_rows(
            df,
            keywords=["CDK2 kinase"],
            target_query="CDK2 kinase",
            organism_filter=None,
            query_slug="cdk2_kinase",
            agent=Mock(model=object()),
            session_state={},
        )

        assert result.summary["metadata_judge_status"] == "not_needed"
        assert result.summary["metadata_judge_row_count"] == 0
        assert result.summary["filtered_row_count"] == 0
        assert result.retained_df["activity_id"].tolist() == [1]

    def test_metadata_filter_rejects_wrong_target_organism(self, monkeypatch):
        toolkit = ChemblToolkit()
        df = pd.DataFrame(
            [
                {
                    "activity_id": 1,
                    "assay_chembl_id": "CHEMBL_A1",
                    "query_keywords": "BRAF human",
                    "target_pref_name": None,
                    "target_type": "SINGLE PROTEIN",
                    "target_organism": "Mus musculus",
                    "description": "BRAF biochemical assay",
                    "smi": "CCO",
                }
            ]
        )
        saved = []

        def fake_save(self, filtered_df, *, query_slug, session_state):
            saved.append(filtered_df.copy())
            return "filtered.csv"

        def fake_metadata_judge(self, judge_items, **kwargs):
            assert judge_items[0]["fields_to_validate"] == ["target_organism"]
            assert judge_items[0]["target_organism"] == "Mus musculus"
            return {
                item["item_id"]: chembl_module._ChemblJudgeDecision(
                    item_id=item["item_id"],
                    keep=False,
                    explanation="Requested human target, but target organism is mouse.",
                )
                for item in judge_items
            }

        monkeypatch.setattr(ChemblToolkit, "_save_filtered_rows", fake_save)
        monkeypatch.setattr(ChemblToolkit, "_run_chembl_metadata_judge", fake_metadata_judge)

        result = toolkit._filter_suspicious_short_keyword_rows(
            df,
            keywords=["BRAF human"],
            target_query="BRAF human",
            organism_filter="Homo sapiens",
            query_slug="braf_human",
            agent=Mock(model=object()),
            session_state={},
        )

        assert result.summary["metadata_filtered_row_count"] == 1
        assert result.retained_df.empty
        assert saved[0]["judge_basis"].tolist() == ["target_organism"]

    def test_metadata_filter_keeps_rows_when_target_organism_missing(self, monkeypatch):
        toolkit = ChemblToolkit()
        df = pd.DataFrame(
            [
                {
                    "activity_id": 1,
                    "assay_chembl_id": "CHEMBL_A1",
                    "query_keywords": "HIV antiviral",
                    "target_pref_name": None,
                    "target_type": "ORGANISM",
                    "target_organism": "",
                    "description": "Antiviral assay",
                    "smi": "CCO",
                }
            ]
        )

        def unexpected_metadata_judge(self, judge_items, **kwargs):
            raise AssertionError("Rows with empty judged metadata should not be judged.")

        monkeypatch.setattr(
            ChemblToolkit,
            "_run_chembl_metadata_judge",
            unexpected_metadata_judge,
        )

        result = toolkit._filter_suspicious_short_keyword_rows(
            df,
            keywords=["HIV antiviral"],
            target_query="HIV antiviral",
            organism_filter="HIV-1",
            query_slug="hiv_antiviral",
            agent=Mock(model=object()),
            session_state={},
        )

        assert result.summary["metadata_judge_status"] == "not_needed"
        assert result.summary["metadata_judge_row_count"] == 0
        assert result.summary["filtered_row_count"] == 0
        assert result.retained_df["activity_id"].tolist() == [1]

    def test_metadata_filter_unavailable_judge_keeps_rows(self):
        toolkit = ChemblToolkit()
        df = pd.DataFrame(
            [
                {
                    "activity_id": 1,
                    "assay_chembl_id": "CHEMBL_A1",
                    "query_keywords": "CDK2 kinase",
                    "target_pref_name": "Cyclin-dependent kinase 2",
                    "target_type": "SINGLE PROTEIN",
                    "target_organism": None,
                    "description": "CDK2 biochemical assay",
                    "smi": "CCO",
                }
            ]
        )

        result = toolkit._filter_suspicious_short_keyword_rows(
            df,
            keywords=["CDK2 kinase"],
            target_query="CDK2 kinase",
            organism_filter=None,
            query_slug="cdk2_kinase",
            agent=None,
            session_state={},
        )

        assert result.summary["metadata_judge_status"] == "unavailable"
        assert result.summary["metadata_judge_row_count"] == 1
        assert result.summary["filtered_row_count"] == 0
        assert result.retained_df["activity_id"].tolist() == [1]

    def test_retrieval_filter_combines_short_keyword_and_metadata_judges(
        self, monkeypatch
    ):
        toolkit = ChemblToolkit()
        df = pd.DataFrame(
            [
                {
                    "activity_id": 1,
                    "assay_chembl_id": "CHEMBL_A1",
                    "query_keywords": "BRAF",
                    "target_pref_name": "Beta-secretase 1",
                    "target_type": "SINGLE PROTEIN",
                    "target_organism": None,
                    "description": "BACE1 biochemical assay",
                    "smi": "CCO",
                },
                {
                    "activity_id": 2,
                    "assay_chembl_id": "CHEMBL_A2",
                    "query_keywords": "kinase",
                    "target_pref_name": "Tyrosine-protein kinase ABL1",
                    "target_type": "SINGLE PROTEIN",
                    "target_organism": None,
                    "description": "ABL1 biochemical assay",
                    "smi": "CCC",
                },
            ]
        )
        saved = []

        def fake_save(self, filtered_df, *, query_slug, session_state):
            saved.append(filtered_df.copy())
            return "filtered.csv"

        def reject_short_keyword(self, judge_items, **kwargs):
            assert len(judge_items) == 1
            return {
                item["item_id"]: chembl_module._ChemblJudgeDecision(
                    item_id=item["item_id"],
                    keep=False,
                    explanation="Wrong short-keyword hit.",
                )
                for item in judge_items
            }

        def reject_metadata(self, judge_items, **kwargs):
            assert len(judge_items) == 1
            assert judge_items[0]["target_pref_name"] == "Tyrosine-protein kinase ABL1"
            return {
                item["item_id"]: chembl_module._ChemblJudgeDecision(
                    item_id=item["item_id"],
                    keep=False,
                    explanation="Wrong protein metadata.",
                )
                for item in judge_items
            }

        monkeypatch.setattr(ChemblToolkit, "_save_filtered_rows", fake_save)
        monkeypatch.setattr(ChemblToolkit, "_run_chembl_retrieval_judge", reject_short_keyword)
        monkeypatch.setattr(ChemblToolkit, "_run_chembl_metadata_judge", reject_metadata)

        result = toolkit._filter_suspicious_short_keyword_rows(
            df,
            keywords=["BRAF", "kinase"],
            target_query="BRAF kinase",
            organism_filter=None,
            query_slug="braf_kinase",
            agent=Mock(model=object()),
            session_state={},
        )

        assert result.summary["judge_status"] == "completed"
        assert result.summary["metadata_judge_status"] == "completed"
        assert result.summary["filtered_row_count"] == 2
        assert result.summary["metadata_filtered_row_count"] == 1
        assert result.retained_df.empty
        assert saved[0]["filter_reason"].tolist() == [
            "llm_judge_rejected",
            "metadata_llm_judge_rejected",
        ]

    def test_short_keyword_filter_unavailable_judge_filters_rows(self, monkeypatch):
        toolkit = ChemblToolkit()
        df = pd.DataFrame(
            [
                {
                    "activity_id": 1,
                    "assay_chembl_id": "CHEMBL_A1",
                    "query_keywords": "CDK2",
                    "target_pref_name": "Cyclin-dependent kinase 2",
                    "target_type": "SINGLE PROTEIN",
                    "description": "CDK2 biochemical assay",
                    "smi": "CCO",
                }
            ]
        )
        saved = []

        def fake_save(self, filtered_df, *, query_slug, session_state):
            saved.append(filtered_df.copy())
            return "filtered.csv"

        monkeypatch.setattr(ChemblToolkit, "_save_filtered_rows", fake_save)

        result = toolkit._filter_suspicious_short_keyword_rows(
            df,
            keywords=["CDK2"],
            target_query="CDK2",
            organism_filter=None,
            query_slug="cdk2",
            agent=None,
            session_state={},
        )

        assert result.summary["judge_status"] == "unavailable"
        assert result.summary["filtered_row_count"] == 1
        assert result.retained_df.empty
        assert saved[0]["filter_reason"].tolist() == ["judge_unavailable"]

    def test_retrieval_filtering_report_section_is_appended(self, tmp_path):
        toolkit = ChemblToolkit()
        report_path = tmp_path / "report.md"
        report_path.write_text("# Dataset Standardization Report\n")

        toolkit._append_retrieval_filtering_report(
            str(report_path),
            {
                "suspicious_row_count": 2,
                "filtered_row_count": 1,
                "retained_row_count": 10,
                "retrieved_row_count": 11,
                "judge_status": "completed",
                "fallback_policy": "filter_rows",
                "filtered_rows_path": "filtered.csv",
                "decision_counts": {"llm_judge_rejected": 1},
            },
        )

        report = report_path.read_text()
        assert "## ChEMBL Retrieval Filtering" in report
        assert "Filtered rows artifact: `filtered.csv`" in report

    @patch.object(ChemblToolkit, "_ensure_client")
    @patch("cs_copilot.tools.databases.chembl.S3")
    def test_fetch_compounds_multi_keyword_tracks_keywords(
        self, mock_s3, mock_ensure_client, monkeypatch
    ):
        """Test that query_keywords column tracks which keywords retrieved each row."""
        mock_client = Mock()

        # Two keywords: "cdk2" and "kinase"
        # activity_id=1 appears for both keywords (overlap)
        # activity_id=2 only from "cdk2", activity_id=3 only from "kinase"
        assays_cdk2 = [
            {
                "assay_chembl_id": "CHEMBL_A1",
                "target_pref_name": "Cyclin-dependent kinase 2",
                "target_type": "SINGLE PROTEIN",
                "description": "CDK2 biochemical assay",
            }
        ]
        assays_kinase = [
            {
                "assay_chembl_id": "CHEMBL_A2",
                "target_pref_name": "Protein kinase",
                "target_type": "SINGLE PROTEIN",
                "description": "Kinase biochemical assay",
            }
        ]

        activities_cdk2 = [
            {
                "activity_id": 1,
                "assay_chembl_id": "CHEMBL_A1",
                "molecule_chembl_id": "MOL1",
                "standard_value": 10.0,
                "standard_units": "nM",
                "standard_type": "IC50",
                "canonical_smiles": "CCO",
            },
            {
                "activity_id": 2,
                "assay_chembl_id": "CHEMBL_A1",
                "molecule_chembl_id": "MOL2",
                "standard_value": 20.0,
                "standard_units": "nM",
                "standard_type": "IC50",
                "canonical_smiles": "CCC",
            },
        ]
        activities_kinase = [
            {
                "activity_id": 1,
                "assay_chembl_id": "CHEMBL_A2",
                "molecule_chembl_id": "MOL1",
                "standard_value": 10.0,
                "standard_units": "nM",
                "standard_type": "IC50",
                "canonical_smiles": "CCO",
            },
            {
                "activity_id": 3,
                "assay_chembl_id": "CHEMBL_A2",
                "molecule_chembl_id": "MOL3",
                "standard_value": 30.0,
                "standard_units": "nM",
                "standard_type": "IC50",
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
        mock_s3.path.side_effect = lambda rel: rel

        saved_dfs = []
        original_prepare = chembl_module.prepare_clean_dataset

        def capture_prepare(df, *args, **kwargs):
            saved_dfs.append(df.copy())
            return original_prepare(df, *args, **kwargs)

        def keep_short_keyword_hits(self, judge_items, **kwargs):
            return {
                item["item_id"]: chembl_module._ChemblJudgeDecision(
                    item_id=item["item_id"],
                    keep=True,
                    explanation="Test keeps short-keyword matches.",
                )
                for item in judge_items
            }

        def keep_metadata_hits(self, judge_items, **kwargs):
            return {
                item["item_id"]: chembl_module._ChemblJudgeDecision(
                    item_id=item["item_id"],
                    keep=True,
                    explanation="Test keeps target metadata matches.",
                )
                for item in judge_items
            }

        monkeypatch.setattr(chembl_module, "prepare_clean_dataset", capture_prepare)
        monkeypatch.setattr(
            ChemblToolkit,
            "_run_chembl_retrieval_judge",
            keep_short_keyword_hits,
        )
        monkeypatch.setattr(
            ChemblToolkit,
            "_run_chembl_metadata_judge",
            keep_metadata_hits,
        )

        toolkit = ChemblToolkit()
        toolkit.fetch_compounds("cdk2, kinase", agent=Mock(model=object(), session_state={}))

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

        result = toolkit.convert_to_chembl_query("phosphodiesterase 4A")

        assert "ChEMBL's `assay_description__icontains` filter" in result
        assert "phosphodiesterase 4A" in result
        # The new example in the tool's return message uses PDE4A as the abbreviation.
        assert "pde4" in result.lower()

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
        mock_s3.path.side_effect = lambda rel: rel

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
        mock_s3.path.side_effect = lambda rel: rel

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

    @patch.object(ChemblToolkit, "_ensure_client")
    @patch("cs_copilot.tools.databases.chembl.S3")
    def test_fetch_compounds_uses_regex_for_multitoken(self, mock_s3, mock_ensure_client):
        """A multi-token keyword should produce ONE REST call with iregex."""
        mock_client = Mock()
        mock_client.assay.filter.return_value = [{"assay_chembl_id": "CHEMBL1"}]
        mock_client.activity.filter.return_value.only.return_value = [
            {
                "activity_id": 1,
                "assay_chembl_id": "CHEMBL1",
                "molecule_chembl_id": "CHEMBL1",
                "standard_value": 10.0,
                "canonical_smiles": "CCO",
            },
        ]
        mock_ensure_client.return_value = mock_client

        mock_file = MagicMock()
        mock_s3.open.return_value.__enter__.return_value = mock_file
        mock_s3.open.return_value.__exit__.return_value = False
        mock_s3.path.side_effect = lambda rel: rel

        toolkit = ChemblToolkit()
        toolkit.fetch_compounds("epidermal growth factor receptor")

        # Should be called exactly once (not 8 times).
        assert mock_client.assay.filter.call_count == 1
        call_kwargs = mock_client.assay.filter.call_args.kwargs
        assert "description__iregex" in call_kwargs
        assert "description__icontains" not in call_kwargs
        assert call_kwargs["description__iregex"] == r"epidermal[- ]growth[- ]factor[- ]receptor"


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
        mock_s3.path.side_effect = lambda rel: rel

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
            fetch_result = toolkit.fetch_compounds("kinase")
            assert "✅ Fetched" in fetch_result

            # Step 2: Describe dataset
            desc_result = toolkit.describe_dataset("chembl_kinase.csv")
            assert isinstance(desc_result, str)


class TestPunctuationRegex:
    """Test the regex builder for punctuation-variant matching."""

    def test_symmetric_cyclin_dependent(self):
        """The user's explicit invariant: equivalent spellings yield the same regex."""
        from cs_copilot.tools.databases.chembl import _build_punctuation_regex

        a = _build_punctuation_regex("cyclin-dependent kinase 2")
        b = _build_punctuation_regex("cyclin dependent kinase 2")
        c = _build_punctuation_regex("cyclin-dependent kinase-2")
        d = _build_punctuation_regex("cyclin dependent kinase-2")

        assert a == b == c == d
        assert a == r"cyclin[- ]dependent[- ]kinase[- ]2"

    def test_four_token_required_separator(self):
        """4-token phrases use required separator and match all hyphen/space combos."""
        from cs_copilot.tools.databases.chembl import _build_punctuation_regex

        p = _build_punctuation_regex("epidermal growth factor receptor")
        assert p == r"epidermal[- ]growth[- ]factor[- ]receptor"
        assert re.search(p, "epidermal growth factor receptor", re.IGNORECASE)
        assert re.search(p, "epidermal-growth-factor-receptor", re.IGNORECASE)
        assert re.search(p, "epidermal-growth factor receptor", re.IGNORECASE)

    def test_four_token_does_not_match_concat(self):
        """4-token phrases require a separator — concatenated form should not match."""
        from cs_copilot.tools.databases.chembl import _build_punctuation_regex

        p = _build_punctuation_regex("epidermal growth factor receptor")
        assert not re.search(p, "epidermalgrowthfactorreceptor", re.IGNORECASE)

    def test_two_token_optional_separator(self):
        """2-token phrases use optional separator, matching space/hyphen/concat."""
        from cs_copilot.tools.databases.chembl import _build_punctuation_regex

        p = _build_punctuation_regex("serotonin 2A")
        assert p == r"serotonin[- ]?2A"
        assert re.search(p, "serotonin 2A", re.IGNORECASE)
        assert re.search(p, "serotonin-2A", re.IGNORECASE)
        assert re.search(p, "serotonin2A", re.IGNORECASE)

    def test_compact_5ht2a(self):
        """Hyphen-separated compact names like 5-HT2A sub-split at letter→digit."""
        from cs_copilot.tools.databases.chembl import _build_punctuation_regex

        p = _build_punctuation_regex("5-HT2A")
        # "HT2A" sub-splits to ["HT", "2A"] → 3 total tokens → optional sep
        assert p == r"5[- ]?HT[- ]?2A"
        assert re.search(p, "5-HT2A")
        assert re.search(p, "5 HT2A")
        assert re.search(p, "5HT2A")
        assert re.search(p, "5-HT-2A")

    def test_three_token_optional_separator(self):
        """3-token phrases also use optional separator."""
        from cs_copilot.tools.databases.chembl import _build_punctuation_regex

        p = _build_punctuation_regex("B-Raf kinase")
        assert p == r"B[- ]?Raf[- ]?kinase"
        assert re.search(p, "B-Raf kinase")
        assert re.search(p, "B Raf kinase")
        assert re.search(p, "BRafkinase")

    def test_single_token_no_boundary_returns_none(self):
        """All-letter tokens with no letter→digit boundary return None."""
        from cs_copilot.tools.databases.chembl import _build_punctuation_regex

        assert _build_punctuation_regex("EGFR") is None
        assert _build_punctuation_regex("BRAF") is None
        assert _build_punctuation_regex("kinase") is None

    def test_single_token_letter_digit_split(self):
        """Single tokens with letter→digit boundaries produce a regex."""
        from cs_copilot.tools.databases.chembl import _build_punctuation_regex

        # CDK2 → ["CDK", "2"] → CDK[- ]?2
        p = _build_punctuation_regex("CDK2")
        assert p == r"CDK[- ]?2"
        assert re.search(p, "CDK2", re.IGNORECASE)
        assert re.search(p, "CDK-2", re.IGNORECASE)
        assert re.search(p, "CDK 2", re.IGNORECASE)

        # PDE4A → ["PDE", "4A"] → PDE[- ]?4A
        p = _build_punctuation_regex("PDE4A")
        assert p == r"PDE[- ]?4A"
        assert re.search(p, "PDE4A", re.IGNORECASE)
        assert re.search(p, "PDE-4A", re.IGNORECASE)
        assert re.search(p, "PDE 4A", re.IGNORECASE)

        # JAK2 → ["JAK", "2"]
        p = _build_punctuation_regex("JAK2")
        assert p == r"JAK[- ]?2"

        # IL6 → ["IL", "6"]
        p = _build_punctuation_regex("IL6")
        assert p == r"IL[- ]?6"

    def test_empty_returns_none(self):
        """Empty and whitespace-only inputs return None."""
        from cs_copilot.tools.databases.chembl import _build_punctuation_regex

        assert _build_punctuation_regex("") is None
        assert _build_punctuation_regex("   ") is None

    def test_metacharacter_escaping(self):
        """Regex metacharacters in tokens are properly escaped."""
        from cs_copilot.tools.databases.chembl import _build_punctuation_regex

        p = _build_punctuation_regex("IL-1beta (receptor)")
        assert p is not None
        assert r"\(" in p
        assert r"\)" in p
        assert re.search(p, "IL-1beta (receptor)", re.IGNORECASE)
        assert re.search(p, "IL 1beta (receptor)", re.IGNORECASE)
