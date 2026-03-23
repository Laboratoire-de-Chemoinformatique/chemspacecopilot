#!/usr/bin/env python
# coding: utf-8
"""
Tests for the SQL ChEMBL data fetcher and backend selection.
"""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from cs_copilot.tools.databases.base import DatabaseError
from cs_copilot.tools.databases.chembl import ChemblToolkit
from cs_copilot.tools.databases.chembl_fetcher import (
    ChemblDataFetcher,
    RestChemblFetcher,
    SqlChemblFetcher,
)


class TestRestChemblFetcher:
    """Tests for RestChemblFetcher."""

    def test_fetch_assays_basic(self):
        mock_client = Mock()
        mock_assays = [{"assay_chembl_id": "CHEMBL1", "description": "kinase assay"}]
        mock_client.assay.filter.return_value = mock_assays

        fetcher = RestChemblFetcher(lambda: mock_client)
        result = fetcher.fetch_assays("kinase")

        assert result == mock_assays
        mock_client.assay.filter.assert_called_with(description__icontains="kinase")

    def test_fetch_assays_with_organism(self):
        mock_client = Mock()
        mock_client.assay.filter.return_value = []

        fetcher = RestChemblFetcher(lambda: mock_client)
        fetcher.fetch_assays("kinase", organism="Homo sapiens")

        mock_client.assay.filter.assert_called_with(
            description__icontains="kinase",
            target_organism__icontains="Homo sapiens",
        )

    def test_fetch_activities(self):
        mock_client = Mock()
        mock_activities = [{"activity_id": 1, "molecule_chembl_id": "CHEMBL1"}]
        mock_client.activity.filter.return_value.only.return_value = mock_activities

        fetcher = RestChemblFetcher(lambda: mock_client)
        result = fetcher.fetch_activities(
            ["CHEMBL_A1"], ["B", "F"], ["activity_id", "molecule_chembl_id"]
        )

        assert result == mock_activities
        mock_client.activity.filter.assert_called_with(
            assay_chembl_id__in=["CHEMBL_A1"], assay_type__in=["B", "F"]
        )

    def test_fetch_activities_no_type_filter(self):
        mock_client = Mock()
        mock_client.activity.filter.return_value.only.return_value = []

        fetcher = RestChemblFetcher(lambda: mock_client)
        fetcher.fetch_activities(["CHEMBL_A1"], [], ["activity_id"])

        mock_client.activity.filter.assert_called_with(assay_chembl_id__in=["CHEMBL_A1"])

    def test_ping_success(self):
        mock_client = Mock()
        mock_client.status = "OK"
        fetcher = RestChemblFetcher(lambda: mock_client)
        assert fetcher.ping()

    def test_ping_failure(self):
        mock_client = Mock()
        mock_client.status = None
        fetcher = RestChemblFetcher(lambda: mock_client)
        assert not fetcher.ping()

    def test_ping_exception(self):
        fetcher = RestChemblFetcher(Mock(side_effect=Exception("unavailable")))
        assert not fetcher.ping()

    def test_connect_success(self):
        mock_client = Mock()
        mock_target = Mock()
        mock_target.__getitem__ = Mock(return_value=[{"target_chembl_id": "CHEMBL1"}])
        mock_client.target.filter.return_value.only.return_value = mock_target
        fetcher = RestChemblFetcher(lambda: mock_client)
        fetcher.connect()  # Should not raise

    def test_close_is_noop(self):
        fetcher = RestChemblFetcher(lambda: Mock())
        fetcher.close()  # Should not raise


class TestSqlChemblFetcher:
    """Tests for SqlChemblFetcher with mocked engine."""

    def _make_fetcher(self, query_results=None):
        """Create a SqlChemblFetcher with a mocked engine."""
        engine = MagicMock()
        conn = MagicMock()
        engine.connect.return_value.__enter__ = Mock(return_value=conn)
        engine.connect.return_value.__exit__ = Mock(return_value=False)

        if query_results is not None:
            mock_rows = []
            for row_dict in query_results:
                mock_row = MagicMock()
                mock_row._mapping = row_dict
                mock_rows.append(mock_row)
            conn.execute.return_value = mock_rows

        return SqlChemblFetcher(engine), conn

    def test_fetch_assays_basic(self):
        assay_data = [
            {
                "assay_chembl_id": "CHEMBL123",
                "description": "CDK2 kinase assay",
                "assay_type": "B",
                "assay_organism": "Homo sapiens",
                "target_chembl_id": "CHEMBL2345",
            }
        ]
        fetcher, conn = self._make_fetcher(assay_data)
        result = fetcher.fetch_assays("CDK2")

        assert len(result) == 1
        assert result[0]["assay_chembl_id"] == "CHEMBL123"
        assert result[0]["description"] == "CDK2 kinase assay"

        # Verify SQL was called with keyword parameter
        call_args = conn.execute.call_args
        params = call_args[0][1]
        assert params["keyword_pattern"] == "%CDK2%"

    def test_fetch_assays_with_organism(self):
        fetcher, conn = self._make_fetcher([])
        fetcher.fetch_assays("kinase", organism="Homo sapiens")

        call_args = conn.execute.call_args
        params = call_args[0][1]
        assert params["keyword_pattern"] == "%kinase%"
        assert params["organism_pattern"] == "%Homo sapiens%"

    def test_fetch_activities_basic(self):
        activity_data = [
            {
                "activity_id": 42,
                "assay_chembl_id": "CHEMBL123",
                "molecule_chembl_id": "CHEMBL456",
                "canonical_smiles": "CCO",
                "standard_type": "IC50",
                "standard_value": 10.0,
                "standard_units": "nM",
                "pchembl_value": 8.0,
                "activity_comment": None,
                "data_validity_comment": None,
                "potential_duplicate": 0,
            }
        ]
        fetcher, conn = self._make_fetcher(activity_data)
        result = fetcher.fetch_activities(
            ["CHEMBL123"], ["B"], ["activity_id", "molecule_chembl_id"]
        )

        assert len(result) == 1
        assert result[0]["activity_id"] == 42
        assert result[0]["canonical_smiles"] == "CCO"

    def test_fetch_activities_empty_assay_ids(self):
        fetcher, _ = self._make_fetcher([])
        result = fetcher.fetch_activities([], ["B"], ["activity_id"])
        assert result == []

    def test_fetch_activities_no_type_filter(self):
        fetcher, conn = self._make_fetcher([])
        fetcher.fetch_activities(["CHEMBL123"], [], ["activity_id"])

        call_args = conn.execute.call_args
        sql_text = str(call_args[0][0])
        assert "assay_type" not in sql_text.split("WHERE")[1] if "WHERE" in sql_text else True

    def test_fetch_activities_with_type_filter(self):
        fetcher, conn = self._make_fetcher([])
        fetcher.fetch_activities(["CHEMBL123"], ["B", "F"], ["activity_id"])

        call_args = conn.execute.call_args
        params = call_args[0][1]
        assert params["at_0"] == "B"
        assert params["at_1"] == "F"

    def test_ping_success(self):
        fetcher, conn = self._make_fetcher()
        conn.execute.return_value = None
        assert fetcher.ping()

    def test_ping_failure(self):
        engine = MagicMock()
        engine.connect.side_effect = Exception("Connection refused")
        fetcher = SqlChemblFetcher(engine)
        assert not fetcher.ping()

    def test_connect_success(self):
        fetcher, conn = self._make_fetcher()
        conn.execute.return_value = None
        fetcher.connect()  # Should not raise

    def test_connect_failure(self):
        engine = MagicMock()
        engine.connect.side_effect = Exception("Connection refused")
        fetcher = SqlChemblFetcher(engine)
        with pytest.raises(DatabaseError, match="ChEMBL MySQL connection failed"):
            fetcher.connect()

    def test_close_disposes_engine(self):
        engine = MagicMock()
        fetcher = SqlChemblFetcher(engine)
        fetcher.close()
        engine.dispose.assert_called_once()

    def test_from_env_missing_pymysql(self):
        with patch.dict(os.environ, {"CHEMBL_MYSQL_HOST": "localhost"}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'pymysql'")):
                with pytest.raises(ImportError, match="pymysql is required"):
                    SqlChemblFetcher.from_env()

    def test_from_env_reads_env_vars(self):
        """Test that from_env correctly reads environment variables."""
        import sys

        env = {
            "CHEMBL_MYSQL_HOST": "db.example.com",
            "CHEMBL_MYSQL_PORT": "3307",
            "CHEMBL_MYSQL_USER": "chembl_user",
            "CHEMBL_MYSQL_PASSWORD": "secret",
            "CHEMBL_MYSQL_DATABASE": "chembl_34",
        }
        # Mock pymysql so the import check passes
        mock_pymysql = MagicMock()
        with patch.dict(sys.modules, {"pymysql": mock_pymysql}):
            with patch.dict(os.environ, env, clear=False):
                from sqlalchemy import create_engine

                with patch(
                    "sqlalchemy.create_engine", return_value=MagicMock()
                ) as mock_ce:
                    fetcher = SqlChemblFetcher.from_env()
                    mock_ce.assert_called_once()
                    url = mock_ce.call_args[0][0]
                    assert "db.example.com" in url
                    assert "3307" in url
                    assert "chembl_user" in url
                    assert "chembl_34" in url


class TestBackendSelection:
    """Tests for ChemblToolkit backend auto-detection and selection."""

    def test_default_is_rest_when_no_env(self):
        """Without CHEMBL_MYSQL_HOST, auto selects REST."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove CHEMBL_MYSQL_HOST if present
            os.environ.pop("CHEMBL_MYSQL_HOST", None)
            toolkit = ChemblToolkit()
            assert isinstance(toolkit._fetcher, RestChemblFetcher)

    def test_explicit_rest_backend(self):
        toolkit = ChemblToolkit(backend="rest")
        assert isinstance(toolkit._fetcher, RestChemblFetcher)

    @patch.object(SqlChemblFetcher, "from_env")
    def test_explicit_mysql_backend(self, mock_from_env):
        mock_from_env.return_value = MagicMock(spec=SqlChemblFetcher)
        toolkit = ChemblToolkit(backend="mysql")
        assert mock_from_env.called

    @patch.object(SqlChemblFetcher, "from_env")
    def test_auto_detects_mysql_from_env(self, mock_from_env):
        mock_from_env.return_value = MagicMock(spec=SqlChemblFetcher)
        with patch.dict(os.environ, {"CHEMBL_MYSQL_HOST": "localhost"}):
            toolkit = ChemblToolkit(backend="auto")
            assert mock_from_env.called

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown ChEMBL backend"):
            ChemblToolkit(backend="postgres")

    @patch.object(SqlChemblFetcher, "from_env")
    def test_mysql_backend_reports_both_capabilities(self, mock_from_env):
        """When MySQL is configured, both supports_http_api and supports_sql must be True."""
        mock_from_env.return_value = MagicMock(spec=SqlChemblFetcher)
        toolkit = ChemblToolkit(backend="mysql")
        assert toolkit.config.supports_sql is True
        assert toolkit.config.supports_http_api is True

    @patch.object(SqlChemblFetcher, "from_env")
    def test_get_capabilities_active_backend_mysql(self, mock_from_env):
        """get_capabilities reports active_backend='mysql' when MySQL is configured."""
        mock_from_env.return_value = MagicMock(spec=SqlChemblFetcher)
        toolkit = ChemblToolkit(backend="mysql")
        caps = toolkit.get_capabilities()
        assert caps["supports_sql"] is True
        assert caps["supports_http_api"] is True
        assert caps["active_backend"] == "mysql"

    def test_get_capabilities_active_backend_rest(self):
        """get_capabilities reports active_backend='rest' for REST backend."""
        toolkit = ChemblToolkit(backend="rest")
        caps = toolkit.get_capabilities()
        assert caps["supports_sql"] is False
        assert caps["supports_http_api"] is True
        assert caps["active_backend"] == "rest"

    @patch.object(ChemblToolkit, "_ensure_client")
    @patch("cs_copilot.tools.databases.chembl.S3")
    def test_fetch_compounds_uses_fetcher(self, mock_s3, mock_ensure_client):
        """Verify fetch_compounds delegates to the fetcher."""
        mock_fetcher = MagicMock(spec=ChemblDataFetcher)
        mock_fetcher.fetch_assays.return_value = [
            {"assay_chembl_id": "CHEMBL1", "description": "test"}
        ]
        mock_fetcher.fetch_activities.return_value = [
            {
                "activity_id": 1,
                "assay_chembl_id": "CHEMBL1",
                "molecule_chembl_id": "MOL1",
                "canonical_smiles": "CCO",
                "standard_value": 10.0,
            }
        ]

        mock_file = MagicMock()
        mock_s3.open.return_value.__enter__.return_value = mock_file
        mock_s3.open.return_value.__exit__.return_value = False

        toolkit = ChemblToolkit(backend="rest")
        toolkit._fetcher = mock_fetcher

        result = toolkit.fetch_compounds("kinase")

        assert "✅ Fetched" in result
        mock_fetcher.fetch_assays.assert_called_once_with("kinase", None)
        mock_fetcher.fetch_activities.assert_called_once()
