#!/usr/bin/env python
# coding: utf-8
"""
Tests for the SQL ChEMBL data fetcher and backend selection.
"""

import os
import sys
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

    def test_fetch_assays_enriches_target_metadata(self):
        mock_client = Mock()
        mock_client.assay.filter.return_value = [
            {
                "assay_chembl_id": "CHEMBL_A1",
                "description": "CDK2 biochemical assay",
                "target_chembl_id": "CHEMBL_TARGET_1",
            }
        ]
        mock_client.target.filter.return_value.only.return_value = [
            {
                "target_chembl_id": "CHEMBL_TARGET_1",
                "pref_name": "Cyclin-dependent kinase 2",
                "target_type": "SINGLE PROTEIN",
                "organism": "Homo sapiens",
                "tax_id": 9606,
                "species_group_flag": 0,
            }
        ]

        fetcher = RestChemblFetcher(lambda: mock_client)
        result = fetcher.fetch_assays("CDK2")

        assert result[0]["target_pref_name"] == "Cyclin-dependent kinase 2"
        assert result[0]["target_type"] == "SINGLE PROTEIN"
        assert result[0]["target_organism"] == "Homo sapiens"
        assert result[0]["target_tax_id"] == 9606
        assert result[0]["target_species_group_flag"] == 0
        mock_client.target.filter.assert_called_once_with(target_chembl_id__in=["CHEMBL_TARGET_1"])

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

    def test_fetch_assays_with_regex(self):
        """When regex_pattern is passed, should use description__iregex."""
        mock_client = Mock()
        mock_client.assay.filter.return_value = [{"assay_chembl_id": "CHEMBL1"}]

        fetcher = RestChemblFetcher(lambda: mock_client)
        fetcher.fetch_assays(
            "epidermal growth factor receptor",
            regex_pattern=r"epidermal[- ]growth[- ]factor[- ]receptor",
        )

        mock_client.assay.filter.assert_called_with(
            description__iregex=r"epidermal[- ]growth[- ]factor[- ]receptor"
        )

    def test_fetch_assays_regex_with_organism(self):
        """Regex and organism filter should combine correctly."""
        mock_client = Mock()
        mock_client.assay.filter.return_value = []

        fetcher = RestChemblFetcher(lambda: mock_client)
        fetcher.fetch_assays(
            "PDE4",
            organism="Homo sapiens",
            regex_pattern=r"PDE[- ]?4",
        )

        mock_client.assay.filter.assert_called_with(
            description__iregex=r"PDE[- ]?4",
            target_organism__icontains="Homo sapiens",
        )


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
                "target_pref_name": "Cyclin-dependent kinase 2",
                "target_type": "SINGLE PROTEIN",
                "target_organism": "Homo sapiens",
                "target_tax_id": 9606,
                "target_species_group_flag": 0,
            }
        ]
        fetcher, conn = self._make_fetcher(assay_data)
        result = fetcher.fetch_assays("CDK2")

        assert len(result) == 1
        assert result[0]["assay_chembl_id"] == "CHEMBL123"
        assert result[0]["description"] == "CDK2 kinase assay"
        assert result[0]["target_pref_name"] == "Cyclin-dependent kinase 2"

        # Verify SQL was called with keyword parameter
        call_args = conn.execute.call_args
        sql_text = str(call_args[0][0])
        params = call_args[0][1]
        assert "td.pref_name AS target_pref_name" in sql_text
        assert "td.target_type" in sql_text
        assert "td.organism AS target_organism" in sql_text
        assert "td.tax_id AS target_tax_id" in sql_text
        assert "td.species_group_flag AS target_species_group_flag" in sql_text
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
        fetcher = SqlChemblFetcher(engine, backend_label="MySQL")
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
                    SqlChemblFetcher.from_mysql_env()

    def test_from_env_reads_env_vars(self):
        """Test that from_mysql_env correctly reads environment variables."""
        env = {
            "CHEMBL_MYSQL_HOST": "db.example.com",
            "CHEMBL_MYSQL_PORT": "3307",
            "CHEMBL_MYSQL_USER": "chembl_user",
            "CHEMBL_MYSQL_PASSWORD": "secret",
            "CHEMBL_MYSQL_DATABASE": "chembl_34",
        }
        mock_pymysql = MagicMock()
        with patch.dict(sys.modules, {"pymysql": mock_pymysql}):
            with patch.dict(os.environ, env, clear=False):
                with patch("sqlalchemy.create_engine", return_value=MagicMock()) as mock_ce:
                    fetcher = SqlChemblFetcher.from_mysql_env()
                    mock_ce.assert_called_once()
                    url = mock_ce.call_args[0][0]
                    assert url.drivername == "mysql+pymysql"
                    assert url.host == "db.example.com"
                    assert url.port == 3307
                    assert url.username == "chembl_user"
                    assert url.password == "secret"
                    assert url.database == "chembl_34"
                    assert url.render_as_string(hide_password=False).startswith(
                        "mysql+pymysql://chembl_user:secret@db.example.com:3307/chembl_34"
                    )
                    assert fetcher._backend_label == "MySQL"

    def test_from_env_preserves_special_chars_in_mysql_password(self):
        env = {
            "CHEMBL_MYSQL_HOST": "localhost",
            "CHEMBL_MYSQL_PORT": "3306",
            "CHEMBL_MYSQL_USER": "aorlov",
            "CHEMBL_MYSQL_PASSWORD": "AxelMySQL7140@",
            "CHEMBL_MYSQL_DATABASE": "chembl_36",
        }
        mock_pymysql = MagicMock()
        with patch.dict(sys.modules, {"pymysql": mock_pymysql}):
            with patch.dict(os.environ, env, clear=False):
                with patch("sqlalchemy.create_engine", return_value=MagicMock()) as mock_ce:
                    SqlChemblFetcher.from_mysql_env()
                    url = mock_ce.call_args[0][0]
                    assert url.host == "localhost"
                    assert url.port == 3306
                    assert url.username == "aorlov"
                    assert url.password == "AxelMySQL7140@"
                    assert url.database == "chembl_36"

    def test_from_env_backward_compat(self):
        """Test that from_env still works as alias for from_mysql_env."""
        mock_pymysql = MagicMock()
        env = {"CHEMBL_MYSQL_HOST": "localhost"}
        with patch.dict(sys.modules, {"pymysql": mock_pymysql}):
            with patch.dict(os.environ, env, clear=False):
                with patch("sqlalchemy.create_engine", return_value=MagicMock()):
                    fetcher = SqlChemblFetcher.from_env()
                    assert fetcher._backend_label == "MySQL"

    def test_fetch_assays_with_regex_mysql(self):
        """MySQL backend should use REGEXP for regex_pattern."""
        fetcher, conn = self._make_fetcher([])
        fetcher._backend_label = "MySQL"
        fetcher.fetch_assays(
            "epidermal growth factor receptor",
            regex_pattern=r"epidermal[- ]growth[- ]factor[- ]receptor",
        )

        call_args = conn.execute.call_args
        sql_text = str(call_args[0][0])
        params = call_args[0][1]
        assert "REGEXP" in sql_text
        assert params["regex_pattern"] == r"epidermal[- ]growth[- ]factor[- ]receptor"
        assert "keyword_pattern" not in params

    def test_fetch_assays_with_regex_postgres(self):
        """PostgreSQL backend should use ~* for regex_pattern."""
        fetcher, conn = self._make_fetcher([])
        fetcher._backend_label = "PostgreSQL"
        fetcher.fetch_assays(
            "JAK2 kinase",
            regex_pattern=r"JAK[- ]?2[- ]?kinase",
        )

        call_args = conn.execute.call_args
        sql_text = str(call_args[0][0])
        params = call_args[0][1]
        assert "~*" in sql_text
        assert params["regex_pattern"] == r"JAK[- ]?2[- ]?kinase"

    def test_fetch_assays_with_regex_sqlite(self):
        """SQLite backend should use regexp() function for regex_pattern."""
        fetcher, conn = self._make_fetcher([])
        fetcher._backend_label = "SQLite"

        # Mock the connection chain for _ensure_sqlite_regexp
        mock_dbapi = MagicMock()
        conn.connection.dbapi_connection = mock_dbapi

        fetcher.fetch_assays(
            "PDE4A",
            regex_pattern=r"PDE[- ]?4A",
        )

        call_args = conn.execute.call_args
        sql_text = str(call_args[0][0])
        params = call_args[0][1]
        assert "regexp(" in sql_text.lower()
        assert params["regex_pattern"] == r"PDE[- ]?4A"
        # Verify REGEXP function was registered on the SQLite connection
        mock_dbapi.create_function.assert_called_once()

    def test_fetch_assays_no_regex_uses_like(self):
        """Without regex_pattern, SQL backend should use LIKE as before."""
        fetcher, conn = self._make_fetcher([])
        fetcher._backend_label = "MySQL"
        fetcher.fetch_assays("kinase")

        call_args = conn.execute.call_args
        params = call_args[0][1]
        assert params["keyword_pattern"] == "%kinase%"
        assert "regex_pattern" not in params


class TestSqlChemblFetcherSQLite:
    """Tests for SqlChemblFetcher SQLite backend."""

    def test_from_sqlite_with_path(self):
        with patch("sqlalchemy.create_engine", return_value=MagicMock()) as mock_ce:
            fetcher = SqlChemblFetcher.from_sqlite("/path/to/chembl_36.db")
            mock_ce.assert_called_once()
            url = mock_ce.call_args[0][0]
            assert url == "sqlite:////path/to/chembl_36.db"
            assert fetcher._backend_label == "SQLite"

    def test_from_sqlite_env_var(self):
        with patch.dict(os.environ, {"CHEMBL_SQLITE_PATH": "/data/chembl.db"}):
            with patch("sqlalchemy.create_engine", return_value=MagicMock()) as mock_ce:
                fetcher = SqlChemblFetcher.from_sqlite()
                url = mock_ce.call_args[0][0]
                assert "/data/chembl.db" in url
                assert fetcher._backend_label == "SQLite"

    def test_from_sqlite_no_path_raises(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CHEMBL_SQLITE_PATH", None)
            with pytest.raises(ValueError, match="SQLite path required"):
                SqlChemblFetcher.from_sqlite()

    def test_connect_failure_uses_sqlite_label(self):
        engine = MagicMock()
        engine.connect.side_effect = Exception("file not found")
        fetcher = SqlChemblFetcher(engine, backend_label="SQLite")
        with pytest.raises(DatabaseError, match="ChEMBL SQLite connection failed"):
            fetcher.connect()


class TestSqlChemblFetcherPostgres:
    """Tests for SqlChemblFetcher PostgreSQL backend."""

    def test_from_postgres_env_reads_env_vars(self):
        env = {
            "CHEMBL_PG_HOST": "pg.example.com",
            "CHEMBL_PG_PORT": "5433",
            "CHEMBL_PG_USER": "pg_user",
            "CHEMBL_PG_PASSWORD": "pg_pass",
            "CHEMBL_PG_DATABASE": "chembl_36",
        }
        mock_psycopg2 = MagicMock()
        with patch.dict(sys.modules, {"psycopg2": mock_psycopg2}):
            with patch.dict(os.environ, env, clear=False):
                with patch("sqlalchemy.create_engine", return_value=MagicMock()) as mock_ce:
                    fetcher = SqlChemblFetcher.from_postgres_env()
                    mock_ce.assert_called_once()
                    url = mock_ce.call_args[0][0]
                    assert url.drivername == "postgresql+psycopg2"
                    assert url.host == "pg.example.com"
                    assert url.port == 5433
                    assert url.username == "pg_user"
                    assert url.password == "pg_pass"
                    assert url.database == "chembl_36"
                    assert fetcher._backend_label == "PostgreSQL"

    def test_from_postgres_env_preserves_special_chars_in_password(self):
        env = {
            "CHEMBL_PG_HOST": "localhost",
            "CHEMBL_PG_PORT": "5432",
            "CHEMBL_PG_USER": "chembl",
            "CHEMBL_PG_PASSWORD": "pg_pass@local",
            "CHEMBL_PG_DATABASE": "chembl_36",
        }
        mock_psycopg2 = MagicMock()
        with patch.dict(sys.modules, {"psycopg2": mock_psycopg2}):
            with patch.dict(os.environ, env, clear=False):
                with patch("sqlalchemy.create_engine", return_value=MagicMock()) as mock_ce:
                    SqlChemblFetcher.from_postgres_env()
                    url = mock_ce.call_args[0][0]
                    assert url.host == "localhost"
                    assert url.port == 5432
                    assert url.username == "chembl"
                    assert url.password == "pg_pass@local"
                    assert url.database == "chembl_36"

    def test_from_postgres_env_missing_psycopg2(self):
        with patch("builtins.__import__", side_effect=ImportError("No psycopg2")):
            with pytest.raises(ImportError, match="psycopg2 is required"):
                SqlChemblFetcher.from_postgres_env()

    def test_connect_failure_uses_postgresql_label(self):
        engine = MagicMock()
        engine.connect.side_effect = Exception("Connection refused")
        fetcher = SqlChemblFetcher(engine, backend_label="PostgreSQL")
        with pytest.raises(DatabaseError, match="ChEMBL PostgreSQL connection failed"):
            fetcher.connect()


class TestBackendSelection:
    """Tests for ChemblToolkit backend auto-detection and selection."""

    def _clear_chembl_env(self):
        """Remove all ChEMBL backend env vars."""
        for key in ("CHEMBL_MYSQL_HOST", "CHEMBL_PG_HOST", "CHEMBL_SQLITE_PATH"):
            os.environ.pop(key, None)

    def test_default_is_rest_when_no_env(self):
        """Without any ChEMBL env vars, auto selects REST."""
        with patch.dict(os.environ, {}, clear=False):
            self._clear_chembl_env()
            toolkit = ChemblToolkit()
            assert isinstance(toolkit._fetcher, RestChemblFetcher)

    def test_explicit_rest_backend(self):
        toolkit = ChemblToolkit(backend="rest")
        assert isinstance(toolkit._fetcher, RestChemblFetcher)

    @patch.object(SqlChemblFetcher, "from_mysql_env")
    def test_explicit_mysql_backend(self, mock_from_mysql):
        mock_from_mysql.return_value = MagicMock(spec=SqlChemblFetcher)
        ChemblToolkit(backend="mysql")
        assert mock_from_mysql.called

    @patch.object(SqlChemblFetcher, "from_mysql_env")
    def test_auto_detects_mysql_from_env(self, mock_from_mysql):
        mock_from_mysql.return_value = MagicMock(spec=SqlChemblFetcher)
        with patch.dict(os.environ, {"CHEMBL_MYSQL_HOST": "localhost"}):
            self._clear_chembl_env()
            os.environ["CHEMBL_MYSQL_HOST"] = "localhost"
            ChemblToolkit(backend="auto")
            assert mock_from_mysql.called

    @patch.object(
        SqlChemblFetcher,
        "from_mysql_env",
        side_effect=ImportError("pymysql is required for MySQL backend"),
    )
    def test_auto_falls_back_to_rest_when_mysql_driver_missing(self, mock_from_mysql):
        with patch.dict(os.environ, {"CHEMBL_MYSQL_HOST": "localhost"}, clear=False):
            self._clear_chembl_env()
            os.environ["CHEMBL_MYSQL_HOST"] = "localhost"
            toolkit = ChemblToolkit(backend="auto")

        assert mock_from_mysql.called
        assert isinstance(toolkit._fetcher, RestChemblFetcher)
        assert toolkit._active_backend == "rest"
        assert toolkit.config.supports_sql is False

    @patch.object(SqlChemblFetcher, "from_mysql_env")
    @patch.object(
        SqlChemblFetcher,
        "from_postgres_env",
        side_effect=ImportError("psycopg2 is required for PostgreSQL backend"),
    )
    def test_auto_falls_back_from_postgresql_to_mysql(self, mock_from_pg, mock_from_mysql):
        mock_from_mysql.return_value = MagicMock(spec=SqlChemblFetcher)
        env = {"CHEMBL_PG_HOST": "localhost", "CHEMBL_MYSQL_HOST": "localhost"}
        with patch.dict(os.environ, env, clear=False):
            self._clear_chembl_env()
            os.environ.update(env)
            toolkit = ChemblToolkit(backend="auto")

        assert mock_from_pg.called
        assert mock_from_mysql.called
        assert toolkit._active_backend == "mysql"

    @patch.object(
        SqlChemblFetcher,
        "from_mysql_env",
        side_effect=ImportError("pymysql is required for MySQL backend"),
    )
    def test_explicit_mysql_backend_still_raises_when_driver_missing(self, mock_from_mysql):
        with pytest.raises(ImportError, match="pymysql is required for MySQL backend"):
            ChemblToolkit(backend="mysql")

        assert mock_from_mysql.called

    @patch.object(SqlChemblFetcher, "from_sqlite")
    def test_auto_detects_sqlite_from_env(self, mock_from_sqlite):
        mock_from_sqlite.return_value = MagicMock(spec=SqlChemblFetcher)
        with patch.dict(os.environ, {"CHEMBL_SQLITE_PATH": "/tmp/chembl.db"}):
            ChemblToolkit(backend="auto")
            assert mock_from_sqlite.called

    @patch.object(SqlChemblFetcher, "from_postgres_env")
    def test_auto_detects_postgresql_from_env(self, mock_from_pg):
        mock_from_pg.return_value = MagicMock(spec=SqlChemblFetcher)
        with patch.dict(os.environ, {}, clear=False):
            self._clear_chembl_env()
            os.environ["CHEMBL_PG_HOST"] = "localhost"
            ChemblToolkit(backend="auto")
            assert mock_from_pg.called

    @patch.object(SqlChemblFetcher, "from_sqlite")
    def test_auto_priority_sqlite_over_pg_and_mysql(self, mock_from_sqlite):
        """SQLite takes priority when multiple env vars are set."""
        mock_from_sqlite.return_value = MagicMock(spec=SqlChemblFetcher)
        env = {
            "CHEMBL_SQLITE_PATH": "/tmp/chembl.db",
            "CHEMBL_PG_HOST": "localhost",
            "CHEMBL_MYSQL_HOST": "localhost",
        }
        with patch.dict(os.environ, env, clear=False):
            toolkit = ChemblToolkit(backend="auto")
            assert mock_from_sqlite.called
            assert toolkit._active_backend == "sqlite"

    @patch.object(SqlChemblFetcher, "from_postgres_env")
    def test_auto_priority_pg_over_mysql(self, mock_from_pg):
        """PostgreSQL takes priority over MySQL."""
        mock_from_pg.return_value = MagicMock(spec=SqlChemblFetcher)
        with patch.dict(os.environ, {}, clear=False):
            self._clear_chembl_env()
            os.environ["CHEMBL_PG_HOST"] = "localhost"
            os.environ["CHEMBL_MYSQL_HOST"] = "localhost"
            toolkit = ChemblToolkit(backend="auto")
            assert mock_from_pg.called
            assert toolkit._active_backend == "postgresql"

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown ChEMBL backend"):
            ChemblToolkit(backend="postgres")

    @patch.object(SqlChemblFetcher, "from_sqlite")
    def test_explicit_sqlite_backend(self, mock_from_sqlite):
        mock_from_sqlite.return_value = MagicMock(spec=SqlChemblFetcher)
        toolkit = ChemblToolkit(backend="sqlite")
        assert mock_from_sqlite.called
        assert toolkit._active_backend == "sqlite"
        assert toolkit.config.supports_sql is True

    @patch.object(SqlChemblFetcher, "from_postgres_env")
    def test_explicit_postgresql_backend(self, mock_from_pg):
        mock_from_pg.return_value = MagicMock(spec=SqlChemblFetcher)
        toolkit = ChemblToolkit(backend="postgresql")
        assert mock_from_pg.called
        assert toolkit._active_backend == "postgresql"
        assert toolkit.config.supports_sql is True

    @patch.object(SqlChemblFetcher, "from_mysql_env")
    def test_mysql_backend_reports_both_capabilities(self, mock_from_mysql):
        """When MySQL is configured, both supports_http_api and supports_sql must be True."""
        mock_from_mysql.return_value = MagicMock(spec=SqlChemblFetcher)
        toolkit = ChemblToolkit(backend="mysql")
        assert toolkit.config.supports_sql is True
        assert toolkit.config.supports_http_api is True

    @patch.object(SqlChemblFetcher, "from_mysql_env")
    def test_get_capabilities_active_backend_mysql(self, mock_from_mysql):
        """get_capabilities reports active_backend='mysql' when MySQL is configured."""
        mock_from_mysql.return_value = MagicMock(spec=SqlChemblFetcher)
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

    @patch.object(SqlChemblFetcher, "from_sqlite")
    def test_get_capabilities_active_backend_sqlite(self, mock_from_sqlite):
        mock_from_sqlite.return_value = MagicMock(spec=SqlChemblFetcher)
        toolkit = ChemblToolkit(backend="sqlite")
        caps = toolkit.get_capabilities()
        assert caps["supports_sql"] is True
        assert caps["active_backend"] == "sqlite"

    @patch.object(SqlChemblFetcher, "from_postgres_env")
    def test_get_capabilities_active_backend_postgresql(self, mock_from_pg):
        mock_from_pg.return_value = MagicMock(spec=SqlChemblFetcher)
        toolkit = ChemblToolkit(backend="postgresql")
        caps = toolkit.get_capabilities()
        assert caps["supports_sql"] is True
        assert caps["active_backend"] == "postgresql"

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
        mock_fetcher.fetch_assays.assert_called_once_with("kinase", None, regex_pattern=None)
        mock_fetcher.fetch_activities.assert_called_once()
