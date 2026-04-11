#!/usr/bin/env python
# coding: utf-8
"""
Strategy interfaces and implementations for ChEMBL data fetching.

Provides pluggable backends (REST API, MySQL) for the ChemblToolkit.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from .base import DatabaseError

logger = logging.getLogger(__name__)


class ChemblDataFetcher(ABC):
    """Abstract strategy interface for ChEMBL data backends."""

    @abstractmethod
    def fetch_assays(
        self,
        keyword: str,
        organism: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch assays matching keyword and optional organism filter.

        Returns list of assay dicts with at minimum:
        assay_chembl_id, description, assay_type, assay_organism.
        """

    @abstractmethod
    def fetch_activities(
        self,
        assay_ids: List[str],
        assay_type_codes: List[str],
        fields: List[str],
    ) -> List[Dict[str, Any]]:
        """Fetch activity records for given assay IDs with optional type filter.

        Returns list of activity dicts matching the requested fields.
        """

    @abstractmethod
    def connect(self) -> None:
        """Establish and validate connection. Raises DatabaseError on failure."""

    @abstractmethod
    def ping(self) -> bool:
        """Test connectivity. Returns True if reachable."""

    @abstractmethod
    def close(self) -> None:
        """Release resources."""


class RestChemblFetcher(ChemblDataFetcher):
    """Fetches ChEMBL data via the REST API (chembl_webresource_client)."""

    def __init__(self, ensure_client_fn: Callable):
        self._ensure_client = ensure_client_fn

    def fetch_assays(self, keyword, organism=None):
        client = self._ensure_client()
        filter_kwargs: Dict[str, Any] = {"description__icontains": keyword}
        if organism:
            filter_kwargs["target_organism__icontains"] = organism
        return list(client.assay.filter(**filter_kwargs))

    def fetch_activities(self, assay_ids, assay_type_codes, fields):
        client = self._ensure_client()
        filter_kwargs: Dict[str, Any] = {"assay_chembl_id__in": assay_ids}
        if assay_type_codes:
            filter_kwargs["assay_type__in"] = assay_type_codes
        return list(client.activity.filter(**filter_kwargs).only(fields))

    def connect(self):
        client = self._ensure_client()
        list(client.target.filter(target_chembl_id="CHEMBL1").only(["target_chembl_id"])[:1])

    def ping(self):
        try:
            client = self._ensure_client()
            return client.status is not None
        except Exception:
            return False

    def close(self):
        pass  # REST client is stateless


class SqlChemblFetcher(ChemblDataFetcher):
    """Fetches ChEMBL data from a local SQL database via SQLAlchemy.

    Works with MySQL, PostgreSQL, and SQLite backends — the SQL queries
    use only ANSI-standard syntax.  Use the ``from_mysql_env``,
    ``from_postgres_env``, or ``from_sqlite`` classmethods to create an
    instance from environment configuration.
    """

    def __init__(self, engine, backend_label: str = "SQL"):
        self._engine = engine
        self._backend_label = backend_label

    @staticmethod
    def _build_connection_url(
        drivername: str,
        host: str,
        port: str,
        user: str,
        password: str,
        database: str,
    ):
        """Build a SQLAlchemy URL while preserving special characters in credentials."""
        from sqlalchemy.engine import URL

        return URL.create(
            drivername=drivername,
            username=user,
            password=password,
            host=host,
            port=int(port),
            database=database,
        )

    @classmethod
    def from_mysql_env(cls) -> "SqlChemblFetcher":
        """Create from CHEMBL_MYSQL_* environment variables."""
        try:
            import pymysql  # noqa: F401
        except ImportError:
            raise ImportError(
                "pymysql is required for MySQL backend. Install it with: uv sync --extra mysql"
            ) from None

        from sqlalchemy import create_engine

        host = os.getenv("CHEMBL_MYSQL_HOST", "localhost")
        port = os.getenv("CHEMBL_MYSQL_PORT", "3306")
        user = os.getenv("CHEMBL_MYSQL_USER", "root")
        password = os.getenv("CHEMBL_MYSQL_PASSWORD", "")
        database = os.getenv("CHEMBL_MYSQL_DATABASE", "chembl_35")

        url = cls._build_connection_url(
            drivername="mysql+pymysql",
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
        )
        engine = create_engine(url, pool_pre_ping=True, pool_size=5)
        logger.info(f"Created MySQL engine for ChEMBL: {host}:{port}/{database}")
        return cls(engine, backend_label="MySQL")

    @classmethod
    def from_env(cls) -> "SqlChemblFetcher":
        """Backward-compatible alias for :meth:`from_mysql_env`."""
        return cls.from_mysql_env()

    @classmethod
    def from_postgres_env(cls) -> "SqlChemblFetcher":
        """Create from CHEMBL_PG_* environment variables."""
        try:
            import psycopg2  # noqa: F401
        except ImportError:
            raise ImportError(
                "psycopg2 is required for PostgreSQL backend. "
                "Install it with: uv sync --extra postgresql"
            ) from None

        from sqlalchemy import create_engine

        host = os.getenv("CHEMBL_PG_HOST", "localhost")
        port = os.getenv("CHEMBL_PG_PORT", "5432")
        user = os.getenv("CHEMBL_PG_USER", "chembl")
        password = os.getenv("CHEMBL_PG_PASSWORD", "")
        database = os.getenv("CHEMBL_PG_DATABASE", "chembl_36")

        url = cls._build_connection_url(
            drivername="postgresql+psycopg2",
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
        )
        engine = create_engine(url, pool_pre_ping=True, pool_size=5)
        logger.info(f"Created PostgreSQL engine for ChEMBL: {host}:{port}/{database}")
        return cls(engine, backend_label="PostgreSQL")

    @classmethod
    def from_sqlite(cls, path: Optional[str] = None) -> "SqlChemblFetcher":
        """Create from a SQLite file path or CHEMBL_SQLITE_PATH env var."""
        from sqlalchemy import create_engine

        db_path = path or os.getenv("CHEMBL_SQLITE_PATH")
        if not db_path:
            raise ValueError("SQLite path required. Pass path= or set CHEMBL_SQLITE_PATH env var.")

        url = f"sqlite:///{db_path}"
        engine = create_engine(url)
        logger.info(f"Created SQLite engine for ChEMBL: {db_path}")
        return cls(engine, backend_label="SQLite")

    def fetch_assays(self, keyword, organism=None):
        from sqlalchemy import text

        sql = """
            SELECT
                a.chembl_id AS assay_chembl_id,
                a.description,
                a.assay_type,
                a.assay_organism,
                td.chembl_id AS target_chembl_id
            FROM assays a
            LEFT JOIN target_dictionary td ON a.tid = td.tid
            WHERE a.description LIKE :keyword_pattern
        """
        params: Dict[str, Any] = {"keyword_pattern": f"%{keyword}%"}

        if organism:
            sql += " AND td.organism LIKE :organism_pattern"
            params["organism_pattern"] = f"%{organism}%"

        with self._engine.connect() as conn:
            result = conn.execute(text(sql), params)
            return [dict(row._mapping) for row in result]

    def fetch_activities(self, assay_ids, assay_type_codes, fields):
        from sqlalchemy import text

        if not assay_ids:
            return []

        # Build parameterized IN clause
        aid_placeholders = ", ".join([f":aid_{i}" for i in range(len(assay_ids))])
        params: Dict[str, Any] = {f"aid_{i}": aid for i, aid in enumerate(assay_ids)}

        sql = f"""
            SELECT
                act.activity_id,
                ass.chembl_id AS assay_chembl_id,
                md.chembl_id AS molecule_chembl_id,
                cs.canonical_smiles,
                act.standard_type,
                act.standard_value,
                act.standard_units,
                act.pchembl_value,
                act.activity_comment,
                act.data_validity_comment,
                act.potential_duplicate
            FROM activities act
            JOIN assays ass ON act.assay_id = ass.assay_id
            JOIN molecule_dictionary md ON act.molregno = md.molregno
            LEFT JOIN compound_structures cs ON md.molregno = cs.molregno
            WHERE ass.chembl_id IN ({aid_placeholders})
        """

        if assay_type_codes:
            type_placeholders = ", ".join([f":at_{i}" for i in range(len(assay_type_codes))])
            sql += f" AND ass.assay_type IN ({type_placeholders})"
            params.update({f"at_{i}": code for i, code in enumerate(assay_type_codes)})

        with self._engine.connect() as conn:
            result = conn.execute(text(sql), params)
            return [dict(row._mapping) for row in result]

    def connect(self):
        try:
            with self._engine.connect() as conn:
                from sqlalchemy import text

                conn.execute(text("SELECT 1"))
            logger.info(f"Connected to ChEMBL {self._backend_label} database successfully")
        except Exception as e:
            raise DatabaseError(f"ChEMBL {self._backend_label} connection failed: {e}") from e

    def ping(self):
        try:
            with self._engine.connect() as conn:
                from sqlalchemy import text

                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def close(self):
        self._engine.dispose()
