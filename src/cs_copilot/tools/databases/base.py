#!/usr/bin/env python
# coding: utf-8
"""
Base database toolkit with abstract interface for all database implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence

from agno.tools import Toolkit

from .types import DatabaseRecord, DBConfig, PaginationMode, QueryParams, ResultPage

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception for database operations."""

    pass


class ConnectionError(DatabaseError):
    """Database connection failed."""

    pass


class NotFound(DatabaseError):
    """Requested resource not found."""

    pass


class ValidationError(DatabaseError):
    """Invalid query parameters or data."""

    pass


class RateLimited(DatabaseError):
    """Rate limit exceeded."""

    pass


class QueryTimeout(DatabaseError):
    """Query execution timed out."""

    pass


class BaseDatabaseToolkit(Toolkit, ABC):
    """
    Abstract foundation for database toolkits.

    The base class intentionally stays agnostic to protocol or backend so it can
    support SQL engines, document stores, REST APIs, or in-memory sources. Only a
    minimal set of lifecycle and pagination helpers are provided; concrete
    toolkits must implement the query logic appropriate for their data source.
    """

    def __init__(self, config: DBConfig, name: str = "database_toolkit", **toolkit_kwargs):
        self.config = config
        self._connected = False

        toolkit_instructions = toolkit_kwargs.pop(
            "instructions",
            f"Database toolkit for {config.uri}. Provides database query and data access capabilities.",
        )
        super().__init__(name=name, instructions=toolkit_instructions, **toolkit_kwargs)
        self.register(self.ping)
        self.register(self.get_capabilities)

    def connect(self) -> None:
        """Mark the toolkit as connected.

        Subclasses can override this to perform real connection checks or
        initialisation for their specific backend.
        """

        logger.info(f"Connecting to database: {self.config.uri}")
        self._connected = True

    def close(self) -> None:
        """Close the database connection."""

        self._connected = False

    def ping(self) -> bool:
        """
        Test database connectivity.

        Returns:
            True if database is reachable, False otherwise
        """

        try:
            return self._connected
        except Exception as e:
            logger.warning(f"Ping failed: {e}")
            return False

    @abstractmethod
    def query(self, params: QueryParams) -> ResultPage:
        """
        Execute a single-page query against the underlying data source.

        Concrete toolkits must implement this and return a ``ResultPage``
        describing the slice of results and pagination pointers for subsequent
        calls.
        """

    def fetch_one(self, params: QueryParams) -> Optional[DatabaseRecord]:
        """
        Fetch a single record.

        Args:
            params: Query parameters

        Returns:
            Single record or None if not found
        """
        params.limit = 1
        result = self.query(params)
        return result.records[0] if result.records else None

    def fetch_many(
        self, params: QueryParams, max_records: Optional[int] = None
    ) -> List[DatabaseRecord]:
        """
        Fetch multiple records, handling pagination automatically.

        Args:
            params: Query parameters
            max_records: Maximum number of records to return

        Returns:
            List of records
        """
        records = []
        fetched = 0

        for record in self.fetch_all(params):
            records.append(record)
            fetched += 1

            if max_records and fetched >= max_records:
                break

        return records

    def fetch_all(self, params: QueryParams) -> Iterable[DatabaseRecord]:
        """
        Fetch all records, yielding them one by one with automatic pagination.

        Args:
            params: Query parameters

        Yields:
            Individual records
        """
        current_params = QueryParams(
            filters=params.filters.copy(),
            fields=params.fields,
            sort=params.sort,
            limit=params.limit or self.config.page_size,
            offset=params.offset,
            cursor=params.cursor,
            page=params.page,
            extra_params=params.extra_params.copy(),
        )

        while True:
            page = self.query(current_params)

            if not page.records:
                break

            for record in page.records:
                yield record

            # Check if there are more pages
            if not page.has_more:
                break

            # Update pagination parameters based on mode
            if self.config.pagination_mode == PaginationMode.OFFSET_LIMIT:
                if page.next_offset is not None:
                    current_params.offset = page.next_offset
                else:
                    break
            elif self.config.pagination_mode == PaginationMode.CURSOR:
                if page.next_cursor is not None:
                    current_params.cursor = page.next_cursor
                else:
                    break
            elif self.config.pagination_mode == PaginationMode.PAGE_NUMBER:
                if page.next_page is not None:
                    current_params.page = page.next_page
                else:
                    break
            else:
                # Unknown pagination mode, stop
                break

    def to_dataframe(self, records: Sequence[DatabaseRecord], normalize: bool = True) -> Any:
        """
        Convert records to a pandas DataFrame.

        Args:
            records: List of database records
            normalize: Whether to normalize nested structures

        Returns:
            pandas DataFrame

        Raises:
            ImportError: If pandas is not available
            NotImplementedError: If not implemented by subclass
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas is required for DataFrame conversion") from e

        if not records:
            return pd.DataFrame()

        if normalize:
            # Flatten nested structures by default
            return pd.json_normalize(records)
        else:
            return pd.DataFrame(records)

    def normalize_params(self, params: QueryParams) -> QueryParams:
        """
        Normalize query parameters for this database type.

        Args:
            params: Original query parameters

        Returns:
            Normalized parameters
        """
        # Default implementation returns params as-is
        return params

    def map_fields(self, record: DatabaseRecord) -> DatabaseRecord:
        """
        Map database-specific field names to standard field names.

        Args:
            record: Raw database record

        Returns:
            Record with mapped field names
        """
        # Default implementation returns record as-is
        return record

    def handle_error(self, error: Exception) -> DatabaseError:
        """
        Map database-specific errors to standard error types.

        Args:
            error: Original exception

        Returns:
            Mapped database error
        """
        if "timeout" in str(error).lower():
            return QueryTimeout(str(error))
        elif "not found" in str(error).lower():
            return NotFound(str(error))
        elif "rate limit" in str(error).lower():
            return RateLimited(str(error))
        elif "invalid" in str(error).lower() or "validation" in str(error).lower():
            return ValidationError(str(error))
        else:
            return DatabaseError(str(error))

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get information about database capabilities.

        Returns:
            Dictionary of capability flags and limits
        """
        return {
            "supports_sql": self.config.supports_sql,
            "supports_http_api": self.config.supports_http_api,
            "pagination_mode": self.config.pagination_mode.value,
            "max_page_size": self.config.page_size,
            "supports_sorting": True,  # Most databases support this
            "supports_filtering": True,  # Most databases support this
            "rate_limit": self.config.rate_limit,
        }

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
