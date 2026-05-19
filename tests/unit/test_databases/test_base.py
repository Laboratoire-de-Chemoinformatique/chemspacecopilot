#!/usr/bin/env python
# coding: utf-8
"""
Tests for the base database toolkit.
"""

from cs_copilot.tools.databases.base import (
    BaseDatabaseToolkit,
    DatabaseError,
    NotFound,
    QueryTimeout,
    RateLimited,
    ValidationError,
)
from cs_copilot.tools.databases.types import DBConfig, PaginationMode, QueryParams, ResultPage


class MockDatabaseToolkit(BaseDatabaseToolkit):
    """Mock implementation for testing."""

    def __init__(self, config: DBConfig):
        super().__init__(config, name="mock_database_toolkit")
        self._query_results = []
        self._call_count = 0

    def query(self, params: QueryParams) -> ResultPage:
        """Implement abstract query method."""
        import time

        self._call_count += 1
        start_time = time.perf_counter()

        if self._query_results:
            result = self._query_results.pop(0)
            # Ensure query_time_ms is set
            if result.query_time_ms is None:
                result.query_time_ms = (time.perf_counter() - start_time) * 1000
            return result

        # Default response
        limit = params.limit or 10
        query_time_ms = (time.perf_counter() - start_time) * 1000
        return ResultPage(
            records=[{"id": i, "name": f"record_{i}"} for i in range(limit)],
            has_more=False,
            query_time_ms=query_time_ms,
        )

    def set_mock_results(self, results):
        """Set mock results for testing."""
        self._query_results = results


class TestBaseDatabaseToolkit:
    def test_init(self):
        """Test toolkit initialization."""
        config = DBConfig(uri="test://localhost")
        toolkit = MockDatabaseToolkit(config)

        assert toolkit.config == config
        assert not toolkit._connected

    def test_connect_and_close(self):
        """Test connection management."""
        config = DBConfig(uri="test://localhost")
        toolkit = MockDatabaseToolkit(config)

        toolkit.connect()
        assert toolkit._connected

        toolkit.close()
        assert not toolkit._connected

    def test_reconnect(self):
        """Test reconnecting after close."""
        config = DBConfig(uri="test://localhost")
        toolkit = MockDatabaseToolkit(config)

        # First connection
        toolkit.connect()
        assert toolkit._connected

        # Close
        toolkit.close()
        assert not toolkit._connected

        # Reconnect
        toolkit.connect()
        assert toolkit._connected

    def test_ping_when_connected(self):
        """Test ping returns True when connected."""
        config = DBConfig(uri="test://localhost")
        toolkit = MockDatabaseToolkit(config)

        toolkit.connect()
        assert toolkit.ping()

    def test_ping_when_not_connected(self):
        """Test ping returns False when not connected."""
        config = DBConfig(uri="test://localhost")
        toolkit = MockDatabaseToolkit(config)

        assert not toolkit.ping()

    def test_context_manager(self):
        """Test context manager functionality."""
        config = DBConfig(uri="test://localhost")

        with MockDatabaseToolkit(config) as toolkit:
            assert toolkit._connected
        assert not toolkit._connected

    def test_context_manager_with_exception(self):
        """Test context manager closes connection even on exception."""
        config = DBConfig(uri="test://localhost")
        toolkit = MockDatabaseToolkit(config)

        try:
            with toolkit:
                assert toolkit._connected
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Connection should be closed even after exception
        assert not toolkit._connected

    def test_query_basic(self):
        """Test basic query execution."""
        config = DBConfig(uri="test://localhost")
        toolkit = MockDatabaseToolkit(config)

        params = QueryParams(filters={"name": "test"}, limit=5)
        result = toolkit.query(params)

        assert isinstance(result, ResultPage)
        assert len(result.records) == 5
        assert result.query_time_ms is not None

    def test_fetch_one(self):
        """Test fetching a single record."""
        config = DBConfig(uri="test://localhost")
        toolkit = MockDatabaseToolkit(config)

        params = QueryParams(filters={"id": 1})
        record = toolkit.fetch_one(params)

        assert record is not None
        assert record["id"] == 0  # Mock returns records starting from 0

    def test_fetch_one_not_found(self):
        """Test fetch_one when no records found."""
        config = DBConfig(uri="test://localhost")
        toolkit = MockDatabaseToolkit(config)

        # Mock empty result
        toolkit.set_mock_results([ResultPage(records=[], has_more=False)])

        params = QueryParams(filters={"id": 999})
        record = toolkit.fetch_one(params)

        assert record is None

    def test_fetch_many(self):
        """Test fetching multiple records."""
        config = DBConfig(uri="test://localhost")
        toolkit = MockDatabaseToolkit(config)

        params = QueryParams(filters={"type": "test"})
        records = toolkit.fetch_many(params, max_records=3)

        assert len(records) == 3

    def test_fetch_all_pagination(self):
        """Test fetch_all with pagination."""
        config = DBConfig(uri="test://localhost", page_size=2)
        toolkit = MockDatabaseToolkit(config)

        # Mock paginated results
        toolkit.set_mock_results(
            [
                ResultPage(records=[{"id": 0}, {"id": 1}], has_more=True, next_offset=2),
                ResultPage(records=[{"id": 2}, {"id": 3}], has_more=True, next_offset=4),
                ResultPage(records=[{"id": 4}], has_more=False),
            ]
        )

        params = QueryParams(filters={"type": "test"})
        all_records = list(toolkit.fetch_all(params))

        assert len(all_records) == 5
        assert [r["id"] for r in all_records] == [0, 1, 2, 3, 4]

    def test_to_dataframe(self):
        """Test DataFrame conversion."""
        config = DBConfig(uri="test://localhost")
        toolkit = MockDatabaseToolkit(config)

        records = [{"id": 1, "name": "test1", "value": 10}, {"id": 2, "name": "test2", "value": 20}]

        df = toolkit.to_dataframe(records)

        assert len(df) == 2
        assert list(df.columns) == ["id", "name", "value"]
        assert df.iloc[0]["name"] == "test1"

    def test_to_dataframe_empty(self):
        """Test DataFrame conversion with empty records."""
        config = DBConfig(uri="test://localhost")
        toolkit = MockDatabaseToolkit(config)

        df = toolkit.to_dataframe([])

        assert len(df) == 0

    def test_error_handling(self):
        """Test error mapping."""
        config = DBConfig(uri="test://localhost")
        toolkit = MockDatabaseToolkit(config)

        # Test different error types
        timeout_error = Exception("Connection timeout occurred")
        mapped_error = toolkit.handle_error(timeout_error)
        assert isinstance(mapped_error, QueryTimeout)

        not_found_error = Exception("Resource not found")
        mapped_error = toolkit.handle_error(not_found_error)
        assert isinstance(mapped_error, NotFound)

        rate_limit_error = Exception("Rate limit exceeded")
        mapped_error = toolkit.handle_error(rate_limit_error)
        assert isinstance(mapped_error, RateLimited)

        validation_error = Exception("Invalid parameter")
        mapped_error = toolkit.handle_error(validation_error)
        assert isinstance(mapped_error, ValidationError)

        generic_error = Exception("Something went wrong")
        mapped_error = toolkit.handle_error(generic_error)
        assert isinstance(mapped_error, DatabaseError)

    def test_get_capabilities(self):
        """Test capabilities reporting."""
        config = DBConfig(
            uri="test://localhost",
            supports_sql=True,
            supports_http_api=False,
            pagination_mode=PaginationMode.CURSOR,
            page_size=50,
            rate_limit=100.0,
        )
        toolkit = MockDatabaseToolkit(config)

        capabilities = toolkit.get_capabilities()

        assert capabilities["supports_sql"]
        assert not capabilities["supports_http_api"]
        assert capabilities["pagination_mode"] == "cursor"
        assert capabilities["max_page_size"] == 50
        assert capabilities["rate_limit"] == 100.0

    def test_normalize_params(self):
        """Test parameter normalization."""
        config = DBConfig(uri="test://localhost")
        toolkit = MockDatabaseToolkit(config)

        params = QueryParams(filters={"name": "test"})
        normalized = toolkit.normalize_params(params)

        # Default implementation should return params as-is
        assert normalized == params

    def test_map_fields(self):
        """Test field mapping."""
        config = DBConfig(uri="test://localhost")
        toolkit = MockDatabaseToolkit(config)

        record = {"id": 1, "name": "test"}
        mapped = toolkit.map_fields(record)

        # Default implementation should return record as-is
        assert mapped == record

    def test_query_timing(self):
        """Test that query timing is recorded."""
        config = DBConfig(uri="test://localhost")
        toolkit = MockDatabaseToolkit(config)

        params = QueryParams(filters={"name": "test"})
        result = toolkit.query(params)

        assert result.query_time_ms is not None
        assert result.query_time_ms >= 0

    def test_connection_reuse(self):
        """Test that connection can be reused for multiple queries."""
        config = DBConfig(uri="test://localhost")
        toolkit = MockDatabaseToolkit(config)

        toolkit.connect()

        # Execute multiple queries on same connection
        params1 = QueryParams(filters={"name": "test1"})
        result1 = toolkit.query(params1)

        params2 = QueryParams(filters={"name": "test2"})
        result2 = toolkit.query(params2)

        assert len(result1.records) > 0
        assert len(result2.records) > 0
        assert toolkit._connected  # Still connected

    def test_query_without_connection(self):
        """Test that queries work even without explicit connect() call."""
        config = DBConfig(uri="test://localhost")
        toolkit = MockDatabaseToolkit(config)

        # Query without calling connect() first
        params = QueryParams(filters={"name": "test"})
        result = toolkit.query(params)

        assert isinstance(result, ResultPage)
        assert len(result.records) > 0

    def test_multiple_context_managers(self):
        """Test that toolkit can be used in multiple context manager blocks."""
        config = DBConfig(uri="test://localhost")
        toolkit = MockDatabaseToolkit(config)

        # First context
        with toolkit:
            assert toolkit._connected
            params = QueryParams(filters={"name": "test1"})
            toolkit.query(params)

        assert not toolkit._connected

        # Second context
        with toolkit:
            assert toolkit._connected
            params = QueryParams(filters={"name": "test2"})
            toolkit.query(params)

        assert not toolkit._connected


class TestDBConfig:
    def test_default_values(self):
        """Test DBConfig default values."""
        config = DBConfig(uri="test://localhost")

        assert config.uri == "test://localhost"
        assert config.api_key is None
        assert config.timeout_s == 30.0
        assert config.retries == 3
        assert config.page_size == 100
        assert config.rate_limit is None
        assert config.headers == {}
        assert not config.supports_sql
        assert config.supports_http_api
        assert config.pagination_mode == PaginationMode.OFFSET_LIMIT

    def test_custom_values(self):
        """Test DBConfig with custom values."""
        config = DBConfig(
            uri="postgres://localhost:5432/test",
            api_key="secret",
            timeout_s=60.0,
            retries=5,
            page_size=200,
            rate_limit=50.0,
            headers={"User-Agent": "TestClient"},
            supports_sql=True,
            supports_http_api=False,
            pagination_mode=PaginationMode.PAGE_NUMBER,
        )

        assert config.uri == "postgres://localhost:5432/test"
        assert config.api_key == "secret"
        assert config.timeout_s == 60.0
        assert config.retries == 5
        assert config.page_size == 200
        assert config.rate_limit == 50.0
        assert config.headers == {"User-Agent": "TestClient"}
        assert config.supports_sql
        assert not config.supports_http_api
        assert config.pagination_mode == PaginationMode.PAGE_NUMBER


class TestQueryParams:
    def test_default_values(self):
        """Test QueryParams default values."""
        params = QueryParams()

        assert params.filters == {}
        assert params.fields is None
        assert params.sort is None
        assert params.limit is None
        assert params.offset == 0
        assert params.cursor is None
        assert params.page is None
        assert params.extra_params == {}

    def test_custom_values(self):
        """Test QueryParams with custom values."""
        params = QueryParams(
            filters={"name": "test", "active": True},
            fields=["id", "name", "created_at"],
            sort=[("created_at", "desc"), ("name", "asc")],
            limit=50,
            offset=100,
            cursor="abc123",
            page=2,
            extra_params={"include_deleted": False},
        )

        assert params.filters == {"name": "test", "active": True}
        assert params.fields == ["id", "name", "created_at"]
        assert params.sort == [("created_at", "desc"), ("name", "asc")]
        assert params.limit == 50
        assert params.offset == 100
        assert params.cursor == "abc123"
        assert params.page == 2
        assert params.extra_params == {"include_deleted": False}


class TestErrorHandling:
    """Test error handling and retries."""

    def test_timeout_error_handling(self):
        """Test timeout error is properly mapped."""
        config = DBConfig(uri="test://localhost", timeout_s=1.0)
        toolkit = MockDatabaseToolkit(config)

        timeout_error = Exception("Connection timeout occurred")
        mapped = toolkit.handle_error(timeout_error)

        assert isinstance(mapped, QueryTimeout)
        assert "timeout" in str(mapped).lower()

    def test_rate_limit_error_handling(self):
        """Test rate limit error is properly mapped."""
        config = DBConfig(uri="test://localhost", rate_limit=10.0)
        toolkit = MockDatabaseToolkit(config)

        rate_error = Exception("Rate limit exceeded, please slow down")
        mapped = toolkit.handle_error(rate_error)

        assert isinstance(mapped, RateLimited)
        assert "rate limit" in str(mapped).lower()

    def test_not_found_error_handling(self):
        """Test not found error is properly mapped."""
        config = DBConfig(uri="test://localhost")
        toolkit = MockDatabaseToolkit(config)

        not_found_error = Exception("Resource not found in database")
        mapped = toolkit.handle_error(not_found_error)

        assert isinstance(mapped, NotFound)
        assert "not found" in str(mapped).lower()

    def test_validation_error_handling(self):
        """Test validation error is properly mapped."""
        config = DBConfig(uri="test://localhost")
        toolkit = MockDatabaseToolkit(config)

        validation_error = Exception("Invalid query parameter: field name")
        mapped = toolkit.handle_error(validation_error)

        assert isinstance(mapped, ValidationError)
        assert "invalid" in str(mapped).lower()

    def test_generic_error_handling(self):
        """Test generic errors are mapped to DatabaseError."""
        config = DBConfig(uri="test://localhost")
        toolkit = MockDatabaseToolkit(config)

        generic_error = Exception("Something unexpected happened")
        mapped = toolkit.handle_error(generic_error)

        assert isinstance(mapped, DatabaseError)
        assert not isinstance(mapped, (QueryTimeout, RateLimited, NotFound, ValidationError))

    def test_config_timeout_setting(self):
        """Test timeout configuration."""
        config = DBConfig(uri="test://localhost", timeout_s=60.0)
        toolkit = MockDatabaseToolkit(config)

        assert toolkit.config.timeout_s == 60.0

    def test_config_retries_setting(self):
        """Test retry configuration."""
        config = DBConfig(uri="test://localhost", retries=5)
        toolkit = MockDatabaseToolkit(config)

        assert toolkit.config.retries == 5

    def test_config_rate_limit_setting(self):
        """Test rate limit configuration."""
        config = DBConfig(uri="test://localhost", rate_limit=100.0)
        toolkit = MockDatabaseToolkit(config)

        assert toolkit.config.rate_limit == 100.0


class TestResultPage:
    def test_default_values(self):
        """Test ResultPage default values."""
        records = [{"id": 1}, {"id": 2}]
        page = ResultPage(records=records)

        assert page.records == records
        assert page.total is None
        assert page.next_offset is None
        assert page.next_cursor is None
        assert page.next_page is None
        assert not page.has_more
        assert page.query_time_ms is None
        assert page.metadata == {}

    def test_custom_values(self):
        """Test ResultPage with custom values."""
        records = [{"id": 1}, {"id": 2}]
        metadata = {"source": "test_db"}

        page = ResultPage(
            records=records,
            total=100,
            next_offset=20,
            next_cursor="xyz789",
            next_page=3,
            has_more=True,
            query_time_ms=150.5,
            metadata=metadata,
        )

        assert page.records == records
        assert page.total == 100
        assert page.next_offset == 20
        assert page.next_cursor == "xyz789"
        assert page.next_page == 3
        assert page.has_more
        assert page.query_time_ms == 150.5
        assert page.metadata == metadata
