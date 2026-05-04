# Unit Tests - Code Validity

This directory contains unit tests that verify the **correctness** and **validity** of individual ChemSpace Copilot components.

## Purpose

Unit tests ensure that:
- Functions and classes behave according to their specifications
- Edge cases and errors are handled correctly
- APIs and interfaces work as documented
- Data transformations are accurate
- External integrations (ChEMBL, S3, Hugging Face) function properly

## Test Organization

```
unit/
├── test_autoencoder.py        # Autoencoder toolkit (download, initialization)
├── test_gtm_sampling.py       # GTM sampling helpers and utilities
├── test_s3_integration.py     # S3 storage operations
└── test_databases/
    ├── test_base.py          # Base database toolkit functionality
    └── test_chembl.py        # ChEMBL-specific toolkit functionality
```

## Test Files

### `test_autoencoder.py`

Tests the autoencoder toolkit, focusing on:
- Hugging Face model downloading
- Model file validation
- Error handling for missing/incomplete files
- Path and directory management

**Key test classes:**
- `TestAutoencoderDownload` - Model download from Hugging Face

**Example:**
```bash
uv run pytest tests/unit/test_autoencoder.py -v
```

### `test_gtm_sampling.py`

Tests GTM (Generative Topographic Mapping) sampling helpers:
- Dense node sampling
- Activity landscape node sampling
- Molecule-level activity ranking
- Coordinate-based sampling
- Data format conversions (DataFrame, SMILES)

**Key test functions:**
- `test_sample_dense_nodes_prioritizes_filtered_density`
- `test_sample_activity_landscape_nodes_infers_probability_column`
- `test_sample_by_coordinates_uses_lookup_table`

**Example:**
```bash
uv run pytest tests/unit/test_gtm_sampling.py -v
```

### `test_s3_integration.py`

Tests S3 storage operations:
- CSV file read/write
- Binary file operations (pickle)
- Gzipped file operations
- Context manager functionality

**Key test function:**
- `test_s3_operations` - Comprehensive S3 operation test

**Example:**
```bash
uv run pytest tests/unit/test_s3_integration.py -v
```

**Note:** Requires S3 credentials in `.env` file.

### `test_databases/`

Tests database toolkit functionality.

#### `test_base.py`

Tests the base database toolkit abstraction:
- Connection management
- Query building and execution
- Pagination (offset, cursor, page-based)
- Error handling and mapping
- DataFrame conversion
- Context manager support

**Key test classes:**
- `TestBaseDatabaseToolkit` - Core functionality
- `TestDBConfig` - Configuration management
- `TestQueryParams` - Query parameter handling
- `TestErrorHandling` - Error mapping
- `TestResultPage` - Result pagination

**Example:**
```bash
uv run pytest tests/unit/test_databases/test_base.py -v
```

#### `test_chembl.py`

Tests ChEMBL-specific functionality:
- API connection and authentication
- Resource-specific queries (activity, molecule, assay, target)
- Field mapping and data transformation
- Rate limiting and error handling
- Compound fetching workflow
- Dataset description

**Key test classes:**
- `TestChemblToolkit` - Core ChEMBL operations
- `TestChemblConnectionManagement` - Connection lifecycle
- `TestChemblRateLimiting` - Rate limit handling
- `TestChemblIntegration` - Full workflow tests

**Example:**
```bash
uv run pytest tests/unit/test_databases/test_chembl.py -v
```

## Running Unit Tests

### Run All Unit Tests

```bash
uv run pytest tests/unit/ -v
```

### Run Specific Test File

```bash
uv run pytest tests/unit/test_autoencoder.py -v
```

### Run Specific Test Class or Function

```bash
# Specific class
uv run pytest tests/unit/test_databases/test_base.py::TestBaseDatabaseToolkit -v

# Specific test
uv run pytest tests/unit/test_autoencoder.py::TestAutoencoderDownload::test_download_from_huggingface_when_files_missing -v
```

### Run with Coverage

```bash
uv run pytest tests/unit/ --cov=src/cs_copilot --cov-report=html
```

Coverage report will be in `htmlcov/index.html`.

### Run with Output Capture Disabled

```bash
uv run pytest tests/unit/test_s3_integration.py -v -s
```

Useful for debugging print statements.

## Test Characteristics

All unit tests follow these principles:

### 1. Fast Execution
- Each test runs in < 1 second
- Use mocks for slow operations (API calls, file I/O)
- Create minimal test data in-memory

### 2. Isolated
- Tests don't depend on each other
- Each test can run independently
- Use fixtures and mocks for external dependencies

### 3. Deterministic
- Same input always produces same output
- No randomness (or use fixed seeds)
- No time-dependent behavior (mock time if needed)

### 4. Focused
- Each test verifies one specific behavior
- Test names clearly describe what's being tested
- Arrange-Act-Assert pattern

## Writing New Unit Tests

### Test Structure

```python
def test_function_name_specific_behavior():
    """Test that function_name does X when given Y."""
    # Arrange - Set up test data and conditions
    input_data = ...
    expected_output = ...

    # Act - Execute the function being tested
    result = function_name(input_data)

    # Assert - Verify the result
    assert result == expected_output
```

### Using Mocks

```python
from unittest.mock import Mock, patch, MagicMock

# Mock a function return value
@patch('cs_copilot.tools.external_api')
def test_my_function(mock_api):
    mock_api.return_value = "expected"
    result = my_function()
    assert result == "expected"
    mock_api.assert_called_once()

# Mock a class method
mock_obj = Mock()
mock_obj.method.return_value = 42
```

### Fixtures

```python
import pytest

@pytest.fixture
def sample_dataframe():
    """Provide a sample DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3],
        'value': [10, 20, 30]
    })

def test_process_dataframe(sample_dataframe):
    result = process_dataframe(sample_dataframe)
    assert len(result) == 3
```

### Test Parametrization

```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_double(input, expected):
    assert double(input) == expected
```

## Testing Best Practices

### DO:
- ✅ Test both success and failure cases
- ✅ Test edge cases (empty input, None, large values)
- ✅ Use descriptive test names
- ✅ Keep tests simple and readable
- ✅ Mock external dependencies
- ✅ Test public APIs, not implementation details

### DON'T:
- ❌ Test multiple behaviors in one test
- ❌ Depend on test execution order
- ❌ Make real API calls or file I/O (use mocks)
- ❌ Use sleep() or time delays
- ❌ Leave commented-out code
- ❌ Test third-party library functionality

## Mocking External Dependencies

### ChEMBL API

```python
@patch.object(ChemblToolkit, "_ensure_client")
def test_chembl_query(mock_ensure_client):
    mock_client = Mock()
    mock_client.activity.filter.return_value.only.return_value = [
        {"activity_id": 1, "value": 10.0}
    ]
    mock_ensure_client.return_value = mock_client

    toolkit = ChemblToolkit()
    result = toolkit.query(...)
    assert len(result.records) == 1
```

### S3 Storage

```python
@patch("cs_copilot.storage.S3")
def test_file_save(mock_s3):
    mock_file = MagicMock()
    mock_s3.open.return_value.__enter__.return_value = mock_file

    # Test code that uses S3.open()
    save_to_s3("data.csv")

    mock_s3.open.assert_called_with("data.csv", "w")
```

### Hugging Face Hub

```python
@patch("huggingface_hub.snapshot_download")
def test_model_download(mock_download):
    def create_files(repo_id, cache_dir, local_dir, resume_download):
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "model.pt").touch()
        return str(local_dir)

    mock_download.side_effect = create_files

    # Test download logic
    download_model()
    assert mock_download.called
```

## Debugging Failed Tests

### View Full Output

```bash
uv run pytest tests/unit/test_autoencoder.py -v -s
```

### Run Single Test with Debugging

```bash
uv run pytest tests/unit/test_autoencoder.py::TestAutoencoderDownload::test_download_from_huggingface_when_files_missing -v -s --pdb
```

### Check Coverage for Specific Module

```bash
uv run pytest tests/unit/test_databases/ --cov=src/cs_copilot/tools/databases --cov-report=term-missing
```

### Run Only Failed Tests

```bash
uv run pytest --lf  # last failed
uv run pytest --ff  # failed first
```

## Environment Setup

### Required Dependencies

Unit tests require:
- `pytest` - Test framework
- `pytest-cov` - Coverage reporting
- All ChemSpace Copilot dependencies from `pyproject.toml`

Install with:
```bash
uv sync
```

### Optional: S3 Tests

To run S3 integration tests, create `.env` file:
```bash
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=your-bucket
```

Or skip S3 tests:
```bash
uv run pytest tests/unit/ -v -k "not s3"
```

## Continuous Integration

Unit tests run automatically on:
- Every push to any branch
- Every pull request
- Before merging to main

CI configuration in `.github/workflows/tests.yml`.

## Test Maintenance

### When to Update Tests

- ✅ When adding new features
- ✅ When fixing bugs (add regression test)
- ✅ When changing APIs or interfaces
- ✅ When refactoring (tests should still pass)

### When to Delete Tests

- ✅ When removing deprecated functionality
- ✅ When tests become redundant after refactoring
- ❌ Don't delete failing tests - fix them or fix the code

## Related Documentation

- **Robustness Tests:** See [../robustness/README.md](../robustness/README.md)
- **Main Test Suite:** See [../README.md](../README.md)
- **Contributing Guide:** See [../../README.md](../../README.md)

## Questions?

- **Test failures:** Check mock configuration and dependencies
- **Import errors:** Ensure `uv sync` has been run
- **Slow tests:** Check if external dependencies need mocking
- **Coverage gaps:** Run with `--cov-report=html` to see uncovered lines
