# ChemSpace Copilot Test Suite

This directory contains all tests for the ChemSpace Copilot project, organized by test type and purpose.

## Directory Structure

```
tests/
├── unit/                    # Unit/validity tests (code correctness)
│   ├── test_autoencoder.py
│   ├── test_gtm_sampling.py
│   ├── test_s3_integration.py
│   └── test_databases/
│       ├── test_base.py
│       └── test_chembl.py
│
└── robustness/             # Robustness tests (multi-agent system reliability)
    ├── test_pipeline_robustness.py
    ├── test_autoencoder_robustness.py
    ├── prompt_variations.py
    ├── comparators.py
    ├── metrics.py
    └── fixtures/
```

## Test Types

### 1. Unit Tests (`tests/unit/`)

**Purpose:** Verify that individual components work correctly according to their specifications.

**What they test:**
- Individual functions, classes, and modules
- API contracts and interfaces
- Error handling and edge cases
- Data validation and transformations
- Integration with external services (ChEMBL, S3, Hugging Face)

**When to run:**
- During development (continuously)
- Before committing code
- In CI/CD pipelines (always)

**Example:**
```bash
# Run all unit tests
uv run pytest tests/unit/ -v

# Run specific test module
uv run pytest tests/unit/test_autoencoder.py -v

# Run with coverage
uv run pytest tests/unit/ --cov=src/cs_copilot --cov-report=html
```

**Characteristics:**
- Fast execution (< 1 second per test)
- Isolated (use mocks for external dependencies)
- Deterministic (same input = same output)
- Focus on code validity and correctness

### 2. Robustness Tests (`tests/robustness/`)

**Purpose:** Assess how reliably ChemSpace Copilot's multi-agent system performs across semantically equivalent but syntactically different prompts.

**What they test:**
- Consistency of outputs given prompt variations
- Stability of the multi-agent workflow
- Agent decision-making under different phrasings
- Output quality across different invocations
- Prompt engineering robustness

**When to run:**
- Before major releases
- After significant prompt changes
- Weekly/periodically for monitoring
- When investigating user-reported inconsistencies

**Example:**
```bash
# Run all robustness tests (requires API key, takes longer)
uv run pytest tests/robustness/ -v

# Run specific robustness test
uv run pytest tests/robustness/test_pipeline_robustness.py::TestPipelineRobustness::test_full_pipeline_robustness -v

# Skip robustness tests in quick runs
uv run pytest tests/unit/ -v  # Only unit tests
```

**Characteristics:**
- Slower execution (10+ seconds per test, often minutes)
- Requires API keys (OpenAI, Anthropic, etc.)
- Stochastic (involves LLM calls)
- Focus on system reliability and consistency
- Generates detailed reports and artifacts

## Key Differences

| Aspect | Unit Tests | Robustness Tests |
|--------|-----------|------------------|
| **Goal** | Code correctness | System reliability |
| **Speed** | Fast (< 1s) | Slow (minutes) |
| **Dependencies** | Mocked | Real (LLM APIs) |
| **Determinism** | Deterministic | Stochastic |
| **Scope** | Single component | End-to-end workflow |
| **Run frequency** | Every commit | Periodically |
| **Cost** | Free | Costs API credits |

## Running Tests

### Quick Test (Unit Only)

```bash
uv run pytest tests/unit/ -v
```

### Full Test Suite

```bash
# Set API keys first
export OPENAI_API_KEY="your-key-here"

# Run all tests
uv run pytest tests/ -v
```

### Test Specific Components

```bash
# Database tests only
uv run pytest tests/unit/test_databases/ -v

# Molecular Designer/autoencoder-engine tests only (unit + robustness)
uv run pytest tests/unit/test_autoencoder.py tests/robustness/test_autoencoder_robustness.py -v

# GTM sampling tests
uv run pytest tests/unit/test_gtm_sampling.py -v
```

### Coverage Report

```bash
uv run pytest tests/unit/ --cov=src/cs_copilot --cov-report=html
# Open htmlcov/index.html in browser
```

## CI/CD Integration

In CI/CD pipelines:
1. **Always run unit tests** - They're fast and catch bugs early
2. **Conditionally run robustness tests** - Only on main branch or releases
3. **Monitor robustness scores over time** - Track degradation

Example GitHub Actions:
```yaml
- name: Run Unit Tests
  run: uv run pytest tests/unit/ -v

- name: Run Robustness Tests (main branch only)
  if: github.ref == 'refs/heads/main'
  run: uv run pytest tests/robustness/ -v
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## Adding New Tests

### Adding Unit Tests

1. Create test file in `tests/unit/` matching the module name:
   - Module: `src/cs_copilot/tools/chemistry/new_toolkit.py`
   - Test: `tests/unit/test_new_toolkit.py`

2. Follow naming conventions:
   ```python
   def test_function_name_behavior():
       """Test that function_name does X when Y."""
       # Arrange
       ...
       # Act
       ...
       # Assert
       ...
   ```

3. Use mocks for external dependencies:
   ```python
   from unittest.mock import patch, Mock

   @patch('module.external_api_call')
   def test_my_function(mock_api):
       mock_api.return_value = "expected"
       ...
   ```

### Adding Robustness Tests

1. Add prompt variations to `tests/robustness/fixtures/prompt_templates.yaml`:
   ```yaml
   prompts:
     your_new_scenario:
       base: "Base prompt description"
       variations:
         - "Variation 1"
         - "Variation 2"
         # ... 8 more
   ```

2. Create test in `tests/robustness/test_*_robustness.py`:
   ```python
   def test_new_scenario_robustness(
       self, agent_team, prompt_generator, comparator, metrics_calculator
   ):
       """Test robustness of new scenario."""
       variations = prompt_generator.get_variations("your_new_scenario", n=10)
       # ... run tests, collect outputs, compare, assert
   ```

3. See `tests/robustness/README.md` for detailed framework documentation.

## Test Data and Fixtures

- **Unit tests:** Use small, synthetic data created in-memory
- **Robustness tests:** Use fixtures defined in `tests/robustness/fixtures/`
- **Integration tests:** May use small real datasets from `data/` directory

## Troubleshooting

### Unit Tests Failing

1. Check imports after code moves
2. Verify mocks are correctly configured
3. Check for environment-specific issues (paths, OS)

### Robustness Tests Failing

1. Ensure API keys are set: `echo $OPENAI_API_KEY`
2. Check API rate limits
3. Review generated reports in `tests/robustness/reports/`
4. Examine artifacts in `tests/robustness/artifacts/`

### Import Errors

If you see import errors after moving tests:
```bash
# Reinstall package in development mode
uv sync

# Or explicitly add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

## Documentation

- **Unit Tests:** See individual test files and module docstrings
- **Robustness Tests:** See [tests/robustness/README.md](robustness/README.md)
- **Prompt Templates:** See [tests/robustness/fixtures/prompt_templates.yaml](robustness/fixtures/prompt_templates.yaml)

## Contributing

When contributing tests:

1. **Always write unit tests** for new code
2. **Consider robustness tests** for new agent workflows or prompts
3. **Follow naming conventions** (`test_*` files, `test_*` functions)
4. **Document test purpose** in docstrings
5. **Keep tests focused** (one concept per test)
6. **Avoid test interdependencies** (tests should run in any order)

## Questions?

- For unit test questions: Review existing tests in `tests/unit/`
- For robustness test questions: See `tests/robustness/README.md`
- For general testing best practices: See project contributing guide
