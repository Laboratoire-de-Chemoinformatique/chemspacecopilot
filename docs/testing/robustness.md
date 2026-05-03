# Robustness Tests

The `tests/robustness/` framework tests output consistency across prompt variations.

## Running Tests

### Standalone Mode (Recommended)

```bash
# Run as script (shows detailed output)
uv run python tests/robustness/test_chembl_interactivity.py

# With timeout
timeout 600 uv run python tests/robustness/test_chembl_interactivity.py
```

### Pytest Mode

```bash
uv run pytest tests/robustness/ -v

# Run specific test file
uv run pytest tests/robustness/test_chembl_interactivity.py -v
```

!!! warning
    Pytest mode may exhaust memory (exit code 137). Use standalone mode if this happens.

### Config-Driven Mode

```bash
# Run specific tests with custom variations
uv run python tests/robustness/robustness_minimal_example.py \
  --test chembl_download --n-variations 3

# List available tests
uv run python tests/robustness/robustness_minimal_example.py --list-tests
```

## Session Isolation

Each prompt variation runs in complete isolation:

- **Fresh Agent Teams**: Each variation creates a new agent team from scratch
- **Disabled Memory**: Agent memory is disabled (`enable_memory=False`)
- **Isolated S3 Storage**: Each variation gets a unique S3 prefix
- **No State Leakage**: Agents cannot see results from previous variations

This ensures tests measure robustness to prompt variation, not side effects from memory or shared state.

## Robustness Score

```
Score = 0.4 × Data + 0.3 × Semantic + 0.2 × Process + 0.1 × Visual
```

| Rating | Score |
|--------|-------|
| Excellent | >= 0.90 |
| Good | >= 0.80 |
| Acceptable | >= 0.70 |
| Concerning | < 0.70 |

## Framework Structure

```
tests/robustness/
├── test_chembl_interactivity.py       # ChEMBL clarification flow tests
├── test_pipeline_robustness.py        # Full pipeline robustness tests
├── test_autoencoder_robustness.py     # Molecular Designer autoencoder-engine tests
├── robustness_minimal_example.py      # Config-driven test runner
├── conftest.py                        # Shared pytest fixtures
├── test_utils.py                      # Core utilities
├── tool_tracker.py                    # Tool sequence tracking
├── config_schema.py                   # Configuration validation
├── prompt_variations.py               # Prompt variation generator
├── comparators.py                     # Output comparison utilities
├── metrics.py                         # Robustness scoring
├── robustness_config.yaml             # Test configuration
└── fixtures/
    └── prompt_templates.yaml          # Prompt variations database
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Tests hang | Add `timeout 300` wrapper |
| Exit code 137 (OOM) | Use standalone mode instead of pytest |
| "Database is locked" | Remove `.agno/` directory |
| MinIO not accessible | Set `USE_S3=false` |
