# Prompt Robustness Testing Framework

This directory contains automated tests to assess the robustness of the ChemSpace Copilot pipeline to prompt variations.

## Overview

The framework tests whether semantically equivalent but syntactically different prompts yield consistent outputs across:
- Individual agents (ChEMBL download, GTM optimization, density analysis, etc.)
- The full end-to-end pipeline
- Molecular Designer autoencoder-engine operations (sampling, interpolation, latent exploration)

**Key Feature**: Each prompt variation runs in a **completely isolated session** with:
- Fresh agent teams (no shared state)
- Disabled memory (no recall of previous runs)
- Unique S3 storage prefixes (no file conflicts)
- Independent execution contexts

This ensures tests measure true robustness to prompt variation, not side effects from shared state or memory.

## Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv add --dev sentence-transformers Pillow ImageHash scikit-image PyYAML

# Or install from requirements file
uv add --dev $(cat tests/robustness/requirements.txt | grep -v "#" | xargs)
```

### 2. Configure Tests

Edit `robustness_config.yaml` to enable/disable tests and configure parameters:

```yaml
general:
  n_variations: 5
  debug_mode: false

tests:
  chembl_download:
    enabled: true
  autoencoder_sampling:
    enabled: true
```

### 3. Run the Robustness Runner (Recommended)

The config-driven runner is the primary way to run robustness tests:

```bash
# Run with default configuration
uv run python tests/robustness/robustness_minimal_example.py

# Run with custom configuration file
uv run python tests/robustness/robustness_minimal_example.py --config custom_config.yaml

# Run specific tests (overrides config)
uv run python tests/robustness/robustness_minimal_example.py --test chembl_download --test autoencoder_sampling

# Run with fewer variations (for quick testing)
uv run python tests/robustness/robustness_minimal_example.py --test chembl_download --n-variations 3

# Enable debug mode
uv run python tests/robustness/robustness_minimal_example.py --debug

# List available tests
uv run python tests/robustness/robustness_minimal_example.py --list-tests

# List available prompt categories
uv run python tests/robustness/robustness_minimal_example.py --list-prompts
```

### 4. Alternative: Run pytest-based Tests

```bash
# Run all robustness tests via pytest
uv run pytest tests/robustness/ -v

# Run specific test class
uv run pytest tests/robustness/test_pipeline_robustness.py::TestPipelineRobustness -v

# Run ChEMBL clarification flow test (requires API key + network access)
uv run pytest tests/robustness/test_chembl_interactivity.py -v

# Run with coverage
uv run pytest tests/robustness/ --cov=src/cs_copilot --cov-report=html
```

### 5. View Reports

Reports are generated in `tests/robustness/reports/<timestamp>/`:
- `report.md` - Comprehensive markdown robustness report
- `summary.json` - JSON summary for programmatic access
- `<test_name>/` - Per-test artifacts and detailed results

**NEW:** Test results are also automatically saved to S3 (when enabled):
- See [S3 Results Integration](S3_RESULTS_INTEGRATION.md) for full details
- Results saved in multiple formats (JSON, CSV, TXT) under session-scoped paths
- Example: `s3://{bucket}/sessions/{session_id}/robustness_tests/chembl_interactivity/{timestamp}/`

## Session Isolation

### How It Works

Every test variation creates a completely isolated execution environment:

```python
# Each variation gets:
1. Fresh agent team (no shared state between runs)
2. Disabled memory (enable_memory=False)
3. Unique S3 prefix (sessions/robustness_TIMESTAMP_pN_vM_UUID/)
4. Independent session ID
```

### Implementation Details

**Agent Isolation**:
- Tests use `agent_team_factory()` instead of reusing a single team
- Factory creates new teams with `enable_memory=False`
- No SQLite database persistence between runs

**S3 Isolation**:
- Each variation sets `S3Client.prefix = "sessions/{unique_id}"`
- Results saved to separate paths prevent overwrites
- Original prefix restored after each run

**Why This Matters**:
Without isolation, agents "remember" previous runs:
- ❌ `"Download ChEMBL data"` → Agent recalls your past queries
- ❌ File conflicts when multiple variations save to same path
- ❌ Session state leaks between supposedly independent tests

With isolation:
- ✅ Each variation is truly independent
- ✅ Tests are reproducible regardless of execution order
- ✅ Measures actual prompt robustness, not memory effects

### Disabling Isolation (Not Recommended)

If you need to test with shared state:

```python
# In test file
agent_team_factory = lambda: get_cs_copilot_agent_team(
    model=model,
    enable_memory=True,  # Enable shared memory
)

# Skip S3 isolation
run_test(..., s3_session_isolation=False)
```

## Framework Architecture

The robustness testing infrastructure consists of **shared utilities** (Phase 1-4 refactoring) and **specialized components** (comparators, metrics, etc.).

### Core Infrastructure (Shared Utilities)

Located in the root of `tests/robustness/`:

#### `conftest.py` - Shared Pytest Fixtures

Provides fixtures automatically available to all test files:

```python
def test_something(agent_team_factory, s3_session_manager, comparator):
    # Fixtures auto-injected by pytest - no imports needed!
    team = agent_team_factory()
    with s3_session_manager.create_isolated_session("test", 0, 0) as session_id:
        result = team.run("test prompt")
```

**Available Fixtures:**
- `model_loader` - Session-scoped model configuration loader (cached)
- `model` - LLM model instance (cached, reused across tests)
- `agent_team_factory` - Factory for creating fresh agent teams
- `s3_session_manager` - S3 session isolation with automatic cleanup
- `prompt_generator` - Prompt variation generator
- `comparator` - Output comparison utilities
- `metrics_calculator` - Robustness metrics calculator
- `response_parser` - Response parsing utilities (class)
- `test_validator` - Test validation utilities (class)

#### `test_utils.py` - Core Utilities

Centralized implementations to eliminate code duplication:

**ModelLoader** - Load and cache LLM models
```python
from test_utils import ModelLoader

loader = ModelLoader.from_config(Path("robustness_config.yaml"))
model = loader.load_model()  # Cached, validated
```

**S3SessionManager** - Safe S3 session isolation
```python
from test_utils import S3SessionManager

manager = S3SessionManager()
with manager.create_isolated_session("test", 0, 0) as session_id:
    # S3.prefix automatically set and isolated
    result = agent.run(prompt)
# Cleanup guaranteed via finally block, even on exceptions
```

**ResponseParser** - Extract information from responses
```python
from test_utils import ResponseParser

files = ResponseParser.extract_files(response_text)  # Set of file paths
smiles = ResponseParser.extract_smiles(response_text)  # List of SMILES
success = ResponseParser.check_success(response_text)  # Boolean
count = ResponseParser.extract_row_count(response_text)  # Optional[int]
```

**Benefits:**
- ~300 lines of duplication eliminated
- Single source of truth for common operations
- Automatic cleanup via context managers (no S3 state corruption)

#### `tool_tracker.py` - Tool Sequence Tracking

Compare tool call sequences across prompt variations:

```python
from tool_tracker import ToolSequenceComparator

# Extract sequences from agent responses
sequences = [
    ToolSequenceComparator.extract_tool_sequence(response)
    for response in responses
]

# Calculate similarity (0.0 to 1.0)
similarity = ToolSequenceComparator.compare_sequences(sequences)

# Analyze patterns
patterns = ToolSequenceComparator.analyze_sequence_patterns(sequences)
```

**Replaces:** Hardcoded `0.90` placeholder in `tool_sequence_similarity`

#### `config_schema.py` - Configuration Validation

Comprehensive validation for `robustness_config.yaml`:

```python
from config_schema import ConfigValidator

# Validate before loading
data = ConfigValidator.load_and_validate(Path("robustness_config.yaml"))

# Check dependency graph (detect circular dependencies)
execution_order = ConfigValidator.validate_dependencies(data)
```

**Validates:**
- Model configuration (provider, model_id, API key existence)
- Metrics configuration (required weights, threshold ordering)
- Test configuration (prompt keys, dependencies)
- General settings (value ranges, types)

**Catches errors early:**
```
Configuration validation failed:
  - [model] API key not found in environment variable: DEEPSEEK_API_KEY
  - [metrics] Missing required weight: data_similarity
  - [tests.my_test] references undefined prompt_key: invalid_key
```

### Migration to New Infrastructure

Existing tests can migrate incrementally. See [MIGRATION.md](MIGRATION.md) for details.

**Code reduction after migration:**
- ~300 lines of duplicated fixtures eliminated
- ~100 lines of S3 management code simplified
- ~50 lines of response parsing patterns consolidated
- **Total:** ~450 lines removed, replaced by reusable infrastructure

## Framework Components

### 1. Prompt Variations (`prompt_variations.py`)

Manages prompt templates and generates variations:

```python
from tests.robustness.prompt_variations import PromptVariationGenerator

generator = PromptVariationGenerator()
variations = generator.get_variations("full_pipeline", n=10)
```

**Available tests and prompt keys:**

| Test Name | Prompt Key | Description |
|-----------|------------|-------------|
| `full_pipeline` | `full_pipeline` | Complete ChemSpace Copilot workflow |
| `chembl_download` | `chembl_download` | ChEMBL data fetching |
| `gtm_optimization` | `gtm_optimization` | GTM model building |
| `density_analysis` | `density_analysis` | Density landscape analysis |
| `activity_analysis` | `activity_analysis` | Activity landscape analysis |
| `chemotype_analysis` | `chemotype_analysis` | Chemotype/scaffold analysis |
| `autoencoder_sampling` | `autoencoder_sampling` | Basic Molecular Designer autoencoder-engine generation |
| `gtm_guided_sampling` | `gtm_guided_sampling` | GTM-guided Molecular Designer generation |
| `interpolation` | `interpolation` | Molecule interpolation in latent space |
| `latent_exploration` | `latent_exploration` | Latent space neighborhood exploration |

### 2. Output Comparators (`comparators.py`)

Compares outputs across runs:

```python
from tests.robustness.comparators import OutputComparator

comparator = OutputComparator()

# Compare DataFrames
similarity = comparator.compare_dataframes([df1, df2, df3])

# Compare text responses
similarity = comparator.compare_text_outputs([text1, text2, text3])

# Compare images
similarity = comparator.compare_images([plot1_path, plot2_path])
```

**Metrics provided:**
- **DataFrames:** Jaccard similarity, column match, value stability, KS test
- **Text:** Semantic similarity, entity overlap, numeric consistency
- **Images:** Perceptual hash, SSIM
- **GTM models:** Structure match, projection correlation, density correlation

### 3. Metrics Calculator (`metrics.py`)

Aggregates metrics into robustness scores:

```python
from tests.robustness.metrics import RobustnessMetrics

calculator = RobustnessMetrics()
score = calculator.calculate_robustness_score(comparison_results)
report = calculator.generate_report(results_dict)
```

**Robustness Score Calculation:**
```
Score = 0.4 × Data_Similarity + 0.3 × Semantic_Similarity
        + 0.2 × Process_Consistency + 0.1 × Visual_Similarity
```

**Thresholds:**
- **Excellent:** ≥ 0.90
- **Good:** ≥ 0.80
- **Acceptable:** ≥ 0.70
- **Concerning:** < 0.70

## Adding New Prompt Variations

Edit `fixtures/prompt_templates.yaml`:

```yaml
prompts:
  your_new_category:
    base: "Your base prompt here"
    variations:
      - "First variation"
      - "Second variation"
      # ... 8 more
```

The framework validates that variations maintain semantic similarity (cosine > 0.70) to the base prompt.

### Selecting a Prompt Templates File

Set `CS_COPILOT_PROMPT_TEMPLATES` to point at an alternate YAML file:

```bash
export CS_COPILOT_PROMPT_TEMPLATES=tests/robustness/fixtures/prompt_templates_short.yaml
```

## S3 Results Storage

Robustness tests automatically save results to S3 when enabled. See [S3 Results Integration](S3_RESULTS_INTEGRATION.md) for comprehensive documentation.

### Quick Setup

```bash
# Enable S3 storage
export USE_S3=true
export S3_ENDPOINT_URL=https://your-minio-server:9000
export S3_BUCKET_NAME=chatbot-assets
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key

# Optional: Set custom session ID for result organization
export SESSION_ID=my-experiment-001
```

### Results Structure

```
s3://{bucket}/sessions/{session_id}/robustness_tests/
└── chembl_interactivity/
    └── 20251231_103045/
        ├── results.json    # Full test results with metadata
        ├── summary.csv     # Tabular data for analysis
        └── summary.txt     # Human-readable report
```

### Accessing Results

```python
from cs_copilot.storage import S3
import pandas as pd

# Read CSV summary
with S3.open("robustness_tests/chembl_interactivity/{timestamp}/summary.csv", "r") as f:
    df = pd.read_csv(f)
```

### Disable S3 (Local Only)

```bash
export USE_S3=false
# Or run with save_to_s3=False in code
```

## Configuration

### Custom Weights

```python
metrics = RobustnessMetrics(
    weights={
        "data_similarity": 0.5,
        "semantic_similarity": 0.3,
        "process_consistency": 0.15,
        "visual_similarity": 0.05,
    }
)
```

### Custom Thresholds

```python
metrics = RobustnessMetrics(
    thresholds={
        "excellent": 0.95,
        "good": 0.85,
        "acceptable": 0.75,
    }
)
```

## CI/CD Integration

Add to your GitHub Actions workflow:

```yaml
- name: Run Robustness Tests
  run: |
    uv run pytest tests/robustness/ -v --tb=short
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## Troubleshooting

### "Need at least 2 texts to compare"

Ensure your test generated multiple outputs. Check that the agent completed all variations.

### "Similarity validation disabled - no model loaded"

Install sentence-transformers:
```bash
uv add --dev sentence-transformers
```

### "Images have different shapes, skipping SSIM"

This is expected if plots have different dimensions. Perceptual hash comparison will still work.

## Configuration File

The `robustness_config.yaml` file controls all aspects of robustness testing:

### General Settings
```yaml
general:
  n_variations: 5          # Number of prompt variations per test
  debug_mode: false        # Verbose agent output
  output_dir: "reports"    # Where to save reports
  save_artifacts: true     # Save per-run details
  s3_session_isolation: true  # Isolate S3 storage per run
```

### Model Configuration
```yaml
model:
  provider: "deepseek"     # "deepseek", "openai", or "anthropic"
  model_id: "deepseek-chat"
  api_key_env: "DEEPSEEK_API_KEY"  # Environment variable name
```

### Metrics Configuration
```yaml
metrics:
  weights:
    data_similarity: 0.4
    semantic_similarity: 0.3
    process_consistency: 0.2
    visual_similarity: 0.1
  thresholds:
    excellent: 0.90
    good: 0.80
    acceptable: 0.70
  pass_threshold: 0.75
```

### Test Selection
```yaml
tests:
  chembl_download:
    enabled: true
    prompt_key: "chembl_download"
    description: "Test ChEMBL data fetching robustness"

  interpolation:
    enabled: true
    prompt_key: "interpolation"
    params:
      molecule_a: "CCO"
      molecule_b: "CCCCCCCCCC"
```

## Architecture

```
tests/robustness/
├── __init__.py                     # Package initialization
├── robustness_minimal_example.py   # Main config-driven runner script
├── robustness_config.yaml          # Configuration file
├── prompt_variations.py            # Prompt management
├── comparators.py                  # Output comparison
├── metrics.py                      # Robustness scoring
├── test_pipeline_robustness.py     # pytest-based end-to-end tests
├── test_autoencoder_robustness.py  # pytest-based autoencoder tests
├── test_chembl_interactivity.py    # pytest-based clarification flow tests
├── fixtures/
│   └── prompt_templates.yaml       # Prompt variations (100+ variations)
├── reports/                        # Generated reports (by timestamp)
│   └── <YYYYMMDD_HHMMSS>/
│       ├── report.md
│       ├── summary.json
│       └── <test_name>/
│           ├── comparison.json
│           └── run_<N>/
└── README.md                       # This file
```

## Future Enhancements

- [ ] LLM-based prompt generation
- [ ] Historical trend analysis
- [ ] Ablation studies (remove specific instructions)
- [ ] Cross-model comparisons (GPT-4 vs Claude vs DeepSeek)
- [ ] Fine-grained outlier analysis
- [ ] Automatic prompt optimization suggestions

## References

- [Sentence-BERT Paper](https://arxiv.org/abs/1908.10084)
- [Perceptual Hashing](http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html)

## Support

For issues or questions, please open an issue on the repository or contact the maintainers.
