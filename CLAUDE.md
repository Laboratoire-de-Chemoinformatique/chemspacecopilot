# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChemSpace Copilot is an LLM-powered agent system for GTM (Generative Topographic Mapping) based chemical space analysis. It integrates multiple AI agents that coordinate to download chemical data, build dimensionality reduction models, analyze molecular properties, and generate novel molecules using autoencoders.

## Development Setup

### Installation

```bash
# Install dependencies
uv sync
```

### Environment Configuration

Required environment variables (create `.env` file):

```bash
# Required - LLM API key
DEEPSEEK_API_KEY=your-api-key-here

# Optional - S3/MinIO storage (can be disabled with USE_S3=false)
USE_S3=true
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=cs_copilot
MINIO_SECRET_KEY=chempwd123
ASSETS_BUCKET=chatbot-assets

# Optional - ChEMBL local MySQL (auto-detected when CHEMBL_MYSQL_HOST is set)
# Download dump: https://chembl.gitbook.io/chembl-interface-documentation/downloads
# CHEMBL_MYSQL_HOST=localhost
# CHEMBL_MYSQL_PORT=3306
# CHEMBL_MYSQL_USER=chembl
# CHEMBL_MYSQL_PASSWORD=
# CHEMBL_MYSQL_DATABASE=chembl_36

# Optional - Postgres for Chainlit history
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/chainlit
```

### Running Application

```bash
# Run Chainlit (chat interface with history)
uv run chainlit run chainlit_app.py -w
```

### Docker Services (Optional)

```bash
# MinIO (S3-compatible storage)
docker run -d --name minio -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=cs_copilot -e MINIO_ROOT_PASSWORD=chempwd123 \
  minio/minio server /data --console-address ":9001"

# Or restart existing container
docker start minio

# PostgreSQL (for Chainlit history)
docker run --name chainlit-pg -p 5432:5432 -d \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_USER=postgres \
  -e POSTGRES_DB=chainlit postgres:16

# Or restart existing container
docker start chainlit-pg
```

## Testing

### Quick Validation

```bash
# Test basic infrastructure (3 seconds)
uv run python test_simple.py

# Test single prompt robustness (15 seconds)
uv run python test_robustness_minimal.py
```

### Unit Tests

```bash
# Run all unit tests
uv run pytest tests/unit/ -v

# Run specific test file
uv run pytest tests/unit/test_gtm_sampling.py -v

# Run with coverage
uv run pytest tests/unit/ --cov=src/cs_copilot --cov-report=html
```

### Robustness Tests

```bash
# Run robustness tests (tests prompt variation handling)
uv run python tests/robustness/test_chembl_interactivity.py

# Run via pytest
uv run pytest tests/robustness/ -v

# Config-driven test runner
uv run python tests/robustness/robustness_minimal_example.py --test chembl_download --n-variations 3

# List available tests
uv run python tests/robustness/robustness_minimal_example.py --list-tests
```

**Session Isolation**: All robustness tests use complete session isolation:
- Each prompt variation creates a fresh agent team with memory disabled
- Unique S3 storage prefix per variation prevents file conflicts
- No state leakage between test runs
- Tests measure true prompt robustness, not memory effects

**Important**: Robustness tests can be long-running. Use timeouts:

```bash
timeout 600 uv run python tests/robustness/test_chembl_interactivity.py
```

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files

# Skip hooks for a commit (not recommended)
git commit --no-verify
```

Pre-commit runs:
- Black (code formatting on `src/` and `tests/` only)
- File checks (trailing whitespace, large files, merge conflicts)
- Unit tests (`tests/unit/`)

### Linting and Formatting

```bash
# Format code with Black
uv run black src/ tests/

# Run Ruff linter
uv run ruff check src/ tests/ --fix

# Sort imports
uv run isort src/ tests/
```

## Architecture Overview

### System Layers

```
┌─────────────────────────────────────────┐
│  UI Layer (Chainlit)                    │  Entry point
├─────────────────────────────────────────┤
│  Agent Orchestration (teams.py)         │  Multi-agent coordination
├─────────────────────────────────────────┤
│  Specialized Agents (factories.py)      │  8 domain-specific agents
├─────────────────────────────────────────┤
│  Tools + Storage (toolkits + S3)        │  Domain logic & persistence
└─────────────────────────────────────────┘
```

### Agent System

**Location**: `src/cs_copilot/agents/`

The system uses a **Factory Pattern + Registry** for agent creation:

- `factories.py` - 8 factory classes (ChEMBLDownloaderFactory, GTMOptimizationFactory, etc.)
- `registry.py` - Dynamic agent registry with auto-discovery
- `teams.py` - Multi-agent team coordination using Agno framework
- `prompts.py` - Agent instructions and system prompts

**8 Specialized Agents**:

1. **ChEMBL Downloader** - Downloads bioactivity data from ChEMBL database
2. **GTM Optimization** - Builds/optimizes Generative Topographic Maps
3. **GTM Density Analysis** - Analyzes compound distributions on GTM maps
4. **GTM Activity Analysis** - Creates activity-density landscapes for SAR
5. **GTM Loading** - Loads pre-existing GTM models from storage
6. **GTM Chemotype Analysis** - Analyzes scaffold distributions and chemotypes
7. **Autoencoder** - Molecular generation via LSTM autoencoders (sampling, interpolation)
8. **Autoencoder GTM Sampling** - Combines autoencoder with GTM for targeted generation

### Tools System

**Location**: `src/cs_copilot/tools/`

Tools are organized as **Toolkit classes** that inherit from `Toolkit` (Agno framework):

```
tools/
├── databases/          Database integrations
│   ├── base.py        BaseDatabaseToolkit (abstract)
│   ├── chembl.py      ChemblToolkit (ChEMBL REST API)
│   └── types.py       Query types and configurations
│
├── chemography/       Dimensionality reduction
│   ├── gtm.py         GTMToolkit (high-level interface)
│   └── gtm_operations.py  Core GTM implementations
│
├── chemistry/         Molecular operations
│   ├── similarity_toolkit.py      Similarity calculations
│   ├── autoencoder_toolkit.py     LSTM autoencoder operations
│   └── descriptors.py             Molecular descriptors
│
├── io/                I/O and formatting
│   ├── pointer_pandas_tools.py   DataFrame ops + S3 integration
│   └── formatting.py              SMILES → images, markdown
│
└── constants.py       Configuration constants
```

**Key Pattern**: Each toolkit registers methods as tools via `self.register(method)`. Agents call these tools via the Agno tool-calling mechanism.

### Storage System

**Location**: `src/cs_copilot/storage/`

Provides a **unified S3/local filesystem abstraction** with session-scoped storage:

```python
from cs_copilot.storage import S3

# Relative paths are session-scoped: sessions/{SESSION_ID}/results.csv
S3.open("results.csv", "w")

# Get full S3 URL for a relative path
S3.path("results.csv")  # → s3://bucket/sessions/{SESSION_ID}/results.csv

# Absolute S3 URLs work directly
S3.open("s3://bucket/data.csv", "r")

# Local absolute paths work too
S3.open("/tmp/data.csv", "r")
```

**Key Features**:
- **Session ID**: Auto-generated (timestamp + UUID) or env-configured
- **Backend Toggle**: S3/MinIO when `USE_S3=true`, local filesystem when `false`
- **Configuration Fallbacks**: Supports multiple env var names (MINIO_ENDPOINT, S3_ENDPOINT_URL, etc.)

### Agent Coordination

**Location**: `src/cs_copilot/agents/teams.py`

The `get_cs_copilot_agent_team()` creates a coordinated team:

```python
team = get_cs_copilot_agent_team(model)
# Creates Team with:
# - 8 specialized agents
# - Shared SqliteDb for memory persistence
# - Context management (num_history_runs=5)
# - Member interaction sharing
# - Streaming event propagation
```

**Capabilities**:
- **Multi-Agent Memory**: Session history persisted in SQLite
- **Context Sharing**: Agents access each other's outputs via `session_state`
- **Streaming**: Real-time event propagation from member agents to UI
- **Agentic Memory**: User preferences and past interactions remembered across runs

### Entry Point

**Chainlit** (`chainlit_app.py`):
- WebSocket-based real-time chat interface
- Per-session agent teams with authentication
- Tool call visualization as steps
- SMILES → inline molecule images
- Streaming response display
- PostgreSQL-backed chat history
- File upload support with S3 integration

## Common Workflows

### Adding a New Agent

1. Create a factory in `src/cs_copilot/agents/factories.py`:

```python
class MyNewAgentFactory(BaseAgentFactory):
    def create_agent(self, model, **kwargs):
        config = AgentConfig(
            name="my_new_agent",
            description="What this agent does",
            instructions="Detailed instructions here",
            tools=[MyToolkit(), ...],
            model=model,
            **kwargs
        )
        return self._create_agent(config)
```

2. The registry auto-discovers it via `AgentRegistry.discover_factories()`
3. Add to team in `teams.py` if needed

### Adding a New Tool

1. Create a toolkit in `src/cs_copilot/tools/`:

```python
from agno import Toolkit

class MyNewToolkit(Toolkit):
    def __init__(self):
        super().__init__(name="my_new_toolkit")
        self.register(self.my_tool_function)

    def my_tool_function(self, param: str) -> str:
        """Tool description for LLM."""
        return f"Result: {param}"
```

2. Import and pass to agent factory's `tools` parameter

### Running Prompt Robustness Tests

Robustness tests verify that semantically similar prompts produce consistent outputs:

1. Add prompt variations to `tests/robustness/fixtures/prompt_templates.yaml`
2. Configure tests in `tests/robustness/robustness_config.yaml`
3. Run: `uv run python tests/robustness/robustness_minimal_example.py --test my_test`

**Session Isolation**: Each prompt variation runs in complete isolation:
```python
# Each variation gets:
- Fresh agent team created from factory
- Disabled memory (enable_memory=False)
- Unique S3 prefix: sessions/robustness_{timestamp}_p{N}_v{M}_{uuid}/
- No access to previous runs or session state
```

This ensures tests measure robustness to prompt variation, not side effects from memory or shared state.

Results are saved to:
- Local: `tests/robustness/reports/<timestamp>/`
- S3: `sessions/{SESSION_ID}/robustness_tests/<test_name>/<timestamp>/`

## Key Files for Common Tasks

| Task | Files to Modify |
|------|-----------------|
| Add/modify agent | `src/cs_copilot/agents/factories.py`, `prompts.py` |
| Add new tool | `src/cs_copilot/tools/<subsystem>/` |
| Modify team coordination | `src/cs_copilot/agents/teams.py` |
| Change storage behavior | `src/cs_copilot/storage/client.py`, `config.py` |
| Update UI (Chainlit) | `chainlit_app.py` |
| Add robustness tests | `tests/robustness/fixtures/prompt_templates.yaml` |
| Configure tests | `tests/robustness/robustness_config.yaml` |

## Important Architectural Notes

### Agent State Management

Agents use `session_state` (a persistent dict) to pass data between runs and between agents:

```python
# Save in one agent
agent.session_state["data_path"] = "results.csv"

# Access in another agent (same team)
path = agent.session_state.get("data_path")
```

### S3 Integration Patterns

All file I/O should use the storage abstraction:

```python
from cs_copilot.storage import S3

# ✅ Good - session-scoped, works with S3 or local
with S3.open("output.csv", "w") as f:
    df.to_csv(f)

# ❌ Bad - hardcoded local path
with open("/tmp/output.csv", "w") as f:
    df.to_csv(f)
```

### Streaming Response Pattern

Chainlit uses streaming for real-time display:

```python
for chunk in agent.run(prompt, stream=True):
    if is_tool_event(chunk):
        display_as_step(chunk)  # Tool calls shown as Chainlit Steps
    elif is_text_chunk(chunk):
        stream_to_ui(chunk)     # Text streamed to message
```

This pattern allows users to see progress as agents work, rather than waiting for completion.

## Robustness Testing System

The `tests/robustness/` framework tests output consistency across prompt variations:

**Components**:
- `prompt_variations.py` - Generates semantically equivalent prompt variations
- `comparators.py` - Compares outputs (DataFrames, text, images, GTM models)
- `metrics.py` - Calculates robustness scores (0.0-1.0)
- `robustness_minimal_example.py` - Config-driven test runner
- `fixtures/prompt_templates.yaml` - 100+ prompt variations across 11 categories

**Robustness Score**:
```
Score = 0.4×Data + 0.3×Semantic + 0.2×Process + 0.1×Visual
```

**Thresholds**:
- Excellent: ≥ 0.90
- Good: ≥ 0.80
- Acceptable: ≥ 0.70
- Concerning: < 0.70

**Usage**: Run tests after modifying agent instructions to ensure behavior remains consistent.

## Known Issues and Workarounds

### Pytest Memory Issues (Exit Code 137)

Robustness tests can exhaust memory in pytest mode. Use standalone mode:

```bash
# ✅ Standalone mode (recommended)
uv run python tests/robustness/test_chembl_interactivity.py

# ❌ Pytest mode (may OOM)
uv run pytest tests/robustness/test_chembl_interactivity.py
```

### Long-Running Tests

Robustness tests involve multiple LLM API calls. Use timeouts:

```bash
timeout 600 uv run python tests/robustness/test_chembl_interactivity.py
```

### S3 Optional for Testing

S3/MinIO is optional. Disable for simpler testing:

```bash
export USE_S3=false
uv run python tests/robustness/test_chembl_interactivity.py
```

Results will only be shown in console (not persisted).

## Code Style and Conventions

- **Formatting**: Black with 100-character line length (auto-applied by pre-commit)
- **Imports**: isort with Black profile
- **Linting**: Ruff (E, W, F, I, B, C4 rules)
- **Python Version**: 3.11 only (specified in pyproject.toml)
- **Type Hints**: Encouraged but not strictly enforced
- **Docstrings**: Required for toolkit methods (visible to LLM as tool descriptions)

**Agent Instructions**: Written in `prompts.py` should be:
- Clear and specific (LLMs will follow them literally)
- Include examples where helpful
- Specify output format expectations
- Reference available tools explicitly

## Dependencies

**Core**:
- `agno` - Agent framework (orchestration, memory, tool calling)
- `rdkit` - Chemistry toolkit (molecule handling, descriptors)
- `torch` - Deep learning (autoencoder models)
- `chemographykit` - Generative Topographic Mapping ([piwheels](https://www.piwheels.org/project/chemographykit/))
- `chembl-webresource-client` - ChEMBL API client

**UI**:
- `chainlit` - WebSocket-based chat interface with authentication, PostgreSQL history, and file upload

**Storage**:
- `s3fs` - S3 filesystem integration
- `lancedb` - Vector database (for embeddings)

**Testing**:
- `pytest` - Unit and integration tests
- `sentence-transformers` - Semantic similarity for robustness tests
- `scikit-image`, `imagehash` - Image comparison for plot validation

See `pyproject.toml` for full dependency list.

## CI/CD Integration

Pre-commit hooks run automatically on commit. For CI/CD pipelines:

```yaml
- name: Run Tests
  run: |
    export USE_S3=false  # Disable S3 for CI
    uv run pytest tests/unit/ -v --tb=short
  env:
    DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
```

Robustness tests are optional in CI due to long runtime and API costs.

## Additional Documentation

- `README.md` - Project overview and quick start
- `tests/robustness/README.md` - Robustness framework documentation
