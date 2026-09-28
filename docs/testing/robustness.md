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

### Manuscript Reliability Mode

```bash
uv run python tests/robustness/robustness_minimal_example.py \
  --config tests/robustness/manuscript_reliability.yaml \
  --tier live --repetitions 3
```

This suite validates the four manuscript cases using explicit artifact and
tool-call acceptance criteria. A fixture-backed `frozen` tier is available for
larger repeatability runs; see
`tests/robustness/fixtures/manuscript/README.md` for its required snapshot and
checksum variables. Publication-facing reports are written below the run's
`reliability/` directory as normalized JSONL, a statistical summary with Wilson
intervals, an environment manifest, and a blinded human-review sheet.

### Multi-agent Architecture Ablation

Run the four frozen manuscript cases through both the seven-specialist team and
the flat all-tools agent:

```bash
uv run python tests/robustness/robustness_minimal_example.py \
  --config tests/robustness/manuscript_reliability.yaml \
  --tier frozen --system both --arm-order team-first \
  --n-variations 3 --repetitions 10 \
  --test frozen_case_1_seh_analysis \
  --test frozen_case_2_seh_generation \
  --test frozen_case_3_retrosynthesis \
  --test frozen_case_4_peptide_design
```

Repeat independent batches with `--arm-order single-agent-first` to reduce
temporal service bias. The cross-arm `comparison.md` and `comparison.json` pair
runs by case, prompt variant, and repetition. They report objective success with
Wilson 95% intervals, latency, tokens, tool use and errors, incorrect selection,
optional estimated cost, and the older prompt-robustness score as a secondary
metric.

The comparison isolates agentic structure, not scientific algorithms: both arms
use the same model configuration, prompts, frozen inputs, tool implementations,
and validators. The flat agent namespaces peptide tools as `peptide_*` so
same-named molecular and peptide methods remain available in one context.
Runs blocked before either arm executes by a missing common fixture or
prerequisite are listed as not evaluated and excluded from architecture success,
latency, token, and pairing metrics.

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
