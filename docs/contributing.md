# Contributing

Contributions are welcome. Create a focused branch, add tests with the change,
and keep documentation aligned with the owning source of truth.

## Development checks

```bash
uv run black src/ tests/
uv run ruff check src/ tests/
uv run isort src/ tests/
uv run pytest tests/unit/ -v
uv run --group docs mkdocs build --strict
```

The project targets Python 3.11, Black's 100-character line length, the Black
isort profile, and the configured Ruff rules. Toolkit method docstrings are
required because they are visible to language models and generated API
documentation.

## Agent descriptions and instructions

Agent text is split by responsibility:

- Add identity, role, and domain ownership to
  `src/cs_copilot/agents/descriptions.py`.
- Add cross-task routing, evidence, state, and output rules to
  `src/cs_copilot/agents/instructions.py`.
- Put reusable scientific procedures in `skills/<slug>/SKILL.md`.
- Put MCP-facing tool, permission, task, and artifact contracts in
  `workflow_catalog/<slug>/WORKFLOW.md`.

Do not duplicate long tool sequences across agent instructions, workflow
contracts, and plugins. See [Skills and Workflows](architecture/catalogs.md)
for the ownership model.

## Extending the system

- [Adding an agent](architecture/agents.md#adding-an-agent)
- [Adding a tool](architecture/tools.md#adding-a-tool)
- [Changing a procedure](architecture/catalogs.md#changing-a-procedure)
- [MCP profiles and contracts](mcp.md#capability-profiles)

Before submitting a pull request, ensure the strict documentation build,
focused unit tests, and pre-commit checks pass.
