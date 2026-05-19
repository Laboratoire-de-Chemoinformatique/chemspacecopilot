# MCP server (optional)

ChemSpace Copilot ships an **optional** Model Context Protocol server that
exposes the same chemistry, chemography, ChEMBL, design, and reporting
toolkits as the Chainlit app — but driven by an external MCP client such as
[Codex](https://github.com/openai/codex) or
[Claude Code](https://docs.anthropic.com/en/docs/claude-code) instead of the
Agno multi-agent team.

In MCP mode the external client is the reasoning engine. The ChemSpace MCP
server only surfaces primitives — it never instantiates the Agno team, the
agent factories, or the configured model backend (DeepSeek / Ollama /
OpenRouter). The default Chainlit and CLI runtimes are unaffected.

## Install

The MCP server lives under `src/cs_copilot/mcp/`. It depends on the official
`mcp` Python SDK, which is gated behind an optional extra:

```sh
uv sync --extra mcp
# or, on a normal pip install
pip install "cs_copilot[mcp]"
```

Importing `cs_copilot` itself does not pull in `mcp` — the SDK is only
imported when you run the server.

## Run

The package installs a `cscopilot-mcp` console script that speaks stdio:

```sh
cscopilot-mcp --session-id my-session --workflow-slug chemical_space
```

Flags:

| Flag | Default | Effect |
|------|---------|--------|
| `--session-id` | auto-generated | Storage prefix used as the session root. |
| `--workflow-slug` | `workflow` | Workflow folder inside the session layout. |
| `--log-level` | `info` | Logger level (logs to stderr). |
| `--no-tools` | False | Skip MCP tool registration. |
| `--no-prompts` | False | Skip MCP prompt registration. |
| `--no-resources` | False | Skip session-artifact resources. |

You can also run the package directly: `python -m cs_copilot.mcp …`.

## Claude Code config

Add an entry under `mcpServers` in `~/.claude.json` (or `~/.claude/.mcp.json`,
depending on your Claude Code version):

```jsonc
{
  "mcpServers": {
    "chemspace": {
      "command": "cscopilot-mcp",
      "args": ["--session-id", "demo", "--workflow-slug", "chemical_space"],
      "env": { "SESSION_ID": "demo", "USE_S3": "false" }
    }
  }
}
```

A reference snippet lives at `examples/mcp/claude_code.json`.

## Codex config

Codex reads `~/.codex/config.toml`. Add a server entry like this:

```toml
[mcp_servers.chemspace]
command = "cscopilot-mcp"
args = ["--session-id", "demo", "--workflow-slug", "chemical_space"]

[mcp_servers.chemspace.env]
SESSION_ID = "demo"
USE_S3 = "false"
```

A reference snippet lives at `examples/mcp/codex.toml`.

## What the server exposes

### Tools

ChemSpace toolkit methods are wrapped one-to-one as MCP tools, namespaced
by toolkit (`chembl_*`, `gtm_*`, `chem_*`, `session_*`, `report_*`,
`robustness_*`). Tool arguments mirror the toolkit method signatures, with
the `agent` / `session_state` parameters injected by the server and hidden
from the public schema.

Use `cscopilot-mcp` once and then `list_tools` from your MCP client to see
the current set (there are ~50 tools).

### Prompts

The agent and team instruction sets from `cs_copilot.agents.prompts` are
exposed as MCP prompts so the client can adopt a ChemSpace "persona":

| Prompt | Role |
|--------|------|
| `chemspace_workflow` | Top-level orchestration. Use this first. |
| `chembl_agent` | ChEMBL data retrieval + validation workflow. |
| `gtm_agent` | GTM build / load / project / sample workflow. |
| `chemoinformatician_agent` | Scaffold, clustering, SAR analyses. |
| `molecular_designer_agent` | Autoencoder + LLM small-molecule design. |
| `peptide_designer_agent` | WAE + LLM peptide design. |
| `synplanner_agent` | Retrosynthetic planning. |
| `report_generator_agent` | Markdown / rich report generation. |
| `robustness_evaluation` | Review robustness test results. |
| `handling_new_files` | `<file>...</file>` sharing convention. |
| `chembl_retrieval_judge` | Parameterised judge prompt (target_query, keywords, organism_filter, items). |
| `chembl_metadata_judge` | Parameterised metadata judge prompt. |

### Resources

Session artifacts (datasets, plots, reports written by tools) are listed and
read via the `cscopilot://session/<rel_path>` URI scheme. Local and S3
backends are surfaced identically — the server delegates reads to
`cs_copilot.storage.S3`.

The synthetic `cscopilot://session/manifest.json` resource always exists and
describes the active session prefix and workflow layout version.

## ChEMBL LLM-as-judge in MCP mode

The Chainlit / CLI runtime runs an LLM-as-judge step on `chembl_fetch_compounds`
results to filter rows pulled in by short / ambiguous keywords. That step
uses the configured Agno model.

In MCP mode the judge is **disabled by default**: the external reasoner is
already in the loop, so the in-process judge would be redundant and would
also break the "the MCP server must not execute the Agno team" invariant.
The toolkit returns the unfiltered rows and reports
`judge_status: "disabled"` in the summary.

To recreate the equivalent filtering with the MCP client's own reasoning,
fetch the `chembl_retrieval_judge` / `chembl_metadata_judge` prompts. They
accept `target_query`, `keywords`, `organism_filter`, and `items` arguments
and return the exact template the in-process judge would have used.

## Architectural invariants

The MCP package is intentionally isolated from the Agno team:

- `cs_copilot.mcp.*` never imports `cs_copilot.agents.teams`,
  `cs_copilot.agents.factories`, `cs_copilot.agents.registry`,
  `cs_copilot.model_config`, or `chainlit_app`.
- Importing `cs_copilot` or `cs_copilot.tools.*` does not require the `mcp`
  extra to be installed.
- The MCP server runs as one OS process per stdio session. Concurrent
  launches are independent — each one has its own `SESSION_ID`, S3 prefix,
  and shared `session_state`.

A guard test (`tests/unit/mcp/test_no_team_imports.py`) AST-walks the package
and fails CI if any of the forbidden imports reappear.
