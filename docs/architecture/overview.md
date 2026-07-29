# Architecture Overview

ChemSpace Copilot has two execution paths over the same scientific tools and
artifact storage. The important difference is which component performs the
reasoning and coordination.

## System map

```text
                 ┌──────────────────────┐
                 │ Chainlit UI / CLI    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Agno coordinator     │
                 │ + 7 specialists     │
                 └──────────┬───────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        │            shared core                │
        ▼                                       ▼
┌──────────────────┐                  ┌──────────────────────┐
│ Skills catalog   │                  │ Deterministic tools  │
│ Workflow catalog │                  │ + policy checks      │
└────────┬─────────┘                  └──────────┬───────────┘
         │                                       │
         └───────────────────┬───────────────────┘
                             ▼
                 ┌────────────────────────┐
                 │ Run events + artifacts │
                 │ S3 / local storage     │
                 └────────────────────────┘
                             ▲
                             │
                 ┌───────────┴───────────┐
                 │ MCP server            │
                 │ capability profiles   │
                 └───────────▲───────────┘
                             │
                 ┌───────────┴───────────┐
                 │ External MCP client   │
                 │ (reasoning engine)    │
                 └───────────────────────┘
```

The repository plugin is a delivery layer for the MCP path. It starts the
server and provides a bootstrap skill; it does not duplicate scientific
procedures or tool implementations.

## Execution paths

### Chainlit and CLI

`chainlit_app.py` and the CLI create the Agno team through
`get_cs_copilot_agent_team()`. The coordinator interprets the request and
delegates bounded tasks to seven specialists. Deterministic toolkit methods
perform database access, descriptor calculation, GTM operations, molecular or
peptide generation, retrosynthesis, and report export.

The default team has:

- SQLite-backed, session-local conversation history;
- a shared `session_state` for current-session working references;
- role-specific tool allowlists;
- structured handoffs with objectives, constraints, artifact references,
  acceptance criteria, and budgets;
- no cross-session user or agent memory; and
- streaming member and tool events for the Chainlit UI.

The default Chainlit and CLI constructors do not supply a v2 `RunContext`.
Their handoffs are validated but remain process-local. A caller that explicitly
supplies a `RunContext` also receives pinned task-contract validation and
durable handoff events.

### MCP

The optional MCP server exposes capability-filtered cs_copilot tools, prompts,
and resources to clients such as Codex, Claude Code, and ChatGPT. The external
client is the supervisor and reasoning engine; the server does not start the
Agno team unless the separately gated compatibility tool is enabled.

The selected MCP profile is both a discovery and invocation boundary. The
adapter applies tool annotations, policy checks, normalized errors, execution
limits, task scope, idempotency, and artifact registration around the same
underlying scientific toolkits used by the Agno agents.

See [MCP server](../mcp.md) for transports, profiles, tools, and deployment.

## Procedures and scientific state

Reusable procedures live in `skills/*/SKILL.md`. MCP-facing workflow contracts
live in `workflow_catalog/*/WORKFLOW.md`. Both catalogs are parsed and validated
by Python registries; their frontmatter declares versions, dependencies,
permissions, profiles, and artifact contracts.

The v2 workflow runtime snapshots the selected workflow contract, records
immutable events, and registers checksummed artifacts under a run-scoped
layout. Events are the replayable source of truth; manifests and artifact
indexes are rebuildable views.

Strict persisted task-DAG enforcement currently applies only to the
`chembl-to-gtm-report` MCP pilot. Other catalog workflows retain taskless
execution while still using run and artifact contracts. See
[Agentic Runtime v2](agentic-runtime-v2.md) for the lifecycle and integrity
model, and [Skills and Workflows](catalogs.md) for catalog ownership.

## Observability

`src/cs_copilot/tracking/` provides optional MLflow tracking for sessions,
agent runs, prompts, and tool calls. Tracking is observational: it does not
replace workflow events or artifact storage as scientific run state.
