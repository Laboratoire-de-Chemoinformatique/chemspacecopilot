# ChemSpace Copilot

**LLM-powered agent system for chemical-space analysis**

ChemSpace Copilot combines language-model coordination with deterministic
cheminformatics tools. Its default Agno team coordinates seven specialists for
ChEMBL retrieval, GTM workflows, downstream analysis, reporting, molecular
design, peptide design, and retrosynthesis. A separate robustness agent and a
single-agent ablation baseline are registered outside the production team.

## How it works

```text
Chainlit / CLI ──► Agno coordinator ──► specialist agents ──┐
                                                           │
External MCP client ──► MCP profiles and adapters ─────────┤
                                                           ▼
            skills + workflow contracts ──► deterministic tools
                                                           │
                                                           ▼
                                  run events + artifacts + storage
```

The Chainlit/CLI path uses the Agno coordinator as its reasoning engine. In MCP
mode the external client is the supervisor, and the server exposes only the
capabilities selected at startup. Both paths share the scientific toolkits,
catalog procedures, S3/local storage abstraction, and optional MLflow
observability.

Workflow events and checksummed artifacts are the reproducible source of
scientific state. Conversation history is session-local, and cross-session
agent memory is disabled.

Read the [Architecture Overview](architecture/overview.md), then continue with
[Agents](architecture/agents.md), [Skills and Workflows](architecture/catalogs.md),
or the [MCP server guide](mcp.md).

## Highlights

- Generative Topographic Mapping through
  [chemographykit](https://www.piwheels.org/project/chemographykit/)
- ChEMBL REST and optional local SQL backends
- Molecular and peptide generation with standalone and GTM-guided modes
- Event-backed workflow runs and integrity-checked artifacts
- S3/MinIO storage with a local filesystem fallback
- Chainlit streaming, file upload, and inline molecule rendering
- Prompt-robustness and multi-agent-versus-single-agent evaluation

## Quick start

Use the [Installation Guide](getting-started/installation.md) for a local
environment or the [Docker Deployment Guide](getting-started/docker.md) for the
containerized application.
