# ChemSpace Copilot

**LLM-powered agent system for GTM-based chemical space analysis**

ChemSpace Copilot is a multi-agent system powered by the [Agno](https://docs.agno.com/) framework. The default runtime team coordinates seven specialized AI agents for ChEMBL bioactivity download, unified GTM workflows, downstream chemoinformatics, report generation, small-molecule design, peptide generation, and retrosynthetic planning. A separate robustness evaluation agent is available for analyzing prompt-robustness test outputs. The GTM engine is provided by [chemographykit](https://www.piwheels.org/project/chemographykit/).

## Features

- **7 Runtime Agents + 1 Evaluation Agent** — ChEMBL data download, unified GTM operations, chemoinformatics analysis, report generation, small-molecule design, peptide design workflows, retrosynthetic planning, and robustness evaluation
- **Generative Topographic Mapping** — Dimensionality reduction and visualization of chemical space via chemographykit
- **Molecular and Peptide Generation** — Molecular Designer small-molecule generation with autoencoder and LLM engines plus Peptide Designer generation with WAE and LLM engines, interpolation, and GTM-guided targeting
- **S3/MinIO Integration** — Session-scoped cloud storage with local filesystem fallback
- **Chainlit Interface** — WebSocket-based real-time chat with password authentication, file upload, and inline molecule rendering
- **Agentic Memory** — SQLite-backed agentic state and recent session history shared across agent workflows
- **Robustness Testing** — Framework for validating prompt variation handling with semantic similarity scoring

## Architecture

```
┌─────────────────────────────────────────┐
│  UI Layer (Chainlit)                    │  Real-time chat interface
├─────────────────────────────────────────┤
│  Agent Orchestration (teams.py)         │  Multi-agent coordination
├─────────────────────────────────────────┤
│  Specialized Agents (factories.py)      │  7 runtime agents + 1 evaluation agent
├─────────────────────────────────────────┤
│  Tools + Storage (toolkits + S3)        │  Domain logic & persistence
└─────────────────────────────────────────┘
```

## Quick Start

Get started with the [Installation Guide](getting-started/installation.md) or the [Docker Deployment Guide](getting-started/docker.md).
