<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/logo_light.png">
    <img src="docs/logo_light.png" alt="ChemSpace Copilot Logo" width="200"/>
  </picture>
</p>

<h1 align="center">ChemSpace Copilot</h1>

<p align="center">
  <strong>Multi-agent system for chemical space analysis</strong><br>
  <a href="https://laboratoire-de-chemoinformatique.github.io/chemspacecopilot/">Documentation</a> ·
  <a href="https://chemrxiv.org/doi/full/10.26434/chemrxiv.15000527/v1">Preprint (ChemRxiv)</a>
</p>

<p align="center">
  <a href="https://github.com/Laboratoire-de-Chemoinformatique/chemspacecopilot/actions/workflows/ci.yml"><img src="https://github.com/Laboratoire-de-Chemoinformatique/chemspacecopilot/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/Laboratoire-de-Chemoinformatique/chemspacecopilot/commits/main"><img src="https://img.shields.io/github/last-commit/Laboratoire-de-Chemoinformatique/chemspacecopilot" alt="Last commit"></a>
  <a href="https://github.com/Laboratoire-de-Chemoinformatique/chemspacecopilot/stargazers"><img src="https://img.shields.io/github/stars/Laboratoire-de-Chemoinformatique/chemspacecopilot" alt="Stars"></a>
  <a href="https://github.com/Laboratoire-de-Chemoinformatique/chemspacecopilot/issues"><img src="https://img.shields.io/github/issues/Laboratoire-de-Chemoinformatique/chemspacecopilot" alt="Issues"></a>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11-blue" alt="Python 3.11"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Laboratoire-de-Chemoinformatique/chemspacecopilot" alt="License"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
  <a href="https://pre-commit.com/"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit" alt="Pre-commit"></a>
  <a href="https://docs.agno.com/"><img src="https://img.shields.io/badge/framework-Agno-purple" alt="Framework: Agno"></a>
  <a href="https://laboratoire-de-chemoinformatique.github.io/chemspacecopilot/"><img src="https://img.shields.io/badge/docs-mkdocs-blue" alt="Docs"></a>
</p>

> **Warning**
> This repository is under active development. APIs, agent behavior, and project structure may change without notice.

---

## Overview

ChemSpace Copilot is a multi-agent system powered by the [Agno](https://docs.agno.com/) framework. The default runtime team coordinates seven specialized AI agents for ChEMBL bioactivity download, unified GTM workflows, downstream chemoinformatics, report generation, small-molecule generation, peptide generation, and retrosynthetic planning. A separate robustness evaluation agent is available for analyzing prompt-robustness test outputs. The GTM engine is provided by [chemographykit](https://www.piwheels.org/project/chemographykit/).

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

## Features

- **7 Runtime Agents + 1 Evaluation Agent** — ChEMBL data download, unified GTM operations, chemoinformatics analysis, report generation, small-molecule generation, peptide WAE workflows, retrosynthetic planning, and robustness evaluation
- **Generative Topographic Mapping** — Dimensionality reduction and visualization of chemical space via [chemographykit](https://www.piwheels.org/project/chemographykit/)
- **Molecular and Peptide Generation** — LSTM autoencoder-based small-molecule generation plus peptide WAE generation, interpolation, and GTM-guided targeting
- **S3/MinIO Integration** — Session-scoped cloud storage with local filesystem fallback
- **Chainlit Interface** — WebSocket-based real-time chat with password authentication, file upload, and inline molecule rendering
- **Agentic Memory** — SQLite-backed agentic state and recent session history shared across agent workflows
- **Robustness Testing** — Framework for validating prompt variation handling with semantic similarity scoring

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Build containers
docker compose build chainlit-app

# Run (prompts for DEEPSEEK_API_KEY only when using the DeepSeek provider)
./docker-start.sh
```

Access the application at **http://localhost:8000**

See the [Docker guide](docs/getting-started/docker.md) for the full Docker deployment guide.

### Option 2: Local Installation

<details>
<summary><strong>Environment setup</strong></summary>

For file-based configuration, copy `.env.example` to `.env` in the project root:

```bash
# Required only for the default DeepSeek provider
DEEPSEEK_API_KEY=your-api-key-here

# Optional model overrides (otherwise .modelconf is used)
# MODEL_PROVIDER=deepseek
# MODEL_ID=deepseek-chat
# OLLAMA_HOST=http://localhost:11434

# Optional — S3/MinIO storage (set USE_S3=true only when you want remote storage)
USE_S3=false
# When enabled:
S3_ENDPOINT_URL=http://localhost:9000
MINIO_ACCESS_KEY=cs_copilot
MINIO_SECRET_KEY=chempwd123
ASSETS_BUCKET=chatbot-assets

# Optional — ChEMBL local MySQL (faster queries, offline use)
# Download dump: https://chembl.gitbook.io/chembl-interface-documentation/downloads
# CHEMBL_MYSQL_HOST=localhost
# CHEMBL_MYSQL_PORT=3306
# CHEMBL_MYSQL_USER=chembl
# CHEMBL_MYSQL_PASSWORD=
# CHEMBL_MYSQL_DATABASE=chembl_36
```

The repository also includes a tracked `.modelconf` file. Edit it if you want to switch from the default DeepSeek backend to a local Ollama model.

</details>

<details>
<summary><strong>Install dependencies</strong></summary>

```bash
uv sync
```

</details>

<details>
<summary><strong>S3/MinIO setup (optional)</strong></summary>

```bash
# Run the interactive setup script
python scripts/setup_s3.py

# Or start MinIO manually
docker run -d --name minio \
  -p 9000:9000 -p 9001:9001 \
  -v /mnt/data:/data \
  -e MINIO_ROOT_USER=cs_copilot \
  -e MINIO_ROOT_PASSWORD=chempwd123 \
  minio/minio server /data --console-address ":9001"
```

If the container already exists: `docker start minio`

</details>

<details>
<summary><strong>Optional Chainlit Persistence</strong></summary>

Chainlit persistence is disabled by default in `chainlit.toml`. Only set up PostgreSQL if you plan to enable Chainlit persistence manually.

```bash
docker run --name chainlit-pg -p 5432:5432 -d \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_DB=chainlit \
  postgres:16

export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/chainlit"
```

If the container already exists: `docker start chainlit-pg`

</details>

## Usage

### Chainlit App

```bash
uv run chainlit run chainlit_app.py -w
```

Notes:
- The bundled `chainlit.toml` currently has `[persistence] enabled = false`.
- The app sets a per-thread title from your first message; you can rename it in the UI.

### Jupyter Notebook

An example workflow is available in `notebooks/cs_copilot.ipynb`.

## Architecture

The system uses a **Factory Pattern + Registry** for agent creation. The default team orchestrator coordinates seven runtime agents, and an eighth agent is available separately for robustness analysis:

### Runtime Team

| Agent | Role |
|-------|------|
| **ChEMBL Downloader** | Downloads and filters bioactivity data from ChEMBL (REST API by default; optional [local MySQL backend](https://chembl.gitbook.io/chembl-interface-documentation/downloads)) |
| **GTM Agent** | Unified GTM operations: build, load, density analysis, activity landscapes, projection, and GTM sampling support |
| **Chemoinformatician** | Downstream chemoinformatics analysis including scaffold, similarity, clustering, and SAR workflows |
| **Report Generator** | Formats analysis results into reports and visual outputs |
| **Autoencoder** | Small-molecule generation via LSTM autoencoders, including standalone and GTM-guided modes |
| **Peptide WAE** | Peptide sequence generation, latent-space GTM workflows, and DBAASP-backed peptide activity landscapes |
| **SynPlanner** | Retrosynthetic planning and route visualization for target molecules |

### Separate Evaluation Agent

| Agent | Role |
|-------|------|
| **Robustness Evaluation** | Analyzes robustness test runs, score distributions, failures, and trends |


Agents share state via `session_state` and persist memory in SQLite. All file I/O goes through a unified S3/local storage abstraction.

For full architectural details, see the [documentation](https://laboratoire-de-chemoinformatique.github.io/chemspacecopilot/).

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Ensure code passes `pre-commit run --all-files`
4. Submit a pull request

See the [Contributing Guide](https://laboratoire-de-chemoinformatique.github.io/chemspacecopilot/contributing/) for code style conventions and detailed guidelines.
