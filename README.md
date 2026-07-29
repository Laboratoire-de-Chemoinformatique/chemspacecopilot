<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo_dark_readme.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/logo_light_readme.png">
    <img src="docs/logo_light_readme.png" alt="ChemSpace Copilot" width="720"/>
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

ChemSpace Copilot combines language-model coordination with deterministic
cheminformatics tools. The default [Agno](https://docs.agno.com/) team
coordinates seven specialists for ChEMBL retrieval, GTM workflows, downstream
analysis, reporting, molecular design, peptide design, and retrosynthesis. An
optional MCP server exposes the same scientific core to external reasoning
clients. A separate robustness agent and a single-agent ablation baseline are
registered outside the production team. The GTM engine is provided by
[chemographykit](https://www.piwheels.org/project/chemographykit/).

```text
Chainlit / CLI ──► Agno coordinator ──► 7 specialists ──┐
                                                       ├─► tools ─► events/artifacts
External client ──► MCP profiles and adapters ─────────┘
                         ▲                    ▲
                         └── skills/workflows ┘
```

## Features

- **7 Runtime Agents + Evaluation/Ablation Factories** — Seven production specialists, a separate robustness evaluator, and a controlled single-agent baseline
- **Generative Topographic Mapping** — Dimensionality reduction and visualization of chemical space via [chemographykit](https://www.piwheels.org/project/chemographykit/)
- **Molecular and Peptide Generation** — Molecular Designer small-molecule generation with autoencoder and LLM engines plus Peptide Designer generation with WAE and LLM engines, interpolation, and GTM-guided targeting
- **S3/MinIO Integration** — Session-scoped cloud storage with local filesystem fallback
- **Chainlit Interface** — WebSocket-based real-time chat with password authentication, file upload, and inline molecule rendering
- **Artifact-backed Runs** — Event-sourced workflow state, checksummed artifacts, and
  session-local chat context; cross-session agent memory is disabled
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

To remove the docker-compose installation and rebuild it from scratch use:
```bash
docker compose down --volumes --remove-orphans --rmi all
```
WARNING! This would wipe all the previously generated assets on S3 and the previously run sessions of the chatbot.

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

SynPlanner and CGRtools are included in the default installation.

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

### MCP server (optional)

An optional Model Context Protocol server lets external MCP clients
(Codex, Claude Code) drive cs_copilot toolkits, prompts, and session
artifacts directly — the external client is the reasoning engine; the Agno
multi-agent team is not invoked. The default Chainlit and CLI runtimes are
unaffected.

```bash
uv sync --extra mcp

# Preflight the remote HTTP path ChatGPT/tunnels will call
cscopilot-mcp-check

# Local stdio MCP clients (Codex, Claude Code)
cscopilot-mcp --profile gtm-analysis --session-id demo --workflow-slug gtm-density-landscape

# Remote MCP clients (ChatGPT apps, browser-hosted clients)
cscopilot-mcp-serve --profile gtm-analysis --session-id demo --workflow-slug gtm-density-landscape --host 127.0.0.1 --port 8000
# Add --allowed-host <your-host> when a reverse proxy preserves the public Host header.

# Optional bearer-token protection for HTTP clients/proxies that send Authorization
CS_COPILOT_MCP_AUTH_TOKEN=change-me cscopilot-mcp-serve --profile standard \
  --session-id demo --workflow-slug chembl-to-gtm-report \
  --host 127.0.0.1 --port 8000 --allowed-host <your-host>
```

For ChatGPT, expose the streamable HTTP endpoint (`/mcp`) through a reachable
HTTPS URL or an approved secure tunnel. If a proxy forwards public Host or
Origin headers, pass them with `--allowed-host` / `--allowed-origin`. The
server includes read-only
`search` / `fetch` tools for ChatGPT data-only/deep-research compatibility,
plus the full cs_copilot tool catalog for full MCP developer-mode clients. Tool
descriptors include MCP `readOnlyHint` annotations so ChatGPT can distinguish
pure lookup tools from state-changing cs_copilot workflows; network-capable
tools also advertise `openWorldHint` for approval policy. The selected
startup profile is a strict discovery/invocation boundary; narrower profiles
include `chembl-retrieval`, `gtm-analysis`, `chemoinformatics`, `reporting`,
`molecular-design`, `peptide-design`, `retrosynthesis`, and `robustness`.
By default, LLM-dependent MCP tool paths create `llm_*` tasks for the external
client to complete. Trusted private deployments can pass
`--llm-policy agno-model` to load only the configured Agno model for toolkit
LLM calls, without enabling the Agno team.

Example client configs ship under `examples/mcp/`
(`claude_code.json`, `codex.toml`, `codex_http.toml`, `chatgpt_remote.md`,
`codex_subscription_smoke.md`, `secure_mcp_tunnel.md`). See `docs/mcp.md`
for the full tool / prompt / resource catalog and the ChEMBL LLM-as-judge
gating contract.

The repository also contains a Codex plugin under
`plugins/chemspace-copilot/`. It packages the standard MCP profile and a thin
bootstrap skill; scientific procedures remain in `skills/` and
`workflow_catalog/`, not in the plugin. This is intentionally a repo-local
marketplace delivery surface (`.agents/plugins/marketplace.json`), not Python
wheel content. Before installing it in Codex, install the MCP extra
(`uv sync --extra mcp`, or `pip install "cs_copilot[mcp]"`) and ensure
`cscopilot-mcp` is on the environment's `PATH`. Then open that marketplace
file in Codex using its absolute checkout path and install
`chemspace-copilot`; a portable deep link cannot be committed because every
clone has a different local path.

## Architecture

The agent registry contains nine factories: seven production team members, a
separate robustness evaluator, and a single-agent architecture-ablation
baseline. The Chainlit/CLI path uses the Agno coordinator as its reasoning
engine; in MCP mode, the external client supervises capability-filtered tools.

### Runtime Team

| Agent | Role |
|-------|------|
| **ChEMBL Downloader** | Downloads and filters bioactivity data from ChEMBL (REST API by default; optional [local MySQL backend](https://chembl.gitbook.io/chembl-interface-documentation/downloads)) |
| **GTM Agent** | Unified GTM operations: build, load, density analysis, activity landscapes, projection, and GTM sampling support |
| **Chemoinformatician** | Downstream chemoinformatics analysis including scaffold, similarity, clustering, and SAR workflows |
| **Report Generator** | Formats analysis results into reports and visual outputs |
| **Molecular Designer** | Small-molecule design via autoencoder and LLM engines, including standalone and GTM-guided modes |
| **Peptide Designer** | Peptide design via WAE and LLM engines, latent-space GTM workflows, and DBAASP-backed peptide activity landscapes |
| **SynPlanner** | Retrosynthetic planning and route visualization for target molecules |

### Separate Evaluation Agent

| Agent | Role |
|-------|------|
| **Robustness Evaluation** | Analyzes robustness test runs, score distributions, failures, and trends |

### Single-Agent Baseline

The `single_agent` factory combines the production tool surface into one flat
agent for controlled multi-agent-versus-single-agent robustness experiments. It
is not part of the production team.


Agents share within-session state via `session_state` and can persist bounded
conversation history in SQLite. Cross-session user and agentic memories are
disabled. The default Chainlit and CLI team uses structured but process-local
ad-hoc handoffs. When a v2 `RunContext` is supplied—or an MCP client drives the
v2 lifecycle—scientific run state is stored as replayable events and
checksummed artifacts through the unified S3/local storage abstraction. Strict
persisted task-DAG enforcement is currently the `chembl-to-gtm-report` MCP
pilot; other catalog workflows remain taskless/legacy.

For full architectural details, see the
[Architecture Overview](docs/architecture/overview.md) and
[Skills and Workflows](docs/architecture/catalogs.md).

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Ensure code passes `pre-commit run --all-files`
4. Submit a pull request

See the [Contributing Guide](https://laboratoire-de-chemoinformatique.github.io/chemspacecopilot/contributing/) for code style conventions and detailed guidelines.
