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
# Add --extra synplanner only if you need the optional retrosynthesis backend
# and SynPlanner/CGRtools wheels are available for your platform.
# or, on a normal pip install
pip install "cs_copilot[mcp]"
```

Importing `cs_copilot` itself does not pull in `mcp` — the SDK is only
imported when you run the server.

## Readiness check

Before wiring ChatGPT or a tunnel, run the local preflight check:

```sh
cscopilot-mcp-check
```

The check starts a temporary streamable HTTP server, connects with the MCP
client, verifies the core ChemSpace tools, validates the MCP server
instructions and ChatGPT-facing tool annotations, lists prompts and resources,
and uses the ChatGPT-compatible `search` / `fetch` pair to fetch both
`chembl_fetch_compounds` documentation and the `chemspace_workflow`
orchestration prompt. It also prints ChatGPT connector metadata and a smoke
prompt with expected evidence for the final connector test. It prints the local
`http://127.0.0.1:<port>/mcp` endpoint that must be exposed as HTTPS or
attached to Secure MCP Tunnel for ChatGPT.

After exposing a real endpoint through HTTPS, a reverse proxy, or a tunnel URL
that is reachable from your shell, probe that exact URL before creating the
ChatGPT connector:

```sh
cscopilot-mcp-check --url https://<your-host>/mcp
```

For a pure Secure MCP Tunnel flow where ChatGPT selects **Tunnel** instead of
a public URL, `tunnel-client doctor --profile <profile> --explain` and a
healthy `tunnel-client run --profile <profile>` are the tunnel-side gates.

## Run

The package installs a `cscopilot-mcp` console script. By default it speaks
stdio for local MCP clients:

```sh
cscopilot-mcp --session-id my-session --workflow-slug chemical_space
```

For remote MCP clients such as ChatGPT apps, use streamable HTTP or SSE:

```sh
# Streamable HTTP (recommended default for remote clients)
cscopilot-mcp-serve --session-id demo --workflow-slug chemical_space --host 127.0.0.1 --port 8000
# Equivalent explicit form:
cscopilot-mcp --transport streamable-http --session-id demo --workflow-slug chemical_space --host 127.0.0.1 --port 8000

# SSE fallback for clients that still require it:
cscopilot-mcp --transport sse --session-id demo --workflow-slug chemical_space --host 127.0.0.1 --port 8000
```

The default streamable HTTP URL is `http://127.0.0.1:8000/mcp`. The default
SSE URL is `http://127.0.0.1:8000/sse`.

The MCP SDK enables DNS-rebinding protection for localhost binds. If an HTTPS
reverse proxy preserves the public `Host` header while forwarding to a local
`cscopilot-mcp-serve`, allow that public host explicitly:

```sh
cscopilot-mcp-serve --host 127.0.0.1 --port 8000 \
  --allowed-host <your-host> \
  --allowed-origin https://chatgpt.com
```

For public binds such as `--host 0.0.0.0`, pass the expected public host and
origin instead of leaving DNS-rebinding protection implicit. Use
`--disable-dns-rebinding-protection` only behind a trusted tunnel or proxy that
performs equivalent Host and Origin checks.

Flags:

| Flag | Default | Effect |
|------|---------|--------|
| `--transport` | `stdio` (`streamable-http` for `cscopilot-mcp-serve`) | Serve over `stdio`, `sse`, or `streamable-http`. |
| `--host` | `127.0.0.1` | Host for HTTP transports. Use `0.0.0.0` only behind trusted access control. |
| `--port` | `8000` | Port for HTTP transports. |
| `--mount-path` | `/` | Mount path used when composing SSE endpoints. |
| `--sse-path` | `/sse` | SSE endpoint path. |
| `--message-path` | `/messages/` | SSE POST message endpoint path. |
| `--streamable-http-path` | `/mcp` | Streamable HTTP endpoint path. |
| `--json-response` | False | Use JSON responses where streamable HTTP supports them. |
| `--stateless-http` | False | Create a fresh streamable-HTTP transport per request. |
| `--allowed-host` | none | Host header allowed by MCP DNS-rebinding protection. Repeat for multiple public/proxy hosts. |
| `--allowed-origin` | none | Origin header allowed by MCP DNS-rebinding protection. Repeat for multiple browser/client origins. |
| `--disable-dns-rebinding-protection` | False | Disable MCP Host/Origin checks. Use only behind a trusted proxy or tunnel with equivalent controls. |
| `--auth-token-env` | `CS_COPILOT_MCP_AUTH_TOKEN` | Environment variable containing a required HTTP bearer token. Auth is disabled when the variable is unset. |
| `--auth-token` | unset | Direct bearer token value. Prefer `--auth-token-env` so secrets do not appear in process listings. |
| `--auth-client-id` | `chemspace-mcp-client` | Client id attached to accepted static bearer tokens. |
| `--auth-scope` | none | Required bearer-token scope. Can be supplied multiple times. |
| `--auth-issuer-url` | endpoint URL | Issuer URL advertised in MCP protected-resource metadata. |
| `--auth-resource-url` | endpoint URL | Public MCP resource URL advertised in auth metadata. Set this to the HTTPS URL seen by remote clients. |
| `--session-id` | auto-generated | Storage prefix used as the session root. |
| `--workflow-slug` | `workflow` | Workflow folder inside the session layout. |
| `--log-level` | `info` | Logger level (logs to stderr). |
| `--no-tools` | False | Skip MCP tool registration. |
| `--no-chatgpt-compat` | False | Skip the read-only `search` / `fetch` tools. |
| `--no-prompts` | False | Skip MCP prompt registration. |
| `--no-resources` | False | Skip session-artifact resources. |

The `cscopilot-mcp-check` command accepts `--url`, `--host`, `--port`,
`--path`, `--session-id`, `--workflow-slug`, `--timeout`, `--log-level`,
`--use-s3`, `--json`, `--auth-token-env`, `--auth-token`,
`--auth-client-id`, repeatable `--auth-scope`, and repeatable
`--required-tool` flags for preflight checks.
If a bearer token is configured, the check protects the temporary server and
connects with `Authorization: Bearer <token>`.

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

## HTTP MCP client config

For HTTP-capable MCP clients such as Codex, expose `cscopilot-mcp-serve` over
HTTPS and configure the client with the streamable HTTP URL. If you set
`CS_COPILOT_MCP_AUTH_TOKEN` when starting the server, the MCP SDK requires
`Authorization: Bearer <token>` on every streamable HTTP or SSE request.
Codex can source that token from an environment variable:

```toml
[mcp_servers.chemspace_http]
url = "https://mcp.example.com/mcp"
bearer_token_env_var = "CS_COPILOT_MCP_AUTH_TOKEN"
startup_timeout_sec = 30
tool_timeout_sec = 300
```

A reference snippet lives at `examples/mcp/codex_http.toml`.

For a ChatGPT-authenticated Codex smoke test that proves a subscription-model
reasoning client can call ChemSpace MCP tools over both stdio and streamable
HTTP, see `examples/mcp/codex_subscription_smoke.md`.

## ChatGPT app / connector setup

ChatGPT is a remote MCP client: it cannot launch the local stdio command
itself. Start `cscopilot-mcp-serve` and connect ChatGPT to the HTTP endpoint
through a reachable HTTPS URL. For a local workstation or private network,
use OpenAI Secure MCP Tunnel when available, or a trusted reverse proxy/tunnel
that terminates HTTPS and restricts access.

OpenAI's current ChatGPT app auth guidance says ChatGPT cannot present custom
API keys. Do not enable `CS_COPILOT_MCP_AUTH_TOKEN` for a plain ChatGPT
connector unless your tunnel or reverse proxy injects the header before the
request reaches ChemSpace. For a direct ChatGPT production connector, use
Secure MCP Tunnel / OpenAI client identification controls, or implement proper
OAuth 2.1 resource metadata and per-tool `securitySchemes`. The static bearer
token option in this package is intended for clients such as Codex and for
private proxy/tunnel deployments that can attach `Authorization`.

If you use a reverse proxy for ChatGPT, keep Host/Origin validation aligned
with the external URL. A proxy that forwards `Host: <your-host>` needs
`--allowed-host <your-host>`. If it forwards an `Origin` header, add the exact
origin shown in proxy logs, for example `--allowed-origin https://chatgpt.com`.

For private servers, the current OpenAI Secure MCP Tunnel flow is:

1. Create or select a tunnel in Platform tunnel settings.
2. Configure a local `tunnel-client` profile that can reach ChemSpace Copilot.
3. Run `tunnel-client doctor --profile <profile> --explain`.
4. Keep `tunnel-client run --profile <profile>` healthy while creating or
   testing the ChatGPT connector.
5. In ChatGPT connector settings, select **Tunnel** for the private MCP server.

A ChemSpace-specific tunnel runbook lives at
`examples/mcp/secure_mcp_tunnel.md`.

In ChatGPT settings, create an app from the remote MCP server URL. Use:

- Streamable HTTP: `https://<your-host>/mcp`
- SSE: `https://<your-host>/sse`

Recommended connector metadata:

- Name: `ChemSpace Copilot`
- Description: `Chemistry and chemography MCP tools for ChEMBL retrieval, GTM chemical-space modeling, chemoinformatics analysis, molecular design, peptide design, session artifacts, and report generation.`

Current OpenAI docs say ChatGPT developer mode supports full MCP tools over
SSE and streamable HTTP, and does not require `search` / `fetch` for full
developer-mode apps. ChatGPT data-only apps, company knowledge, and deep
research use the read-only `search` and `fetch` compatibility tools. Plan and
workspace availability changes over time, so verify the current ChatGPT Apps
/ Developer Mode eligibility in OpenAI's docs before relying on a specific
subscription tier.

A runnable command reference lives at `examples/mcp/chatgpt_remote.md`.

## What the server exposes

### Tools

ChemSpace toolkit methods are wrapped one-to-one as MCP tools, namespaced
by toolkit (`chembl_*`, `gtm_*`, `chem_*`, `session_*`, `report_*`,
`robustness_*`). Tool arguments mirror the toolkit method signatures, with
the `agent` / `session_state` parameters injected by the server and hidden
from the public schema.

The server attaches MCP tool annotations for ChatGPT / Apps approval UX:
`readOnlyHint=True` is used only for strict lookup, retrieval, listing, or
pure computation tools. Tools that fetch-and-store data, load mutable GTM
state, sample/register zones, update session memory, train models, or save
reports are advertised as write actions with `readOnlyHint=False`.
`destructiveHint` and `openWorldHint` are currently `False` for every tool
because ChemSpace writes are scoped to private session storage and do not
delete data or publish to public internet state.

The server also exposes two read-only ChatGPT compatibility tools:

| Tool | Purpose |
|------|---------|
| `search` | Search the ChemSpace MCP tool catalog, prompt catalog, and active session artifacts. |
| `fetch` | Fetch a search result by id, returning tool/prompt documentation or text artifact content. |

Use `cscopilot-mcp` once and then `list_tools` from your MCP client to see
the current set (there are ~50 tools plus `search` / `fetch`).

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
