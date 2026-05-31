# Secure MCP Tunnel for ChatGPT

Use this path when ChemSpace Copilot runs on a workstation, devbox, or private
network and you want ChatGPT to use it without making the MCP server public.

OpenAI's current Secure MCP Tunnel flow has three moving parts:

1. A tunnel endpoint created in OpenAI Platform tunnel settings.
2. A local `tunnel-client` profile that can reach ChemSpace Copilot.
3. A ChatGPT connector created from Settings -> Connectors with **Tunnel**
   selected.

## 1. Verify ChemSpace MCP locally

```sh
uv sync --extra mcp
USE_S3=false cscopilot-mcp-check
```

Expected output includes `55` tools, `12` prompts, `server_instructions: ok`, `tool_annotations: ok`, `workflow_prompt: ok`, and `search_fetch: ok`. Add `--json` when you want a machine-readable proof for setup logs, connector metadata, the first ChatGPT smoke prompt, and expected evidence.

## 2. Install tunnel-client

Download the current `tunnel-client` from OpenAI Platform tunnel settings or
from the latest public release:

Visit:

- `https://platform.openai.com/settings/organization/tunnels`
- `https://github.com/openai/tunnel-client/releases/latest`

Then verify the binary:

```sh
tunnel-client help quickstart
```

## 3. Create an OpenAI tunnel endpoint

Create or copy a tunnel id in Platform tunnel settings, then export the values
used below:

```sh
cd /path/to/chemspacecopilot
export CONTROL_PLANE_API_KEY="sk-..."
export OPENAI_MCP_TUNNEL_ID="tunnel_0123456789abcdef0123456789abcdef"
export CHEMSPACE_ROOT="$(pwd)"
```

## 4. Recommended local profile: stdio

This is the most direct private-workstation setup because `tunnel-client`
starts `cscopilot-mcp` itself and ChatGPT talks to the tunnel.

```sh
tunnel-client init \
  --sample sample_mcp_stdio_local \
  --profile chemspace-stdio \
  --tunnel-id "$OPENAI_MCP_TUNNEL_ID" \
  --mcp-command "env USE_S3=false AGNO_TELEMETRY=false $CHEMSPACE_ROOT/.venv/bin/cscopilot-mcp --session-id chatgpt-demo --workflow-slug chemical_space --log-level warning"

tunnel-client doctor --profile chemspace-stdio --explain
tunnel-client run --profile chemspace-stdio
```

Keep `tunnel-client run` healthy while testing from ChatGPT.

## 5. HTTP profile: existing HTTPS MCP endpoint

Use this only when `cscopilot-mcp-serve` is already reachable by
`tunnel-client` through an HTTPS URL, such as an internal reverse proxy:

```sh
USE_S3=false cscopilot-mcp-serve \
  --session-id chatgpt-demo \
  --workflow-slug chemical_space \
  --host 127.0.0.1 \
  --port 8000 \
  --allowed-host mcp.internal.example.com \
  --allowed-origin https://chatgpt.com

tunnel-client init \
  --sample sample_mcp_stdio_local \
  --profile chemspace-http \
  --tunnel-id "$OPENAI_MCP_TUNNEL_ID" \
  --mcp-server-url https://mcp.internal.example.com/mcp

tunnel-client doctor --profile chemspace-http --explain
tunnel-client run --profile chemspace-http
```

Do not point ChatGPT directly at `http://127.0.0.1:8000/mcp`; ChatGPT needs a
reachable HTTPS endpoint or a Secure MCP Tunnel. If your reverse proxy forwards
`Host: mcp.internal.example.com`, keep `--allowed-host mcp.internal.example.com`
on the local server. If it forwards a browser `Origin`, add the exact origin
shown in proxy logs.

If the endpoint is reachable from your shell, probe the exact URL before
creating the connector:

```sh
cscopilot-mcp-check --url https://mcp.internal.example.com/mcp
```

For a pure Secure MCP Tunnel connection selected inside ChatGPT, keep using
`tunnel-client doctor` and the tunnel admin UI as the tunnel health check.

## 6. Create the ChatGPT connector

In ChatGPT:

1. Enable Developer Mode under Settings -> Apps & Connectors -> Advanced
   settings.
2. Go to Settings -> Connectors -> Create.
3. Select **Tunnel** for a private ChemSpace server, or paste the public
   `https://.../mcp` URL for a public HTTPS deployment.
4. Use:

   - Connector name: `ChemSpace Copilot`
   - Description: `Chemistry and chemography MCP tools for ChEMBL retrieval, GTM chemical-space modeling, chemoinformatics analysis, molecular design, peptide design, session artifacts, and report generation.`
   - Connector URL for public HTTPS deployments: `https://<your-host>/mcp`

5. Click Create. A successful connection should show the ChemSpace tool list,
   including `chembl_fetch_compounds`, `gtm_optimization`, `search`, and
   `fetch`.

## 7. First ChatGPT prompt

After adding the connector to a chat, start with a small read-only discovery
request:

```text
Use ChemSpace Copilot. List the available ChemSpace MCP tools and fetch the
chemspace_workflow prompt. Do not run long ChEMBL or GTM jobs yet.
```

Then try a minimal catalog lookup:

```text
Use ChemSpace Copilot to search the MCP catalog for ChEMBL retrieval tools and
explain which tool I should use to fetch CDK2 inhibitor activity data.
```
