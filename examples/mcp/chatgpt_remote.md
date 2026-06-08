# ChatGPT remote MCP endpoint

ChatGPT connects to remote MCP servers; it cannot start the local stdio
`cscopilot-mcp` command. Run cs_copilot with an HTTP transport, then
connect ChatGPT to the resulting HTTPS URL through ChatGPT Apps / Developer
Mode.

## Preflight

```sh
uv sync --extra mcp
cscopilot-mcp-check
```

The preflight starts a temporary streamable HTTP server and verifies the same
MCP path that ChatGPT or Secure MCP Tunnel will call, including server
instructions, tool annotations, and the fetchable `cs_copilot_workflow`
orchestration prompt used by the subscription model as the reasoning layer. For a private workstation
connection, use the full runbook in `examples/mcp/secure_mcp_tunnel.md`.

## Streamable HTTP

```sh
USE_S3=false cscopilot-mcp-serve \
  --session-id demo \
  --workflow-slug chemical_space \
  --host 127.0.0.1 \
  --port 8000 \
  --allowed-host <your-host> \
  --allowed-origin https://chatgpt.com
```

Local endpoint: `http://127.0.0.1:8000/mcp`

For ChatGPT, expose that endpoint as `https://<your-host>/mcp` using OpenAI
Secure MCP Tunnel when available, or a trusted HTTPS reverse proxy/tunnel.
If your proxy preserves the public `Host` header, pass that host with
`--allowed-host`; if it forwards an `Origin` header, allow the exact origin
shown in proxy logs. Do not expose the unauthenticated server publicly. When
the HTTPS URL is reachable from your shell, probe the exact endpoint before
creating the connector:

```sh
cscopilot-mcp-check --url https://<your-host>/mcp
# Optional machine-readable proof for connector setup logs:
cscopilot-mcp-check --url https://<your-host>/mcp --json
```

The report includes `chatgpt_connector_name`,
`chatgpt_connector_description`, `chatgpt_smoke_prompt`, and
`chatgpt_expected_evidence` fields. Use those values for the first ChatGPT
connector smoke test.

If your MCP client or proxy can attach HTTP headers, protect the endpoint with
a static bearer token:

```sh
export CS_COPILOT_MCP_AUTH_TOKEN=change-me
cscopilot-mcp-check --auth-scope mcp:read
USE_S3=false cscopilot-mcp-serve \
  --session-id demo \
  --workflow-slug chemical_space \
  --host 127.0.0.1 \
  --port 8000 \
  --allowed-host <your-host> \
  --allowed-origin https://chatgpt.com \
  --auth-scope mcp:read \
  --auth-resource-url https://<your-host>/mcp
```

Plain ChatGPT connectors cannot present custom API keys directly. Use Secure
MCP Tunnel, OpenAI client identification controls, or a proxy that injects the
header when you enable this static-token mode.

## SSE fallback

```sh
USE_S3=false cscopilot-mcp \
  --transport sse \
  --session-id demo \
  --workflow-slug chemical_space \
  --host 127.0.0.1 \
  --port 8000
```

Local endpoint: `http://127.0.0.1:8000/sse`

## Compatibility tools

The remote server registers:

- `search` and `fetch` for ChatGPT data-only apps, company knowledge, and deep research.
- The full cs_copilot `chembl_*`, `gtm_*`, `chem_*`, `session_*`, `report_*`, and `robustness_*` tools for full MCP developer-mode clients.
- MCP tool annotations: pure lookup/computation tools advertise `readOnlyHint=True`; cs_copilot workflows that store data, update session state, train, sample, or save reports advertise `readOnlyHint=False`.
- cs_copilot workflow prompts and session artifact resources.
