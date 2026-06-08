# Codex ChatGPT subscription MCP smoke test

This runbook verifies the cs_copilot MCP server with a subscription-model
reasoning layer before doing the final ChatGPT web connector test. It uses the
local Codex CLI authenticated with ChatGPT, not an OpenAI API key.

## 1. Confirm Codex is using ChatGPT auth

```sh
codex login status
```

Expected output:

```text
Logged in using ChatGPT
```

## 2. Stdio smoke

This proves the subscription model can reason over cs_copilot MCP tools without
starting an HTTP server:

```sh
codex -a never exec \
  --ignore-user-config \
  --ephemeral \
  --sandbox read-only \
  -C /path/to/chemspacecopilot \
  -c 'mcp_servers.cs_copilot.command="/path/to/chemspacecopilot/.venv/bin/cscopilot-mcp"' \
  -c 'mcp_servers.cs_copilot.args=["--session-id","codex-subscription-smoke","--workflow-slug","chemical_space","--log-level","error"]' \
  -c 'mcp_servers.cs_copilot.cwd="/path/to/chemspacecopilot"' \
  -c 'mcp_servers.cs_copilot.env={USE_S3="false", SESSION_ID="codex-subscription-smoke", AGNO_TELEMETRY="false"}' \
  'Use only the configured cs_copilot MCP server named cs_copilot. Do not run shell commands, inspect files, edit files, or use web search. Use the MCP search tool to search the cs_copilot MCP catalog for ChEMBL retrieval tools. Then use the MCP fetch tool to fetch prompt:cs_copilot_workflow. Answer with whether the MCP server was usable and which cs_copilot tool should fetch CDK2 inhibitor activity data.'
```

Expected evidence in Codex output:

```text
mcp: cs_copilot/search started
mcp: cs_copilot/search (completed)
mcp: cs_copilot/fetch started
mcp: cs_copilot/fetch (completed)
chembl_fetch_compounds
```

Observed locally on 2026-05-31 with Codex `0.135.0`, model `gpt-5.5`,
provider `openai`, and ChatGPT login: Codex called `cs_copilot/search` and
`cs_copilot/fetch`, reported that the MCP server was usable, and selected
`chembl_fetch_compounds` for CDK2 inhibitor activity retrieval.

## 3. Streamable HTTP smoke with `cscopilot-mcp-serve`

Start the server in one terminal:

```sh
env USE_S3=false AGNO_TELEMETRY=false SESSION_ID=codex-http-smoke \
  /path/to/chemspacecopilot/.venv/bin/cscopilot-mcp-serve \
  --session-id codex-http-smoke \
  --workflow-slug chemical_space \
  --host 127.0.0.1 \
  --port 8765 \
  --log-level error
```

Probe the same endpoint locally:

```sh
/path/to/chemspacecopilot/.venv/bin/cscopilot-mcp-check \
  --url http://127.0.0.1:8765/mcp \
  --timeout 30
```

Then run Codex against the HTTP endpoint:

```sh
codex -a never exec \
  --ignore-user-config \
  --ephemeral \
  --sandbox read-only \
  -C /path/to/chemspacecopilot \
  -c 'mcp_servers.cs_copilot_http.url="http://127.0.0.1:8765/mcp"' \
  -c 'mcp_servers.cs_copilot_http.startup_timeout_sec=30' \
  -c 'mcp_servers.cs_copilot_http.tool_timeout_sec=300' \
  'Use only the configured cs_copilot HTTP MCP server named cs_copilot_http. Do not run shell commands, inspect files, edit files, or use web search. Use the MCP search tool to search the cs_copilot MCP catalog for ChEMBL retrieval tools. Then use the MCP fetch tool to fetch prompt:cs_copilot_workflow. Answer with whether the HTTP MCP server was usable and which cs_copilot tool should fetch CDK2 inhibitor activity data.'
```

Expected evidence in Codex output:

```text
mcp: cs_copilot_http/search started
mcp: cs_copilot_http/search (completed)
mcp: cs_copilot_http/fetch started
mcp: cs_copilot_http/fetch (completed)
chembl_fetch_compounds
```

Observed locally on 2026-05-31 with Codex `0.135.0`, model `gpt-5.5`,
provider `openai`, and ChatGPT login: Codex called
`cs_copilot_http/search` and `cs_copilot_http/fetch`, reported that the HTTP MCP
server was usable, and selected `chembl_fetch_compounds`.

## 4. Streamable HTTP domain-tool smoke

After the catalog smoke, run one low-cost read-only cs_copilot tool through the
same HTTP endpoint:

```sh
codex -a never exec \
  --ignore-user-config \
  --ephemeral \
  --sandbox read-only \
  -C /path/to/chemspacecopilot \
  -c 'mcp_servers.cs_copilot_http.url="http://127.0.0.1:8765/mcp"' \
  -c 'mcp_servers.cs_copilot_http.startup_timeout_sec=30' \
  -c 'mcp_servers.cs_copilot_http.tool_timeout_sec=300' \
  'Use only the configured cs_copilot HTTP MCP server named cs_copilot_http. Do not run shell commands, inspect files, edit files, or use web search. Call the MCP tool chem_calculate_tanimoto_similarity with smiles1="CCO", smiles2="CCN", and fp_type="rdkit". Answer with whether the HTTP MCP server was usable, the tool you called, and the numeric similarity returned.'
```

Expected evidence in Codex output:

```text
mcp: cs_copilot_http/chem_calculate_tanimoto_similarity started
mcp: cs_copilot_http/chem_calculate_tanimoto_similarity (completed)
Numeric similarity returned
```

Observed locally on 2026-05-31 with Codex `0.135.0`, model `gpt-5.5`,
provider `openai`, and ChatGPT login: Codex called
`cs_copilot_http/chem_calculate_tanimoto_similarity` over `cscopilot-mcp-serve`
and returned Tanimoto similarity `0.2` for `CCO` vs `CCN` with `rdkit`
fingerprints.

## 5. What this does and does not prove

This proves a ChatGPT-authenticated subscription client can use cs_copilot MCP
tools as its reasoning/tool layer over both stdio and streamable HTTP.

It does not replace the final ChatGPT web app test. For that, expose
`https://<your-host>/mcp` through Secure MCP Tunnel or HTTPS reverse proxy,
create the ChatGPT app in Developer Mode, then run the smoke prompt printed by
`cscopilot-mcp-check --url https://<your-host>/mcp`.
