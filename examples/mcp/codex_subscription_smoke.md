# Codex ChatGPT subscription MCP smoke test

This runbook verifies the ChemSpace MCP server with a subscription-model
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

This proves the subscription model can reason over ChemSpace MCP tools without
starting an HTTP server:

```sh
codex -a never exec \
  --ignore-user-config \
  --ephemeral \
  --sandbox read-only \
  -C /path/to/chemspacecopilot \
  -c 'mcp_servers.chemspace.command="/path/to/chemspacecopilot/.venv/bin/cscopilot-mcp"' \
  -c 'mcp_servers.chemspace.args=["--session-id","codex-subscription-smoke","--workflow-slug","chemical_space","--log-level","error"]' \
  -c 'mcp_servers.chemspace.cwd="/path/to/chemspacecopilot"' \
  -c 'mcp_servers.chemspace.env={USE_S3="false", SESSION_ID="codex-subscription-smoke", AGNO_TELEMETRY="false"}' \
  'Use only the configured ChemSpace MCP server named chemspace. Do not run shell commands, inspect files, edit files, or use web search. Use the MCP search tool to search the ChemSpace MCP catalog for ChEMBL retrieval tools. Then use the MCP fetch tool to fetch prompt:chemspace_workflow. Answer with whether the MCP server was usable and which ChemSpace tool should fetch CDK2 inhibitor activity data.'
```

Expected evidence in Codex output:

```text
mcp: chemspace/search started
mcp: chemspace/search (completed)
mcp: chemspace/fetch started
mcp: chemspace/fetch (completed)
chembl_fetch_compounds
```

Observed locally on 2026-05-31 with Codex `0.135.0`, model `gpt-5.5`,
provider `openai`, and ChatGPT login: Codex called `chemspace/search` and
`chemspace/fetch`, reported that the MCP server was usable, and selected
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
  -c 'mcp_servers.chemspace_http.url="http://127.0.0.1:8765/mcp"' \
  -c 'mcp_servers.chemspace_http.startup_timeout_sec=30' \
  -c 'mcp_servers.chemspace_http.tool_timeout_sec=300' \
  'Use only the configured ChemSpace HTTP MCP server named chemspace_http. Do not run shell commands, inspect files, edit files, or use web search. Use the MCP search tool to search the ChemSpace MCP catalog for ChEMBL retrieval tools. Then use the MCP fetch tool to fetch prompt:chemspace_workflow. Answer with whether the HTTP MCP server was usable and which ChemSpace tool should fetch CDK2 inhibitor activity data.'
```

Expected evidence in Codex output:

```text
mcp: chemspace_http/search started
mcp: chemspace_http/search (completed)
mcp: chemspace_http/fetch started
mcp: chemspace_http/fetch (completed)
chembl_fetch_compounds
```

Observed locally on 2026-05-31 with Codex `0.135.0`, model `gpt-5.5`,
provider `openai`, and ChatGPT login: Codex called
`chemspace_http/search` and `chemspace_http/fetch`, reported that the HTTP MCP
server was usable, and selected `chembl_fetch_compounds`.

## 4. Streamable HTTP domain-tool smoke

After the catalog smoke, run one low-cost read-only ChemSpace tool through the
same HTTP endpoint:

```sh
codex -a never exec \
  --ignore-user-config \
  --ephemeral \
  --sandbox read-only \
  -C /path/to/chemspacecopilot \
  -c 'mcp_servers.chemspace_http.url="http://127.0.0.1:8765/mcp"' \
  -c 'mcp_servers.chemspace_http.startup_timeout_sec=30' \
  -c 'mcp_servers.chemspace_http.tool_timeout_sec=300' \
  'Use only the configured ChemSpace HTTP MCP server named chemspace_http. Do not run shell commands, inspect files, edit files, or use web search. Call the MCP tool chem_calculate_tanimoto_similarity with smiles1="CCO", smiles2="CCN", and fp_type="rdkit". Answer with whether the HTTP MCP server was usable, the tool you called, and the numeric similarity returned.'
```

Expected evidence in Codex output:

```text
mcp: chemspace_http/chem_calculate_tanimoto_similarity started
mcp: chemspace_http/chem_calculate_tanimoto_similarity (completed)
Numeric similarity returned
```

Observed locally on 2026-05-31 with Codex `0.135.0`, model `gpt-5.5`,
provider `openai`, and ChatGPT login: Codex called
`chemspace_http/chem_calculate_tanimoto_similarity` over `cscopilot-mcp-serve`
and returned Tanimoto similarity `0.2` for `CCO` vs `CCN` with `rdkit`
fingerprints.

## 5. What this does and does not prove

This proves a ChatGPT-authenticated subscription client can use ChemSpace MCP
tools as its reasoning/tool layer over both stdio and streamable HTTP.

It does not replace the final ChatGPT web app test. For that, expose
`https://<your-host>/mcp` through Secure MCP Tunnel or HTTPS reverse proxy,
create the ChatGPT app in Developer Mode, then run the smoke prompt printed by
`cscopilot-mcp-check --url https://<your-host>/mcp`.
