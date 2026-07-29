# MCP server (optional)

cs_copilot ships an **optional** Model Context Protocol server that
exposes the same chemistry, chemography, ChEMBL, design, and reporting
toolkits as the Chainlit app — but driven by an external MCP client such as
[Codex](https://github.com/openai/codex) or
[Claude Code](https://docs.anthropic.com/en/docs/claude-code) instead of the
Agno multi-agent team.

In MCP mode the external client is the reasoning engine. The cs_copilot MCP
server only surfaces primitives — it never instantiates the Agno team, the
agent factories, or the configured model backend (DeepSeek / Ollama /
OpenRouter). The default Chainlit and CLI runtimes are unaffected.

## Install

The MCP server lives under `src/cs_copilot/mcp/`. It depends on the official
`mcp` Python SDK, which is gated behind an optional extra:

```sh
uv sync --extra mcp
# or, on a normal pip install
pip install "cs_copilot[mcp]"
```

Importing `cs_copilot` itself does not pull in `mcp` — the SDK is only
imported when you run the server.

## Readiness check

Before wiring ChatGPT or a tunnel, run the local preflight check:

```sh
cscopilot-mcp-check --profile standard
```

The check starts a temporary streamable HTTP server, connects with the MCP
client, verifies the core cs_copilot tools, validates the MCP server
instructions and ChatGPT-facing tool annotations, lists prompts and resources,
and uses the ChatGPT-compatible `search` / `fetch` pair to fetch both
`chembl_fetch_compounds` documentation and the `cs_copilot_mcp_workflow`
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
cscopilot-mcp --profile gtm-analysis --session-id my-session \
  --workflow-slug gtm-density-landscape
```

Starting with a catalog workflow validates that every required tool is
available in the selected profile before a run is created. To resume the exact
event-backed run later, keep the same session and pass its run id:

```sh
cscopilot-mcp --profile gtm-analysis --session-id my-session \
  --run-id <run-id>
```

For remote MCP clients such as ChatGPT apps, use streamable HTTP or SSE:

```sh
# Streamable HTTP (recommended default for remote clients)
cscopilot-mcp-serve --profile gtm-analysis --session-id demo \
  --workflow-slug gtm-density-landscape --host 127.0.0.1 --port 8000
# Equivalent explicit form:
cscopilot-mcp --transport streamable-http --profile gtm-analysis \
  --session-id demo --workflow-slug gtm-density-landscape \
  --host 127.0.0.1 --port 8000

# SSE fallback for clients that still require it:
cscopilot-mcp --transport sse --profile gtm-analysis --session-id demo \
  --workflow-slug gtm-density-landscape --host 127.0.0.1 --port 8000
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
| `--auth-client-id` | `cs_copilot-mcp-client` | Client id attached to accepted static bearer tokens. |
| `--auth-scope` | none | Required bearer-token scope. Can be supplied multiple times. |
| `--auth-issuer-url` | endpoint URL | Issuer URL advertised in MCP protected-resource metadata. |
| `--auth-resource-url` | endpoint URL | Public MCP resource URL advertised in auth metadata. Set this to the HTTPS URL seen by remote clients. |
| `--session-id` | auto-generated | Storage prefix used as the session root. |
| `--run-id` | new run | Resume an existing v2 run in the selected session. Its stored workflow must be compatible with the selected profile. |
| `--profile` | `standard` | Strict tool discovery and invocation allowlist. `standard` is the all-capabilities profile; domain profiles are least-privilege surfaces. Can also be set with `CS_COPILOT_MCP_PROFILE`. |
| `--workflow-slug` | ad hoc `mcp-session` | Select and validate a catalog workflow contract for the new run. |
| `--log-level` | `info` | Logger level (logs to stderr). |
| `--llm-policy` | `external` | LLM behavior for MCP tools: `external` stores pending `llm_*` tasks for the client, `agno-model` loads only the configured Agno model for toolkit LLM calls, and `disabled` rejects LLM-dependent work. Can also be set with `CS_COPILOT_MCP_LLM_POLICY`. |
| `--no-tools` | False | Skip MCP tool registration. |
| `--no-chatgpt-compat` | False | Skip the read-only `search` / `fetch` tools. |
| `--no-prompts` | False | Skip MCP prompt registration. |
| `--no-resources` | False | Skip session-artifact resources. |

The `cscopilot-mcp-check` command accepts `--url`, `--host`, `--port`,
`--path`, `--session-id`, `--profile`, `--workflow-slug`, `--timeout`, `--log-level`,
`--use-s3`, `--json`, `--auth-token-env`, `--auth-token`,
`--auth-client-id`, repeatable `--auth-scope`, and repeatable
`--required-tool` flags for preflight checks.
If a bearer token is configured, the check protects the temporary server and
connects with `Authorization: Bearer <token>`.

You can also run the package directly: `python -m cs_copilot.mcp …`.

## Capability profiles

Profiles are server-side allowlists: tools outside the selected profile are
neither discoverable nor invocable. Catalog workflows declare their
least-privilege profile, while `standard` is the deliberate all-capabilities
superset.

| Profile | Intended surface |
|---------|------------------|
| `bootstrap` | Read-only `mcp_bootstrap` and catalog discovery, plus write-capable ChEMBL/GTM preflights that record plan artifacts |
| `standard` | Every stable catalog tool |
| `chembl-retrieval` | ChEMBL retrieval, judging, and run artifacts |
| `gtm-analysis` | GTM, tabular preparation, and reports |
| `chemoinformatics` | Similarity/chemotype analysis, tables, and reports |
| `reporting` | Run inspection and report generation |
| `molecular-design` | Small-molecule design, validation, GTM, and reports |
| `peptide-design` | Peptide design, validation, GTM, and reports |
| `retrosynthesis` | Candidate resolution and SynPlanner routes |
| `robustness` | Robustness analysis and report export |

The same role names and profile assignments are used by the in-process agent
factories. Factory construction fails when a role is configured with a
toolkit outside its allowlist.

## Claude Code config

Add an entry under `mcpServers` in `~/.claude.json` (or `~/.claude/.mcp.json`,
depending on your Claude Code version):

```jsonc
{
  "mcpServers": {
    "cs_copilot": {
      "command": "cscopilot-mcp",
      "args": [
        "--profile", "gtm-analysis",
        "--session-id", "demo",
        "--workflow-slug", "gtm-density-landscape"
      ],
      "env": { "SESSION_ID": "demo", "USE_S3": "false" }
    }
  }
}
```

A reference snippet lives at `examples/mcp/claude_code.json`.

## Codex config

Codex reads `~/.codex/config.toml`. Add a server entry like this:

```toml
[mcp_servers.cs_copilot]
command = "cscopilot-mcp"
args = [
  "--profile", "gtm-analysis",
  "--session-id", "demo",
  "--workflow-slug", "gtm-density-landscape",
]

[mcp_servers.cs_copilot.env]
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
[mcp_servers.cs_copilot_http]
url = "https://mcp.example.com/mcp"
bearer_token_env_var = "CS_COPILOT_MCP_AUTH_TOKEN"
startup_timeout_sec = 30
tool_timeout_sec = 300
```

A reference snippet lives at `examples/mcp/codex_http.toml`.

For a ChatGPT-authenticated Codex smoke test that proves a subscription-model
reasoning client can call cs_copilot MCP tools over both stdio and streamable
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
request reaches cs_copilot. For a direct ChatGPT production connector, use
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
2. Configure a local `tunnel-client` profile that can reach cs_copilot.
3. Run `tunnel-client doctor --profile <profile> --explain`.
4. Keep `tunnel-client run --profile <profile>` healthy while creating or
   testing the ChatGPT connector.
5. In ChatGPT connector settings, select **Tunnel** for the private MCP server.

A cs_copilot-specific tunnel runbook lives at
`examples/mcp/secure_mcp_tunnel.md`.

In ChatGPT settings, create an app from the remote MCP server URL. Use:

- Streamable HTTP: `https://<your-host>/mcp`
- SSE: `https://<your-host>/sse`

Recommended connector metadata:

- Name: `cs_copilot`
- Description: `Chemistry and chemography MCP tools for ChEMBL retrieval, GTM chemical-space modeling, chemoinformatics analysis, molecular design, peptide design, session artifacts, and report generation.`

Current OpenAI docs say ChatGPT developer mode supports full MCP tools over
SSE and streamable HTTP, and does not require `search` / `fetch` for full
developer-mode apps. ChatGPT data-only apps, company knowledge, and deep
research use the read-only `search` and `fetch` compatibility tools. Plan and
workspace availability changes over time, so verify the current ChatGPT Apps
/ Developer Mode eligibility in OpenAI's docs before relying on a specific
subscription tier.

A runnable command reference lives at `examples/mcp/chatgpt_remote.md`.

## Skills, workflows, and manifests

For full MCP clients, the recommended first call for a new user request is:

```text
mcp_bootstrap(user_request="<the user's request>")
```

The read-only bootstrap tool is an organization step. It returns the
MCP-native prompt to use, matched workflow and skill documents to fetch,
declared scientific preflight tools to run, and an ordered action plan. It is advisory
only: it does not mutate session state, execute a workflow, or call the Agno
team.

Bootstrap-level questions are limited to bootstrap blockers such as an empty
request or an unknown explicit workflow slug. Domain questions belong to the
preflight tools. If `chembl_prepare_retrieval` or
`chemspace_plan_analysis` returns `needs_clarification=true`, ask the returned
questions before calling mutating tools. Do not infer missing target, organism,
assay type, mechanism, analysis intent, dataset source, or workflow details
just to avoid asking the user.

The MCP server also exposes the shared cs_copilot skill catalog through the
ChatGPT-compatible `search` / `fetch` tools. Use `fetch("catalog:skills")`
to list reusable skills, then fetch `skill:<slug>` for the full `SKILL.md`
procedure and required tool names. The same catalog is available to the Agno
team through its read-only Skills toolkit.

The MCP server also exposes reusable workflow contracts. Use
`fetch("catalog:workflows")` or the direct read-only `workflow_list`,
`workflow_search`, and `workflow_fetch` tools to discover workflow metadata,
semantic version, dependencies, profiles, permissions, task DAG, artifact
contracts, preflight tools, required tools, and the recommended MCP prompt.
Workflow contracts guide the external MCP client; they do not execute the
Agno team or replace the Chainlit runtime.

Every server process creates or resumes an explicit v2 run identified by
`session_id`, `run_id`, and `workflow_slug`. `workflow_start_run` replaces the
untouched startup placeholder with the selected catalog run, pins the selected
workflow plus its transitive dependency contracts, and materializes any task
DAG declared by those contracts. Required request-like `workflow_inputs` are
serialized as checksummed, run-scoped JSON artifacts. File-backed inputs must
first be registered inside the run and then referenced as
`{"artifact_id": "<id>"}`. On an active run, omit `constraints`, `budget`, and
`workflow_inputs` to reuse their pinned values; explicit empty mappings request
an actual change and conflict with non-empty pinned values.

Strict persisted task-DAG and task-contract enforcement is currently the
`chembl-to-gtm-report` pilot. Other catalog workflows remain taskless/legacy:
they still receive durable run, event, artifact, and pinned workflow contracts,
but clients must not invent task records or task handoffs that the fetched
contract does not declare.

The run's authoritative state is an immutable sequence of `events/*.jsonl`;
`manifest.json` and `artifacts/index.json` are rebuildable snapshots. Tool calls
append redacted observations with timing, attempts, idempotency/cache metadata,
active task/role, and output summaries. Registered artifacts carry SHA-256,
size, media type, producer, trust, and provenance. Paths returned by tools are
registered automatically when they resolve to an existing file inside the
active run root.

Use `workflow_start_run`, `workflow_get_run`, `workflow_add_task`, `workflow_transition_run`,
`workflow_transition_task`, `workflow_record_handoff`,
`workflow_register_artifact`, `workflow_verify_artifact`, and
`workflow_complete_run` to drive the explicit lifecycle. Completion becomes
`partial` when required catalog tasks, root inputs, or output artifacts are
missing. In a workflow with a declared DAG, each task requires a structured,
bounded handoff before it can run; a failed retry requires a fresh handoff.
Domain calls require an active running task and are checked against that
task's exact pinned handoff, role, profile, selected input artifacts, and tool
allowlist. Selected inputs are checksum-verified again before every domain
call. Explicit and automatic artifact producer ids must match that
authoritative active `RUNNING` task. Lifecycle transitions, including
transitions to `failed`, are rejected while a domain call is in flight; cancel
or await the call first so publication and registration reach a durable
terminal state.

`workflow_abandon_tool_invocation` is an explicit high-risk, supervisor
crash-recovery operation, not a normal timeout or cancellation path. Use it
only for a durable invocation left in flight after a process crash, and only
after independently verifying that no live server or worker still owns the
span. The caller must then set `confirm_not_running=true`; the operation
refuses spans that are still active in the current server process and appends
an `abandoned` event when accepted. Afterwards, transition the affected task
to `failed` or `cancelled`. Retry it only with a fresh handoff.

Every direct tool returns the same schema-v2 envelope:

```json
{
  "schema_version": 2,
  "status": "success",
  "data": {},
  "artifact_ids": [],
  "warnings": [],
  "error": null,
  "metrics": {
    "duration_ms": 12.3,
    "attempts": 1,
    "retries": 0,
    "cached": false,
    "output_bytes": 42
  },
  "trace": {
    "run_id": "<run-id>",
    "trace_id": "<trace-id>",
    "span_id": "<span-id>",
    "parent_span_id": null
  }
}
```

Errors use stable codes: `invalid_input`, `permission_denied`,
`transient_external`, `timeout`, `resource_limit`,
`scientific_validation`, and `internal`. Only idempotent, retryable calls are
retried. Idempotent tools accept an optional `idempotency_key`; repeated calls
with the same task-scoped key and arguments during the server process coalesce
or return the cached successful envelope; reusing a key with different
arguments is rejected. Async tools and subprocess workers may declare
cancellable timeouts. In-process synchronous tools intentionally do not claim
a cancellable deadline, and caller cancellation waits for their worker thread
to finish so a mutation cannot outlive its envelope. The remaining catalog
handoff deadline bounds async calls and subprocess workers while they run. A
synchronous overrun is reported only after its worker thread has safely
finished; it is never abandoned in the background. The server also enforces
the handoff's observable tool-call budget, while its token budget is enforced
by the external reasoning client. The default one-megabyte inline output cap
keeps responses bounded; larger scientific outputs should be persisted and
returned as artifact ids.

For every tool with `write_scope="session"`, client-selected output paths are
confined to the active run. The storage client rechecks the boundary at the
actual local write with descriptor-relative, no-follow opens. Absolute and
`file://` paths, parent traversal, symlink swaps, runtime-metadata aliases,
encoded traversal, foreign S3 buckets/sessions/runs, Python-literal parameter
bag escapes, unsafe pandas constructors, and unsupported open-ended
serializers are rejected. Files already registered in the run's artifact
ledger are protected from later domain-tool writes. Scoped writes are
create-only and invocation-transactional. Worker processes publish first to a
job-specific staging prefix; the parent promotes checksum-bound bytes only
after accepting the result. If required artifact registration fails, newly
published but unregistered bytes are released without deleting files already
bound by an artifact event.

OS-level termination can interrupt a worker while an S3 multipart upload is in
progress, before the invocation cleanup path can run. Production S3 and MinIO
buckets therefore need a lifecycle rule that aborts incomplete multipart
uploads; this repository recommends one day. See
[Incomplete multipart upload lifecycle](architecture/storage.md#incomplete-multipart-upload-lifecycle).

## LLM policy

The MCP server has one LLM policy shared by every MCP tool:

| Policy | Behavior |
|--------|----------|
| `external` | Default. Tools that need LLM reasoning create pending tasks in session state. The MCP client reads them with `llm_get_task`, reasons with its own LLM, and submits results with `llm_submit_task_result` or a domain-specific submit tool. |
| `agno-model` | Loads only the configured Agno model into `MCPAgentContext.model`. Toolkit LLM calls such as ChEMBL judges and `engine="llm"` design can run in-process, but the Agno team is still not constructed or invoked. |
| `disabled` | Rejects LLM-dependent work and keeps deterministic/non-LLM tools available. |

The generic task lifecycle tools are `llm_create_task`,
`llm_list_pending_tasks`, `llm_get_task`, `llm_submit_task_result`, and
`llm_cancel_task`. Small-molecule and peptide design tools return
`status: "needs_external_llm"` in default mode when called with
`engine="llm"`.

## Private Agno delegation

By default the MCP server keeps external-client reasoning separate from the
Agno team. Trusted private deployments can opt into one coarse delegation tool:

```sh
cscopilot-mcp-serve --enable-agno-team-tool
```

This registers `agno_team_run(prompt)`, which loads the configured Agno model
and delegates the prompt to the cs_copilot Agno team. Prefer fine-grained MCP
skills and tools for normal external clients; use this flag only where the
client is trusted and model/API access is intentional.

## What the server exposes

### Tools

cs_copilot toolkit methods and workflow preflight helpers are exposed as MCP
tools, namespaced by toolkit or workflow (`chembl_*`, `chemspace_*`, `gtm_*`,
`chem_*`, `session_*`, `report_*`, `robustness_*`, `llm_*`). Tool arguments mirror
the toolkit method signatures, with the `agent` / `session_state` parameters
injected by the server and hidden from the public schema.

Each tool declares whether it may use the network (for example, ChEMBL or
PubChem access and Hugging Face model fallback). This contract is included in
ChatGPT-compatible tool documentation; a locally cached model can avoid an
actual request without changing the tool's network-capable classification.

For vague ChEMBL or chemical-space requests, call the preflight tools before
the longer-running execution tools. `mcp_bootstrap` itself is read-only.
`chembl_prepare_retrieval` and `chemspace_plan_analysis` persist validated plan
artifacts in the active run, so they are write-capable. Consequently, the
`bootstrap` profile as a whole must not be treated as read-only.

| Tool | Purpose |
|------|---------|
| `mcp_bootstrap` | Recommend the MCP-native prompt, workflow contract, skills, preflight tools, and ordered action plan for a user request. |
| `chembl_prepare_retrieval` | Validate retrieval requirements and persist the plan before `chembl_fetch_compounds`. |
| `chemspace_plan_analysis` | Classify chemical-space intent, identify missing details, and persist the plan before execution tools. |

The server attaches MCP tool annotations for ChatGPT / Apps approval UX:
`readOnlyHint=True` is used only for strict lookup, retrieval, listing, or
pure computation tools. Tools that fetch-and-store data, load mutable GTM
state, sample/register zones, update session memory, train models, or save
reports are advertised as write actions with `readOnlyHint=False`.
`destructiveHint` remains `False` for ordinary create-only workflow tools.
The explicit `workflow_abandon_tool_invocation` recovery action advertises
`destructiveHint=True` because it releases an in-flight lifecycle guard and
must receive deliberate approval, even though it does not delete an artifact.
Tools that may contact ChEMBL, PubChem, Hugging Face, or another external
service advertise `openWorldHint=True` and high risk even when a local cache
can satisfy a particular invocation.
`idempotentHint` is derived from the same execution contract used by retries
and idempotency-key handling.

The server also exposes two read-only ChatGPT compatibility tools:

| Tool | Purpose |
|------|---------|
| `search` | Search the cs_copilot MCP tool, prompt, skill, workflow, and active session artifact catalogs. |
| `fetch` | Fetch a search result by id, returning tool/prompt documentation or text artifact content. |

Use `cscopilot-mcp` once and then `list_tools` from your MCP client to see
the current set of direct cs_copilot tools plus the read-only `search` /
`fetch` compatibility tools.

### Prompts

The MCP server exposes a native external-client orchestration prompt plus the
agent and team instruction sets from `cs_copilot.agents.instructions`:

| Prompt | Role |
|--------|------|
| `cs_copilot_mcp_workflow` | MCP-native top-level orchestration. Use this first with `mcp_bootstrap`. |
| `cs_copilot_workflow` | Legacy team-level orchestration prompt, kept for compatibility. |
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

Only v2 run state and registered artifacts are exposed. Local and S3 backends
use the same canonical URIs:

- `cscopilot://runs/<run-id>/manifest.json`
- `cscopilot://runs/<run-id>/events/<event-id>.jsonl`
- `cscopilot://runs/<run-id>/artifacts/<artifact-id>`

Artifact URIs resolve through the artifact index and verify checksum and size
before returning content. Arbitrary session files are not resources. A new run
therefore exposes its manifest and events immediately, while artifact
resources appear only after registration.

## ChEMBL LLM-as-judge in MCP mode

The Chainlit / CLI runtime runs an LLM-as-judge step on `chembl_fetch_compounds`
results to filter rows pulled in by short / ambiguous keywords. That step
uses the configured Agno model.

In default MCP mode (`--llm-policy external`) the in-process judge is skipped:
the external MCP client is already the reasoning layer. The toolkit returns
rows that would otherwise need in-process judging and reports
`judge_status: "disabled"` / `metadata_judge_status: "disabled"` when relevant.

Use `chembl_create_external_judge_task` to create a pending retrieval or
metadata judge task from the same prompt templates as the in-process judge,
then submit validated decisions with `chembl_submit_external_judge_result`
or the generic `llm_submit_task_result`. The parameterised
`chembl_retrieval_judge` / `chembl_metadata_judge` prompts remain available
for clients that want to render the prompt directly.

For trusted private deployments, `--llm-policy agno-model` enables the
existing in-process ChEMBL LLM-as-judge code using the configured model only.
It does not enable `agno_team_run` and does not instantiate the Agno team.

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
- Toolkit specifications backed by the same factory share one instance inside
  a server, while separate server constructions receive separate instances.
  Factory functions therefore do not retain process-global mutable toolkits.

A guard test (`tests/unit/mcp/test_no_team_imports.py`) AST-walks the package
and fails CI if any of the forbidden imports reappear.
