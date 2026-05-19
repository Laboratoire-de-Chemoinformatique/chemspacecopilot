# Logging

ChemSpace Copilot ships an opt-in **advanced logging layer** that captures the
full JSON payloads exchanged between every agent and the LLM, as well as
inter-agent coordination events emitted by the Team. Logging is off by default
and adds no overhead until activated.

## What gets captured

When activated, the system emits **JSON Lines** (one JSON object per line)
covering five event types:

| Event                | Emitted by                        | Key payload fields                                                        |
| -------------------- | --------------------------------- | -------------------------------------------------------------------------- |
| `team.run.start`     | Team coordinator pre-hook         | `input`, `session_id`, `user_id`, `metadata`                              |
| `team.run.end`       | Team coordinator post-hook        | `messages` (full system + user + assistant + tool roles), `tool_calls`, `content`, `metrics`, `model`, `status` |
| `agent.run.start`    | Each member Agent pre-hook        | same shape as `team.run.start`                                            |
| `agent.run.end`      | Each member Agent post-hook       | same shape as `team.run.end`                                              |
| `tool.call.start` / `tool.call.end` | Tool middleware around every function call | `tool`, `arguments`, `result`, `duration_ms`, `status` (`ok` / `error`) |

The `messages` array is produced by Agno's own `Message.to_dict()` and is
structurally identical to the OpenAI chat-completions wire format — system
prompt, user input, assistant content, `tool_calls`, `tool` responses, and
token usage are all present. That is the JSON the LLM actually saw and
returned.

Each event also carries:

- `ts` — UTC ISO-8601 timestamp
- `session_id` — the Chainlit thread id or CLI session id (events from
  different chat threads land in different files automatically)
- `actor` — agent or team name

An optional **second layer** records OpenTelemetry spans (trace-id /
parent-span-id hierarchy) for the same operations, including the raw
`request_params` Agno passed to the provider's HTTP client.

## How it is implemented

The implementation lives entirely on Agno's documented extension points and
does not monkey-patch any private API.

### Layer 1 — JSONL hooks (no extra dependencies)

`src/cs_copilot/tracking/agno_logging.py` defines three hook factories and a
shared sink:

- `make_pre_hook(sink, scope=...)` — logs the run start with the resolved
  input.
- `make_post_hook(sink, scope=...)` — logs the run end with
  `run_output.messages`, `run_output.tools`, `run_output.metrics`, and the
  model id / provider.
- `make_tool_hook(sink)` — Agno middleware that wraps every function call,
  capturing `arguments`, the return value, and wall-clock duration.

`attach_agno_hooks(target, scope=...)` mutates the target's
`pre_hooks` / `post_hooks` / `tool_hooks` lists in place. It is called from:

- `src/cs_copilot/agents/factories.py` — right after each `Agent(...)`.
- `src/cs_copilot/agents/teams.py` — right after the `Team(...)`.

Because the call is unconditional, but `attach_agno_hooks` is a no-op unless
`CS_COPILOT_AGNO_LOG=1`, there is zero runtime cost when logging is disabled.

The `JsonlSink` is a process-wide singleton that opens **one append-only
file per `session_id`** on first use. Concurrent Chainlit threads therefore
write to separate files without any per-thread plumbing.

### Layer 2 — OpenTelemetry (optional, install `[otel]` extra)

`src/cs_copilot/tracking/agno_otel.py` calls
[`AgnoInstrumentor().instrument()`](https://docs.agno.com/observability) from
the official `openinference-instrumentation-agno` package. Spans cover every
Team run → Agent run → model invocation → tool call, with the raw HTTP
request parameters attached as span attributes.

Without an OTLP endpoint, spans are written to
`$CS_COPILOT_AGNO_LOG_DIR/spans.jsonl` by a built-in file exporter, so the
OTel layer is useful in fully offline / Docker-only setups. With an endpoint
set, they ship to Langfuse, Phoenix, OpenLIT, or any OTLP-compatible
collector.

### File layout

```
logs/agno/
├── <session-id-1>.jsonl    # one line per event, per session
├── <session-id-2>.jsonl
└── spans.jsonl             # OTel spans (only if CS_COPILOT_OTEL=1
                            # and no OTLP endpoint is configured)
```

## How to use it

### Direct (`uv run`)

```bash
# JSONL message logging only
CS_COPILOT_AGNO_LOG=1 uv run cscopilot

# Custom log location
CS_COPILOT_AGNO_LOG=1 \
CS_COPILOT_AGNO_LOG_DIR=/var/log/cscopilot/agno \
uv run cscopilot

# Also enable OpenTelemetry tracing
uv sync --extra otel
CS_COPILOT_AGNO_LOG=1 CS_COPILOT_OTEL=1 uv run cscopilot

# Ship spans to a remote backend (e.g. self-hosted Langfuse)
CS_COPILOT_OTEL=1 \
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:3000/api/public/otel \
OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic $(echo -n "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" | base64)" \
uv run cscopilot
```

### Docker Compose

The compose stack reads the same env vars from your shell or `.env` and
already mounts `./logs:/app/logs` into the `chainlit-app` container.

```bash
# In .env (or exported in the shell before `docker compose up`)
CS_COPILOT_AGNO_LOG=1
CS_COPILOT_AGNO_LOG_DIR=/app/logs/agno     # default inside the container
# CS_COPILOT_AGNO_LOG_TRUNCATE=4000        # cap long content fields

# Optional OTel layer
# CS_COPILOT_OTEL=1
# OTEL_EXPORTER_OTLP_ENDPOINT=...
# OTEL_EXPORTER_OTLP_HEADERS=...
```

```bash
docker compose up -d
# logs appear on the host under ./logs/agno/<thread-id>.jsonl
tail -F logs/agno/*.jsonl
```

To enable the OTel layer inside Docker, edit `Dockerfile` to add `--extra otel`
to the `uv sync` invocation, or rebuild after running
`uv sync --extra otel` locally so the lockfile picks up the optional packages.

### Environment variable reference

| Variable                       | Default            | Effect                                                                 |
| ------------------------------ | ------------------ | ---------------------------------------------------------------------- |
| `CS_COPILOT_AGNO_LOG`          | `0`                | `1`/`true` activates the JSONL hook logger.                            |
| `CS_COPILOT_AGNO_LOG_DIR`      | `./logs/agno`      | Output directory for `*.jsonl` (and OTel `spans.jsonl`).               |
| `CS_COPILOT_AGNO_LOG_TRUNCATE` | `0`                | If `>0`, truncates string fields above N characters per event.         |
| `CS_COPILOT_OTEL`              | `0`                | `1`/`true` activates OpenInference auto-instrumentation.               |
| `OTEL_EXPORTER_OTLP_ENDPOINT`  | unset              | If set, OTel spans ship via OTLP/HTTP; else written to `spans.jsonl`.  |
| `OTEL_EXPORTER_OTLP_HEADERS`   | unset              | Used for auth headers (Langfuse, Phoenix, etc.).                       |

## Reading the logs

Each line is a self-contained JSON object. Typical recipes:

```bash
# All system prompts seen this run
jq 'select(.event=="agent.run.end") | .messages[] | select(.role=="system") | .content' \
  logs/agno/<session>.jsonl | head

# All tool calls with their args + duration
jq 'select(.event=="tool.call.end") | {tool, duration_ms, status, args:.arguments}' \
  logs/agno/<session>.jsonl

# Replay the full transcript of one agent run
jq 'select(.event=="agent.run.end" and .actor=="GTM Agent") | .messages' \
  logs/agno/<session>.jsonl

# Aggregate token usage per agent
jq -r 'select(.event=="agent.run.end") | [.actor, .metrics.input_tokens, .metrics.output_tokens] | @tsv' \
  logs/agno/<session>.jsonl
```

## Caveats

- **Sensitive data.** The JSONL captures the *entire* conversation including
  user input and any uploaded SMILES, chemistry data, or API responses. Treat
  the log directory the same way you treat `.env` — exclude it from VCS and
  rotate it periodically. The default `./logs/` directory is already covered
  by the project `.gitignore`.
- **Size.** With long ChEMBL downloads or large tool results, individual
  events can be several MB. Set `CS_COPILOT_AGNO_LOG_TRUNCATE` to cap them if
  you only need a structural trace.
- **Streaming.** Hook-based logging fires once per *run*, after the agent has
  finished, not chunk-by-chunk. For live token-level visibility, switch to the
  OTel layer and ship spans to Langfuse or Phoenix.
- **MLflow.** The existing MLflow wrapper in `factories.py` is untouched. It
  continues to log per-run metrics and prompt-registry versions, complementing
  (rather than duplicating) the JSONL message exchange.
