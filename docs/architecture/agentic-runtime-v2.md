# Agentic Runtime v2

ChemSpace Copilot 0.2 introduces explicit workflow-run contracts that can be
used by the Agno team and external MCP clients. The language model chooses and
coordinates work; deterministic tools, artifact storage, policy checks, and
event replay remain ordinary Python infrastructure. Strict persisted task-DAG
and task-contract enforcement is currently implemented as the
`chembl-to-gtm-report` MCP pilot. Other catalog workflows retain their
taskless, legacy execution shape while still benefiting from run and artifact
contracts.

## Identity and lifecycle

A `session_id` identifies the chat or storage session, a `workflow_slug`
identifies a catalog procedure, and a `run_id` identifies one execution of
that procedure. Runs move through `submitted`, `planning`, `running`,
`input_required`, and a terminal `completed`, `partial`, `failed`, or
`cancelled` state. Workflows that declare tasks also give each task a validated
lifecycle and assigned role/profile.

A structured handoff carries only the objective, constraints, acceptance criteria, trace identifiers, and input/output artifact contracts needed by the receiving role. Full histories and private reasoning are not handoff data.

At run creation, the selected workflow and all transitive dependencies are
snapshotted with their semantic versions, Markdown procedures, and SHA-256
contract hashes. Resume and task validation use this pinned contract rather
than silently adopting later catalog edits.

## Artifact-backed source of truth

Each run writes beneath `workflows/<run_id>/`:

- `manifest.json`: rebuildable current-state snapshot.
- `events/*.jsonl`: immutable state-transition records used for replay.
- `artifacts/index.json`: checksummed artifact provenance.
- Operation folders for chemical space, design, retrosynthesis, and reports.

The event stream is authoritative. Manifest and artifact-index resources are
replayed from those immutable events, so a missing, stale, or failed snapshot
refresh cannot change run truth. SQLite stores conversation history, while
workflow events and artifacts provide reproducible scientific state.
Cross-session agent memory remains disabled.

Registered artifacts must remain inside their run root and record SHA-256, MIME type, producer task/tool, size, trust classification, and creation time. Local
artifact verification and reads use no-follow, descriptor-relative traversal;
MCP session-write arguments are normalized into the active run, and the
storage layer repeats that boundary check at the actual descriptor-relative,
no-follow open. Absolute, traversal, symlink-swap, reserved-runtime-file,
encoded-path, hard-link, and foreign-S3 escapes fail closed. Paths already
registered in the artifact ledger are immutable during domain-tool execution.
For the task-DAG pilot, required task inputs are selected explicitly, persisted
on the task, and checksum-verified before handoff, activation, and every domain
call. A workflow cannot complete until its declared run, task, and artifact
contracts are satisfied.

Domain writes use create-only invocation transactions. In-process files remain
private until the tool returns successfully; subprocess files remain beneath a
job-specific `.staging/` prefix until the parent accepts their checksums,
output limits, task epoch, and state merge. Registration failures release only
the new, checksum-matching files that have no authoritative artifact event.
Task and run lifecycle transitions, including failure transitions, wait until
each domain invocation reaches a durable terminal stage: `result_accepted` for
a successful registered result, `cache_hit`, `failed`, `cancelled`, or a
supervisor-confirmed `abandoned`. The later `completed` observation is
best-effort because result acceptance has already linearized publication and
registration. This prevents status changes from racing an in-flight mutation.

## Capability profiles

`cscopilot-mcp --profile <name>` registers only the selected profile's tools. The shipped profiles are `bootstrap`, `standard`, `chembl-retrieval`, `gtm-analysis`, `chemoinformatics`, `reporting`, `molecular-design`, `peptide-design`, `retrosynthesis`, and `robustness`. Unknown profiles and workflows whose required tools are unavailable fail before execution.

Parallel role policies enforce per-role toolkit allowlists for the in-process
coordinator and every specialist. For catalog workflows with a task DAG—currently the
`chembl-to-gtm-report` pilot—the MCP adapter also requires an active running
task and enforces its role, profile, and tool allowlist before domain
execution. Tool contracts include read/write,
destructive, open-world and idempotency annotations, execution limits, and
normalized errors. Network-capable tools are always advertised as open-world,
high-risk capabilities. Workflow loading also rejects permission metadata
that omits the network, compute, or artifact access implied by its declared
tools and artifact contracts.

The Agno team remains the central in-process supervisor. Member transcript
broadcasting is disabled, role factories reject tools outside their allowlists,
and the delegation guard validates handoff schema, private-context exclusion,
receiver role, and declared budgets. When the caller supplies a v2
`RunContext`, Agno additionally records the handoff through that runtime and
therefore receives its durable pinned-task validation. The default Chainlit
and CLI team constructors do not supply a `RunContext`; their structured
handoffs are ad hoc and process-local rather than durable workflow events.

For the MCP task-DAG pilot, handoffs are validated against the pinned task role
and exact capability, selected-input, output, and acceptance-criteria
contracts. Catalog tasks cannot start without an attempt-bound handoff, and
resuming from `input_required` or retrying a failed task requires a fresh one.
In MCP mode the server enforces its observable tool-call and elapsed-time
limits; the external reasoning client remains responsible for the declared
token budget because its token usage is not visible to the server. External
MCP clients are supervisors in MCP mode; the server never starts the Agno team
unless the separately gated compatibility tool is enabled.

Concurrent MCP calls with the same task-scoped idempotency key and arguments
coalesce behind one execution; key reuse with different arguments is rejected.
Retries remain idempotent-only. Async calls and subprocess workers have
cancellable deadlines, including the remaining handoff deadline. In-process
synchronous methods do not advertise a false cancellable timeout: cancellation
waits for their worker thread to finish, and a handoff overrun is rejected only
after that safe drain, so a mutation cannot continue invisibly and race a
retry. Subprocess session-state results use a three-way merge and reject
concurrent collisions or stale task scope atomically. Output limits still
bound inline responses. The replay evaluator checks semantic milestones,
role/tool permissions, preflight ordering, duplicate successful calls,
artifact types, and terminal status instead of comparing generated prose.

## Catalog and plugin boundaries

`skills/*/SKILL.md` and `workflow_catalog/*/WORKFLOW.md` remain the source of scientific procedures. Their metadata declares semantic versions, dependencies, required profiles/permissions, and input/output artifact contracts.

The repo plugin in `plugins/chemspace-copilot` is a delivery layer. It starts the MCP server and teaches Codex to bootstrap and fetch catalog procedures; it does not duplicate chemistry logic.

## Breaking migration from 0.1

Version 0.2 does not project v1 `session_state` keys or `cscopilot://session/` resources into the new schema. Existing files are left untouched but cannot be resumed as v2 runs. New resource URIs are run-scoped:

- `cscopilot://runs/<run_id>/manifest.json`
- `cscopilot://runs/<run_id>/events/<event-id>.jsonl`
- `cscopilot://runs/<run_id>/artifacts/<artifact-id>`
