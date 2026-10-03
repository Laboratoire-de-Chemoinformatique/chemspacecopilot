"""Runtime-neutral tool execution kernel.

Every runtime that lets a reasoner call cs_copilot toolkits drives each call
through the same pipeline: catalog-task authorization and budgets, read and
write boundaries that confine files to the active workflow run, idempotency,
checksummed artifact registration and rollback, durable ``tool_progress`` /
``tool_call_recorded`` events, and a stable result envelope.

* :mod:`cs_copilot.mcp.tool_adapter` exposes the pipeline to external MCP
  clients.
* The in-process Agno runtime uses the synchronous orchestrator in
  :mod:`cs_copilot.execution.runner`.

This package must stay importable without either runtime: it never imports
``agno``, ``cs_copilot.agents``, ``cs_copilot.mcp``, or the ``mcp`` SDK, and its
modules import storage, workflows, and toolkit helpers lazily because
``cs_copilot.storage`` reads ``SESSION_ID`` at import time. Import from the
submodules directly; this package initializer intentionally exports nothing.
"""
