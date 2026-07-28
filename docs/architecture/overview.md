# Architecture Overview

## System Layers

```
┌─────────────────────────────────────────┐
│  UI Layer (Chainlit)                    │  Entry point
├─────────────────────────────────────────┤
│  Agent Orchestration (teams.py)         │  Multi-agent coordination
├─────────────────────────────────────────┤
│  Specialized Agents (factories.py)      │  7 runtime agents + 1 evaluation agent
├─────────────────────────────────────────┤
│  Tools + Storage (toolkits + S3)        │  Domain logic & persistence
└─────────────────────────────────────────┘
```

## Entry Point

**Chainlit** (`chainlit_app.py`):

- WebSocket-based real-time chat interface
- Per-session agent teams with authentication
- Tool call visualization as steps
- SMILES to inline molecule images
- Streaming response display
- Chainlit persistence disabled by default
- File upload support with S3 integration

## Streaming Response Pattern

Chainlit uses streaming for real-time display:

```python
for chunk in agent.run(prompt, stream=True):
    if is_tool_event(chunk):
        display_as_step(chunk)  # Tool calls shown as Chainlit Steps
    elif is_text_chunk(chunk):
        stream_to_ui(chunk)     # Text streamed to message
```

This allows users to see progress as agents work, rather than waiting for completion.

## Agent State Management

Agents use `session_state` (a persistent dict) to pass data between runs and between agents:

```python
# Save in one agent
agent.session_state["data_path"] = "results.csv"

# Access in another agent (same team)
path = agent.session_state.get("data_path")
```

## Agent Coordination

The `get_cs_copilot_agent_team()` function in `teams.py` creates a coordinated team:

```python
team = get_cs_copilot_agent_team(model)
# Creates Team with:
# - 7 runtime agents
# - SQLite persistence for session-local chat history
# - Bounded context management (num_history_runs=5)
# - Structured, artifact-referenced member handoffs
# - No cross-session user/agent memory
# - Streaming event propagation
```

Capabilities:

- **Session History**: Recent history can be persisted in SQLite and reused
  only inside the active session
- **Context Sharing**: Specialists receive explicit handoff facts and
  artifact references; coordinator state and full transcripts are not injected
  into member prompts
- **Bounded Handoffs**: Member transcripts are not broadcast; specialists
  receive structured task facts, artifact ids, acceptance criteria, and budgets
- **Streaming**: Real-time event propagation from member agents to UI
- **Scientific State**: V2 workflow events and checksummed artifacts, rather
  than agent memory, are the replayable source of truth

The default Chainlit and CLI team runs in ad-hoc handoff mode: its structured
handoffs are schema-, role-, and budget-validated, but remain process-local.
Callers that explicitly construct the team with a v2 `RunContext` additionally
receive durable, pinned task-contract validation and handoff events.
