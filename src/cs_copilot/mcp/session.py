"""Bootstrap MCP storage, workflow-run state, and execution context."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class BootstrapConfig:
    """Parsed values describing how to bootstrap the MCP server process."""

    session_id: Optional[str] = None
    run_id: Optional[str] = None
    workflow_slug: Optional[str] = None
    profile: str = "standard"
    log_level: str = "info"
    llm_policy: str = "external"


_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def configure_logging(level: str) -> None:
    """Configure stderr logging without contaminating stdio JSON-RPC."""

    numeric = _LEVELS.get(level.lower(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=__import__("sys").stderr,
        force=True,
    )


def apply_session_id(session_id: Optional[str]) -> None:
    """Persist ``session_id`` before importing storage-dependent toolkits."""

    if session_id:
        os.environ["SESSION_ID"] = session_id


def bootstrap(config: BootstrapConfig):
    """Create or resume a v2 workflow run and bind the MCP agent shim."""

    from cs_copilot.storage import S3
    from cs_copilot.workflows import RunContext

    from .context import (
        MCPAgentContext,
        restore_active_task_scope,
        set_current_context,
    )
    from .llm import LLMBroker, normalize_llm_policy
    from .profiles import (
        get_profile,
        validate_pinned_workflow_profile,
        validate_workflow_profile,
    )

    requested_session = config.session_id or os.environ.get("SESSION_ID", "").strip()
    if requested_session:
        S3.set_session_prefix(f"sessions/{requested_session}")

    profile = get_profile(config.profile)
    if config.workflow_slug and not config.run_id:
        validate_workflow_profile(profile, config.workflow_slug)
    llm_policy = normalize_llm_policy(config.llm_policy)
    model = None
    if llm_policy == "agno-model":
        from .llm.agno_model import load_configured_model

        model = load_configured_model()

    ctx = MCPAgentContext(model=model, llm_policy=llm_policy)
    ctx.llm = LLMBroker(ctx)
    workflow_slug = config.workflow_slug or "mcp-session"
    if config.run_id:
        run_context = RunContext.resume(
            config.run_id,
            session_id=requested_session or config.session_id,
        )
        if config.workflow_slug and run_context.run.workflow_slug != config.workflow_slug:
            raise ValueError(
                f"Run {config.run_id!r} belongs to workflow "
                f"{run_context.run.workflow_slug!r}, not {config.workflow_slug!r}"
            )
        if run_context.run.workflow_slug != "mcp-session":
            validate_pinned_workflow_profile(
                profile,
                run_context.run.workflow_contract,
            )
        run_context.bind_session_state(ctx.session_state)
        restore_active_task_scope(ctx.session_state, run_context.run)
    else:
        run_context = RunContext.create(
            workflow_slug,
            session_state=ctx.session_state,
            session_id=requested_session or config.session_id,
        )

    # MCPAgentContext intentionally remains a light shim. These dynamic
    # attributes are process-local; only serializable identity is stored in
    # session_state.
    ctx.run_context = run_context
    ctx.mcp_profile = profile.name
    ctx.session_state["mcp_profile"] = profile.name
    set_current_context(ctx)
    return ctx
