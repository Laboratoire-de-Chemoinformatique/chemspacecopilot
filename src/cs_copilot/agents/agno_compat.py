"""Compatibility fixes for the pinned Agno release.

Agno 2.1.9's ``Model.arun_function_call`` leaves its ``result`` unassigned
when a tool's pre-hook raises an ``AgentRunException`` (``RetryAgentRun`` /
``StopAgentRun``), so the async team path crashes with ``UnboundLocalError``
instead of turning the exception into a retry request for the model, as the
synchronous ``run_function_call`` does. The structured-handoff guard rejects
malformed delegations exactly that way, so without this fix any rejected
delegation in an async chat (Chainlit, the CLI) aborts the whole turn.

:func:`patch_async_function_call_retry` installs the upstream method with the
single missing initialization. It only applies to the pinned version and is
idempotent.
"""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version

logger = logging.getLogger(__name__)

PATCHED_AGNO_VERSIONS = frozenset({"2.1.9"})
_PATCH_MARKER = "__cs_copilot_retry_fix__"


def patch_async_function_call_retry() -> bool:
    """Make async tool pre-hook retries behave like the synchronous path.

    Returns whether the patch is active.
    """

    from agno.models.base import Model

    if getattr(Model.arun_function_call, _PATCH_MARKER, False):
        return True
    try:
        installed = version("agno")
    except PackageNotFoundError:  # pragma: no cover - agno is a hard dependency
        return False
    if installed not in PATCHED_AGNO_VERSIONS:
        logger.warning(
            "Not applying the async tool-retry fix for agno %s (validated for %s); "
            "check whether Model.arun_function_call still needs it.",
            installed,
            ", ".join(sorted(PATCHED_AGNO_VERSIONS)),
        )
        return False

    from inspect import isasyncgenfunction, iscoroutine, iscoroutinefunction

    from agno.exceptions import AgentRunException
    from agno.tools.function import FunctionExecutionResult
    from agno.utils.log import log_error
    from agno.utils.timer import Timer

    async def arun_function_call(self, function_call):
        """Run a single function call and return its success status, timer, and the FunctionCall object."""

        function_call_timer = Timer()
        function_call_timer.start()
        success = False
        # The only change from agno 2.1.9: a pre-hook AgentRunException must
        # still return a result, exactly like the synchronous path.
        result = FunctionExecutionResult(status="failure")

        try:
            if (
                iscoroutinefunction(function_call.function.entrypoint)
                or isasyncgenfunction(function_call.function.entrypoint)
                or iscoroutine(function_call.function.entrypoint)
            ):
                result = await function_call.aexecute()
                success = result.status == "success"

            # If any of the hooks are async, we need to run the function call asynchronously
            elif function_call.function.tool_hooks is not None and any(
                iscoroutinefunction(f) for f in function_call.function.tool_hooks
            ):
                result = await function_call.aexecute()
                success = result.status == "success"
            else:
                import asyncio

                result = await asyncio.to_thread(function_call.execute)
                success = result.status == "success"
        except AgentRunException as e:
            success = e
        except Exception as e:
            log_error(f"Error executing function {function_call.function.name}: {e}")
            success = False
            raise e

        function_call_timer.stop()
        return success, function_call_timer, function_call, result

    setattr(arun_function_call, _PATCH_MARKER, True)
    Model.arun_function_call = arun_function_call
    return True
