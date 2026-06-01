"""JSONL logging hooks for Agno Agents and Teams.

Captures the full message exchange (system prompt, user input, assistant
content, tool calls, token usage) for every Agent / Team run using Agno's
native ``pre_hooks`` / ``post_hooks`` / ``tool_hooks`` extension points. One
JSON object per event is appended to a per-session JSONL file.

Activation
----------
Controlled by a single env var:

* ``CS_COPILOT_AGNO_LOG=1`` enables logging (default: off).
* ``CS_COPILOT_AGNO_LOG_DIR`` sets the output directory (default
  ``./logs/agno``).
* ``CS_COPILOT_AGNO_LOG_TRUNCATE`` truncates string fields above N chars
  (default 0 = no truncation).

Wire-in
-------
``attach_agno_hooks(agent_or_team)`` mutates the object's hook lists in place,
so calling it after ``Agent(...)`` / ``Team(...)`` is enough.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import threading
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def is_enabled() -> bool:
    """Return True iff the JSONL hook logger should be installed."""
    return _env_flag("CS_COPILOT_AGNO_LOG", default=False)


def _log_dir() -> Path:
    return Path(os.getenv("CS_COPILOT_AGNO_LOG_DIR", "./logs/agno")).expanduser()


def _truncate_chars() -> int:
    try:
        return int(os.getenv("CS_COPILOT_AGNO_LOG_TRUNCATE", "0"))
    except ValueError:
        return 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_dump(obj: Any, _depth: int = 0) -> Any:
    """Best-effort recursive serializer used as ``json.dumps(default=...)``.

    Handles pydantic models, dataclasses, sets, bytes, and arbitrary objects
    with ``to_dict`` / ``model_dump`` methods. Falls back to ``repr``.
    """
    if _depth > 8:
        return f"<truncated depth {_depth}>"
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe_dump(x, _depth + 1) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_dump(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, set):
        return [_safe_dump(x, _depth + 1) for x in obj]
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return f"<{len(obj)} bytes>"
    if isinstance(obj, datetime):
        return obj.isoformat()
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return _safe_dump(fn(), _depth + 1)
            except Exception:
                continue
    if is_dataclass(obj):
        try:
            return _safe_dump(asdict(obj), _depth + 1)
        except Exception:
            pass
    return repr(obj)


def _maybe_truncate(value: Any, limit: int) -> Any:
    if limit <= 0:
        return value
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"…[truncated {len(value) - limit} chars]"
    if isinstance(value, list):
        return [_maybe_truncate(v, limit) for v in value]
    if isinstance(value, dict):
        return {k: _maybe_truncate(v, limit) for k, v in value.items()}
    return value


class JsonlSink:
    """Append-only JSONL writer that routes events to per-session files.

    A process-wide singleton instance is shared by all hooks. Each event
    carries a ``session_id`` field; the sink opens (and caches) one append-only
    file handle per distinct session id, so concurrent chat threads in
    Chainlit each get their own JSONL log without manual plumbing.
    """

    _instance: Optional["JsonlSink"] = None
    _lock = threading.Lock()

    def __init__(self, log_dir: Path, default_session_id: Optional[str] = None):
        self.log_dir = log_dir
        self.default_session_id = (
            default_session_id or os.getenv("SESSION_ID") or "default"
        )
        self._handles: Dict[str, Any] = {}
        self._fh_lock = threading.Lock()
        self._truncate = _truncate_chars()

    @classmethod
    def get(cls) -> "JsonlSink":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(_log_dir())
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            if cls._instance is not None:
                cls._instance.close()
            cls._instance = None

    def path_for(self, session_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in session_id) or "default"
        return self.log_dir / f"{safe}.jsonl"

    def _handle_for(self, session_id: str):
        fh = self._handles.get(session_id)
        if fh is not None:
            return fh
        self.log_dir.mkdir(parents=True, exist_ok=True)
        fh = open(self.path_for(session_id), "a", encoding="utf-8")
        self._handles[session_id] = fh
        return fh

    def write(self, event: Dict[str, Any]) -> None:
        try:
            event.setdefault("ts", _now_iso())
            sid = event.get("session_id") or self.default_session_id
            event["session_id"] = sid
            payload = _safe_dump(event)
            if self._truncate:
                payload = _maybe_truncate(payload, self._truncate)
            line = json.dumps(payload, ensure_ascii=False, default=str)
            with self._fh_lock:
                fh = self._handle_for(sid)
                fh.write(line)
                fh.write("\n")
                fh.flush()
        except Exception as exc:
            logger.warning("agno_logging: failed to write event: %s", exc)

    def close(self) -> None:
        with self._fh_lock:
            for fh in list(self._handles.values()):
                try:
                    fh.close()
                except Exception:
                    pass
            self._handles.clear()


def _actor_name(agent: Any = None, team: Any = None) -> str:
    if team is not None:
        return getattr(team, "name", None) or "team"
    if agent is not None:
        return getattr(agent, "name", None) or "agent"
    return "unknown"


def _input_to_payload(run_input: Any) -> Any:
    if run_input is None:
        return None
    for attr in ("model_dump", "to_dict"):
        fn = getattr(run_input, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                continue
    return _safe_dump(run_input)


def _messages_to_payload(messages: Any) -> Any:
    if not messages:
        return []
    out = []
    for m in messages:
        to_dict = getattr(m, "to_dict", None)
        if callable(to_dict):
            try:
                out.append(to_dict())
                continue
            except Exception:
                pass
        out.append(_safe_dump(m))
    return out


def make_pre_hook(sink: JsonlSink, *, scope: str) -> Callable:
    """Build a pre_hook that logs the agent/team run start.

    ``scope`` is "agent" or "team" — used in the event name so consumers can
    distinguish coordinator events from member-agent events.
    """

    def pre_hook(  # type: ignore[no-untyped-def]
        run_input=None,
        agent=None,
        session=None,
        session_state=None,
        metadata=None,
        user_id=None,
        **kwargs,
    ):
        try:
            sink.write(
                {
                    "event": f"{scope}.run.start",
                    "actor": _actor_name(agent=agent, team=kwargs.get("team")),
                    "session_id": getattr(session, "session_id", None)
                    or sink.default_session_id,
                    "user_id": user_id,
                    "input": _input_to_payload(run_input),
                    "metadata": metadata,
                }
            )
        except Exception as exc:
            logger.warning("agno_logging pre_hook failed: %s", exc)

    pre_hook.__name__ = f"cs_copilot_{scope}_pre_log"
    return pre_hook


def _metrics_payload(metrics: Any) -> Any:
    if metrics is None:
        return None
    for attr in ("model_dump", "to_dict"):
        fn = getattr(metrics, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                continue
    return _safe_dump(metrics)


def _tools_payload(tools: Any) -> Any:
    if not tools:
        return []
    out = []
    for t in tools:
        to_dict = getattr(t, "to_dict", None)
        if callable(to_dict):
            try:
                out.append(to_dict())
                continue
            except Exception:
                pass
        out.append(_safe_dump(t))
    return out


def make_post_hook(sink: JsonlSink, *, scope: str) -> Callable:
    """Build a post_hook that logs the full message exchange after a run."""

    def post_hook(  # type: ignore[no-untyped-def]
        run_output=None,
        agent=None,
        session=None,
        session_state=None,
        metadata=None,
        user_id=None,
        **kwargs,
    ):
        try:
            sink.write(
                {
                    "event": f"{scope}.run.end",
                    "actor": _actor_name(agent=agent, team=kwargs.get("team")),
                    "run_id": getattr(run_output, "run_id", None),
                    "session_id": getattr(session, "session_id", None)
                    or getattr(run_output, "session_id", None)
                    or sink.default_session_id,
                    "user_id": user_id,
                    "status": str(getattr(run_output, "status", "")) or None,
                    "model": {
                        "id": getattr(run_output, "model", None),
                        "provider": getattr(run_output, "model_provider", None),
                    },
                    "messages": _messages_to_payload(
                        getattr(run_output, "messages", None)
                    ),
                    "tool_calls": _tools_payload(getattr(run_output, "tools", None)),
                    "content": getattr(run_output, "content", None),
                    "metrics": _metrics_payload(getattr(run_output, "metrics", None)),
                    "metadata": metadata,
                }
            )
        except Exception as exc:
            logger.warning("agno_logging post_hook failed: %s", exc)

    post_hook.__name__ = f"cs_copilot_{scope}_post_log"
    return post_hook


def make_tool_hook(sink: JsonlSink) -> Callable:
    """Build a tool_hook (middleware) that logs every tool call I/O.

    Agno passes hooks ``(function_name, function_call, arguments, **kw)`` and
    expects them to invoke ``function_call(**arguments)`` and return the
    result (see ``agno/tools/function.py:_build_hook_args``).

    In Agno's async execution chain, ``function_call`` (the inner ``next_func``)
    is always ``async def`` — even for sync tool entrypoints — so calling it
    returns a coroutine that must be awaited. We stay a sync hook (so the
    sync chain still runs us) and, when we detect a coroutine, hand back an
    awaitable that performs the await + result logging once the chain awaits
    us.
    """

    def tool_hook(  # type: ignore[no-untyped-def]
        function_name=None,
        function_call=None,
        arguments=None,
        agent=None,
        team=None,
        session_state=None,
        **kwargs,
    ):
        actor = _actor_name(agent=agent, team=team)
        started = time.perf_counter()
        sink.write(
            {
                "event": "tool.call.start",
                "actor": actor,
                "tool": function_name,
                "arguments": _safe_dump(arguments),
            }
        )

        def _log_end(status: str, *, result: Any = None, error: Optional[BaseException] = None) -> None:
            payload: Dict[str, Any] = {
                "event": "tool.call.end",
                "actor": actor,
                "tool": function_name,
                "status": status,
                "duration_ms": int((time.perf_counter() - started) * 1000),
            }
            if status == "ok":
                payload["result"] = _safe_dump(result)
            else:
                payload["error"] = repr(error)
            sink.write(payload)

        try:
            outcome = function_call(**(arguments or {}))
        except Exception as exc:
            _log_end("error", error=exc)
            raise

        if inspect.iscoroutine(outcome):
            async def _await_and_log():  # type: ignore[no-untyped-def]
                try:
                    value = await outcome
                except Exception as exc:
                    _log_end("error", error=exc)
                    raise
                _log_end("ok", result=value)
                return value

            return _await_and_log()

        _log_end("ok", result=outcome)
        return outcome

    tool_hook.__name__ = "cs_copilot_tool_log"
    return tool_hook


def _append(hooks_attr_value, new_hook):
    if hooks_attr_value is None:
        return [new_hook]
    if isinstance(hooks_attr_value, list):
        # Avoid duplicate installation (e.g. if attach_agno_hooks is called twice)
        if any(getattr(h, "__name__", "") == new_hook.__name__ for h in hooks_attr_value):
            return hooks_attr_value
        return list(hooks_attr_value) + [new_hook]
    return [hooks_attr_value, new_hook]


def attach_agno_hooks(
    target: Any,
    *,
    scope: str,
    sink: Optional[JsonlSink] = None,
) -> None:
    """Install JSONL hooks on an Agno Agent or Team.

    ``scope`` should be ``"agent"`` or ``"team"`` and becomes part of the
    event name.

    No-op if ``CS_COPILOT_AGNO_LOG`` is not enabled, so callers may invoke it
    unconditionally.
    """
    if not is_enabled():
        return
    sink = sink or JsonlSink.get()

    pre = make_pre_hook(sink, scope=scope)
    post = make_post_hook(sink, scope=scope)
    tool = make_tool_hook(sink)

    try:
        target.pre_hooks = _append(getattr(target, "pre_hooks", None), pre)
        target.post_hooks = _append(getattr(target, "post_hooks", None), post)
        target.tool_hooks = _append(getattr(target, "tool_hooks", None), tool)
    except Exception as exc:
        logger.warning("agno_logging: failed to attach hooks to %s: %s", target, exc)
        return

    # If the agent's tools have already been normalised into Function objects
    # they hold their own tool_hooks list. Propagate the new hook there too.
    tools = getattr(target, "tools", None) or []
    for t in tools:
        # Toolkits expose .functions: Dict[str, Function]
        fns = getattr(t, "functions", None)
        if isinstance(fns, dict):
            for fn in fns.values():
                existing = getattr(fn, "tool_hooks", None)
                fn.tool_hooks = _append(existing, tool)


def configure_session(session_id: str) -> JsonlSink:
    """Set the default session id used when an event has no explicit one."""
    sink = JsonlSink.get()
    sink.default_session_id = session_id
    return sink
