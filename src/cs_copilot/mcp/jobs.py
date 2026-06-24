"""Subprocess execution support for heavy MCP tools."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from cs_copilot.storage import S3

from .context import MCPAgentContext
from .errors import MCPToolError

DEFAULT_WORKER_TIMEOUT_S = 900.0
_MAX_ERROR_CHARS = 4000


def run_tool_job(spec: Any, kwargs: Mapping[str, Any], ctx: MCPAgentContext) -> Any:
    """Run a heavy MCP tool in a dedicated subprocess and merge session state."""

    timeout_s = float(spec.worker_timeout_s or DEFAULT_WORKER_TIMEOUT_S)
    payload = {
        "tool_name": spec.mcp_name,
        "kwargs": dict(kwargs),
        "session_state": ctx.session_state,
        "session_prefix": S3.current_prefix(),
        "llm_policy": getattr(ctx, "llm_policy", "external"),
    }

    with tempfile.TemporaryDirectory(prefix=f"cscopilot-mcp-{spec.mcp_name}-") as tmp:
        tmp_path = Path(tmp)
        job_path = tmp_path / "job.json"
        result_path = tmp_path / "result.json"
        _write_json(job_path, payload)

        stdout, stderr = _run_worker_process(
            job_path=job_path,
            result_path=result_path,
            timeout_s=timeout_s,
        )

        if not result_path.exists():
            details = _format_worker_details(stdout=stdout, stderr=stderr)
            raise MCPToolError(f"{spec.mcp_name} worker did not write a result file.{details}")

        result_payload = _read_json(result_path)
        if not isinstance(result_payload, dict):
            raise MCPToolError(f"{spec.mcp_name} worker returned a malformed result.")

        session_state = result_payload.get("session_state")
        if result_payload.get("ok") and isinstance(session_state, dict):
            ctx.session_state.clear()
            ctx.session_state.update(session_state)

        if not result_payload.get("ok"):
            error = str(result_payload.get("error") or "unknown worker error")
            traceback_text = str(result_payload.get("traceback") or "")
            details = _short_text(
                "\n".join(part for part in (error, traceback_text, stderr) if part)
            )
            raise MCPToolError(f"{spec.mcp_name} worker failed: {details}")

        return result_payload.get("result")


def _run_worker_process(
    *,
    job_path: Path,
    result_path: Path,
    timeout_s: float,
) -> tuple[str, str]:
    env = os.environ.copy()
    src_root = Path(__file__).resolve().parents[2]
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(src_root)
        if not existing_pythonpath
        else os.pathsep.join([str(src_root), existing_pythonpath])
    )

    cmd = [
        sys.executable,
        "-m",
        "cs_copilot.mcp.worker",
        "--job",
        str(job_path),
        "--result",
        str(result_path),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=os.getcwd(),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(proc)
        stdout, stderr = proc.communicate()
        details = _format_worker_details(stdout=stdout, stderr=stderr)
        raise MCPToolError(
            f"MCP worker timed out after {timeout_s:.0f}s for job {job_path.name}.{details}"
        ) from exc

    if proc.returncode != 0:
        details = _format_worker_details(stdout=stdout, stderr=stderr)
        raise MCPToolError(
            f"MCP worker exited with code {proc.returncode} for job {job_path.name}.{details}"
        )
    return stdout, stderr


def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.write("\n")
    except TypeError as exc:
        raise MCPToolError(f"MCP worker job payload is not JSON-serializable: {exc}") from exc


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _format_worker_details(*, stdout: str, stderr: str) -> str:
    details = []
    if stdout:
        details.append(f"stdout:\n{_short_text(stdout)}")
    if stderr:
        details.append(f"stderr:\n{_short_text(stderr)}")
    return "" if not details else "\n" + "\n".join(details)


def _short_text(text: str, max_chars: int = _MAX_ERROR_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."
