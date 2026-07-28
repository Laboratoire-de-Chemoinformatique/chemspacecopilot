"""Subprocess execution support for heavy MCP tools."""

from __future__ import annotations

import errno
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, MutableMapping

from cs_copilot.storage import S3

from .context import MCPAgentContext
from .errors import MCPErrorCode, MCPToolError

DEFAULT_WORKER_TIMEOUT_S = 900.0
_MAX_ERROR_CHARS = 4000
_MISSING = object()
_SESSION_STATE_MERGE_LOCK = threading.RLock()
_EXECUTION_SCOPE_KEYS = frozenset(
    {
        "output_context",
        "active_task_id",
        "active_role",
        "active_profile",
        "active_task_attempt",
        "active_handoff_id",
        "mcp_profile",
    }
)


@dataclass
class DeferredToolJob:
    """Worker result whose state and artifacts await parent-side acceptance."""

    result: Any
    ctx: MCPAgentContext
    base_session_state: Mapping[str, Any]
    worker_session_state: Mapping[str, Any]
    tool_name: str
    retryable: bool
    publications: Mapping[str, Mapping[str, Any]]
    write_boundary: str | None
    staging_id: str | None
    settled: bool = False

    def accept(self) -> None:
        if self.settled:
            return
        promoted = False
        try:
            if self.publications:
                S3.promote_staged_publications(self.publications)
                promoted = True
            if self.write_boundary and self.staging_id:
                S3.discard_staging_prefix(self.write_boundary, self.staging_id)
            _merge_worker_session_state(
                target=self.ctx.session_state,
                base=self.base_session_state,
                worker=self.worker_session_state,
                tool_name=self.tool_name,
                retryable=self.retryable,
            )
            self.settled = True
        except BaseException:
            if promoted:
                S3.rollback_promoted_publications(self.publications)
            self.abort()
            raise

    def abort(self) -> None:
        if self.settled:
            return
        if self.write_boundary and self.staging_id:
            S3.discard_staging_prefix(self.write_boundary, self.staging_id)
        self.settled = True


def _validate_staged_publications(
    publications: Mapping[str, Any],
    *,
    write_boundary: str | None,
    staging_id: str | None,
    protected_paths: Any,
) -> dict[str, dict[str, Any]]:
    """Bind worker-returned paths to its exact run/job staging namespace."""

    if write_boundary is None or staging_id is None:
        if publications:
            raise ValueError("staged publications require a captured write boundary and job id")
        return {}
    boundary = _safe_relative_path(write_boundary, field="write_boundary")
    job_id = _safe_relative_path(staging_id, field="staging_id")
    if len(job_id.parts) != 1:
        raise ValueError("staging_id must be one path component")
    protected = {
        _safe_relative_path(path, field="protected_path").as_posix() for path in protected_paths
    }
    validated: dict[str, dict[str, Any]] = {}
    seen_staged: set[str] = set()
    for raw_final, raw_metadata in publications.items():
        if not isinstance(raw_metadata, Mapping):
            raise ValueError(f"publication metadata for {raw_final!r} must be an object")
        if set(raw_metadata) != {"staged_path", "sha256", "size_bytes"}:
            raise ValueError(f"publication metadata for {raw_final!r} has unexpected fields")
        raw_staged = raw_metadata["staged_path"]
        sha256 = raw_metadata["sha256"]
        size_bytes = raw_metadata["size_bytes"]
        if (
            not isinstance(raw_staged, str)
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            raise ValueError(f"publication integrity metadata for {raw_final!r} is invalid")
        final = _safe_relative_path(raw_final, field="publication final")
        staged = _safe_relative_path(raw_staged, field="publication staged path")
        try:
            relative = final.relative_to(boundary)
        except ValueError as exc:
            raise ValueError(f"final path {final!s} is outside the active run") from exc
        if not relative.parts or relative.parts[0] in {".staging", "events"}:
            raise ValueError(f"final path {final!s} targets a reserved run namespace")
        if relative in {
            PurePosixPath("manifest.json"),
            PurePosixPath("artifacts", "index.json"),
        }:
            raise ValueError(f"final path {final!s} targets runtime metadata")
        if final.as_posix() in protected:
            raise ValueError(f"final path {final!s} is an immutable registered artifact")
        expected_staged = PurePosixPath(
            boundary,
            ".staging",
            job_id,
            relative,
        )
        if staged != expected_staged:
            raise ValueError(f"staged path for {final!s} must be exactly {expected_staged!s}")
        if staged.as_posix() in seen_staged:
            raise ValueError(f"duplicate staged publication path: {staged!s}")
        seen_staged.add(staged.as_posix())
        validated[final.as_posix()] = {
            "staged_path": staged.as_posix(),
            "sha256": sha256,
            "size_bytes": size_bytes,
        }
    return validated


def _safe_relative_path(value: Any, *, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip() or "\\" in value:
        raise ValueError(f"{field} must be a safe relative path")
    path = PurePosixPath(value.strip())
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must be a safe relative path")
    return path


def run_tool_job(
    spec: Any,
    kwargs: Mapping[str, Any],
    ctx: MCPAgentContext,
    *,
    before_state_merge: Callable[[], None] | None = None,
    defer_commit: bool = False,
) -> Any:
    """Run a heavy MCP tool in a dedicated subprocess and merge session state."""

    timeout_s = float(spec.worker_timeout_s or DEFAULT_WORKER_TIMEOUT_S)
    base_session_state = _snapshot_session_state(ctx.session_state)
    write_boundary = S3.current_write_boundary()
    staging_id = (
        f"job-{uuid.uuid4().hex}" if defer_commit and isinstance(write_boundary, str) else None
    )
    payload = {
        "tool_name": spec.mcp_name,
        "kwargs": dict(kwargs),
        "session_state": base_session_state,
        "session_prefix": S3.current_prefix(),
        "write_boundary": write_boundary,
        "write_protected_paths": sorted(S3.current_write_protected_paths()),
        "verified_artifact_reads": {
            path: [sha256, size_bytes]
            for path, (sha256, size_bytes) in S3.current_verified_artifact_reads().items()
        },
        "llm_policy": getattr(ctx, "llm_policy", "external"),
        "staging_id": staging_id,
    }

    def discard_staging() -> None:
        if isinstance(write_boundary, str) and staging_id is not None:
            S3.discard_staging_prefix(write_boundary, staging_id)

    with tempfile.TemporaryDirectory(prefix=f"cscopilot-mcp-{spec.mcp_name}-") as tmp:
        tmp_path = Path(tmp)
        job_path = tmp_path / "job.json"
        result_path = tmp_path / "result.json"
        _write_json(job_path, payload)

        try:
            stdout, stderr = _run_worker_process(
                job_path=job_path,
                result_path=result_path,
                timeout_s=timeout_s,
            )
        except BaseException:
            discard_staging()
            raise

        if not result_path.exists():
            discard_staging()
            details = _format_worker_details(stdout=stdout, stderr=stderr)
            raise MCPToolError(
                f"{spec.mcp_name} worker did not write a result file.{details}",
                code=MCPErrorCode.INTERNAL,
            )

        try:
            result_payload = _read_json(result_path)
        except (json.JSONDecodeError, UnicodeError) as exc:
            discard_staging()
            raise MCPToolError(
                f"{spec.mcp_name} worker wrote a malformed JSON result: {exc}",
                code=MCPErrorCode.INTERNAL,
            ) from exc
        except MemoryError as exc:
            discard_staging()
            raise MCPToolError(
                f"{spec.mcp_name} worker result exceeded available memory.",
                code=MCPErrorCode.RESOURCE_LIMIT,
            ) from exc
        except OSError as exc:
            discard_staging()
            raise MCPToolError(
                f"{spec.mcp_name} worker result could not be read: {exc}",
                code=_filesystem_error_code(exc),
            ) from exc

        if not isinstance(result_payload, dict):
            discard_staging()
            raise MCPToolError(
                f"{spec.mcp_name} worker returned a malformed result.",
                code=MCPErrorCode.INTERNAL,
            )

        ok = result_payload.get("ok")
        if not isinstance(ok, bool):
            discard_staging()
            raise MCPToolError(
                f"{spec.mcp_name} worker result omitted the boolean 'ok' field.",
                code=MCPErrorCode.INTERNAL,
            )

        session_state = result_payload.get("session_state")
        if ok and not isinstance(session_state, dict):
            discard_staging()
            raise MCPToolError(
                f"{spec.mcp_name} worker returned malformed session state.",
                code=MCPErrorCode.INTERNAL,
            )
        staged_publications = result_payload.get("staged_publications", {})
        if not isinstance(staged_publications, dict) or any(
            not isinstance(key, str) for key in staged_publications
        ):
            discard_staging()
            raise MCPToolError(
                f"{spec.mcp_name} worker returned malformed staged publications.",
                code=MCPErrorCode.INTERNAL,
            )
        if ok:
            if defer_commit:
                try:
                    validated_publications = _validate_staged_publications(
                        staged_publications,
                        write_boundary=write_boundary,
                        staging_id=staging_id,
                        protected_paths=payload["write_protected_paths"],
                    )
                except (TypeError, ValueError) as exc:
                    discard_staging()
                    raise MCPToolError(
                        f"{spec.mcp_name} worker returned unsafe staged " f"publications: {exc}",
                        code=MCPErrorCode.PERMISSION_DENIED,
                    ) from exc
                return DeferredToolJob(
                    result=result_payload.get("result"),
                    ctx=ctx,
                    base_session_state=base_session_state,
                    worker_session_state=session_state,
                    tool_name=spec.mcp_name,
                    retryable=bool(getattr(spec, "idempotent", False)),
                    publications=validated_publications,
                    write_boundary=write_boundary,
                    staging_id=staging_id,
                )
            if before_state_merge is not None:
                before_state_merge()
            _merge_worker_session_state(
                target=ctx.session_state,
                base=base_session_state,
                worker=session_state,
                tool_name=spec.mcp_name,
                retryable=bool(getattr(spec, "idempotent", False)),
            )

        if not ok:
            discard_staging()
            error = str(result_payload.get("error") or "unknown worker error")
            traceback_text = str(result_payload.get("traceback") or "")
            details = _short_text(
                "\n".join(part for part in (error, traceback_text, stderr) if part)
            )
            code, retryable = _worker_error_metadata(result_payload)
            raise MCPToolError(
                f"{spec.mcp_name} worker failed: {details}",
                code=code,
                retryable=retryable,
            )

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
            f"MCP worker timed out after {timeout_s:.0f}s for job {job_path.name}.{details}",
            code=MCPErrorCode.TIMEOUT,
            retryable=True,
        ) from exc

    if proc.returncode != 0:
        details = _format_worker_details(stdout=stdout, stderr=stderr)
        raise MCPToolError(
            f"MCP worker exited with code {proc.returncode} for job {job_path.name}.{details}",
            code=MCPErrorCode.INTERNAL,
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
    except (TypeError, ValueError) as exc:
        raise MCPToolError(
            f"MCP worker job payload is not JSON-serializable: {exc}",
            code=MCPErrorCode.INVALID_INPUT,
        ) from exc
    except (MemoryError, OverflowError, RecursionError) as exc:
        raise MCPToolError(
            "MCP worker job payload exceeded an available serialization resource.",
            code=MCPErrorCode.RESOURCE_LIMIT,
        ) from exc
    except OSError as exc:
        raise MCPToolError(
            f"MCP worker job payload could not be written: {exc}",
            code=_filesystem_error_code(exc),
        ) from exc


def _snapshot_session_state(session_state: Mapping[str, Any]) -> dict[str, Any]:
    """Return the detached JSON state that a worker will receive."""

    try:
        snapshot = json.loads(json.dumps(session_state))
    except (TypeError, ValueError) as exc:
        raise MCPToolError(
            f"MCP worker session state is not JSON-serializable: {exc}",
            code=MCPErrorCode.INVALID_INPUT,
        ) from exc
    except (MemoryError, OverflowError, RecursionError) as exc:
        raise MCPToolError(
            "MCP worker session-state snapshot exceeded an available serialization resource.",
            code=MCPErrorCode.RESOURCE_LIMIT,
        ) from exc
    except RuntimeError as exc:
        raise MCPToolError(
            "MCP worker session state changed while its snapshot was being created.",
            code=MCPErrorCode.INTERNAL,
        ) from exc
    if not isinstance(snapshot, dict):  # pragma: no cover - the input contract is a mapping
        raise MCPToolError(
            "MCP worker session-state snapshot was not an object.",
            code=MCPErrorCode.INTERNAL,
        )
    return snapshot


def _merge_worker_session_state(
    *,
    target: MutableMapping[str, Any],
    base: Mapping[str, Any],
    worker: Mapping[str, Any],
    tool_name: str,
    retryable: bool,
) -> None:
    """Atomically apply a worker's top-level delta using optimistic concurrency.

    ``base`` is the detached state sent to the worker. Worker changes are only
    applied where the parent still matches that base. Convergent changes are
    accepted, while divergent same-key changes reject the entire merge. The
    execution-scope keys additionally act as guards: changing one while a
    worker is running makes all of that worker's output stale.
    """

    with _SESSION_STATE_MERGE_LOCK:
        changes = {
            key: worker.get(key, _MISSING)
            for key in base.keys() | worker.keys()
            if not _state_slots_equal(base, worker, key)
        }
        parent_scope_changes = {
            key for key in _EXECUTION_SCOPE_KEYS if not _state_slots_equal(base, target, key)
        }
        worker_scope_changes = {
            key for key in _EXECUTION_SCOPE_KEYS if not _state_slots_equal(base, worker, key)
        }
        conflicts = parent_scope_changes | worker_scope_changes

        for key in changes:
            if key in conflicts or _state_slots_equal(base, target, key):
                continue
            if _state_slots_equal(worker, target, key):
                # Both sides independently reached the same value or deletion.
                continue
            conflicts.add(key)

        if conflicts:
            conflict_keys = _format_conflict_keys(conflicts)
            can_retry = retryable and conflicts.isdisjoint(_EXECUTION_SCOPE_KEYS)
            raise MCPToolError(
                f"{tool_name} worker session-state merge conflicted with concurrent "
                f"parent updates for keys: {conflict_keys}. Parent state was preserved.",
                code=MCPErrorCode.INTERNAL,
                retryable=can_retry,
            )

        for key, value in changes.items():
            if _state_slots_equal(worker, target, key):
                continue
            if value is _MISSING:
                target.pop(key, None)
            else:
                target[key] = value


def _state_slots_equal(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    key: str,
) -> bool:
    left_value = left.get(key, _MISSING)
    right_value = right.get(key, _MISSING)
    if left_value is _MISSING or right_value is _MISSING:
        return left_value is right_value
    try:
        return json.dumps(
            left_value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ) == json.dumps(
            right_value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, OverflowError, RecursionError, RuntimeError):
        # A non-JSON or concurrently mutating parent value cannot be proven
        # unchanged across a JSON worker boundary, so the merge fails closed.
        return False


def _format_conflict_keys(keys: set[str]) -> str:
    ordered = sorted(keys)
    shown = ordered[:10]
    suffix = f", ... (+{len(ordered) - len(shown)} more)" if len(ordered) > len(shown) else ""
    return ", ".join(shown) + suffix


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


def _worker_error_metadata(payload: Mapping[str, Any]) -> tuple[MCPErrorCode, bool]:
    """Return validated error metadata supplied by a worker result."""

    raw_code = payload.get("error_code", MCPErrorCode.INTERNAL.value)
    try:
        code = MCPErrorCode(str(raw_code))
    except ValueError:
        return MCPErrorCode.INTERNAL, False

    raw_retryable = payload.get("retryable", False)
    if not isinstance(raw_retryable, bool):
        return MCPErrorCode.INTERNAL, False
    return code, raw_retryable


def _filesystem_error_code(exc: OSError) -> MCPErrorCode:
    """Classify local quota/size exhaustion separately from other I/O faults."""

    resource_errnos = {
        value
        for value in (
            getattr(errno, "EDQUOT", None),
            getattr(errno, "EFBIG", None),
            getattr(errno, "ENOSPC", None),
        )
        if value is not None
    }
    if exc.errno in resource_errnos:
        return MCPErrorCode.RESOURCE_LIMIT
    return MCPErrorCode.INTERNAL
