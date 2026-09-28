"""Worker-process entry point for heavy MCP tool calls."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Mapping


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(dotenv_path=Path.cwd() / ".env")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )


def _build_context(payload: Mapping[str, Any]):
    from .context import MCPAgentContext
    from .llm import LLMBroker, normalize_llm_policy

    llm_policy = normalize_llm_policy(str(payload.get("llm_policy") or "external"))
    model = None
    if llm_policy == "agno-model":
        from .llm.agno_model import load_configured_model

        model = load_configured_model()

    session_state = payload.get("session_state")
    ctx = MCPAgentContext(
        session_state=dict(session_state) if isinstance(session_state, dict) else {},
        model=model,
        llm_policy=llm_policy,
    )
    ctx.llm = LLMBroker(ctx)
    return ctx


def _run_chembl_fetch_compounds(kwargs: Mapping[str, Any], ctx: Any) -> Any:
    from .facades.chembl import ChemblMCPFacade

    call_kwargs = dict(kwargs)
    call_kwargs.pop("agent", None)
    call_kwargs.pop("session_state", None)
    return ChemblMCPFacade().fetch_compounds(
        **call_kwargs,
        agent=ctx,
        session_state=ctx.session_state,
    )


_DISPATCH: dict[str, Callable[[Mapping[str, Any], Any], Any]] = {
    "chembl_fetch_compounds": _run_chembl_fetch_compounds,
}


def run_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    from cs_copilot.storage import S3

    session_prefix = payload.get("session_prefix")
    if isinstance(session_prefix, str) and session_prefix:
        S3.set_session_prefix(session_prefix)

    tool_name = str(payload.get("tool_name") or "")
    dispatch = _DISPATCH.get(tool_name)
    if dispatch is None:
        raise ValueError(f"Unsupported MCP worker tool: {tool_name!r}")

    kwargs = payload.get("kwargs")
    if not isinstance(kwargs, dict):
        raise ValueError("Worker payload field 'kwargs' must be an object.")

    ctx = _build_context(payload)
    write_boundary = payload.get("write_boundary")
    verified_payload = payload.get("verified_artifact_reads", {})
    if not isinstance(verified_payload, dict):
        raise ValueError("Worker verified_artifact_reads must be an object.")
    verified_reads: dict[str, tuple[str, int]] = {}
    for path, metadata in verified_payload.items():
        if (
            not isinstance(path, str)
            or not isinstance(metadata, list)
            or len(metadata) != 2
            or not isinstance(metadata[0], str)
            or not isinstance(metadata[1], int)
        ):
            raise ValueError("Worker verified artifact metadata is malformed.")
        verified_reads[path] = (metadata[0], metadata[1])

    publications: dict[str, dict[str, Any]] = {}
    staging_id = payload.get("staging_id")
    with ExitStack() as staging_stack:
        if staging_id is not None:
            if not isinstance(staging_id, str):
                raise ValueError("Worker staging_id must be a string.")
            staging_stack.enter_context(S3.stage_publications(staging_id))
        with ExitStack() as execution_stack:
            if verified_reads:
                execution_stack.enter_context(S3.confine_artifact_reads(verified_reads))
            if isinstance(write_boundary, str):
                protected_paths = payload.get("write_protected_paths", [])
                if not isinstance(protected_paths, list) or any(
                    not isinstance(path, str) for path in protected_paths
                ):
                    raise ValueError(
                        "Worker payload field 'write_protected_paths' " "must be a string list."
                    )
                execution_stack.enter_context(
                    S3.confine_writes(
                        write_boundary,
                        protected_paths=protected_paths,
                    )
                )
            result = dispatch(kwargs, ctx)
        if staging_id is not None:
            publications = S3.staged_publication_metadata()
    return {
        "ok": True,
        "result": result,
        "session_state": ctx.session_state,
        "staged_publications": publications,
    }


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.write("\n")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one heavy cs_copilot MCP worker job.")
    parser.add_argument("--job", required=True, help="Path to the input JSON job payload.")
    parser.add_argument("--result", required=True, help="Path for the output JSON result.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    _configure_logging()
    args = _parse_args(argv)
    result_path = Path(args.result)
    tool_name: str | None = None

    try:
        payload = _read_json(Path(args.job))
        if isinstance(payload, Mapping):
            tool_name = str(payload.get("tool_name") or "") or None
        result = run_payload(payload)
    except Exception as exc:  # noqa: BLE001
        from .errors import normalize_error

        normalized = normalize_error(
            exc,
            tool_name=tool_name,
            # Preserve intrinsic transient/timeout retryability in the worker
            # payload. The parent adapter applies the ToolSpec idempotence gate.
            idempotent=True,
        )
        result = {
            "ok": False,
            "error": normalized.message,
            "error_code": normalized.code,
            "retryable": normalized.retryable,
            "traceback": traceback.format_exc(),
            "session_state": {},
        }

    try:
        _write_json(result_path, result)
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
