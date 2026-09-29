"""Confine client-selected output destinations to the active workflow run."""

from __future__ import annotations

import copy
import json
import re
import typing
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from urllib.parse import unquote, urlsplit

from .context import ExecutionContext, _active_output_layout
from .errors import ToolErrorCode, ToolExecutionError
from .paths import (
    _decode_parameter_bag,
    _format_argument_location,
    _normalized_argument_key,
    _normalized_pandas_creation_function,
)
from .spec import _ARTIFACT_TYPE_RE, ToolSpec, _is_control_plane_spec

_WRITE_PATH_KEYS = frozenset(
    {
        "destination",
        "destination_file",
        "destination_filename",
        "destination_path",
        "dest_file",
        "dest_path",
        "export_file",
        "export_filename",
        "export_path",
        "output_file",
        "output_filename",
        "output_path",
        "save_file",
        "save_filename",
        "save_path",
        "save_to",
        "target_file",
        "target_path",
        "write_path",
    }
)
_WRITE_PATH_CONTAINER_KEYS = frozenset(
    {
        "destination",
        "destinations",
        "exports",
        "output",
        "outputs",
        "sink",
        "sinks",
        "target",
        "targets",
    }
)
_GENERIC_PATH_KEYS = frozenset(
    {
        "file",
        "filename",
        "file_path",
        "filepath",
        "filepath_or_buffer",
        "buf",
        "excel_writer",
        "path",
        "path_or_buffer",
        "path_or_buf",
        "uri",
        "url",
    }
)
_PARAMETER_BAG_KEYS = frozenset(
    {
        "function_parameters",
        "kwargs",
        "operation_parameters",
        "options",
        "parameters",
    }
)
_ENCODED_PATH_SEPARATOR_RE = re.compile(r"%(?:2e|2f|5c)", re.IGNORECASE)
_DERIVED_WRITE_NAME_FIELDS: dict[str, tuple[str, ...]] = {
    "gtm_optimization": ("dataset_name", "gtm_name"),
    "gtm_save_model_and_data": ("dataset_name", "gtm_name"),
    "gtm_train_on_latent_space": ("dataset_name", "gtm_name"),
}
_PANDAS_WRITE_OPERATION_ALIASES = {
    "export_csv": "to_csv",
    "save_csv": "to_csv",
    "write_csv": "to_csv",
}
_PANDAS_EXTERNAL_WRITE_OPERATIONS = frozenset({"to_clipboard", "to_gbq", "to_sql"})
_PANDAS_SAFE_CREATE_FUNCTIONS = frozenset(
    {
        "DataFrame",
        "DataFrame.from_dict",
        "DataFrame.from_records",
        "from_dict",
        "from_file",
        "from_records",
        "from_s3",
        "read_csv",
    }
)


def _enforce_write_boundary(
    spec: ToolSpec,
    public_args: Mapping[str, Any],
    ctx: ExecutionContext,
) -> dict[str, Any]:
    """Confine client-selected write destinations to the active run/session.

    Toolkit methods continue to own scientific behavior and storage formats.
    The MCP boundary only recognizes destination-shaped arguments, including
    nested parameter bags, and rewrites safe relative paths into the active
    workflow root. Read paths and workflow control-plane paths are preserved.
    """

    copied = copy.deepcopy(dict(public_args))
    if spec.write_scope != "session" or spec.read_only or _is_control_plane_spec(spec):
        return copied

    _validate_derived_write_names(spec, copied)
    operation = _normalized_operation(public_args)
    writer_operation = _writer_operation(operation)
    _validate_open_ended_writer(spec, copied, operation=operation)
    return typing.cast(
        dict[str, Any],
        _rewrite_write_paths(
            copied,
            ctx=ctx,
            location=(),
            writer_operation=writer_operation,
            path_container=False,
            parameter_bag=False,
        ),
    )


def _validate_derived_write_names(spec: ToolSpec, arguments: Mapping[str, Any]) -> None:
    """Reject state keys that selected toolkit methods later embed in filenames."""

    for field_name in _DERIVED_WRITE_NAME_FIELDS.get(spec.mcp_name, ()):
        value = arguments.get(field_name)
        if not isinstance(value, str) or not _ARTIFACT_TYPE_RE.fullmatch(value):
            raise ToolExecutionError(
                f"{field_name} must be a safe identifier because {spec.mcp_name} "
                "uses it to derive a run artifact filename",
                code=ToolErrorCode.INVALID_INPUT,
            )


def _normalized_operation(arguments: Mapping[str, Any]) -> str | None:
    operation = arguments.get("operation")
    if not isinstance(operation, str):
        return None
    normalized = operation.strip().lower()
    if normalized.endswith("()"):
        normalized = normalized[:-2]
    normalized = normalized.rsplit(".", 1)[-1]
    return _PANDAS_WRITE_OPERATION_ALIASES.get(normalized, normalized)


def _writer_operation(operation: str | None) -> bool:
    """Return whether an open-ended operation name denotes a serialization sink."""

    if operation is None:
        return False
    return operation.startswith("to_") and operation not in {
        "to_dict",
        "to_numpy",
        "to_records",
    }


def _validate_open_ended_writer(
    spec: ToolSpec,
    arguments: Mapping[str, Any],
    *,
    operation: str | None,
) -> None:
    """Block pandas persistence modes that bypass the session storage adapter."""

    if spec.mcp_name == "pandas_create_dataframe":
        _validate_pandas_creation_function(arguments)
        return
    if spec.mcp_name != "pandas_run_operation" or operation is None:
        return
    if operation in _PANDAS_EXTERNAL_WRITE_OPERATIONS:
        raise ToolExecutionError(
            f"pandas operation {operation!r} writes outside artifact storage and "
            "is not available through MCP",
            code=ToolErrorCode.PERMISSION_DENIED,
        )
    if not _writer_operation(operation) or operation == "to_csv":
        return
    if _contains_write_destination(arguments):
        raise ToolExecutionError(
            f"pandas operation {operation!r} cannot persist through the session "
            "storage adapter; use to_csv for a run-scoped artifact",
            code=ToolErrorCode.PERMISSION_DENIED,
        )


def _validate_pandas_creation_function(arguments: Mapping[str, Any]) -> None:
    """Allow only DataFrame-producing pandas entry points without write side effects."""

    raw_function, function = _normalized_pandas_creation_function(arguments)
    if function not in _PANDAS_SAFE_CREATE_FUNCTIONS:
        raise ToolExecutionError(
            f"pandas creation function {raw_function!r} is not available through MCP; "
            "use an approved DataFrame constructor or registered CSV artifact",
            code=ToolErrorCode.PERMISSION_DENIED,
        )


def _contains_write_destination(value: Any, *, parameter_bag: bool = False) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized_key = _normalized_argument_key(key)
            if normalized_key in _WRITE_PATH_KEYS:
                return child is not None
            if parameter_bag and normalized_key in _GENERIC_PATH_KEYS:
                return child is not None
            if _contains_write_destination(
                child,
                parameter_bag=(parameter_bag or normalized_key in _PARAMETER_BAG_KEYS),
            ):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(
            _contains_write_destination(child, parameter_bag=parameter_bag) for child in value
        )
    if parameter_bag and isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        decoded = _decode_parameter_bag(value)
        if decoded is None:
            return False
        return _contains_write_destination(decoded, parameter_bag=True)
    return False


def _rewrite_write_paths(
    value: Any,
    *,
    ctx: ExecutionContext,
    location: tuple[str, ...],
    writer_operation: bool,
    path_container: bool,
    parameter_bag: bool,
) -> Any:
    if isinstance(value, Mapping):
        rewritten: dict[Any, Any] = {}
        for key, child in value.items():
            normalized_key = _normalized_argument_key(key)
            child_location = (*location, str(key))
            strong_sink = normalized_key in _WRITE_PATH_KEYS
            generic_sink = normalized_key in _GENERIC_PATH_KEYS and (
                path_container or (parameter_bag and writer_operation)
            )
            child_container = path_container or normalized_key in _WRITE_PATH_CONTAINER_KEYS
            child_parameter_bag = parameter_bag or normalized_key in _PARAMETER_BAG_KEYS

            if strong_sink or generic_sink:
                rewritten[key] = _rewrite_path_sink(
                    child,
                    ctx=ctx,
                    location=child_location,
                )
                continue

            if (
                child_parameter_bag
                and isinstance(child, str)
                and child.lstrip().startswith(("{", "["))
            ):
                rewritten[key] = _rewrite_json_parameter_bag(
                    child,
                    ctx=ctx,
                    location=child_location,
                    writer_operation=writer_operation,
                )
                continue

            rewritten[key] = _rewrite_write_paths(
                child,
                ctx=ctx,
                location=child_location,
                writer_operation=writer_operation,
                path_container=child_container,
                parameter_bag=child_parameter_bag,
            )
        return rewritten

    if isinstance(value, list):
        return [
            _rewrite_write_paths(
                child,
                ctx=ctx,
                location=(*location, str(index)),
                writer_operation=writer_operation,
                path_container=path_container,
                parameter_bag=parameter_bag,
            )
            for index, child in enumerate(value)
        ]

    if isinstance(value, tuple):
        return tuple(
            _rewrite_write_paths(
                child,
                ctx=ctx,
                location=(*location, str(index)),
                writer_operation=writer_operation,
                path_container=path_container,
                parameter_bag=parameter_bag,
            )
            for index, child in enumerate(value)
        )

    if path_container and isinstance(value, str):
        return _confine_write_path(value, ctx=ctx, location=location)
    return value


def _rewrite_json_parameter_bag(
    value: str,
    *,
    ctx: ExecutionContext,
    location: tuple[str, ...],
    writer_operation: bool,
) -> str:
    decoded = _decode_parameter_bag(value)
    if decoded is None:
        return value
    rewritten = _rewrite_write_paths(
        decoded,
        ctx=ctx,
        location=location,
        writer_operation=writer_operation,
        path_container=False,
        parameter_bag=True,
    )
    return json.dumps(rewritten, ensure_ascii=False, separators=(",", ":"))


def _rewrite_path_sink(
    value: Any,
    *,
    ctx: ExecutionContext,
    location: tuple[str, ...],
) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping) or isinstance(value, (list, tuple)):
        return _rewrite_write_paths(
            value,
            ctx=ctx,
            location=location,
            writer_operation=True,
            path_container=True,
            parameter_bag=False,
        )
    if not isinstance(value, str):
        raise ToolExecutionError(
            f"write destination {_format_argument_location(location)} must be a path string",
            code=ToolErrorCode.INVALID_INPUT,
        )
    return _confine_write_path(value, ctx=ctx, location=location)


def _confine_write_path(
    value: str,
    *,
    ctx: ExecutionContext,
    location: tuple[str, ...],
) -> str:
    from cs_copilot.storage import S3
    from cs_copilot.storage.layout import normalize_run_relative_path

    candidate = value.strip()
    label = _format_argument_location(location)
    if not candidate:
        raise ToolExecutionError(
            f"write destination {label} cannot be empty",
            code=ToolErrorCode.INVALID_INPUT,
        )
    if "\x00" in candidate or "\\" in candidate:
        raise _write_boundary_error(label)
    if _ENCODED_PATH_SEPARATOR_RE.search(candidate):
        raise _write_boundary_error(label)

    layout = _active_output_layout(ctx)
    run_prefix = layout.run_root if layout is not None else ""
    parsed = urlsplit(candidate)
    if parsed.scheme:
        if parsed.scheme.lower() != "s3":
            raise _write_boundary_error(label)
        _validate_scoped_s3_destination(
            candidate,
            run_prefix=run_prefix,
            label=label,
        )
        return candidate

    if candidate.startswith(("/", "file://")) or PurePosixPath(candidate).is_absolute():
        raise _write_boundary_error(label)

    if layout is not None:
        try:
            relative = normalize_run_relative_path(layout.run_id, candidate)
        except ValueError as exc:
            raise _write_boundary_error(label) from exc
        confined = layout.artifact_rel_path(relative)
    else:
        confined = _normalize_session_relative_path(candidate, label=label)

    _validate_local_destination(
        S3,
        confined=confined,
        run_prefix=run_prefix,
        label=label,
    )
    return confined


def _normalize_session_relative_path(candidate: str, *, label: str) -> str:
    path = PurePosixPath(candidate)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise _write_boundary_error(label)
    if not path.parts:
        raise ToolExecutionError(
            f"write destination {label} cannot be empty",
            code=ToolErrorCode.INVALID_INPUT,
        )
    return path.as_posix()


def _validate_scoped_s3_destination(
    candidate: str,
    *,
    run_prefix: str,
    label: str,
) -> None:
    from cs_copilot.storage import S3

    parsed = urlsplit(candidate)
    if parsed.query or parsed.fragment or not parsed.netloc:
        raise _write_boundary_error(label)
    decoded_path = unquote(parsed.path)
    if "\\" in decoded_path:
        raise _write_boundary_error(label)
    path_parts = PurePosixPath(decoded_path).parts
    if any(part in {"", ".", ".."} for part in path_parts):
        raise _write_boundary_error(label)

    expected = urlsplit(S3.path(run_prefix))
    if expected.scheme.lower() != "s3" or not expected.netloc:
        raise _write_boundary_error(label)
    expected_path = expected.path.rstrip("/")
    if (
        parsed.netloc != expected.netloc
        or not decoded_path.startswith(f"{expected_path}/")
        or decoded_path == expected_path
    ):
        raise _write_boundary_error(label)


def _validate_local_destination(
    storage: Any,
    *,
    confined: str,
    run_prefix: str,
    label: str,
) -> None:
    storage_root = storage.path("")
    destination = storage.path(confined)
    if urlsplit(storage_root).scheme == "s3":
        return
    if urlsplit(destination).scheme:
        raise _write_boundary_error(label)

    session_root = Path(storage_root).resolve(strict=False)
    boundary_root = Path(storage.path(run_prefix)).resolve(strict=False)
    destination_path = Path(destination).resolve(strict=False)
    try:
        boundary_root.relative_to(session_root)
        destination_path.relative_to(boundary_root)
    except ValueError as exc:
        raise _write_boundary_error(label) from exc


def _write_boundary_error(label: str) -> ToolExecutionError:
    return ToolExecutionError(
        f"write destination {label} must remain inside the active workflow run/session; "
        "absolute paths, file:// URLs, traversal, and foreign S3 locations are not allowed",
        code=ToolErrorCode.PERMISSION_DENIED,
    )
