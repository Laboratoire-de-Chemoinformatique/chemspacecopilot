"""Resolve declared scientific file inputs to verified, registered run artifacts."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlsplit

from .context import ExecutionContext, _active_output_layout, _optional_str
from .errors import ToolErrorCode, ToolExecutionError
from .paths import (
    _decode_parameter_bag,
    _format_argument_location,
    _normalized_argument_key,
    _normalized_pandas_creation_function,
)
from .spec import ToolSpec

_PANDAS_FILE_CREATE_FUNCTIONS = frozenset({"from_file", "from_s3", "read_csv"})
_PANDAS_FILE_PARAMETER_KEYS: dict[str, tuple[str, ...]] = {
    "read_csv": (
        "path_or_buf",
        "path_or_buffer",
        "filepath_or_buffer",
        "file_path",
        "filepath",
        "path",
    ),
    "from_s3": ("s3_path",),
    "from_file": ("file_path",),
}
_PANDAS_LOADABLE_SUFFIXES = (
    ".csv",
    ".csv.gz",
    ".tsv",
    ".tab",
    ".txt",
)
_REPORT_FIGURE_PATH_KEYS = frozenset(
    {
        "artifact_path",
        "html_path",
        "image_path",
        "interactive_path",
        "path",
        "png_path",
        "src",
    }
)


@dataclass(frozen=True)
class _ReadBoundaryResult:
    """Canonical arguments plus an optional invocation-local session view."""

    arguments: dict[str, Any]
    session_state: dict[str, Any] | None = None


class _SessionStateReadSnapshot(dict[str, Any]):
    """A live-write session view with selected keys frozen for deterministic reads."""

    def __init__(
        self,
        target: dict[str, Any],
        *,
        frozen: Mapping[str, Any],
    ) -> None:
        super().__init__(target)
        self._target = target
        self._frozen_keys = frozenset(frozen)
        dict.update(self, frozen)

    def __setitem__(self, key: str, value: Any) -> None:
        dict.__setitem__(self, key, value)
        if key not in self._frozen_keys:
            self._target[key] = value

    def __delitem__(self, key: str) -> None:
        dict.__delitem__(self, key)
        if key not in self._frozen_keys:
            del self._target[key]

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key in self:
            return self[key]
        self[key] = default
        return default


def _enforce_read_boundary(
    spec: ToolSpec,
    public_args: Mapping[str, Any],
    ctx: ExecutionContext,
) -> _ReadBoundaryResult:
    """Resolve declared scientific file inputs to verified run artifacts."""

    copied = copy.deepcopy(dict(public_args))
    for field_name in spec.read_artifact_fields:
        candidate = copied.get(field_name)
        if candidate is None:
            continue
        copied[field_name] = _confine_registered_read_path(
            candidate,
            ctx=ctx,
            location=(field_name,),
            trusted_pickle=field_name in spec.trusted_pickle_fields,
        )
    copied, pandas_state = _enforce_pandas_read_boundary(spec, copied, ctx)
    copied, report_state = _enforce_report_read_boundary(spec, copied, ctx)
    copied, candidate_state = _enforce_candidate_reference_boundary(spec, copied, ctx)
    session_views = [
        state for state in (pandas_state, report_state, candidate_state) if state is not None
    ]
    if len(session_views) > 1:  # pragma: no cover - tool names are mutually exclusive
        raise ToolExecutionError(
            f"{spec.mcp_name} requested incompatible session read snapshots",
            code=ToolErrorCode.INTERNAL,
        )
    return _ReadBoundaryResult(
        arguments=copied,
        session_state=session_views[0] if session_views else None,
    )


def _enforce_pandas_read_boundary(
    spec: ToolSpec,
    arguments: dict[str, Any],
    ctx: ExecutionContext,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Require pandas filesystem reads to use verified active-run artifacts."""

    if spec.mcp_name == "pandas_load_dataframe_from_session":
        from cs_copilot.tools.io.session_memory import resolve_loadable_session_data

        session_key = arguments.get("session_key")
        if not isinstance(session_key, str) or not session_key.strip():
            raise ToolExecutionError(
                "session_key must be a non-empty string",
                code=ToolErrorCode.INVALID_INPUT,
            )
        try:
            resolved = resolve_loadable_session_data(ctx.session_state, session_key)
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolExecutionError(str(exc), code=ToolErrorCode.INVALID_INPUT) from exc
        resolved_key = str(resolved.get("session_key") or session_key)
        resolved_value = resolved.get("value")
        if resolved.get("kind") == "csv_path":
            pinned_value: Any = _confine_registered_read_path(
                resolved.get("value"),
                ctx=ctx,
                location=("session_state", resolved_key),
            )
        else:
            # A detached frame prevents a concurrent caller from mutating the
            # in-memory object after it has been selected for this invocation.
            pinned_value = resolved_value.copy(deep=True)
        session_snapshot = dict(ctx.session_state)
        # resolve_loadable_session_data checks exact top-level keys first, so a
        # dotted selected key can be pinned without rewriting its source tree.
        session_snapshot[resolved_key] = pinned_value
        arguments["session_key"] = resolved_key
        return arguments, session_snapshot

    if spec.mcp_name == "pandas_create_dataframe":
        _raw_function, function = _normalized_pandas_creation_function(arguments)
        arguments["create_using_function"] = function
        if function not in _PANDAS_FILE_CREATE_FUNCTIONS:
            return arguments, None
        raw_parameters = arguments.get("function_parameters")
        serialized = isinstance(raw_parameters, str)
        if raw_parameters is None:
            parameters: dict[str, Any] = {}
        elif isinstance(raw_parameters, Mapping):
            parameters = dict(raw_parameters)
        elif serialized:
            decoded = _decode_parameter_bag(raw_parameters)
            if not isinstance(decoded, Mapping):
                raise ToolExecutionError(
                    "function_parameters must decode to an object",
                    code=ToolErrorCode.INVALID_INPUT,
                )
            parameters = dict(decoded)
        else:
            raise ToolExecutionError(
                "function_parameters must be an object or encoded object",
                code=ToolErrorCode.INVALID_INPUT,
            )

        allowed_keys = set(_PANDAS_FILE_PARAMETER_KEYS[function])
        selected_keys = [key for key in parameters if _normalized_argument_key(key) in allowed_keys]
        if len(selected_keys) != 1:
            raise ToolExecutionError(
                f"pandas creation function {function!r} requires exactly one "
                "unambiguous registered artifact path",
                code=ToolErrorCode.INVALID_INPUT,
            )
        selected_key = selected_keys[0]
        parameters[selected_key] = _confine_registered_read_path(
            parameters[selected_key],
            ctx=ctx,
            location=("function_parameters", str(selected_key)),
        )
        arguments["function_parameters"] = (
            json.dumps(parameters, ensure_ascii=False, separators=(",", ":"))
            if serialized
            else parameters
        )
        return arguments, None

    read_field = {
        "pandas_run_operation": "dataframe_name",
        "pandas_normalize_for_analysis": "df_path",
    }.get(spec.mcp_name)
    if read_field is None:
        return arguments, None
    candidate = arguments.get(read_field)
    if isinstance(candidate, str) and _looks_like_pandas_file_source(candidate):
        arguments[read_field] = _confine_registered_read_path(
            candidate,
            ctx=ctx,
            location=(read_field,),
        )
    return arguments, None


def _enforce_report_read_boundary(
    spec: ToolSpec,
    arguments: dict[str, Any],
    ctx: ExecutionContext,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Verify every explicit or session-resolved rich-report figure path."""

    if spec.mcp_name != "report_save_rich":
        return arguments, None

    session_snapshot: _SessionStateReadSnapshot | None = None

    def report_session_snapshot() -> _SessionStateReadSnapshot:
        nonlocal session_snapshot
        if session_snapshot is not None:
            return session_snapshot
        live_memory = ctx.session_state.get("session_objects")
        if not isinstance(live_memory, dict):
            live_memory = {}
            ctx.session_state["session_objects"] = live_memory
        live_figures = live_memory.get("figures")
        frozen_figures = (
            copy.deepcopy(dict(live_figures)) if isinstance(live_figures, Mapping) else {}
        )
        memory_snapshot = _SessionStateReadSnapshot(
            live_memory,
            frozen={"figures": frozen_figures},
        )
        session_snapshot = _SessionStateReadSnapshot(
            ctx.session_state,
            frozen={"session_objects": memory_snapshot},
        )
        return session_snapshot

    def normalize_figures(value: Any, *, location: tuple[str, ...]) -> Any:
        if value is None:
            return None
        figures = value if isinstance(value, (list, tuple)) else [value]
        normalized = [
            normalize_figure(figure, location=(*location, str(index)))
            for index, figure in enumerate(figures)
        ]
        return normalized if isinstance(value, (list, tuple)) else normalized[0]

    def normalize_figure(value: Any, *, location: tuple[str, ...]) -> Any:
        if isinstance(value, str):
            return _confine_registered_read_path(value, ctx=ctx, location=location)
        if not isinstance(value, Mapping):
            return value
        figure = copy.deepcopy(dict(value))
        for key in _REPORT_FIGURE_PATH_KEYS:
            candidate = figure.get(key)
            if candidate:
                figure[key] = _confine_registered_read_path(
                    candidate,
                    ctx=ctx,
                    location=(*location, key),
                )
        normalize_metadata_paths(figure, location=location)
        for metadata_key in ("figure_metadata", "metadata"):
            metadata = figure.get(metadata_key)
            if isinstance(metadata, Mapping):
                metadata_copy = copy.deepcopy(dict(metadata))
                normalize_metadata_paths(
                    metadata_copy,
                    location=(*location, metadata_key),
                )
                figure[metadata_key] = metadata_copy
        validate_session_figure(figure, location=location)
        return figure

    def normalize_metadata_paths(
        metadata: dict[str, Any],
        *,
        location: tuple[str, ...],
    ) -> None:
        paths = metadata.get("paths")
        if not isinstance(paths, Mapping):
            return
        normalized_paths = {}
        for key, candidate in paths.items():
            normalized_paths[key] = (
                _confine_registered_read_path(
                    candidate,
                    ctx=ctx,
                    location=(*location, "paths", str(key)),
                )
                if candidate
                else candidate
            )
        metadata["paths"] = normalized_paths

    def validate_session_figure(
        figure: Mapping[str, Any],
        *,
        location: tuple[str, ...],
    ) -> None:
        figure_id = figure.get("figure_id")
        if figure_id is None:
            figure_id = figure.get("session_figure_id")
        if figure_id is None or not str(figure_id).strip():
            return
        from cs_copilot.tools.io.figure_metadata import session_figure_metadata

        snapshot = report_session_snapshot()
        metadata = session_figure_metadata(snapshot, figure_id)
        if not metadata:
            return
        paths = metadata.get("paths")
        if not isinstance(paths, Mapping):
            return
        normalized_paths: dict[str, Any] = {}
        for key, candidate in paths.items():
            normalized_paths[str(key)] = (
                _confine_registered_read_path(
                    candidate,
                    ctx=ctx,
                    location=(*location, "session_figure", str(figure_id), str(key)),
                )
                if candidate
                else candidate
            )
        memory = snapshot.get("session_objects")
        figures = memory.get("figures") if isinstance(memory, Mapping) else None
        record = figures.get(str(figure_id)) if isinstance(figures, dict) else None
        if isinstance(record, Mapping):
            pinned_record = copy.deepcopy(dict(record))
            pinned_record["paths"] = normalized_paths
            figures[str(figure_id)] = pinned_record

    if "figures" in arguments:
        arguments["figures"] = normalize_figures(
            arguments.get("figures"),
            location=("figures",),
        )

    sections = arguments.get("sections")
    if isinstance(sections, (list, tuple)):
        normalized_sections = []
        for index, section in enumerate(sections):
            if not isinstance(section, Mapping):
                normalized_sections.append(section)
                continue
            section_copy = copy.deepcopy(dict(section))
            if "figures" in section_copy:
                section_copy["figures"] = normalize_figures(
                    section_copy.get("figures"),
                    location=("sections", str(index), "figures"),
                )
            normalized_sections.append(section_copy)
        arguments["sections"] = normalized_sections
    return arguments, session_snapshot


def _enforce_candidate_reference_boundary(
    spec: ToolSpec,
    arguments: dict[str, Any],
    ctx: ExecutionContext,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Pin the exact candidate artifact that the selected toolkit will consume."""

    guarded_tools = {
        "peptide_load_design_candidates",
        "session_load_candidate_set_artifact",
        "session_materialize_candidate_set_dataset",
    }
    if spec.mcp_name not in guarded_tools:
        return arguments, None
    default_reference = {
        "peptide_load_design_candidates": "designed_peptides",
        "session_load_candidate_set_artifact": "top candidates",
        "session_materialize_candidate_set_dataset": "generated compounds",
    }[spec.mcp_name]
    reference = arguments.get("reference", default_reference)
    if not isinstance(reference, str) or not reference.strip():
        raise ToolExecutionError(
            "reference must be a non-empty string",
            code=ToolErrorCode.INVALID_INPUT,
        )
    reference = reference.strip()
    direct_path = _looks_like_candidate_artifact_path(reference)
    selected: tuple[str, tuple[str, ...]] | None = (
        (reference, ("reference",)) if direct_path else None
    )
    candidate_set: Mapping[str, Any] | None = None
    pointer = ctx.session_state.get(reference)
    if selected is None and spec.mcp_name == "peptide_load_design_candidates":
        selected = _first_candidate_path(
            pointer,
            keys=("artifact_rel_path", "artifact_path"),
            location=("session_state", reference),
        )

    if selected is None and spec.mcp_name == "session_load_candidate_set_artifact":
        selected = _first_candidate_path(
            pointer,
            keys=("artifact_rel_path", "artifact_path"),
            location=("session_state", reference),
        )

    if spec.mcp_name.startswith("session_") and selected is None:
        from cs_copilot.tools.io.session_memory import (
            get_session_object,
            resolve_candidate_set,
        )

        if (
            spec.mcp_name == "session_materialize_candidate_set_dataset"
            and isinstance(pointer, Mapping)
            and pointer.get("candidate_set_id")
        ):
            candidate_set = get_session_object(
                ctx.session_state,
                str(pointer["candidate_set_id"]),
            )
        if candidate_set is None:
            resolved = resolve_candidate_set(ctx.session_state, reference)
            candidate_set = resolved.get("candidate_set")
        if isinstance(candidate_set, Mapping):
            if (
                spec.mcp_name == "session_materialize_candidate_set_dataset"
                and arguments.get("top_n") is None
            ):
                selected = _first_candidate_path(
                    candidate_set,
                    keys=("csv_path",),
                    location=("session_candidate_set", reference),
                )
            if selected is None:
                selected = _first_candidate_path(
                    candidate_set,
                    keys=("artifact_rel_path", "artifact_path"),
                    location=("session_candidate_set", reference),
                )

    if (
        spec.mcp_name
        in {
            "peptide_load_design_candidates",
            "session_load_candidate_set_artifact",
        }
        and selected is None
    ):
        raise ToolExecutionError(
            f"candidate reference {reference!r} does not resolve to a registered run artifact",
            code=ToolErrorCode.INVALID_INPUT,
        )

    if selected is None:
        return arguments, None

    canonical = _confine_registered_read_path(
        selected[0],
        ctx=ctx,
        location=selected[1],
    )
    if not direct_path and spec.mcp_name in {
        "peptide_load_design_candidates",
        "session_load_candidate_set_artifact",
    }:
        pinned_pointer = copy.deepcopy(dict(pointer)) if isinstance(pointer, Mapping) else {}
        for path_key in ("artifact_rel_path", "artifact_path", "csv_path"):
            pinned_pointer.pop(path_key, None)
        pinned_pointer["artifact_rel_path"] = canonical
        if isinstance(candidate_set, Mapping) and candidate_set.get("id"):
            pinned_pointer.setdefault("candidate_set_id", str(candidate_set["id"]))
        session_snapshot = dict(ctx.session_state)
        session_snapshot[reference] = pinned_pointer
        return arguments, session_snapshot

    arguments["reference"] = canonical
    return arguments, None


def _first_candidate_path(
    source: Any,
    *,
    keys: tuple[str, ...],
    location: tuple[str, ...],
) -> tuple[str, tuple[str, ...]] | None:
    if not isinstance(source, Mapping):
        return None
    for key in keys:
        candidate = source.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate, (*location, key)
    return None


def _looks_like_candidate_artifact_path(value: str) -> bool:
    candidate = value.strip().lower().split("?", 1)[0]
    return _looks_like_pandas_file_source(candidate) or candidate.endswith((".json", ".json.gz"))


def _looks_like_pandas_file_source(value: str) -> bool:
    candidate = value.strip().lower()
    return (
        "/" in candidate
        or "\\" in candidate
        or bool(urlsplit(candidate).scheme)
        or candidate.endswith(_PANDAS_LOADABLE_SUFFIXES)
    )


def _confine_registered_read_path(
    value: Any,
    *,
    ctx: ExecutionContext,
    location: tuple[str, ...],
    trusted_pickle: bool = False,
) -> str:
    label = _format_argument_location(location)
    if not isinstance(value, str) or not value.strip():
        raise ToolExecutionError(
            f"pandas read source {label} must be a non-empty path string",
            code=ToolErrorCode.INVALID_INPUT,
        )
    candidate = value.strip()
    layout = _active_output_layout(ctx)
    run_context = getattr(ctx, "run_context", None)
    if run_context is not None and hasattr(run_context, "refresh"):
        run_context.refresh()
    run = getattr(run_context, "run", None)
    if layout is None or run_context is None or run is None:
        raise ToolExecutionError(
            "pandas filesystem reads require an active workflow run",
            code=ToolErrorCode.PERMISSION_DENIED,
        )

    record = next(
        (
            artifact
            for artifact in run.artifacts.values()
            if _matches_registered_read_alias(
                candidate,
                run_relative=artifact.relative_path,
                run_scoped=layout.artifact_rel_path(artifact.relative_path),
            )
        ),
        None,
    )
    if record is None:
        raise ToolExecutionError(
            f"read source {label} is not a registered artifact in the active run",
            code=ToolErrorCode.PERMISSION_DENIED,
        )
    workflow_contract = run.workflow_contract if isinstance(run.workflow_contract, Mapping) else {}
    task_contracts = workflow_contract.get("tasks")
    if isinstance(task_contracts, list) and task_contracts:
        state = ctx.session_state if isinstance(ctx.session_state, dict) else {}
        task_id = _optional_str(state.get("active_task_id"))
        task = run.tasks.get(task_id) if task_id is not None else None
        if task is None or getattr(task.status, "value", str(task.status)) != "running":
            raise ToolExecutionError(
                "catalog workflow artifact reads require an active RUNNING task",
                code=ToolErrorCode.PERMISSION_DENIED,
            )
        allowed_artifact_ids = set(task.input_artifact_ids)
        allowed_artifact_ids.update(
            artifact.artifact_id
            for artifact in run.artifacts.values()
            if artifact.producer_task_id == task.task_id
        )
        if record.artifact_id not in allowed_artifact_ids:
            raise ToolExecutionError(
                f"read source {label} was not handed off to active task " f"{task.task_id!r}",
                code=ToolErrorCode.PERMISSION_DENIED,
            )
    if trusted_pickle and (
        getattr(record.trust, "value", str(record.trust)) != "internal"
        or record.producer_tool != "gtm_save_model_and_data"
        or record.artifact_type != "gtm_model_path"
        or not record.relative_path.lower().endswith(".pkl.gz")
    ):
        raise ToolExecutionError(
            f"read source {label} is executable serialized model content; only "
            "an internal gtm_model_path produced by gtm_save_model_and_data is accepted",
            code=ToolErrorCode.PERMISSION_DENIED,
        )
    run_context.verify_artifact(record.artifact_id)
    return layout.artifact_rel_path(record.relative_path)


def _matches_registered_read_alias(
    candidate: str,
    *,
    run_relative: str,
    run_scoped: str,
) -> bool:
    """Match public, storage-expanded, or absolute forms of one artifact path."""

    from cs_copilot.storage import S3

    expanded = S3.path(run_scoped)
    if candidate in {run_relative, run_scoped, expanded}:
        return True

    candidate_url = urlsplit(candidate)
    expanded_url = urlsplit(expanded)
    if candidate_url.scheme or expanded_url.scheme:
        if (
            candidate_url.scheme.lower() == "s3"
            and expanded_url.scheme.lower() == "s3"
            and not candidate_url.query
            and not candidate_url.fragment
        ):
            return candidate_url.netloc == expanded_url.netloc and unquote(
                candidate_url.path
            ) == unquote(expanded_url.path)
        if candidate_url.scheme.lower() != "file" or expanded_url.scheme:
            return False
        candidate_path = Path(unquote(candidate_url.path))
    else:
        candidate_path = Path(candidate)

    if not candidate_path.is_absolute() and ".." in candidate_path.parts:
        return False
    try:
        return candidate_path.resolve(strict=False) == Path(expanded).resolve(strict=False)
    except (OSError, RuntimeError):
        return False
