"""Automatic, checksummed registration of tool result artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .context import ExecutionContext
from .envelopes import _artifact_ids
from .errors import ToolErrorCode, ToolExecutionError
from .spec import ToolSpec

logger = logging.getLogger(__name__)
_BACKTICK_PATH_RE = re.compile(r"`([^`\r\n]+)`")
_KNOWN_LABELED_BACKTICK_RE = re.compile(
    r"(?im)^[^`\r\n]*?(?P<label>"
    r"Clean dataset(?: \([^)]*\))?|Raw dataset|Descriptor Parquet|"
    r"Standardization report|Filtered rows|HTML|PDF|Markdown"
    r"):\s*`(?P<path>[^`\r\n]+)`"
)
_GTM_SAVED_PATHS_RE = re.compile(
    r"^\s*dataset_path:\s*(?P<dataset>[^;\r\n]+?)\s*;" r"\s*gtm_path:\s*(?P<gtm>[^\r\n]+?)\s*$"
)


def _register_result_artifacts(
    spec: ToolSpec,
    value: Any,
    ctx: ExecutionContext,
    *,
    active_task_id: str | None,
    invocation_span_id: str | None,
    publication_leases: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    from cs_copilot.workflows import ArtifactIntegrityError

    artifact_ids = _artifact_ids(value)
    warnings: list[str] = []
    run_context = getattr(ctx, "run_context", None)
    run = getattr(run_context, "run", None)
    if run_context is None or run is None:
        return artifact_ids, warnings

    existing_by_path = {
        record.relative_path: record.artifact_id for record in run.artifacts.values()
    }
    if active_task_id not in run.tasks:
        active_task_id = None
    required_output_types = _required_task_output_types(run, active_task_id)
    owned_publications = dict(publication_leases)

    if spec.result_artifact_type is not None:
        materialized_publications: dict[str, dict[str, Any]] = {}
        try:
            relative, materialized_publications = _materialize_result_json(
                run_context,
                artifact_type=spec.result_artifact_type,
                value=value,
            )
            owned_publications.update(materialized_publications)
            artifact_id = _register_result_path(
                spec,
                run_context,
                existing_by_path,
                relative=relative,
                artifact_type=spec.result_artifact_type,
                mime_type="application/json",
                active_task_id=active_task_id,
                result_field="structured_result",
                invocation_span_id=invocation_span_id,
                publication_leases=owned_publications,
            )
            artifact_ids.append(artifact_id)
        except ArtifactIntegrityError:
            _rollback_unregistered_publications(ctx, materialized_publications)
            raise
        except Exception as exc:  # noqa: BLE001
            _rollback_unregistered_publications(ctx, materialized_publications)
            if spec.result_artifact_type in required_output_types:
                raise ToolExecutionError(
                    f"required artifact {spec.result_artifact_type!r} could not "
                    f"be registered for task {active_task_id!r}: {exc}",
                    code=ToolErrorCode.INTERNAL,
                ) from exc
            warning = (
                f"Could not materialize structured result as "
                f"{spec.result_artifact_type!r}: {exc}"
            )
            warnings.append(warning)
            logger.warning("%s: %s", spec.mcp_name, warning)

    for field_name, candidate in _result_paths(value):
        relative = _run_relative_result_path(run_context, candidate)
        if relative is None:
            continue
        artifact_type = _infer_artifact_type(spec, field_name, relative)
        try:
            artifact_id = _register_result_path(
                spec,
                run_context,
                existing_by_path,
                relative=relative,
                artifact_type=artifact_type,
                mime_type=_infer_mime_type(relative),
                active_task_id=active_task_id,
                result_field=field_name,
                invocation_span_id=invocation_span_id,
                publication_leases=owned_publications,
            )
            artifact_ids.append(artifact_id)
        except ArtifactIntegrityError:
            raise
        except Exception as exc:  # noqa: BLE001
            if artifact_type in required_output_types:
                raise ToolExecutionError(
                    f"required artifact {artifact_type!r} at {relative!r} could "
                    f"not be registered for task {active_task_id!r}: {exc}",
                    code=ToolErrorCode.INTERNAL,
                ) from exc
            warning = (
                f"Could not register run-scoped result path {relative!r} " f"as an artifact: {exc}"
            )
            warnings.append(warning)
            logger.warning("%s: %s", spec.mcp_name, warning)

    return (
        list(dict.fromkeys(artifact_ids)),
        list(dict.fromkeys(warnings)),
    )


def _rollback_unregistered_publications(
    ctx: ExecutionContext,
    publications: Mapping[str, Mapping[str, Any]],
) -> None:
    """Release only invocation-owned bytes that have no durable artifact event."""

    if not publications:
        return
    from cs_copilot.storage import S3

    run_context = getattr(ctx, "run_context", None)
    if run_context is None:
        return
    try:
        run = run_context.refresh()
        registered = {
            run_context.layout.artifact_rel_path(record.relative_path)
            for record in run.artifacts.values()
        }
    except Exception:
        logger.warning(
            "Could not establish authoritative artifact state; publication " "rollback was skipped",
            exc_info=True,
        )
        return
    releasable = {
        path: metadata for path, metadata in publications.items() if path not in registered
    }
    try:
        S3.rollback_promoted_publications(releasable)
    except Exception:
        logger.warning(
            "Could not release unregistered invocation publications",
            exc_info=True,
        )


def _register_result_path(
    spec: ToolSpec,
    run_context: Any,
    existing_by_path: dict[str, str],
    *,
    relative: str,
    artifact_type: str,
    mime_type: str,
    active_task_id: str | None,
    result_field: str,
    invocation_span_id: str | None,
    publication_leases: Mapping[str, Mapping[str, Any]],
) -> str:
    existing_id = existing_by_path.get(relative)
    if existing_id is not None:
        existing = run_context.verify_artifact(existing_id)
        expected_trust = "external" if spec.requires_network else "internal"
        existing_trust = getattr(existing.trust, "value", str(existing.trust))
        existing_provenance = (
            existing.provenance if isinstance(existing.provenance, Mapping) else {}
        )
        if (
            existing.artifact_type != artifact_type
            or existing.mime_type != mime_type
            or existing.producer_task_id != active_task_id
            or existing.producer_tool != spec.mcp_name
            or existing_trust != expected_trust
            or existing_provenance.get("registration") != "automatic"
            or existing_provenance.get("result_field") != result_field
        ):
            raise ValueError(
                f"existing artifact {existing_id!r} at {relative!r} does not "
                "match this tool result's type, producer, trust, or provenance"
            )
        return existing_id
    storage_key = run_context.layout.artifact_rel_path(relative)
    if storage_key not in publication_leases:
        from cs_copilot.workflows import ArtifactIntegrityError

        raise ArtifactIntegrityError(
            f"unregistered result path {relative!r} was not published by this " "tool invocation"
        )
    record = run_context.register_artifact(
        relative,
        artifact_type=artifact_type,
        mime_type=mime_type,
        producer_task_id=active_task_id,
        active_task_id=active_task_id,
        producer_tool=spec.mcp_name,
        provenance={
            "registration": "automatic",
            "result_field": result_field,
            "invocation_span_id": invocation_span_id,
        },
        trust="external" if spec.requires_network else "internal",
    )
    existing_by_path[record.relative_path] = record.artifact_id
    return record.artifact_id


def _register_observed_writes(
    spec: ToolSpec,
    ctx: ExecutionContext,
    created_paths: Iterable[str],
    *,
    producer_task_id: str | None,
    invocation_span_id: str | None,
) -> tuple[list[str], list[dict[str, str]]]:
    """Register files an observed call created inside its run; never raise.

    ``created_paths`` are session-relative keys reported by
    ``S3.observe_writes``. Files outside the run root and files that no longer
    exist are skipped; registration failures are returned for the audit.
    """

    run_context = getattr(ctx, "run_context", None)
    layout = getattr(run_context, "layout", None)
    if run_context is None or layout is None:
        return [], []
    run_root = PurePosixPath(layout.run_root)
    artifact_ids: list[str] = []
    problems: list[dict[str, str]] = []
    for key in created_paths:
        try:
            relative = PurePosixPath(key).relative_to(run_root).as_posix()
        except ValueError:
            continue
        try:
            record = run_context.register_artifact(
                relative,
                artifact_type=_infer_artifact_type(spec, "path", relative),
                mime_type=_infer_mime_type(relative),
                producer_task_id=producer_task_id,
                active_task_id=producer_task_id,
                producer_tool=spec.mcp_name,
                provenance={
                    "registration": "automatic",
                    "result_field": "observed_write",
                    "invocation_span_id": invocation_span_id,
                },
                trust="external" if spec.requires_network else "internal",
            )
        except FileNotFoundError:
            continue
        except Exception as exc:  # noqa: BLE001 - observation never fails the call
            problems.append({"path": relative, "error": str(exc)[:300]})
            continue
        artifact_ids.append(record.artifact_id)
    return list(dict.fromkeys(artifact_ids)), problems


def _required_task_output_types(run: Any, task_id: str | None) -> frozenset[str]:
    """Return the pinned output contracts for one active catalog task."""

    if task_id is None:
        return frozenset()
    contract = getattr(run, "workflow_contract", None)
    if not isinstance(contract, Mapping):
        return frozenset()
    tasks = contract.get("tasks")
    if not isinstance(tasks, list):
        return frozenset()
    globally_required = {
        str(item.get("name"))
        for item in contract.get("output_artifacts", ())
        if isinstance(item, Mapping) and item.get("name") and item.get("required", True)
    }
    for task in tasks:
        if not isinstance(task, Mapping) or str(task.get("task_id") or "") != task_id:
            continue
        outputs = task.get("output_artifacts")
        if not isinstance(outputs, (list, tuple)):
            return frozenset()
        return frozenset(
            str(item) for item in outputs if str(item).strip() and str(item) in globally_required
        )
    return frozenset()


def _materialize_result_json(
    run_context: Any,
    *,
    artifact_type: str,
    value: Any,
) -> tuple[str, dict[str, dict[str, Any]]]:
    from cs_copilot.storage import S3

    serialized = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=str,
    )
    encoded = f"{serialized}\n".encode("utf-8")
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    relative = f"artifacts/contracts/{artifact_type}-{digest[:16]}.json"
    storage_path = run_context.layout.artifact_rel_path(relative)
    created = False
    try:
        with S3.open_atomic(storage_path, "x") as handle:
            handle.write(serialized)
            handle.write("\n")
        created = True
    except (FileExistsError, PermissionError):
        # The deterministic content-addressed path may already have been
        # materialized by an idempotent call. Reuse it only when the pinned
        # bytes are exactly the result being returned.
        from cs_copilot.workflows import ArtifactIntegrityError

        try:
            with S3._open_verified_snapshot(
                storage_path,
                "rb",
                expected_sha256=hashlib.sha256(encoded).hexdigest(),
                expected_size=len(encoded),
            ):
                pass
        except Exception as integrity_exc:
            raise ArtifactIntegrityError(
                f"result artifact path {relative!r} already exists with "
                "different or unverifiable content"
            ) from integrity_exc
    publications = (
        {
            storage_path: {
                "staged_path": storage_path,
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "size_bytes": len(encoded),
            }
        }
        if created
        else {}
    )
    return relative, publications


def _result_paths(value: Any, *, depth: int = 0):
    if depth > 6:
        return
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            if _is_path_result_field(key):
                if isinstance(item, (str, Path)):
                    if str(item).strip():
                        yield key, str(item)
                elif isinstance(item, (list, tuple, set)):
                    for candidate in item:
                        if isinstance(candidate, (str, Path)) and str(candidate).strip():
                            yield key, str(candidate)
            if isinstance(item, (Mapping, list, tuple, set)):
                yield from _result_paths(item, depth=depth + 1)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, (Mapping, list, tuple, set)):
                yield from _result_paths(item, depth=depth + 1)
    elif isinstance(value, str):
        labeled_paths: set[str] = set()
        for matched in _KNOWN_LABELED_BACKTICK_RE.finditer(value):
            candidate = matched.group("path").strip()
            labeled_paths.add(candidate)
            yield _labeled_path_field(matched.group("label")), candidate
        for candidate in _BACKTICK_PATH_RE.findall(value):
            normalized = candidate.strip()
            if normalized and normalized not in labeled_paths:
                yield "backticked_path", normalized
        saved_paths = _GTM_SAVED_PATHS_RE.fullmatch(value)
        if saved_paths is not None:
            yield "dataset_path", saved_paths.group("dataset").strip()
            yield "gtm_path", saved_paths.group("gtm").strip()


def _labeled_path_field(label: str) -> str:
    normalized = label.strip().lower()
    if normalized.startswith("clean dataset"):
        return "clean_dataset_path"
    known = {
        "raw dataset": "raw_dataset_path",
        "descriptor parquet": "descriptor_parquet_path",
        "standardization report": "standardization_report_path",
        "filtered rows": "filtered_rows_path",
        "html": "html_path",
        "pdf": "pdf_path",
        "markdown": "markdown_path",
    }
    return known.get(normalized, "backticked_path")


def _is_path_result_field(field_name: str) -> bool:
    normalized = field_name.strip().lower()
    return normalized in {"path", "paths", "file", "files"} or normalized.endswith(
        ("_path", "_paths")
    )


def _run_relative_result_path(run_context: Any, candidate: str) -> str | None:
    from cs_copilot.storage import S3, normalize_run_relative_path

    run_id = run_context.layout.run_id
    value = str(candidate).strip()
    run_root = S3.path(run_context.layout.run_root).rstrip("/")
    comparable_value = value[7:] if value.startswith("file://") else value
    comparable_root = run_root[7:] if run_root.startswith("file://") else run_root
    prefix = f"{comparable_root}/"
    if comparable_value.startswith(prefix):
        try:
            return normalize_run_relative_path(run_id, comparable_value[len(prefix) :])
        except ValueError:
            return None
    try:
        return normalize_run_relative_path(run_id, value)
    except ValueError:
        return None


def _infer_artifact_type(
    spec: ToolSpec,
    field_name: str,
    relative_path: str,
) -> str:
    normalized = field_name.strip().lower()
    suffix = Path(relative_path).suffix.lower()
    if spec.mcp_name == "gtm_save_model_and_data":
        contract_types = {
            "dataset_path": "projected_dataset_path",
            "gtm_path": "gtm_model_path",
        }
        if normalized in contract_types:
            return contract_types[normalized]
    if spec.mcp_name == "gtm_create_activity_landscapes":
        if suffix == ".csv":
            return "activity_landscape_csv"
        if suffix in {".html", ".png", ".jpg", ".jpeg", ".svg", ".webp"}:
            return "activity_plot_path"
    if spec.mcp_name == "gtm_save_density_plot" and suffix in {
        ".html",
        ".png",
        ".jpg",
        ".jpeg",
        ".svg",
        ".webp",
    }:
        return "density_plot_path"
    if spec.mcp_name == "report_save_rich":
        report_types = {
            ".html": "html_report_path",
            ".pdf": "pdf_report_path",
            ".md": "markdown_report_path",
        }
        if suffix in report_types:
            return report_types[suffix]
    if spec.mcp_name == "report_save_markdown" and suffix == ".md":
        return "markdown_report_path"
    if normalized.endswith("_paths"):
        normalized = f"{normalized[:-6]}_path"
    generic = {
        "path",
        "paths",
        "file",
        "files",
        "relative_path",
        "artifact_path",
        "artifact_rel_path",
    }
    if normalized and normalized not in generic:
        return normalized
    if suffix in {".png", ".jpg", ".jpeg", ".svg", ".webp"}:
        return "visualization"
    if suffix in {".csv", ".tsv", ".parquet", ".feather"}:
        return "dataset"
    if suffix in {".html", ".md", ".pdf", ".docx"}:
        return "report"
    if suffix in {".pkl", ".pickle", ".joblib", ".onnx", ".pt", ".pth"}:
        return "model"
    return "artifact"


def _infer_mime_type(relative_path: str) -> str:
    suffix = Path(relative_path).suffix.lower()
    overrides = {
        ".jsonl": "application/x-ndjson",
        ".parquet": "application/vnd.apache.parquet",
        ".pkl": "application/x-python-pickle",
        ".pickle": "application/x-python-pickle",
    }
    return (
        overrides.get(suffix)
        or mimetypes.guess_type(relative_path)[0]
        or ("application/octet-stream")
    )
