"""MCP resources for versioned workflow runs and registered artifacts.

Resources use stable run and artifact identities rather than exposing arbitrary
session files. Immutable workflow events are authoritative; replaceable
manifest and artifact-index snapshots are never trusted for resource reads.
Artifact content is checksum-verified before it is returned.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, List, Mapping, Optional

logger = logging.getLogger(__name__)

URI_PREFIX = "cscopilot://runs/"
MANIFEST_NAME = "manifest.json"
_TEXT_MIME_RE = re.compile(r"^(text/|application/(?:json|x-ndjson|x-yaml|xml|toml|x-toml))")


@dataclass(frozen=True)
class ResourceEntry:
    """One MCP resource record (URI, name, MIME type, optional size)."""

    uri: str
    name: str
    mime_type: str
    size: Optional[int] = None


def list_entries(run_id: str | None = None) -> List[ResourceEntry]:
    """List canonical resources for one run or every run in this session."""

    from cs_copilot.storage import validate_identifier

    selected = validate_identifier(run_id, field="run_id") if run_id else None
    files = _list_workflow_files()
    run_ids = sorted(
        {
            parts[1]
            for rel in files
            if len((parts := PurePosixPath(rel).parts)) == 4
            and parts[0] == "workflows"
            and parts[2] == "events"
            and parts[3].endswith(".jsonl")
            and (selected is None or parts[1] == selected)
        }
    )
    entries: List[ResourceEntry] = []
    for current_run_id in run_ids:
        try:
            context = _run_context(current_run_id)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            logger.warning("Failed to replay workflow run %s: %s", current_run_id, exc)
            continue
        entries.append(
            ResourceEntry(
                uri=f"{URI_PREFIX}{current_run_id}/{MANIFEST_NAME}",
                name=f"{current_run_id}/{MANIFEST_NAME}",
                mime_type="application/json",
                size=len(_manifest_blob(context)),
            )
        )

        for event in context.events:
            event_id = validate_identifier(event.event_id, field="event_id")
            entries.append(
                ResourceEntry(
                    uri=f"{URI_PREFIX}{current_run_id}/events/{event_id}.jsonl",
                    name=f"{current_run_id}/events/{event_id}.jsonl",
                    mime_type="application/x-ndjson",
                    size=_storage_size(context.layout.event_rel_path(event_id)),
                )
            )

        assert context.run is not None
        for artifact_id in sorted(context.run.artifacts):
            record = context.run.artifacts[artifact_id]
            entries.append(
                ResourceEntry(
                    uri=f"{URI_PREFIX}{current_run_id}/artifacts/{artifact_id}",
                    name=PurePosixPath(record.relative_path).name,
                    mime_type=record.mime_type,
                    size=record.size_bytes,
                )
            )
    return entries


def read_text(uri: str) -> str:
    """Return resource content decoded as UTF-8 text."""

    blob, mime = _read_blob_with_mime(uri)
    if mime != "application/json" and not _TEXT_MIME_RE.match(mime):
        logger.debug("Decoding non-text mime %s as utf-8", mime)
    return blob.decode("utf-8", errors="replace")


def read_blob(uri: str) -> bytes:
    """Return resource content as raw bytes."""

    blob, _ = _read_blob_with_mime(uri)
    return blob


def resource_mime(uri: str) -> str:
    """Return the MIME type associated with a canonical run resource."""

    run_id, kind, identifier = _parse_uri(uri)
    if kind == "manifest":
        return "application/json"
    if kind == "event":
        return "application/x-ndjson"
    return str(_artifact_record(run_id, identifier)["mime_type"])


def is_text_resource(uri: str) -> bool:
    """Return True if the resource is best surfaced as text."""

    return bool(_TEXT_MIME_RE.match(resource_mime(uri)))


def iter_entries(run_id: str | None = None) -> Iterable[ResourceEntry]:
    """Yield canonical run resource entries."""

    yield from list_entries(run_id=run_id)


def _read_blob_with_mime(uri: str) -> tuple[bytes, str]:
    from cs_copilot.storage import (
        S3,
        is_s3_enabled,
        normalize_run_relative_path,
        open_local_run_artifact,
    )

    run_id, kind, identifier = _parse_uri(uri)
    context = _run_context(run_id)
    if kind == "manifest":
        return _manifest_blob(context), "application/json"
    elif kind == "event":
        matches = [event for event in context.events if event.event_id == identifier]
        if not matches:
            raise FileNotFoundError(f"unknown event resource {identifier!r} in run {run_id!r}")
        return _event_blob(matches[0].to_dict()), "application/x-ndjson"
    else:
        assert context.run is not None
        try:
            artifact = context.run.artifacts[identifier]
        except KeyError as exc:
            raise FileNotFoundError(
                f"unknown artifact resource {identifier!r} in run {run_id!r}"
            ) from exc
        relative = normalize_run_relative_path(run_id, artifact.relative_path)
        mime = artifact.mime_type

    if is_s3_enabled():
        handle = S3.open(context.layout.artifact_rel_path(relative), "rb")
    else:
        handle = open_local_run_artifact(context.layout, relative)
    with handle:
        blob = handle.read()
    if isinstance(blob, str):
        blob = blob.encode("utf-8")

    actual_digest = hashlib.sha256(blob).hexdigest()
    if len(blob) != artifact.size_bytes or actual_digest != artifact.sha256:
        raise ValueError(f"artifact {identifier!r} failed checksum/size verification")
    return blob, mime


def _parse_uri(uri: str) -> tuple[str, str, str]:
    from cs_copilot.storage import validate_identifier

    if not isinstance(uri, str) or not uri.startswith(URI_PREFIX):
        raise ValueError(f"Unsupported resource URI: {uri!r}")
    remainder = uri[len(URI_PREFIX) :]
    if "?" in remainder or "#" in remainder or "\\" in remainder:
        raise ValueError(f"Unsupported resource URI: {uri!r}")
    parts = remainder.split("/")
    if len(parts) == 2 and parts[1] == MANIFEST_NAME:
        return validate_identifier(parts[0], field="run_id"), "manifest", MANIFEST_NAME
    if len(parts) == 3 and parts[1] == "events" and parts[2].endswith(".jsonl"):
        event_id = parts[2][: -len(".jsonl")]
        return (
            validate_identifier(parts[0], field="run_id"),
            "event",
            validate_identifier(event_id, field="event_id"),
        )
    if len(parts) == 3 and parts[1] == "artifacts":
        return (
            validate_identifier(parts[0], field="run_id"),
            "artifact",
            validate_identifier(parts[2], field="artifact_id"),
        )
    raise ValueError(f"Unsupported resource URI: {uri!r}")


def _run_context(run_id: str):
    """Replay one run from immutable events."""

    from cs_copilot.workflows.runtime import RunContext

    return RunContext.load(run_id)


def _artifact_index(run_id: str) -> list[Mapping[str, Any]]:
    context = _run_context(run_id)
    assert context.run is not None
    return [
        context.run.artifacts[artifact_id].to_dict()
        for artifact_id in sorted(context.run.artifacts)
    ]


def _artifact_record(run_id: str, artifact_id: str) -> Mapping[str, Any]:
    matches = [item for item in _artifact_index(run_id) if item.get("artifact_id") == artifact_id]
    if not matches:
        raise FileNotFoundError(f"unknown artifact resource {artifact_id!r} in run {run_id!r}")
    if len(matches) > 1:
        raise ValueError(f"duplicate artifact id in index: {artifact_id!r}")
    return matches[0]


def _manifest_blob(context: Any) -> bytes:
    return (json.dumps(context.manifest_payload(), indent=2, sort_keys=True) + "\n").encode("utf-8")


def _event_blob(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _list_workflow_files() -> set[str]:
    """Return paths relative to the active storage session prefix."""

    from cs_copilot.storage import S3, get_s3_config, is_s3_enabled

    if not is_s3_enabled():
        local_root = Path("data") / S3.current_prefix().strip("/")
        workflows_root = local_root / "workflows"
        if not workflows_root.exists():
            return set()
        return {
            path.relative_to(local_root).as_posix()
            for path in workflows_root.rglob("*")
            if path.is_file()
        }

    config = get_s3_config()
    try:
        import s3fs  # type: ignore

        fs = s3fs.S3FileSystem(**config.to_storage_options())
        session_prefix = S3.current_prefix().strip("/")
        bucket_root = f"s3://{config.bucket_name}/{session_prefix}/workflows"
        try:
            files = fs.find(bucket_root)
        except FileNotFoundError:
            return set()
        prefix_strip = f"{config.bucket_name}/{session_prefix}/"
        result: set[str] = set()
        for key in files:
            rel = str(key)
            if rel.startswith("s3://"):
                rel = rel[len("s3://") :]
            if rel.startswith(prefix_strip):
                rel = rel[len(prefix_strip) :]
            result.add(rel)
        return result
    except Exception as exc:  # pragma: no cover - network paths only
        logger.warning("Failed to list S3 workflow resources: %s", exc)
        return set()


def _storage_size(rel_path: str) -> int | None:
    """Return local size when inexpensive; omit it for object-store listings."""

    from cs_copilot.storage import S3, is_s3_enabled

    if is_s3_enabled():
        return None
    try:
        return Path(S3.path(rel_path)).stat().st_size
    except OSError:
        return None
