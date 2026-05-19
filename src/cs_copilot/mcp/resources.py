"""Resource access for the ChemSpace MCP server.

Resources expose session artifacts (datasets, plots, reports) that ChemSpace
toolkits write under the active S3 / local session prefix. We surface them
under a custom ``cscopilot://`` URI scheme so the S3 and local backends share
one representation; the actual read is delegated to
:class:`cs_copilot.storage.client.S3`.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)

URI_PREFIX = "cscopilot://session/"
MANIFEST_PATH = "manifest.json"
_TEXT_MIME_RE = re.compile(r"^(text/|application/(?:json|x-yaml|xml|toml|x-toml))")


@dataclass(frozen=True)
class ResourceEntry:
    """One MCP resource record (URI, name, MIME type, optional size)."""

    uri: str
    name: str
    mime_type: str
    size: Optional[int] = None


def _guess_mime(rel_path: str) -> str:
    mime, _ = mimetypes.guess_type(rel_path)
    return mime or "application/octet-stream"


def list_entries() -> List[ResourceEntry]:
    """List session artifacts as MCP resource entries.

    Local mode: walks the local session directory under ``data/``.
    S3 mode: walks the active bucket prefix via ``s3fs``.
    Returns an empty list when the prefix has no artifacts yet.
    """

    from cs_copilot.storage import S3, get_s3_config, is_s3_enabled

    entries: List[ResourceEntry] = []
    entries.append(
        ResourceEntry(
            uri=f"{URI_PREFIX}{MANIFEST_PATH}",
            name=MANIFEST_PATH,
            mime_type="application/json",
        )
    )

    if not is_s3_enabled():
        local_root = Path("data") / S3.current_prefix().strip("/")
        if local_root.exists():
            for path in sorted(local_root.rglob("*")):
                if path.is_file():
                    rel = path.relative_to(local_root).as_posix()
                    entries.append(
                        ResourceEntry(
                            uri=f"{URI_PREFIX}{rel}",
                            name=path.name,
                            mime_type=_guess_mime(rel),
                            size=path.stat().st_size,
                        )
                    )
        return entries

    config = get_s3_config()
    try:
        import s3fs  # type: ignore

        fs = s3fs.S3FileSystem(**config.to_storage_options())
        bucket_root = f"s3://{config.bucket_name}/{S3.current_prefix().strip('/')}"
        try:
            files = fs.find(bucket_root)
        except FileNotFoundError:
            files = []
        prefix_strip = f"{config.bucket_name}/{S3.current_prefix().strip('/')}/"
        for key in sorted(files):
            rel = key
            if rel.startswith(prefix_strip):
                rel = rel[len(prefix_strip) :]
            elif rel.startswith("s3://"):
                rel = rel.split(prefix_strip, 1)[-1]
            entries.append(
                ResourceEntry(
                    uri=f"{URI_PREFIX}{rel}",
                    name=rel.rsplit("/", 1)[-1],
                    mime_type=_guess_mime(rel),
                )
            )
    except Exception as exc:  # pragma: no cover - network paths only
        logger.warning("Failed to list S3 resources: %s", exc)
    return entries


def _resolve_relative_path(uri: str) -> str:
    if not uri.startswith(URI_PREFIX):
        raise ValueError(f"Unsupported resource URI: {uri!r}")
    return uri[len(URI_PREFIX) :]


def read_text(uri: str) -> str:
    """Return the resource content decoded as UTF-8 text."""

    blob, mime = _read_blob_with_mime(uri)
    if mime != "application/json" and not _TEXT_MIME_RE.match(mime):
        # Best-effort decode for ambiguous types.
        logger.debug("Decoding non-text mime %s as utf-8", mime)
    return blob.decode("utf-8", errors="replace")


def read_blob(uri: str) -> bytes:
    """Return the resource content as raw bytes."""

    blob, _ = _read_blob_with_mime(uri)
    return blob


def resource_mime(uri: str) -> str:
    """Return the MIME type associated with a resource URI."""

    rel_path = _resolve_relative_path(uri)
    if rel_path == MANIFEST_PATH:
        return "application/json"
    return _guess_mime(rel_path)


def is_text_resource(uri: str) -> bool:
    """Return True if the resource is best surfaced as text."""

    return bool(_TEXT_MIME_RE.match(resource_mime(uri)))


def _read_blob_with_mime(uri: str) -> tuple[bytes, str]:
    rel_path = _resolve_relative_path(uri)
    if rel_path == MANIFEST_PATH:
        return _build_manifest().encode("utf-8"), "application/json"

    from cs_copilot.storage import S3

    with S3.open(rel_path, "rb") as handle:
        blob = handle.read()
    if isinstance(blob, str):
        blob = blob.encode("utf-8")
    return blob, _guess_mime(rel_path)


def _build_manifest() -> str:
    from cs_copilot.storage import LAYOUT_VERSION, S3

    payload = {
        "layout_version": LAYOUT_VERSION,
        "session_prefix": S3.current_prefix(),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def iter_entries() -> Iterable[ResourceEntry]:
    """Yield resource entries (alias of :func:`list_entries` for symmetry)."""

    yield from list_entries()
