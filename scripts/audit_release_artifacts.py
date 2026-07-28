#!/usr/bin/env python3
"""Fail when built release archives omit contracts or contain private state."""

from __future__ import annotations

import argparse
import re
import tarfile
import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
STRONG_SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{30,}"),
    re.compile(rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"),
)
FORBIDDEN_COMPONENTS = {
    ".cache",
    ".codex",
    ".git",
    ".pytest_cache",
    ".staging",
    ".vscode",
    "__pycache__",
    "artifacts",
    "events",
    "runs",
    "sessions",
}
FORBIDDEN_ROOTS = {".chainlit", "data", "dist", "notebooks", "site"}
FORBIDDEN_DATABASE_SUFFIXES = (
    ".db",
    ".db-journal",
    ".db-shm",
    ".db-wal",
    ".sqlite",
    ".sqlite-journal",
    ".sqlite-shm",
    ".sqlite-wal",
    ".sqlite3",
    ".sqlite3-journal",
    ".sqlite3-shm",
    ".sqlite3-wal",
)


@dataclass(frozen=True)
class ArchiveEntry:
    """One normalized regular file and its archived bytes."""

    path: str
    content: bytes


def _project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def _one_archive(dist_dir: Path, pattern: str) -> Path:
    matches = sorted(dist_dir.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one archive matching {pattern!r} in {dist_dir}, "
            f"found {[path.name for path in matches]}"
        )
    return matches[0]


def _safe_path(raw_path: str, *, strip_root: bool) -> str:
    candidate = PurePosixPath(raw_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError(f"archive contains an unsafe path: {raw_path!r}")
    parts = candidate.parts[1:] if strip_root else candidate.parts
    if not parts:
        raise RuntimeError(f"archive contains an empty member path: {raw_path!r}")
    return PurePosixPath(*parts).as_posix()


def _zip_entries(path: Path) -> tuple[ArchiveEntry, ...]:
    with zipfile.ZipFile(path) as archive:
        return tuple(
            ArchiveEntry(
                _safe_path(info.filename, strip_root=False),
                archive.read(info),
            )
            for info in archive.infolist()
            if not info.is_dir()
        )


def _tar_entries(path: Path) -> tuple[ArchiveEntry, ...]:
    with tarfile.open(path, "r:gz") as archive:
        members = tuple(archive.getmembers())
        roots = {PurePosixPath(member.name).parts[0] for member in members if member.name}
        if len(roots) != 1:
            raise RuntimeError(f"sdist must have one archive root, found {sorted(roots)}")
        linked = [member.name for member in members if member.issym() or member.islnk()]
        if linked:
            raise RuntimeError(f"sdist contains symbolic or hard links: {linked}")
        entries = []
        for member in members:
            if not member.isfile():
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(f"could not read sdist member {member.name!r}")
            entries.append(
                ArchiveEntry(
                    _safe_path(member.name, strip_root=True),
                    handle.read(),
                )
            )
        return tuple(entries)


def _forbidden_reason(path: str) -> str | None:
    parts = PurePosixPath(path).parts
    lowered = tuple(part.lower() for part in parts)
    name = lowered[-1]
    if lowered[0] in FORBIDDEN_ROOTS:
        return f"forbidden release root {parts[0]!r}"
    if any(part in FORBIDDEN_COMPONENTS for part in lowered):
        return "private/cache/runtime path component"
    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        return "environment-secret filename"
    if name.endswith(FORBIDDEN_DATABASE_SUFFIXES):
        return "database or database-sidecar filename"
    if name.endswith((".key", ".pem")):
        return "private-key filename"
    if name.endswith(".json") and ("credentials" in name or "client-secret" in name):
        return "credential filename"
    return None


def _validate_entries(
    entries: Iterable[ArchiveEntry],
    *,
    archive_name: str,
    required: set[str],
) -> set[str]:
    selected = tuple(entries)
    paths = {entry.path for entry in selected}
    if len(paths) != len(selected):
        raise RuntimeError(f"{archive_name} contains duplicate normalized paths")
    missing = sorted(required - paths)
    if missing:
        raise RuntimeError(f"{archive_name} is missing required files: {missing}")
    forbidden = {
        entry.path: reason
        for entry in selected
        if (reason := _forbidden_reason(entry.path)) is not None
    }
    if forbidden:
        raise RuntimeError(f"{archive_name} contains forbidden files: {forbidden}")
    secret_hits = []
    for entry in selected:
        if any(pattern.search(entry.content) for pattern in STRONG_SECRET_PATTERNS):
            secret_hits.append(entry.path)
    if secret_hits:
        raise RuntimeError(
            f"{archive_name} contains strong secret signatures: {sorted(secret_hits)}"
        )
    return paths


def audit(dist_dir: Path) -> None:
    """Audit the one wheel and sdist for the current project version."""

    version = _project_version()
    wheel = _one_archive(dist_dir, f"*-{version}-*.whl")
    sdist = _one_archive(dist_dir, f"*-{version}.tar.gz")
    wheel_paths = _validate_entries(
        _zip_entries(wheel),
        archive_name=wheel.name,
        required={
            "cs_copilot/skills/catalog/chembl-to-gtm-report/SKILL.md",
            "cs_copilot/workflows/catalog/chembl-to-gtm-report/WORKFLOW.md",
        },
    )
    sdist_paths = _validate_entries(
        _tar_entries(sdist),
        archive_name=sdist.name,
        required={
            ".agents/plugins/marketplace.json",
            ".env.example",
            "plugins/chemspace-copilot/.codex-plugin/plugin.json",
            "plugins/chemspace-copilot/.mcp.json",
            "plugins/chemspace-copilot/skills/chemspace-orchestrate/SKILL.md",
            "plugins/chemspace-copilot/skills/chemspace-orchestrate/agents/openai.yaml",
            "scripts/audit_release_artifacts.py",
        },
    )
    leaked_delivery_files = sorted(
        path for path in wheel_paths if path.startswith((".agents/", "plugins/"))
    )
    if leaked_delivery_files:
        raise RuntimeError(
            "wheel must contain Python/catalog runtime only; found plugin delivery "
            f"files: {leaked_delivery_files}"
        )
    print(
        f"Release archive audit passed: {wheel.name} ({len(wheel_paths)} files), "
        f"{sdist.name} ({len(sdist_paths)} files)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dist_dir",
        nargs="?",
        type=Path,
        default=ROOT / "dist",
        help="Directory containing exactly one current-version wheel and sdist.",
    )
    args = parser.parse_args()
    audit(args.dist_dir.resolve())


if __name__ == "__main__":
    main()
