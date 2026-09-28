#!/usr/bin/env python
"""Bundle explicitly selected evidence and immutable Git source snapshots locally.

The selection file declares source_commits, included_paths, and limitations.
This command does not publish, assign a DOI, or certify scientific completeness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from audit_release_artifacts import STRONG_SECRET_PATTERNS

ROOT = Path(__file__).resolve().parents[1]


def selected_files(root, selections):
    files = {}
    for entry in selections:
        relative = Path(entry)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Selection must be relative to the repository: {entry}")
        path = root / relative
        if not path.exists():
            raise ValueError(f"Selected evidence does not exist: {entry}")
        for item in sorted(path.rglob("*")) if path.is_dir() else [path]:
            # Never follow symlinks, including a directory selected via a symlink.
            if item.is_symlink() or item.resolve() != item.absolute():
                raise ValueError(f"Selected evidence includes a symlink: {item}")
            if item.is_dir():
                continue
            rel = item.relative_to(root)
            if any(part in {".git", ".venv", "__pycache__", "python-deps"} for part in rel.parts):
                raise ValueError(
                    f"Select scientific evidence rather than runtime/cache trees: {rel}"
                )
            if item.name.startswith(".env") or item.suffix.lower() in {
                ".pem",
                ".key",
                ".db",
                ".sqlite",
                ".sqlite3",
                ".pyc",
            }:
                raise ValueError(f"Private or runtime file is not evidence: {rel}")
            files[rel.as_posix()] = item
    if not files:
        raise ValueError("No evidence files selected")
    return files


def package(root, selection_path, output):
    root = root.resolve()
    if output.exists():
        raise ValueError("Refusing to overwrite an existing evidence bundle")
    selection_bytes = selection_path.read_bytes()
    selection = json.loads(selection_bytes)
    if not selection.get("source_commits") or not selection.get("limitations"):
        raise ValueError("Selection must identify source commits and completeness limitations")
    evidence = selected_files(root, selection["included_paths"])
    if output.resolve() in {path.resolve() for path in evidence.values()}:
        raise ValueError("Output overlaps selected evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "chemspacecopilot.evidence_bundle.v1",
        "status": "unpublished_working_revision",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "original_repository_root": str(root),
        "path_mapping": "Evidence preserves repository-relative paths under evidence/. Absolute paths in original records retain the original_repository_root; relocate that prefix when replaying. Original records are not rewritten.",
        "limitations": selection["limitations"],
        "source_snapshots": [],
        "files": [],
    }
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise ValueError("Partial archive already exists; inspect it before choosing a new output")
    try:
        with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED) as archive:

            def add(name, data):
                if any(pattern.search(data) for pattern in STRONG_SECRET_PATTERNS):
                    raise ValueError(f"Possible credential signature in selected evidence: {name}")
                archive.writestr(name, data)
                manifest["files"].append(
                    {
                        "path": name,
                        "size_bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )

            add("selection.json", selection_bytes)
            seen = set()
            for ref in selection["source_commits"]:
                commit = subprocess.check_output(
                    ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=root, text=True
                ).strip()
                if commit in seen:
                    continue
                seen.add(commit)
                source = subprocess.check_output(
                    ["git", "archive", "--format=tar.gz", commit], cwd=root
                )
                name = f"source/{commit}.tar.gz"
                # Snapshot bytes are exactly git archive output; no workspace secrets.
                add(name, source)
                manifest["source_snapshots"].append({"git_commit": commit, "path": name})
            for relative, path in sorted(evidence.items()):
                before = path.stat()
                data = path.read_bytes()
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError(f"Evidence changed while packaging: {relative}")
                add("evidence/" + relative, data)
            # Include a portable verifier with no third-party dependencies.
            verifier = """import hashlib, json
from pathlib import Path
root = Path(__file__).resolve().parent
manifest = json.loads((root / "bundle_manifest.json").read_text())
for record in manifest["files"]:
    data = (root / record["path"]).read_bytes()
    assert len(data) == record["size_bytes"], record["path"]
    assert hashlib.sha256(data).hexdigest() == record["sha256"], record["path"]
print(f"Verified {len(manifest['files'])} bundled files")
"""
            add("verify.py", verifier.encode())
            archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
        # Check CRCs after closing the central directory, before promotion.
        with zipfile.ZipFile(partial) as archive:
            if archive.testzip() is not None:
                raise ValueError("Archive integrity check failed")
        partial.rename(output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    with output.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    output.with_name(output.name + ".sha256").write_text(f"{digest}  {output.name}\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = package(ROOT, args.selection, args.output)
    print(f"Bundled {len(result['files'])} entries at {args.output}; status: {result['status']}")


if __name__ == "__main__":
    main()
