import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "evidence_bundle", SCRIPTS / "package_revision_evidence.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_preserves_raw_evidence_and_pins_source(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "source.py").write_text("print('versioned')\n")
    subprocess.run(["git", "add", "source.py"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "commit",
            "-qm",
            "initial",
        ],
        cwd=root,
        check=True,
    )
    raw = root / "raw.json"
    raw.write_bytes(b'{"unmodified":true}\n')
    selection = root / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "source_commits": ["HEAD"],
                "included_paths": ["raw.json"],
                "limitations": ["Not publication-ready"],
            }
        )
    )
    output = tmp_path / "bundle.zip"
    manifest = module.package(root, selection, output)
    assert manifest["status"] == "unpublished_working_revision"
    assert len(manifest["source_snapshots"][0]["git_commit"]) == 40
    with zipfile.ZipFile(output) as archive:
        assert archive.read("evidence/raw.json") == raw.read_bytes()
        for item in manifest["files"]:
            assert hashlib.sha256(archive.read(item["path"])).hexdigest() == item["sha256"]
    with pytest.raises(ValueError, match="overwrite"):
        module.package(root, selection, output)


def test_rejects_paths_outside_selected_workspace(tmp_path):
    with pytest.raises(ValueError, match="relative"):
        module.selected_files(tmp_path, ["../outside"])
    outside = tmp_path / "outside"
    outside.write_text("private")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "linked").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        module.selected_files(root, ["linked"])


def test_rejects_environment_files(tmp_path):
    (tmp_path / ".env").write_text("SECRET=dummy")
    with pytest.raises(ValueError, match="Private"):
        module.selected_files(tmp_path, [".env"])


@pytest.mark.parametrize(
    "filename, contents, expected",
    [
        (".env", "PROVIDER_TOKEN=plain-unrecognized-format", "Private or runtime source member"),
        ("source.py", "TOKEN='" + "sk-" + "a" * 30 + "'", "credential signature in source member"),
    ],
)
def test_rejects_committed_private_material_before_archive_promotion(
    tmp_path, filename, contents, expected
):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / filename).write_text(contents)
    subprocess.run(["git", "add", filename], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )
    (root / "raw.json").write_text('{"scientific":true}')
    selection = root / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "source_commits": ["HEAD"],
                "included_paths": ["raw.json"],
                "limitations": ["Test fixture"],
            }
        )
    )
    output = tmp_path / "bundle.zip"
    with pytest.raises(ValueError, match=expected):
        module.package(root, selection, output)
    assert not output.exists()
    assert not output.with_name("bundle.zip.partial").exists()
    assert not output.with_name("bundle.zip.sha256").exists()
