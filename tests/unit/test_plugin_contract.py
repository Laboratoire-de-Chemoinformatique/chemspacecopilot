import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "plugins" / "chemspace-copilot"


def test_repo_plugin_manifest_and_mcp_profile_are_thin_and_versioned():
    manifest = json.loads((PLUGIN / ".codex-plugin" / "plugin.json").read_text())
    mcp = json.loads((PLUGIN / ".mcp.json").read_text())

    assert manifest["name"] == "chemspace-copilot"
    assert manifest["version"] == "0.2.0"
    assert manifest["skills"] == "./skills/"
    assert manifest["mcpServers"] == "./.mcp.json"
    assert "apps" not in manifest
    server = mcp["mcpServers"]["chemspace-copilot"]
    assert server["command"] == "cscopilot-mcp"
    assert server["args"] == ["--profile", "standard"]


def test_bridge_skill_has_no_scientific_procedure_copy():
    skill_path = PLUGIN / "skills" / "chemspace-orchestrate" / "SKILL.md"
    text = skill_path.read_text()
    frontmatter = yaml.safe_load(text.split("---", 2)[1])

    assert frontmatter["name"] == "chemspace-orchestrate"
    assert "mcp_bootstrap" in text
    assert "fetch" in text
    assert "gtm_optimization" not in text
    assert "chembl_fetch_compounds" not in text


def test_repo_marketplace_entry_is_installable_on_install():
    marketplace = json.loads((ROOT / ".agents" / "plugins" / "marketplace.json").read_text())
    entry = next(item for item in marketplace["plugins"] if item["name"] == "chemspace-copilot")

    assert entry["source"] == {"source": "local", "path": "./plugins/chemspace-copilot"}
    assert entry["policy"] == {
        "installation": "AVAILABLE",
        "authentication": "ON_INSTALL",
    }


def test_source_distribution_omits_developer_notebooks_and_runtime_state():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    config = project["tool"]["hatch"]["build"]["targets"]["sdist"]

    assert "notebooks" not in config["only-include"]
    assert {"/.env", "/.codex", "/.vscode", "/data", "/dist"}.issubset(config["exclude"])
    assert {
        "/**/.env",
        "/**/.env.*",
        "/**/*.db-wal",
        "/**/*.pem",
        "/**/*credentials*.json",
        "/**/.staging/**",
        "/**/artifacts/**",
        "/**/events/**",
        "/**/runs/**",
        "/**/sessions/**",
    }.issubset(config["exclude"])
    assert (ROOT / "scripts" / "audit_release_artifacts.py").is_file()


def test_release_archive_auditor_rejects_runtime_paths_and_key_signatures():
    audit_path = ROOT / "scripts" / "audit_release_artifacts.py"
    spec = importlib.util.spec_from_file_location("cs_copilot_release_audit", audit_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)

    for path in (
        "docs/artifacts/result.json",
        "examples/events/0001.jsonl",
        "public/runs/run-001/manifest.json",
        "src/package/sessions/session-001/data.csv",
    ):
        assert module._forbidden_reason(path) is not None
    for token in (
        b"sk-" + b"1234567890abcdefghijklmnop",
        b"sk-" + b"proj-" + b"1234567890abcdefghijklmnop",
        b"sk-" + b"svcacct-" + b"1234567890abcdefghijklmnop",
    ):
        assert any(pattern.search(token) for pattern in module.STRONG_SECRET_PATTERNS)
