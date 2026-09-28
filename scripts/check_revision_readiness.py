#!/usr/bin/env python
"""Check manuscript benchmark prerequisites offline, without constructing agents.

Only credential presence is reported. No provider requests, model deserialization,
scientific computation, or implicit asset downloads are performed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"
CORE_PACKAGES = (
    "agno",
    "rdkit",
    "torch",
    "chemographykit",
    "ugtm",
    "deepchemography",
    "optuna",
    "huggingface-hub",
    "transformers",
    "pandas",
    "numpy",
    "PyYAML",
)
FILE_SUFFIXES = (
    ".csv",
    ".csv.gz",
    ".parquet",
    ".json",
    ".pkl",
    ".pkl.gz",
    ".pt",
    ".ckpt",
    ".png",
    ".svg",
    ".pdf",
    ".html",
    ".md",
    ".sdf",
    ".fasta",
)
SECRET_KEY = re.compile(r"api[_-]?key|token|password|secret|credential|authorization", re.I)


def expand(value: str, env: Mapping[str, str], home: Path) -> str:
    value = re.sub(
        r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)",
        lambda m: env.get(m.group(1) or m.group(2), m.group(0)),
        value,
    )
    return str(home) + value[1:] if value == "~" or value.startswith("~/") else value


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def artifact(path: str, *, base: Path, env: Mapping[str, str], home: Path) -> dict[str, Any]:
    expanded = expand(path, env, home)
    if "$" in expanded:
        return {"status": "unresolved_variable"}
    if expanded.startswith(("s3://", "http://", "https://")):
        # URLs may carry access tokens; never echo them into the readiness report.
        return {"status": "remote_not_verified_offline"}
    resolved = Path(expanded.removeprefix("file://"))
    if not resolved.is_absolute():
        resolved = base / resolved
    result: dict[str, Any] = {"path": str(resolved), "status": "missing"}
    try:
        if resolved.is_file():
            result["bytes"] = resolved.stat().st_size
            with resolved.open("rb") as handle:
                is_pointer = handle.read(len(LFS_HEADER)).startswith(LFS_HEADER)
            result["status"] = (
                "git_lfs_pointer" if is_pointer else "present" if result["bytes"] else "empty"
            )
        elif resolved.is_dir():
            result["status"] = "directory"
    except OSError:
        result["status"] = "unreadable"
    return result


def pointers(value: Any, key: str = ""):
    if SECRET_KEY.search(key) or key.lower() in {
        "label",
        "title",
        "description",
        "session_memory_summary",
        "prompt",
        "summary",
        "notes",
        "message",
        "content",
    }:
        return
    if isinstance(value, dict):
        for name, item in value.items():
            yield from pointers(item, str(name))
    elif isinstance(value, list):
        for item in value:
            yield from pointers(item, key)
    elif (
        isinstance(value, str)
        and "\n" not in value
        and (
            key.endswith(("_path", "_uri"))
            or key in {"path", "uri"}
            or value.lower().endswith(FILE_SUFFIXES)
            or value.startswith("s3://")
        )
    ):
        yield value


def check_fixture(
    fixture: Mapping[str, Any], *, config_dir: Path, env: Mapping[str, str], home: Path
) -> dict[str, Any]:
    info = artifact(
        str(fixture.get("session_state_path") or ""), base=config_dir, env=env, home=home
    )
    if info["status"] != "present":
        return info
    path = Path(info["path"])
    expected = expand(str(fixture.get("sha256") or ""), env, home)
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        return {**info, "status": "missing_or_invalid_checksum"}
    actual = digest(path)
    if actual.lower() != expected.lower():
        return {**info, "status": "checksum_mismatch"}
    try:
        loaded = json.loads(path.read_text())
        state = loaded.get("session_state", loaded) if isinstance(loaded, dict) else None
        if not isinstance(state, dict):
            raise ValueError("invalid state")
    except (OSError, ValueError, UnicodeError):
        return {**info, "status": "invalid_snapshot"}
    inputs = [
        artifact(item, base=path.parent, env=env, home=home)
        for item in sorted(set(pointers(state)))
    ]
    return {
        **info,
        "status": (
            "verified"
            if all(item["status"] in {"present", "directory"} for item in inputs)
            else "artifact_unavailable"
        ),
        "sha256": actual,
        "referenced_artifacts": inputs,
        "session_gtm_available": (
            artifact(str(state["_current_gtm_model_path"]), base=path.parent, env=env, home=home)[
                "status"
            ]
            == "present"
            if state.get("_current_gtm_model_path")
            else False
        ),
    }


def check_readiness(
    config_path: Path,
    *,
    project_root: Path,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    tier: str = "both",
    system: str = "team",
    tests: list[str] | None = None,
    repetitions: int | None = None,
    n_variations: int | None = None,
) -> dict[str, Any]:
    env = dict(os.environ if env is None else env)
    home = Path.home() if home is None else home
    config = yaml.safe_load(config_path.read_text())
    general = config.get("general") or {}
    model = config.get("model") or {}
    repeats = repetitions if repetitions is not None else int(general.get("repetitions", 1))
    variants = n_variations if n_variations is not None else int(general.get("n_variations", 1))
    if repeats < 1 or variants < 1:
        raise ValueError("Repetitions and variations must be positive")
    selected = {
        name: spec
        for name, spec in config.get("tests", {}).items()
        if (name in tests if tests is not None else spec.get("enabled", False))
        and (tier == "both" or spec.get("tier", "both") in {tier, "both"})
    }
    unknown = sorted(set(tests or []) - set(config.get("tests", {})))
    blockers = [f"Unknown test: {name}" for name in unknown]
    if not selected:
        blockers.append("No tests selected")
    key_name = str(model.get("api_key_env") or "")
    credential = {"environment_variable": key_name, "present": bool(env.get(key_name))}
    if model.get("provider") != "ollama" and not credential["present"]:
        blockers.append(f"Provider credential environment variable is absent: {key_name}")

    validators = {
        step.get("validator", spec.get("validator"))
        for spec in selected.values()
        for step in (spec.get("steps") or [spec])
    }
    packages = {}
    package_names = list(CORE_PACKAGES)
    if "retrosynthesis" in validators:
        package_names += ["SynPlanner"]
    for package in package_names:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
            blockers.append(f"Missing package: {package}")
    if packages.get("agno") not in {None, "2.1.9"}:
        blockers.append("Agno differs from the benchmark's pinned version 2.1.9")
    if sys.version_info[:2] not in {(3, 11), (3, 12)}:
        blockers.append("The project requires Python 3.11 or 3.12")

    # Both architectures eagerly construct both design toolkits, including for
    # recovery-only prompts. Check this shared startup cost explicitly.
    model_files = {}
    for name, variable, folder, filenames in (
        (
            "autoencoder",
            "AUTOENCODER_MODEL_PATH",
            "autoencoder",
            ("config.pt", "vocab.pt", "model.pt"),
        ),
        ("peptide", "PEPTIDE_DESIGNER_MODEL_PATH", "peptide_designer", ("model.pt", "vocab.dict")),
    ):
        directory = expand(env.get(variable) or f"~/.cache/cs_copilot/models/{folder}", env, home)
        model_files[name] = [
            artifact(str(Path(directory) / item), base=project_root, env=env, home=home)
            for item in filenames
        ]
        if any(item["status"] != "present" for item in model_files[name]):
            blockers.append(
                f"Missing/invalid {name} checkpoint; agent construction would fail or download assets"
            )

    prerequisite_cases = {}
    per_arm_count = 0
    for name, spec in selected.items():
        count = (
            len(spec["steps"])
            if spec.get("steps")
            else min(variants, len(spec.get("prompt_variants") or [None]))
        )
        per_arm_count += count * repeats
        item = {}
        if spec.get("fixture"):
            item["fixture"] = check_fixture(
                spec["fixture"], config_dir=config_path.parent, env=env, home=home
            )
            if item["fixture"]["status"] != "verified":
                blockers.append(f"Fixture unavailable or unverified: {name}")
        item["required_files"] = []
        for requirement in spec.get("required_files") or []:
            path = (
                env.get(requirement.get("env", ""))
                or requirement.get("path")
                or requirement.get("default_path")
                or ""
            )
            checked = artifact(path, base=config_path.parent, env=env, home=home)
            item["required_files"].append({"name": requirement.get("name"), **checked})
            if checked["status"] != "present":
                blockers.append(
                    f"Missing prerequisite for {name}: {requirement.get('name', 'input')}"
                )
        prerequisite_cases[name] = item

    if "seh_analysis" in validators:
        gtm_root = home / ".cache/cs_copilot/models/gtm"
        model_files["cached_gtm"] = [
            artifact(str(item), base=project_root, env=env, home=home)
            for item in sorted(gtm_root.glob("*.pkl*"))
        ]
        cached_map_present = any(item["status"] == "present" for item in model_files["cached_gtm"])
        for name, spec in selected.items():
            analysis = any(
                step.get("validator", spec.get("validator")) == "seh_analysis"
                for step in (spec.get("steps") or [spec])
            )
            fixture_map_present = (
                prerequisite_cases[name].get("fixture", {}).get("session_gtm_available", False)
            )
            if analysis and not cached_map_present and not fixture_map_present:
                blockers.append(
                    f"GTM unavailable for {name}: provide a verified session map pointer or the cached default map"
                )
    if "retrosynthesis" in validators:
        candidates = [
            project_root / "synplan_data",
            Path.cwd() / "synplan_data",
            home / ".synplan_data",
        ]
        explicit_folder = config.get("tool_settings", {}).get("synplanner", {}).get("data_folder")
        if explicit_folder:
            candidates.insert(0, Path(expand(str(explicit_folder), env, home)))
        relative_files = [
            "building_blocks/building_blocks_em_sa_ln.smi",
            "uspto/uspto_reaction_rules.pickle",
            "uspto/weights/ranking_policy_network.ckpt",
        ]
        inventory = [
            [
                artifact(str(folder / filename), base=project_root, env=env, home=home)
                for filename in relative_files
            ]
            for folder in candidates
        ]
        model_files["synplanner"] = inventory
        if not any(all(item["status"] == "present" for item in batch) for batch in inventory):
            blockers.append("SynPlanner rules, policy weights, and building blocks are unavailable")

    executions = per_arm_count * (2 if system == "both" else 1)
    return {
        "schema_version": "1.0",
        "ready_for_pilot": not blockers,
        "scope": "Offline existence/dependency preflight; scientific correctness, provider reachability, and model load compatibility are not tested",
        "project_root": str(project_root),
        "config_path": str(config_path),
        "tier": tier,
        "system": system,
        "model": {
            "provider": model.get("provider"),
            "model_id": model.get("model_id"),
            "credential": credential,
        },
        "packages": packages,
        "models": model_files,
        "cases": prerequisite_cases,
        "planned_task_executions": executions,
        "configured_execution_timeout_ceiling_seconds": executions
        * int(general.get("timeout_seconds", 0)),
        "resources": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "disk_free_bytes": shutil.disk_usage(project_root).free,
        },
        "blockers": blockers,
    }


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=root / "tests/robustness/manuscript_reliability.yaml"
    )
    parser.add_argument("--tier", choices=["live", "frozen", "both"], default="both")
    parser.add_argument("--system", choices=["team", "single_agent", "both"], default="team")
    parser.add_argument("--test", dest="tests", action="append")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--n-variations", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = check_readiness(
        args.config.resolve(),
        project_root=root,
        tier=args.tier,
        system=args.system,
        tests=args.tests,
        repetitions=args.repetitions,
        n_variations=args.n_variations,
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")
    return 0 if report["ready_for_pilot"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
