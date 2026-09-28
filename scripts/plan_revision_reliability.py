#!/usr/bin/env python
"""Prepare an offline, counterbalanced 48+12 execution plan; never call a provider."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

FROZEN = (
    ("frozen_case_1_seh_analysis", "seh_input"),
    ("frozen_case_2_seh_generation", "seh_analysis"),
    ("frozen_case_3_retrosynthesis", "seh_candidates"),
    ("frozen_case_4_peptide_design", "peptide_input"),
)
BUNDLE_FILES = ("landscape.json", "landscape.safetensors", "nodes.parquet", "sampler.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_hashes(root: Path) -> dict[str, str]:
    paths = [
        path
        for directory in (root / "src", root / "tests/robustness")
        for path in directory.rglob("*")
        if path.is_file() and path.suffix in {".py", ".md", ".yaml", ".yml", ".json", ".j2", ".txt"}
    ]
    paths.extend(root / name for name in ("pyproject.toml", "uv.lock") if (root / name).is_file())
    return {str(path.relative_to(root)): sha256(path) for path in sorted(paths)}


def verify_plan(plan: dict, root: Path) -> list[str]:
    """Verify frozen source, scientific inputs and batch configs without inference."""
    mismatches = []
    for group in ("runtime_sha256", "input_sha256"):
        for name, expected in plan[group].items():
            path = Path(name) if group == "input_sha256" else root / name
            if not path.is_file() or sha256(path) != expected:
                mismatches.append(f"{group}: {name}")
    for batch in plan["batches"]:
        path = Path(batch["config_path"])
        if not path.is_file() or sha256(path) != batch["config_sha256"]:
            mismatches.append(f"batch config: {batch['batch_id']}")
    return mismatches


def fixture(path: Path) -> dict:
    return {"required": True, "session_state_path": str(path.resolve()), "sha256": sha256(path)}


def build_matrix(
    base: dict, fixtures: dict, bundle: Path, output: Path, synplanner: Path
) -> list[dict]:
    """Pure configuration transform; one local repetition per global batch."""
    settings = {
        "synplanner": {
            "data_folder": str(synplanner),
            "prefer_gpu": False,
            "max_time": 120,
            "max_iterations": 100,
            "max_depth": 9,
            "max_tree_size": 10000,
            "top_rules": 50,
            "rule_prob_threshold": 0,
            "min_mol_size": 6,
            "enable_retry_profiles": False,
            "default_top_k": 1,
        }
    }
    batches = []
    for repetition in range(3):
        for variant in range(2):
            batch_id = f"frozen_p{variant}_r{repetition}"
            config = copy.deepcopy(base)
            config["general"].update(
                n_variations=1,
                repetitions=1,
                scientific_seed=(11, 22, 33)[repetition],
                stop_on_timeout=True,
                tier="frozen",
                output_dir=str(output / "runs" / batch_id),
            )
            config["tool_settings"] = copy.deepcopy(settings)
            config["tests"] = {name: copy.deepcopy(base["tests"][name]) for name, _ in FROZEN}
            for name, source in FROZEN:
                config["tests"][name]["fixture"] = fixtures[source]
                config["tests"][name]["prompt_variants"] = [
                    base["tests"][name]["prompt_variants"][variant]
                ]
            batches.append(
                {
                    "batch_id": batch_id,
                    "tier": "frozen",
                    "global_prompt_variant": variant,
                    "global_repetition": repetition,
                    "system": "both",
                    "arm_order": "team-first" if variant == 0 else "single-agent-first",
                    "expected_executions": 8,
                    "configuration": config,
                }
            )
    for repetition in range(3):
        batch_id = f"live_r{repetition}"
        config = copy.deepcopy(base)
        config["general"].update(
            n_variations=1,
            repetitions=1,
            scientific_seed=(11, 22, 33)[repetition],
            stop_on_timeout=True,
            tier="live",
            output_dir=str(output / "runs" / batch_id),
        )
        config["tool_settings"] = copy.deepcopy(settings)
        config["tests"] = {
            name: copy.deepcopy(base["tests"][name])
            for name in ("live_seh_workflow", "live_peptide_design")
        }
        config["tests"]["live_seh_workflow"]["fixture"] = fixtures["live_model"]
        peptide = config["tests"]["live_peptide_design"]
        peptide["fixture"] = fixtures["peptide_input"]
        peptide["required_files"] = [
            {"name": f"Pinned public peptide {name}", "path": str(bundle / name)}
            for name in BUNDLE_FILES
        ]
        batches.append(
            {
                "batch_id": batch_id,
                "tier": "live",
                "global_prompt_variant": 0,
                "global_repetition": repetition,
                "system": "team",
                "arm_order": "team-first",
                "expected_executions": 4,
                "configuration": config,
            }
        )
    return batches


def prepare(args):
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    base = yaml.safe_load(args.config.read_text())
    seh = json.loads(args.seh_manifest.read_text())
    fixture_dir = output / "fixtures"
    fixture_dir.mkdir()
    bundle = args.peptide_bundle.resolve()
    for name in BUNDLE_FILES:
        if not (bundle / name).is_file():
            raise FileNotFoundError(bundle / name)
    peptide = fixture_dir / "peptide_input.json"
    peptide.write_text(
        json.dumps(
            {
                "session_state": {
                    "peptide_landscape_bundle": {"bundle_path": str(bundle)},
                    "revision_context": {
                        "scope": "Existing public aggregate landscape; raw DBAASP and verified training structures unavailable; conditional WAE decoding is a new revision capability"
                    },
                }
            },
            indent=2,
        )
        + "\n"
    )
    model = fixture_dir / "live_model.json"
    model_path = Path(
        json.loads(Path(seh["fixtures"]["seh_input"]["path"]).read_text())["session_state"][
            "_current_gtm_model_path"
        ]
    )
    model.write_text(
        json.dumps(
            {
                "session_state": {
                    "_current_gtm_model_path": str(model_path),
                    "default_descriptor": "autoencoder",
                    "map_type": "default_map",
                }
            },
            indent=2,
        )
        + "\n"
    )
    fixtures = {name: fixture(Path(info["path"])) for name, info in seh["fixtures"].items()}
    fixtures.update(peptide_input=fixture(peptide), live_model=fixture(model))
    for name, info in seh["fixtures"].items():
        if fixtures[name]["sha256"] != info["sha256"]:
            raise ValueError(f"sEH fixture source hash changed: {name}")
    batches = build_matrix(base, fixtures, bundle, output, args.synplanner_dir.resolve())
    configuration_dir = output / "configs"
    configuration_dir.mkdir()
    for batch in batches:
        path = configuration_dir / f"{batch['batch_id']}.yaml"
        path.write_text(yaml.safe_dump(batch.pop("configuration"), sort_keys=False))
        batch["config_path"] = str(path)
        batch["config_sha256"] = sha256(path)
        batch["command"] = [
            sys.executable,
            str(root / "tests/robustness/robustness_minimal_example.py"),
            "--config",
            str(path),
            "--tier",
            batch["tier"],
            "--system",
            batch["system"],
            "--arm-order",
            batch["arm_order"],
            "--n-variations",
            "1",
            "--repetitions",
            "1",
        ]
    import check_revision_readiness as readiness

    input_files = set()
    for specification in fixtures.values():
        snapshot = Path(specification["session_state_path"])
        input_files.add(snapshot)
        state = json.loads(snapshot.read_text())["session_state"]
        for value in readiness.pointers(state):
            path = Path(value)
            if not path.is_absolute():
                path = snapshot.parent / path
            if path.is_dir():
                input_files.update(item for item in path.rglob("*") if item.is_file())
            else:
                input_files.add(path)
    for folder in (args.synplanner_dir, args.autoencoder_dir, args.peptide_model_dir):
        input_files.update(item for item in folder.resolve().rglob("*") if item.is_file())
    plan = {
        "schema_version": "1.0",
        "purpose": "Predeclared prospective study, no executions performed by this planning script",
        "git_commit_at_planning": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "runtime_sha256": runtime_hashes(root),
        "input_sha256": {str(path): sha256(path) for path in sorted(input_files)},
        "environment": {
            "PYTHONPATH": str(root / "src"),
            "AUTOENCODER_MODEL_PATH": str(args.autoencoder_dir.resolve()),
            "PEPTIDE_DESIGNER_MODEL_PATH": str(args.peptide_model_dir.resolve()),
            "USE_S3": "false",
            "HF_HUB_OFFLINE": "1",
            "OMP_NUM_THREADS": "1",
            "AGNO_TELEMETRY": "false",
        },
        "provider_credential": "Ambient DEEPSEEK_API_KEY; its value is never recorded",
        "local_index_mapping": "Each config contains one explicit variant and repetition. Aggregate using global indices in this plan; retain original local indices.",
        "frozen_executions": 48,
        "live_stage_executions": 12,
        "total_executions": 60,
        "batches": batches,
    }
    (output / "study_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "plan": str(output / "study_plan.json"),
                "batches": len(batches),
                "executions": sum(batch["expected_executions"] for batch in batches),
            },
            indent=2,
        )
    )


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=root / "tests/robustness/manuscript_reliability.yaml"
    )
    parser.add_argument(
        "--verify-plan", type=Path, help="Check an existing frozen plan only; no inference"
    )
    parser.add_argument("--seh-manifest", type=Path)
    parser.add_argument("--peptide-bundle", type=Path)
    parser.add_argument("--synplanner-dir", type=Path)
    parser.add_argument("--autoencoder-dir", type=Path)
    parser.add_argument("--peptide-model-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.verify_plan:
        mismatches = verify_plan(json.loads(args.verify_plan.read_text()), root)
        print(json.dumps({"verified": not mismatches, "mismatches": mismatches}, indent=2))
        raise SystemExit(1 if mismatches else 0)
    missing = [
        name
        for name in (
            "seh_manifest",
            "peptide_bundle",
            "synplanner_dir",
            "autoencoder_dir",
            "peptide_model_dir",
            "output_dir",
        )
        if getattr(args, name) is None
    ]
    if missing:
        parser.error(
            "Required for planning: " + ", ".join("--" + name.replace("_", "-") for name in missing)
        )
    prepare(args)


if __name__ == "__main__":
    main()
