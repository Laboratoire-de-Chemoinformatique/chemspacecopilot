#!/usr/bin/env python3
"""Run a predeclared SynPlanner target list with one bounded search per target.

No LLM fallback or downloads. Default: 10 DISTINCT targets, one 120-second search
profile per target, 300-second whole-worker guard (setup/rendering included).
Use --expected-target-count for an explicitly separate pilot. A CSV must contain
smiles (and optionally target_id); JSON accepts a list or {"targets": [...]}.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import traceback
from pathlib import Path

from _study_common import apply_seed, file_identity, run_jobs, software_identity, write_json
from summarize_retrosynthesis import summarize_plan


def load_targets(path: Path, expected_count: int) -> list[dict]:
    from rdkit import Chem

    if path.suffix.lower() == ".csv":
        with path.open(newline="") as handle:
            targets = list(csv.DictReader(handle))
    else:
        payload = json.loads(path.read_text())
        targets = payload.get("targets") if isinstance(payload, dict) else payload
    if not isinstance(targets, list) or len(targets) != expected_count or expected_count < 1:
        raise ValueError(
            f"Predeclare exactly {expected_count} targets; do not silently select a subset"
        )
    results, seen = [], set()
    for index, entry in enumerate(targets, start=1):
        item = {"smiles": entry} if isinstance(entry, str) else entry
        if not isinstance(item, dict) or not isinstance(item.get("smiles"), str):
            raise ValueError(f"Target {index} must contain a SMILES string")
        mol = Chem.MolFromSmiles(item["smiles"])
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"Target {index} has invalid SMILES")
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        if canonical in seen:
            raise ValueError(f"Target {index} duplicates an earlier canonical target")
        seen.add(canonical)
        results.append(
            {
                "target_id": str(item.get("target_id") or item.get("id") or f"target_{index:03d}"),
                "smiles": item["smiles"],
                "canonical_smiles": canonical,
            }
        )
    return results


def check_planning_imports() -> None:
    """Do not spend the target denominator on a missing binary/runtime dependency."""
    from cs_copilot.tools.chemistry.synplanner_toolkit import (
        _install_cgrtools_miniracer_compatibility,
    )

    try:
        _install_cgrtools_miniracer_compatibility()
        required = {
            "synplan.mcts.tree": ("Tree",),
            "synplan.mcts.expansion": ("PolicyNetworkFunction",),
            "synplan.chem.utils": ("mol_from_smiles",),
            "synplan.utils.config": (
                "PolicyNetworkConfig",
                "TreeConfig",
            ),
            "synplan.utils.loading": (
                "load_building_blocks",
                "load_reaction_rules",
            ),
        }
        for module_name, symbols in required.items():
            module = importlib.import_module(module_name)
            for symbol in symbols:
                if not hasattr(module, symbol):
                    raise ImportError(f"{module_name}.{symbol} is unavailable")
        config_module = importlib.import_module("synplan.utils.config")
        loading_module = importlib.import_module("synplan.utils.loading")
        if not (
            hasattr(config_module, "RolloutEvaluationConfig")
            and hasattr(loading_module, "load_evaluation_function")
        ):
            config = config_module.TreeConfig(evaluation_type="rollout")
            tree_type = importlib.import_module("synplan.mcts.tree").Tree
            if config.evaluation_type != "rollout" or not callable(
                getattr(tree_type, "_rollout_node", None)
            ):
                raise ImportError("No supported built-in rollout evaluation API")
        # Import success alone does not show that the native depiction backend
        # works; reject missing MiniRacer before consuming any target attempts.
        parsed = importlib.import_module("synplan.chem.utils").mol_from_smiles(
            "CC(=O)Oc1ccccc1C(=O)O", standardize=True, clean_stereo=True, clean2d=True
        )
        if parsed is None or len(parsed) == 0:
            raise ValueError("CGRtools parsing/2D depiction preflight returned no molecule")
    except Exception as exc:
        raise ValueError(
            f"SynPlanner runtime preflight failed before starting any targets: {type(exc).__name__}: {exc}"
        ) from exc


def build_specification(args: argparse.Namespace) -> dict:
    if args.max_time <= 0 or args.case_timeout_seconds <= args.max_time:
        raise ValueError("Search time must be positive and whole-worker timeout must exceed it")
    if args.max_iterations < 1 or args.max_depth < 1 or args.top_rules < 1 or args.min_mol_size < 0:
        raise ValueError(
            "Search iterations/depth/top-rules must be positive; minimum molecule size nonnegative"
        )
    for module in ("synplan", "torch"):
        if importlib.util.find_spec(module) is None:
            raise ValueError(f"Required installed dependency is unavailable: {module}")
    check_planning_imports()
    target_source = file_identity(args.targets)
    targets = load_targets(args.targets, args.expected_target_count)
    if args.seed < 0 or args.seed + len(targets) >= 2**32:
        raise ValueError("Target seeds must be in [0, 2**32)")
    assets = {}
    for key, candidates in {
        "building_blocks": ("building_blocks/building_blocks_em_sa_ln.smi", "building_blocks.smi"),
        "reaction_rules": ("uspto/uspto_reaction_rules.pickle", "uspto_reaction_rules.pickle"),
        "ranking_policy": (
            "uspto/weights/ranking_policy_network.ckpt",
            "ranking_policy_network.ckpt",
        ),
    }.items():
        path = next(
            (
                args.data_dir / candidate
                for candidate in candidates
                if (args.data_dir / candidate).is_file()
            ),
            args.data_dir / candidates[0],
        )
        assets[key] = file_identity(path)
    return {
        "implementation_files": {
            name: file_identity(Path(__file__).resolve().parents[1] / name)
            for name in [
                "src/cs_copilot/tools/chemistry/synplanner_toolkit.py",
                "scripts/summarize_retrosynthesis.py",
            ]
        },
        "study": "retrosynthesis",
        "software": software_identity(),
        "assets": assets,
        "launcher": file_identity(Path(__file__)),
        "ledger_code": file_identity(Path(__file__).with_name("_study_common.py")),
        "target_source": target_source,
        "case_timeout_seconds": args.case_timeout_seconds,
        "configuration": {
            "data_dir": str(args.data_dir.resolve()),
            "max_time": args.max_time,
            "max_iterations": args.max_iterations,
            "max_depth": args.max_depth,
            "max_tree_size": 10000,
            "top_rules": args.top_rules,
            "rule_prob_threshold": 0.0,
            "min_mol_size": args.min_mol_size,
            "enable_retry_profiles": False,
            "top_k": 1,
            "llm_fallback": False,
        },
        "jobs": [
            {"id": f"target_{index:03d}", "seed": args.seed + index - 1, **target}
            for index, target in enumerate(targets, start=1)
        ],
    }


def run_worker(worker_spec: Path) -> None:
    from cs_copilot.tools.chemistry.synplanner_toolkit import SynPlannerToolkit

    context = json.loads(worker_spec.read_text())
    spec, job = context["specification"], context["job"]
    output = Path(context["output_dir"])
    config = spec["configuration"]
    # Check cached files still exist and have the declared size. Hashes were
    # frozen before launch; re-hashing a large building-block library per target
    # would distort the study overhead. Assets must remain immutable during it.
    for asset in spec["assets"].values():
        path = Path(asset["path"])
        if not path.is_file() or path.stat().st_size != asset["size_bytes"]:
            raise ValueError("SynPlanner asset changed after study declaration")
    apply_seed(job["seed"])
    backend = SynPlannerToolkit(
        data_folder=config["data_dir"],
        prefer_gpu=False,
        max_time=config["max_time"],
        max_iterations=config["max_iterations"],
        max_depth=config["max_depth"],
        max_tree_size=config["max_tree_size"],
        top_rules=config["top_rules"],
        rule_prob_threshold=config["rule_prob_threshold"],
        min_mol_size=config["min_mol_size"],
        enable_retry_profiles=False,
        default_top_k=1,
    )
    seed_control = apply_seed(job["seed"])
    state = {}
    plan = backend.plan_synthesis(job["smiles"], top_k=1, session_state=state)
    plan["study_provenance"] = {
        "target_id": job["target_id"],
        "seed_control": seed_control,
        "assets": spec["assets"],
        "configuration": config,
    }
    write_json(output / "plan.json", plan)
    row = summarize_plan(output / "plan.json")
    write_json(
        output / "result.json",
        {
            "scientific_outcome": row["outcome"],
            "target_id": job["target_id"],
            "route_count": row["route_count"],
            "first_route_num_steps": row["first_route_num_steps"],
            "first_route_score": row["first_route_score"],
            "route_length_check": row["route_length_check"],
            "search_seconds": row["search_seconds"],
            "seed_control": seed_control,
            "artifacts": {"plan.json": file_identity(output / "plan.json")},
        },
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-target-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-time", type=int, default=120)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument("--top-rules", type=int, default=50)
    parser.add_argument("--min-mol-size", type=int, default=6)
    parser.add_argument("--case-timeout-seconds", type=float, default=300)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if args.worker_spec:
        try:
            run_worker(args.worker_spec)
        except Exception as exc:
            output = Path(json.loads(args.worker_spec.read_text())["output_dir"])
            write_json(
                output / "result.json",
                {"scientific_outcome": "error", "error": f"{type(exc).__name__}: {exc}"},
            )
            traceback.print_exc()
            raise SystemExit(1) from exc
        return
    if not args.data_dir or not args.targets or not args.output_dir:
        parser.error("--data-dir, --targets and --output-dir are required")
    try:
        spec = build_specification(args)
        summary = run_jobs(
            spec,
            args.output_dir,
            worker_script=Path(__file__),
            resume=args.resume,
            prepare_only=args.prepare_only,
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
