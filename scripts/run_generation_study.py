#!/usr/bin/env python3
"""Run the bounded, seeded autoencoder supplement using existing LOCAL assets only.

PYTHONPATH=src python scripts/run_generation_study.py --model-dir /absolute/model \
    --output-dir /absolute/study

Default: CHEMBL3327073, seeds 11/22/33, 100 neighborhood outputs per seed,
CPU, one Torch thread. No LLM calls or downloads. Training novelty is unavailable
unless --training-corpus supplies the actual generator training corpus.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import traceback
from pathlib import Path

from _study_common import apply_seed, file_identity, run_jobs, software_identity, write_json
from summarize_generation import _training_records, summarize

PARENT_ID = "CHEMBL3327073"
PARENT_SMILES = "CCC(C)C(=O)N1CCC(NC(=O)Nc2ccc(C(F)(C(F)(F)F)C(F)(F)F)cc2)CC1"


def build_specification(args: argparse.Namespace) -> dict:
    from rdkit import Chem

    if (
        args.n_samples < 1
        or args.temperature <= 0
        or args.noise_scale < 0
        or args.case_timeout_seconds <= 0
    ):
        raise ValueError(
            "Counts/timeouts/temperature must be positive; noise scale cannot be negative"
        )
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        raise ValueError("Provide distinct seeds in [0, 2**32)")
    mol = Chem.MolFromSmiles(args.parent_smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Parent must be a valid nonempty SMILES")
    for module in ("torch", "numpy", "deepchemography"):
        if importlib.util.find_spec(module) is None:
            raise ValueError(f"Required installed dependency is unavailable: {module}")
    assets = {
        name: file_identity(args.model_dir / name) for name in ("config.pt", "vocab.pt", "model.pt")
    }
    training = file_identity(args.training_corpus) if args.training_corpus else None
    if training:
        # Fail before generation if the intended novelty denominator cannot be parsed.
        summarize(
            [], training_smiles=_training_records(args.training_corpus, args.training_smiles_column)
        )
    return {
        "implementation_files": {
            name: file_identity(Path(__file__).resolve().parents[1] / name)
            for name in [
                "src/cs_copilot/tools/chemistry/autoencoder_toolkit.py",
                "src/cs_copilot/tools/chemistry/standardize.py",
                "src/cs_copilot/generation_audit.py",
                "scripts/summarize_generation.py",
            ]
        },
        "study": "molecular_generation",
        "software": software_identity(),
        "assets": assets,
        "launcher": file_identity(Path(__file__)),
        "ledger_code": file_identity(Path(__file__).with_name("_study_common.py")),
        "training_corpus": training,
        "training_smiles_column": args.training_smiles_column,
        "configuration": {
            "model_dir": str(args.model_dir.resolve()),
            "parent_id": args.parent_id,
            "parent_smiles": args.parent_smiles,
            "n_samples": args.n_samples,
            "noise_scale": args.noise_scale,
            "temperature": args.temperature,
            "decode_mode": args.decode_mode,
            "device": args.device,
        },
        "case_timeout_seconds": args.case_timeout_seconds,
        "jobs": [{"id": f"seed_{seed}", "seed": seed} for seed in args.seeds],
    }


def run_worker(worker_spec: Path) -> None:
    from cs_copilot.generation_audit import save_generation_audit
    from cs_copilot.tools.chemistry.autoencoder_toolkit import AutoencoderToolkit

    context = json.loads(worker_spec.read_text())
    spec, job = context["specification"], context["job"]
    config = spec["configuration"]
    output = Path(context["output_dir"])
    # Verify the bytes about to be loaded against the frozen study manifest.
    for asset in spec["assets"].values():
        if file_identity(Path(asset["path"])) != asset:
            raise ValueError("Model checkpoint changed after study declaration")
    apply_seed(job["seed"])
    backend = AutoencoderToolkit(model_path=config["model_dir"], device=config["device"])
    seed_control = apply_seed(job["seed"])
    backend.explore_latent_neighborhood(
        base_smiles=config["parent_smiles"],
        n_neighbors=config["n_samples"],
        noise_scale=config["noise_scale"],
        temperature=config["temperature"],
        decode_mode=config["decode_mode"],
    )
    audit = backend.last_generation_audit
    if not audit or not audit.get("raw_outputs_available"):
        raise RuntimeError("Backend did not preserve the unfiltered outputs")
    audit.update(
        {
            "rng_seed": job["seed"],
            "seed_control": seed_control,
            "parent_id": config["parent_id"],
            "seed_smiles": config["parent_smiles"],
            "generation_mode": "analog",
            "neighborhood_noise_scale": config["noise_scale"],
            "model_assets": spec["assets"],
            "rng_state_recorded": False,
        }
    )
    save_generation_audit(audit, audit_path=str(output / "raw_audit.json"))
    training = spec.get("training_corpus")
    if training and file_identity(Path(training["path"])) != training:
        raise ValueError("Training corpus changed after study declaration")
    training_smiles = (
        _training_records(Path(training["path"]), spec["training_smiles_column"])
        if training
        else None
    )
    report = summarize(audit, training_smiles=training_smiles)
    report["training_corpus"] = training
    report["seed_control"] = seed_control
    write_json(output / "metrics.json", report)
    rows = report["records"]
    with (output / "candidates.csv").open("w", newline="") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields or ["record_index", "raw_smiles"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output / "result.json",
        {
            "scientific_outcome": "outputs_measured",
            "observed_output_count": report["observed_record_count"],
            "rdkit_valid_count": report["rdkit_valid_count"],
            "unique_standardized_count": report["unique_standardized_count"],
            "raw_output_validity": report["raw_output_validity"],
            "raw_output_uniqueness": report["raw_output_uniqueness"],
            "novel_to_training_fraction": report["novel_to_training_fraction"],
            "artifacts": {
                name: file_identity(output / name)
                for name in ("raw_audit.json", "metrics.json", "candidates.csv")
            },
        },
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--parent-id", default=PARENT_ID)
    parser.add_argument("--parent-smiles", default=PARENT_SMILES)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--noise-scale", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--decode-mode", choices=("sample", "greedy"), default="sample")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--case-timeout-seconds", type=float, default=300)
    parser.add_argument("--training-corpus", type=Path)
    parser.add_argument("--training-smiles-column", default="smiles")
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
    if not args.model_dir or not args.output_dir:
        parser.error("--model-dir and --output-dir are required")
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
