#!/usr/bin/env python3
"""Run a bounded latent-neighbourhood study with the chembed VAE.

    python scripts/run_chembed_study.py \
        --output-dir /absolute/study --parent-id ID --parent-smiles SMILES \
        --noise-scale 0.5 --n-samples 100

chembed is a SELFIES transformer VAE whose pretrained checkpoint downloads from
HuggingFace on first use. It goes through ``ChembedEngine``, the same path the
toolkit uses, so the study measures the shipped code rather than a parallel
implementation.

Sampling mirrors ``run_generation_study.py``: encode the parent, add Gaussian
noise of the given scale to its latent vector, decode, and keep every raw
output. Metrics reuse ``summarize_generation`` so validity, uniqueness,
standardization and parent similarity keep their definitions.

Two cautions when reading the results:

* SELFIES decode to a valid molecule by construction, so this engine's validity
  is 1.0 by definition and **must not** be compared with a SMILES decoder's
  validity. Compare distinct-structure yield, Tanimoto to the parent and SA.
* Noise scales do not transfer between latent spaces. Against chembed, 0.1
  returns the parent almost every time; 0.5 is the calibrated default and 0.25
  suits large seeds. See ``docs/architecture/tools.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from pathlib import Path

from _study_common import apply_seed, file_identity, run_jobs, software_identity, write_json
from summarize_generation import summarize


def _cuda_available() -> bool:
    """Report CUDA availability without importing torch at module scope."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - absence is the only thing that matters
        return False


def build_specification(args: argparse.Namespace) -> dict:
    from rdkit import Chem

    if args.n_samples < 1 or args.repeats < 1 or args.noise_scale < 0:
        raise ValueError("Counts must be positive and noise scale nonnegative")
    mol = Chem.MolFromSmiles(args.parent_smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Parent must be a valid nonempty SMILES")
    from importlib.metadata import version

    repository = Path(__file__).resolve().parents[1]
    return {
        "study": "chembed_molecular_generation",
        "implementation_files": {
            name: file_identity(repository / name)
            for name in [
                "src/cs_copilot/tools/chemistry/standardize.py",
                "src/cs_copilot/tools/chemistry/chembed_engine.py",
                "src/cs_copilot/generation_audit.py",
                "scripts/summarize_generation.py",
            ]
        },
        "software": software_identity(),
        "launcher": file_identity(Path(__file__)),
        "ledger_code": file_identity(Path(__file__).with_name("_study_common.py")),
        "external_package": {
            "name": "chembed",
            # chembed's __init__ is empty, so there is no __version__ attribute.
            "version": version("chembed"),
            "source": "https://github.com/3BioCompBio/chembed",
            "checkpoint": "HuggingFace 3BioCompBio/chembed-default",
        },
        "configuration": {
            "engine": "chembed",
            "representation": "selfies",
            "parent_id": args.parent_id,
            "parent_smiles": args.parent_smiles,
            "n_samples": args.n_samples,
            "noise_scale": args.noise_scale,
            "device": args.device,
        },
        "case_timeout_seconds": args.case_timeout_seconds,
        "jobs": [{"id": f"run_{i:02d}", "seed": 11 * (i + 1)} for i in range(args.repeats)],
    }


def run_worker(worker_spec: Path) -> None:
    payload = json.loads(worker_spec.read_text())
    specification, job = payload["specification"], payload["job"]
    config = specification["configuration"]
    output_dir = Path(payload["output_dir"])
    apply_seed(job["seed"])

    import numpy as np
    import torch

    from cs_copilot.generation_audit import capture_generation_audit
    from cs_copilot.tools.chemistry.chembed_engine import ChembedEngine

    torch.manual_seed(job["seed"])
    engine = ChembedEngine()
    engine.load(model_path=None, device=torch.device(config["device"]))

    latent = engine.encode([config["parent_smiles"]], batch_size=256)
    rng = np.random.default_rng(job["seed"])
    noise = rng.standard_normal((config["n_samples"], latent.shape[1])) * config["noise_scale"]
    perturbed = (latent + noise).astype(np.float32)
    raw_outputs = engine.decode(perturbed, max_len=100, temp=1.0, decode="greedy")

    audit = capture_generation_audit(
        raw_outputs,
        source="chembed_latent_neighborhood_before_validation",
        requested_count=config["n_samples"],
        raw_outputs_available=True,
    )
    write_json(output_dir / "raw_audit.json", audit)
    report = summarize(audit, parent_smiles=config["parent_smiles"])
    write_json(output_dir / "metrics.json", report)
    with (output_dir / "candidates.csv").open("w", newline="") as handle:
        keys = list(dict.fromkeys(key for row in report["records"] for key in row))
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(report["records"])
    write_json(
        output_dir / "result.json",
        {
            "scientific_outcome": "outputs_measured",
            "observed_output_count": report["observed_record_count"],
            "rdkit_valid_count": report["rdkit_valid_count"],
            "unique_standardized_count": report["unique_standardized_count"],
            "raw_output_validity": report["raw_output_validity"],
            "raw_output_uniqueness": report["raw_output_uniqueness"],
            # Reported so a reader can see whether any molecule was altered to
            # fit chembed's vocabulary.
            "unknown_token_substitutions": engine.unknown_token_substitutions,
            "artifacts": {
                name: file_identity(output_dir / name)
                for name in ("raw_audit.json", "metrics.json", "candidates.csv")
            },
        },
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--parent-id")
    parser.add_argument("--parent-smiles")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--noise-scale", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--case-timeout-seconds", type=float, default=3600)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    if args.worker_spec:
        try:
            run_worker(args.worker_spec)
        except Exception as exc:  # noqa: BLE001 - the ledger records the failure
            write_json(
                Path(json.loads(args.worker_spec.read_text())["output_dir"]) / "result.json",
                {
                    "scientific_outcome": "worker_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
            raise
        return

    if not args.output_dir or not args.parent_id or not args.parent_smiles:
        parser.error("--output-dir, --parent-id and --parent-smiles are required")
    summary = run_jobs(
        build_specification(args),
        args.output_dir,
        worker_script=Path(__file__),
        resume=args.resume,
    )
    print(json.dumps(summary["status_counts"], indent=2))


if __name__ == "__main__":
    main()
