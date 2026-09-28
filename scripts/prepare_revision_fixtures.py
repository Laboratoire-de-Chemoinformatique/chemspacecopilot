#!/usr/bin/env python
"""Build prospective sEH frozen fixtures from curated data and trusted pinned assets.

This performs deterministic projection on a pretrained map, never model fitting or
provider calls. The curator's activity classes are preserved without reclassification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def curated_view(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose curated pIC50/classes without triggering generic raw-IC50 thresholds."""
    required = {"canonical_smiles", "pIC50", "activity_class", "activity_binary"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing curated fields: {sorted(missing)}")
    view = frame.rename(
        columns={"canonical_smiles": "smi", "standard_value": "median_IC50_nM"}
    ).copy()
    view = view.drop(columns=["standard_type", "standard_units"], errors="ignore")
    # Generic raw activity classification must not override the curator's gap.
    view["activity_comment"] = view["activity_class"].where(
        view["activity_class"].isin(["active", "inactive"])
    )
    return view


def build(args) -> dict:
    from cs_copilot.tools.chemography import gtm_operations as ops
    from cs_copilot.tools.io.session_memory import register_session_object

    torch.set_num_threads(1)
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    source = args.curated_csv.resolve()
    model_path = args.gtm_model.resolve()
    candidates_path = args.candidates_csv.resolve()
    inputs = {str(path): checksum(path) for path in (source, model_path, candidates_path)}
    frame = curated_view(pd.read_csv(source))
    view_path = destination / "seh_tool_view.csv"
    frame.to_csv(view_path, index=False)
    state = {
        "_current_gtm_model_path": str(model_path),
        "default_descriptor": "autoencoder",
        "map_type": "default_map",
        "data_file_paths": {
            "dataset_path": str(view_path),
            "clean_dataset_path": str(view_path),
            "curation_provenance_path": str(args.provenance.resolve()),
        },
        "revision_context": {
            "study_type": "prospective reviewer revision; not historical manuscript replay",
            "dataset": "ChEMBL37 human EPHX2/CHEMBL2409; curated epoxide-hydrolase IC50",
            "active_rule": "median IC50 < 1000 nM",
            "inactive_rule": "median IC50 >= 10000 nM",
            "intermediate_rule": "1000 <= median IC50 < 10000 nM; excluded from binary landscape",
            "activity_source": "pIC50 continuous landscape; activity_comment preserves curated binary labels",
            "map_selection": "pinned pretrained map; no new fit or hyperparameter selection",
        },
    }
    register_session_object(
        state,
        "dataset",
        {"dataset_path": str(view_path), "n_compounds": len(frame)},
        label="Curated prospective sEH compounds",
        source_tool="prepare_revision_fixtures",
    )
    input_fixture = destination / "seh_input.json"
    write_json(input_fixture, {"session_state": state})

    model, projected_frame, descriptors, responsibilities = ops.data_load_and_prep(
        str(view_path), str(model_path), descriptor_type="autoencoder"
    )
    if not np.isfinite(responsibilities).all() or responsibilities.shape != (
        len(frame),
        model.num_nodes,
    ):
        raise ValueError("Projection changed row count or produced invalid responsibilities")
    coordinates = ops.calculate_latent_coords(responsibilities, correction=True, return_node=True)
    projection = projected_frame.drop(
        columns=[
            column for column in projected_frame if "latent" in column or "fingerprint" in column
        ],
        errors="ignore",
    ).copy()
    for column in coordinates:
        projection[column] = coordinates[column].to_numpy()
    projection_path = destination / "seh_projection.csv"
    projection.to_csv(projection_path, index=False)
    np.savez_compressed(
        destination / "seh_projection_arrays.npz",
        descriptors=descriptors,
        responsibilities=responsibilities,
    )
    density = ops.get_density_matrix(responsibilities)
    density_table = ops.density_to_table(density, node_threshold=0.1)
    density_path = destination / "seh_density.csv"
    density_table.to_csv(density_path, index=False)
    reg_density, reg_activity = ops.get_reg_density_matrix(
        responsibilities, projected_frame["pIC50"]
    )
    activity_table = ops.reg_density_to_table(reg_density, reg_activity, node_threshold=0.1)
    activity_path = destination / "seh_activity_regression.csv"
    activity_table.to_csv(activity_path, index=False)
    probabilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)
    occupancy = probabilities.sum(axis=0)
    occupancy_probabilities = occupancy / occupancy.sum()
    nonzero = occupancy_probabilities[occupancy_probabilities > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))
    stats = {
        "sample_count": len(frame),
        "node_count": int(model.num_nodes),
        "hard_occupied_nodes": int(np.unique(probabilities.argmax(axis=1)).size),
        "soft_occupancy_sum": float(occupancy.sum()),
        "normalized_occupancy_entropy": entropy / math.log(model.num_nodes),
        "mean_normalized_sample_responsibility_entropy": float(
            -np.sum(
                np.where(
                    probabilities > 0,
                    probabilities * np.log(np.maximum(probabilities, np.finfo(float).tiny)),
                    0,
                ),
                axis=1,
            ).mean()
            / math.log(model.num_nodes)
        ),
        "criterion_scope": "descriptive projection occupancy; not a model-selection score",
        "model_attributes": {
            key: getattr(model, key, None)
            for key in (
                "num_nodes",
                "num_basis_functions",
                "basis_width",
                "reg_coeff",
                "max_iter",
                "standardize",
                "pca_scale",
            )
        },
        "input_dimensions": int(descriptors.shape[1]),
        "model_sha256": inputs[str(model_path)],
    }
    write_json(destination / "projection_summary.json", stats)
    state["analysis_results"] = {
        "projection_csv": str(projection_path),
        "density_csv": str(density_path),
        "activity_csv": str(activity_path),
    }
    state["landscape_files"] = {"landscape_data_csv": str(activity_path)}
    register_session_object(
        state,
        "map",
        {
            "map_type": "gtm",
            "model_path": str(model_path),
            "dataset_path": str(view_path),
            "descriptor_type": "autoencoder",
            "projection_csv": str(projection_path),
            "density_csv": str(density_path),
            "activity_csv": str(activity_path),
        },
        label="Precomputed prospective sEH landscape",
        source_tool="prepare_revision_fixtures",
    )
    analysis_fixture = destination / "seh_analysis.json"
    write_json(analysis_fixture, {"session_state": state})

    generated = pd.read_csv(candidates_path)
    # The frozen order is the published deterministic unique-candidate CSV order.
    candidates = [{"smiles": value} for value in generated["canonical_smiles"].tolist()]
    if not candidates:
        raise ValueError("No generated candidates")
    candidate_artifact = destination / "candidates.json"
    parent = "CCC(C)C(=O)N1CCC(NC(=O)Nc2ccc(C(F)(C(F)(F)F)C(F)(F)F)cc2)CC1"
    metadata = {
        "seed_compound_id": "CHEMBL3327073",
        "seed_smiles": parent,
        "engine": "autoencoder",
        "source_csv": str(candidates_path),
        "source_sha256": inputs[str(candidates_path)],
        "ordering": "source CSV order; first valid candidate is the fixed planning target",
    }
    write_json(candidate_artifact, {"metadata": metadata, "candidates": candidates})
    register_session_object(
        state,
        "candidate_set",
        {"artifact_path": str(candidate_artifact), "count_returned": len(candidates), **metadata},
        label="Measured seeded autoencoder candidates",
        source_tool="prepare_revision_fixtures",
    )
    candidate_fixture = destination / "seh_candidates.json"
    write_json(candidate_fixture, {"session_state": state})
    manifest = {
        "purpose": "prospective frozen inputs, not autonomous benchmark outputs",
        "inputs": inputs,
        "fixtures": {
            name: {"path": str(path), "sha256": checksum(path)}
            for name, path in (
                ("seh_input", input_fixture),
                ("seh_analysis", analysis_fixture),
                ("seh_candidates", candidate_fixture),
            )
        },
        "projection_summary": stats,
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated-csv", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument(
        "--gtm-model", type=Path, required=True, help="Trusted pinned executable pickle checkpoint"
    )
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
