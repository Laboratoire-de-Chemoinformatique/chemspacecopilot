#!/usr/bin/env python
# coding: utf-8
"""Test whether a GTM map of chembed latent space usefully steers generation.

The manuscript states that chemical space maps highlight promising regions
which generative models then target. For small molecules that coupling was
never implemented: the map was built on Morgan fingerprints while generation
sampled the autoencoder latent, two unrelated spaces. chembed makes the
coupling possible because it encodes the large, potent chemotypes the LSTM
autoencoder could not.

The study fits a GTM on the chembed latents of a curated activity set, scores
each node by responsibility-weighted activity, and decodes the most active
nodes back into molecules.

It reports a baseline alongside, and the baseline is the point. Perturbing the
latents of the most active *compounds* costs nothing and needs no map, so the
map has to beat it on something to be worth running. If both arms look alike,
that is the result.

Read the output with two cautions:

* SELFIES decode to a valid molecule by construction, so validity is 1.0 for
  both arms by definition and says nothing. Compare distinct yield, novelty,
  Tanimoto to the actives, and SA.
* A 2D manifold cannot carry a 256D latent faithfully, so decoded node centres
  are regional summaries rather than neighbours of any particular compound, and
  the GTM arm is expected to range wider. Wider is only better if the molecules
  are good. How much the manifold actually loses is measured per run, not
  assumed: ``roundtrip`` in the summary reports the fraction of decoded node
  centres that re-encode back into their own node. Read the two arms in its
  light — a low fraction means the GTM arm was seeded from points the manifold
  cannot place, which is a finding about the map, not about generation.

Usage::

    python scripts/run_chembed_latent_gtm_study.py \\
        --compounds reports/reviewer_revision/seh/clean_compounds.csv \\
        --output-dir reports/chembed_latent_gtm
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("chembed_latent_gtm")

# Fixed rather than Optuna-searched so the run is deterministic and the two arms
# are compared on one manifold. 15**2 basis functions, (15+15)**2 nodes, matching
# the shape of the published map.
GTM_CONFIG = {
    "n_basis_functions_sqrt": 15,
    "n_nodes_n_basis_diff": 15,
    "basis_width": 1.0,
    "reg_coeff": 100.0,
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compounds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smiles-column", default="canonical_smiles")
    parser.add_argument(
        "--activity-column",
        default=None,
        help="Potency column; detected from the file when omitted (pIC50, pchembl_value, ...)",
    )
    parser.add_argument("--top-nodes", type=int, default=10)
    parser.add_argument("--n-candidates", type=int, default=200)
    parser.add_argument("--noise-scale", type=float, default=0.5)
    parser.add_argument(
        "--noise-sweep",
        type=float,
        nargs="+",
        default=None,
        help="Repeat both arms at each scale. A single scale cannot show whether a "
        "result is an artifact of that setting.",
    )
    parser.add_argument("--min-node-mass", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def score_nodes(resps: np.ndarray, activity: np.ndarray) -> pd.DataFrame:
    """Rank nodes by responsibility-weighted activity.

    Args:
        resps: ``(n_compounds, n_nodes)`` responsibilities.
        activity: ``(n_compounds,)`` activity values.

    Returns:
        One row per node with its 1-based id, responsibility mass and weighted
        mean activity.
    """
    mass = resps.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        weighted = (resps * activity[:, None]).sum(axis=0) / mass
    return pd.DataFrame(
        {
            "node_id": np.arange(1, resps.shape[1] + 1),
            "mass": mass,
            "weighted_activity": weighted,
        }
    )


def describe(smiles: Sequence[str], reference_fps, reference_smiles: set) -> Dict[str, Any]:
    """Summarize a set of generated molecules against the training actives."""
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    from chembed.mol_utils import mol_property_from_mol

    generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    canonical, similarities, sa_scores = [], [], []
    for item in smiles:
        mol = Chem.MolFromSmiles(item)
        if mol is None:
            continue
        canonical.append(Chem.MolToSmiles(mol))
        fingerprint = generator.GetFingerprint(mol)
        similarities.append(max(DataStructs.BulkTanimotoSimilarity(fingerprint, reference_fps)))
        try:
            sa_scores.append(mol_property_from_mol(mol, "SA_Score"))
        except Exception:  # noqa: BLE001 - a missing score must not end the run
            pass

    distinct = sorted(set(canonical))
    novel = [item for item in distinct if item not in reference_smiles]
    return {
        "requested": len(smiles),
        "parsed": len(canonical),
        "distinct": len(distinct),
        "novel_vs_training_set": len(novel),
        "max_tanimoto_to_actives_median": float(np.median(similarities)) if similarities else None,
        "max_tanimoto_to_actives_max": float(np.max(similarities)) if similarities else None,
        "sa_score_median": float(np.median(sa_scores)) if sa_scores else None,
        "sa_score_max": float(np.max(sa_scores)) if sa_scores else None,
        "examples": distinct[:10],
    }


def main(argv: Optional[List[str]] = None) -> int:
    import torch

    from cs_copilot.tools.chemistry.activity_schema import require_activity_column
    from cs_copilot.tools.chemistry.chembed_engine import ChembedEngine
    from cs_copilot.tools.chemography.gtm_operations import (
        latent_centers_from_gtm,
        node_roundtrip_fidelity,
        project_latent_on_gtm,
        train_latent_gtm,
    )

    args = parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.compounds)
    # The curated sEH file is keyed on pIC50 or pchembl_value depending on its
    # activity rule, so resolve the name rather than assume one.
    args.activity_column = args.activity_column or require_activity_column(frame)
    frame = frame.dropna(subset=[args.smiles_column, args.activity_column])
    smiles = frame[args.smiles_column].astype(str).tolist()
    activity = frame[args.activity_column].to_numpy(dtype=float)
    logger.info("Loaded %d compounds from %s", len(smiles), args.compounds)

    engine = ChembedEngine()
    engine.load(model_path=None, device=torch.device(device))
    latents = engine.encode(smiles, batch_size=256)
    logger.info(
        "Encoded to %s with %d token substitution(s)",
        latents.shape,
        engine.unknown_token_substitutions,
    )

    gtm, scaler, entropy = train_latent_gtm(
        latents.astype(np.float64), config=GTM_CONFIG, descriptor_backend="chembed"
    )
    resps = project_latent_on_gtm(latents.astype(np.float64), gtm, scaler)
    logger.info("Fitted latent GTM, Shannon entropy %.4f", entropy)

    nodes = score_nodes(resps, activity)
    nodes.to_csv(args.output_dir / "node_activity.csv", index=False)
    eligible = nodes[nodes["mass"] >= args.min_node_mass].dropna(subset=["weighted_activity"])
    selected = eligible.nlargest(args.top_nodes, "weighted_activity")
    logger.info(
        "Selected %d of %d nodes (mass >= %s), weighted activity %.2f-%.2f",
        len(selected),
        len(nodes),
        args.min_node_mass,
        selected["weighted_activity"].min(),
        selected["weighted_activity"].max(),
    )

    # Reference set for novelty and similarity: the actives themselves.
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    reference_mols = [Chem.MolFromSmiles(item) for item in smiles]
    reference_fps = [generator.GetFingerprint(m) for m in reference_mols if m is not None]
    reference_smiles = {Chem.MolToSmiles(m) for m in reference_mols if m is not None}

    def decode_with_noise(centers: np.ndarray, count: int, scale: float) -> List[str]:
        """Spread *count* samples over *centers* and decode them."""
        picks = rng.integers(0, len(centers), size=count)
        noise = rng.standard_normal((count, centers.shape[1])) * scale
        return engine.decode(
            (centers[picks] + noise).astype(np.float32), max_len=100, temp=1.0, decode="greedy"
        )

    node_centers = latent_centers_from_gtm(
        gtm, node_ids=selected["node_id"].tolist(), scaler=scaler
    )

    # What the 2D bottleneck costs this manifold, measured rather than quoted.
    # Noise-free, because this asks about the centres themselves, not about the
    # sampling around them.
    roundtrip = node_roundtrip_fidelity(
        gtm, engine, node_ids=selected["node_id"].tolist(), scaler=scaler
    )
    logger.info(
        "Round trip: %s of %s decoded node centres re-encode into their own node, "
        "median responsibility to it %s",
        roundtrip["same_node"],
        roundtrip["decoded"],
        roundtrip["origin_responsibility_median"],
    )

    top_compounds = np.argsort(activity)[::-1][: len(selected)]
    active_latents = latents[top_compounds].astype(np.float64)

    scales = args.noise_sweep or [args.noise_scale]
    arms: Dict[str, Any] = {}
    for scale in scales:
        # Arm 1 decodes the latent centres of the most active nodes; arm 2 perturbs
        # the latents of the most active compounds. Both get the same scale, which
        # is what makes the comparison fair.
        gtm_smiles = decode_with_noise(node_centers, args.n_candidates, scale)
        baseline_smiles = decode_with_noise(active_latents, args.n_candidates, scale)
        key = f"noise_{scale:g}"
        arms[key] = {
            "gtm_guided": describe(gtm_smiles, reference_fps, reference_smiles),
            "active_ball_baseline": describe(baseline_smiles, reference_fps, reference_smiles),
        }
        pd.DataFrame({"smiles": gtm_smiles}).to_csv(
            args.output_dir / f"gtm_guided_{key}.csv", index=False
        )
        pd.DataFrame({"smiles": baseline_smiles}).to_csv(
            args.output_dir / f"baseline_{key}.csv", index=False
        )

    summary = {
        "compounds": {
            "path": str(args.compounds),
            "count": len(smiles),
            "activity_column": args.activity_column,
        },
        "encoder": {
            "backend": engine.name,
            "checkpoint": engine.checkpoint_path,
            "latent_dimensions": int(latents.shape[1]),
            "unknown_token_substitutions": engine.unknown_token_substitutions,
        },
        "gtm": {
            "config": GTM_CONFIG,
            "shannon_entropy": float(entropy),
            "nodes_total": int(len(nodes)),
            "nodes_selected": int(len(selected)),
            "selected_node_ids": selected["node_id"].tolist(),
            "selected_weighted_activity": selected["weighted_activity"].round(3).tolist(),
        },
        "sampling": {
            "noise_scales": scales,
            "n_candidates": args.n_candidates,
            "seed": args.seed,
        },
        "roundtrip": roundtrip,
        "arms_by_noise_scale": arms,
        "caveats": [
            "SELFIES decode to a valid molecule by construction; validity is not "
            "comparable with a SMILES decoder and is not reported as evidence.",
            "A 2D manifold cannot carry a 256D latent faithfully, so decoded node "
            "centres are regional summaries rather than neighbours of any compound. "
            "'roundtrip' measures what that costs on this manifold: a low "
            "same_node fraction means the GTM arm was seeded from points the map "
            "cannot place, which is a finding about the map, not about generation.",
            "Generated activity is unproven; node enrichment guides sampling only.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for key, arm in arms.items():
        for name, stats in arm.items():
            logger.info(
                "%s %-22s distinct=%3d novel=%3d tanimoto_median=%.3f SA_median=%.2f",
                key,
                name,
                stats["distinct"],
                stats["novel_vs_training_set"],
                stats["max_tanimoto_to_actives_median"] or float("nan"),
                stats["sa_score_median"] or float("nan"),
            )
    print(json.dumps(arms, indent=2))
    logger.info("Wrote %s", args.output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
