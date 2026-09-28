"""Pinned public preset for the SynPlanner 1.7 runtime (not historical evidence)."""

from pathlib import Path

SYNPLANNER_REPO = "Laboratoire-De-Chemoinformatique/SynPlanner-data"
SYNPLANNER_REVISION = "f4ab02bdb1edd9c437f8e2a9de1299ebf16f5a67"
SYNPLANNER_FILES = {
    "reaction_rules": "policy/supervised_gps/v1/reaction_rules.tsv",
    "ranking_policy": "policy/supervised_gps/v1/v1/ranking_policy.ckpt",
    "building_blocks": "building_blocks/emolecules-salt-ln/building_blocks.tsv",
}


def resolve_synplanner_assets(data_folder: str | None = None) -> dict[str, Path]:
    """Resolve the matching GPS preset; explicit local inputs never trigger downloads."""
    root = Path(data_folder) if data_folder is not None else Path("synplan_data")
    paths = {key: root / relative for key, relative in SYNPLANNER_FILES.items()}
    missing = [key for key, path in paths.items() if not path.is_file()]
    if missing and data_folder is not None:
        raise FileNotFoundError(
            f"SynPlanner 1.7 GPS preset is incomplete in {root}: {missing}. "
            "Use the matching TSV rules and policy preset; historical CGRtools "
            "assets require their original environment. Download with "
            "'synplan download_preset --preset synplanner-gps --save_to <directory>'."
        )
    if missing:
        from huggingface_hub import hf_hub_download

        for key in missing:
            paths[key] = Path(
                hf_hub_download(
                    repo_id=SYNPLANNER_REPO,
                    revision=SYNPLANNER_REVISION,
                    filename=SYNPLANNER_FILES[key],
                    local_dir=str(root),
                )
            )
    return paths
