#!/usr/bin/env python
# coding: utf-8
"""HuggingFace-backed aggregate peptide landscape loading and sampling helpers."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from cs_copilot.tools.constants import (
    DEFAULT_PEPTIDE_DESIGNER_DATA_PATH,
    DEFAULT_PEPTIDE_LANDSCAPE_ID,
    HUGGINGFACE_PEPTIDE_DESIGNER_DATA_REPO,
)

logger = logging.getLogger(__name__)

LANDSCAPE_RELATIVE_ROOT = "landscapes"
REQUIRED_LANDSCAPE_FILES = (
    "landscape.json",
    "landscape.safetensors",
    "nodes.parquet",
    "sampler.json",
    "runtime/gtm.pkl.gz",
)
REQUIRED_TENSORS = (
    "gtm.phi",
    "gtm.weights",
    "scaler.mean",
    "scaler.scale",
)


class PeptideLandscapeError(Exception):
    """Raised when an aggregate peptide landscape cannot be loaded or sampled."""


@dataclass(frozen=True)
class PeptideLandscapeBundle:
    """Loaded aggregate peptide landscape bundle."""

    landscape_id: str
    root_path: Path
    manifest: dict[str, Any]
    sampler: dict[str, Any]
    nodes: pd.DataFrame
    tensors: dict[str, np.ndarray]

    @property
    def organisms(self) -> list[str]:
        if "organism" in self.nodes.columns:
            return sorted(str(value) for value in self.nodes["organism"].dropna().unique())
        endpoint = self.manifest.get("activity_endpoint") or {}
        return sorted(str(value) for value in endpoint.get("organisms") or [])

    @property
    def plotted_organisms(self) -> list[str]:
        endpoint = self.manifest.get("activity_endpoint") or {}
        return sorted(str(value) for value in endpoint.get("plotted_organisms") or [])

    @property
    def alphabet(self) -> list[str]:
        return [str(value) for value in self.manifest.get("peptide_alphabet") or []]

    @property
    def latent_dim(self) -> Optional[int]:
        value = self.manifest.get("latent_dim")
        return int(value) if value is not None else None


def peptide_landscape_cache_root(local_dir: Optional[str] = None) -> Path:
    """Return the local cache root for peptide landscape datasets."""

    path = (
        local_dir or os.getenv("PEPTIDE_DESIGNER_DATA_PATH") or DEFAULT_PEPTIDE_DESIGNER_DATA_PATH
    )
    return Path(path).expanduser()


def peptide_landscape_root(
    landscape_id: str = DEFAULT_PEPTIDE_LANDSCAPE_ID,
    *,
    local_dir: Optional[str] = None,
) -> Path:
    """Return the root directory for one cached landscape bundle."""

    return peptide_landscape_cache_root(local_dir) / LANDSCAPE_RELATIVE_ROOT / landscape_id


def required_landscape_paths(
    landscape_id: str = DEFAULT_PEPTIDE_LANDSCAPE_ID,
    *,
    local_dir: Optional[str] = None,
) -> dict[str, Path]:
    """Return required file paths for a cached landscape bundle."""

    root = peptide_landscape_root(landscape_id, local_dir=local_dir)
    return {name: root / name for name in REQUIRED_LANDSCAPE_FILES}


def is_peptide_landscape_cached(
    landscape_id: str = DEFAULT_PEPTIDE_LANDSCAPE_ID,
    *,
    local_dir: Optional[str] = None,
) -> bool:
    """Return True when all required files for the landscape are cached."""

    return all(
        path.exists()
        for path in required_landscape_paths(landscape_id, local_dir=local_dir).values()
    )


def list_cached_peptide_landscapes(*, local_dir: Optional[str] = None) -> list[dict[str, Any]]:
    """Return cached peptide landscapes discovered under the local cache root."""

    base = peptide_landscape_cache_root(local_dir) / LANDSCAPE_RELATIVE_ROOT
    out: list[dict[str, Any]] = []
    if base.exists():
        for manifest_path in sorted(base.glob("*/landscape.json")):
            landscape_id = manifest_path.parent.name
            item: dict[str, Any] = {
                "landscape_id": landscape_id,
                "cached": is_peptide_landscape_cached(landscape_id, local_dir=local_dir),
                "path": str(manifest_path.parent),
            }
            try:
                manifest = json.loads(manifest_path.read_text())
                endpoint = manifest.get("activity_endpoint") or {}
                item.update(
                    {
                        "compatible_decoder_repo": manifest.get("compatible_decoder_repo"),
                        "latent_dim": manifest.get("latent_dim"),
                        "organism_count": len(endpoint.get("organisms") or []),
                        "plotted_organism_count": len(endpoint.get("plotted_organisms") or []),
                    }
                )
            except Exception as exc:
                item["warning"] = f"Could not parse manifest: {exc}"
            out.append(item)

    if not any(item["landscape_id"] == DEFAULT_PEPTIDE_LANDSCAPE_ID for item in out):
        out.append(
            {
                "landscape_id": DEFAULT_PEPTIDE_LANDSCAPE_ID,
                "cached": False,
                "repo_id": HUGGINGFACE_PEPTIDE_DESIGNER_DATA_REPO,
                "path": str(peptide_landscape_root(DEFAULT_PEPTIDE_LANDSCAPE_ID)),
            }
        )
    return out


def ensure_peptide_landscape_cached(
    landscape_id: str = DEFAULT_PEPTIDE_LANDSCAPE_ID,
    *,
    include_plots: bool = False,
    local_dir: Optional[str] = None,
    repo_id: str = HUGGINGFACE_PEPTIDE_DESIGNER_DATA_REPO,
) -> Path:
    """Ensure a peptide landscape is cached locally and return its root path."""

    root = peptide_landscape_root(landscape_id, local_dir=local_dir)
    if is_peptide_landscape_cached(landscape_id, local_dir=local_dir):
        return root

    cache_root = peptide_landscape_cache_root(local_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise PeptideLandscapeError(
            "huggingface_hub is required to download peptide landscape bundles."
        ) from exc

    allow_patterns = _download_allow_patterns(landscape_id, include_plots=include_plots)
    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    try:
        logger.info("Downloading peptide landscape %s from %s", landscape_id, repo_id)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(cache_root),
            allow_patterns=allow_patterns,
            token=token,
        )
    except Exception as exc:
        raise PeptideLandscapeError(
            f"Failed to download peptide landscape '{landscape_id}' from {repo_id}: {exc!r}"
        ) from exc

    missing = [
        str(path)
        for path in required_landscape_paths(landscape_id, local_dir=local_dir).values()
        if not path.exists()
    ]
    if missing:
        raise PeptideLandscapeError(
            "Downloaded peptide landscape bundle is incomplete. Missing: " + ", ".join(missing)
        )
    return root


def load_peptide_landscape_bundle(
    landscape_id: str = DEFAULT_PEPTIDE_LANDSCAPE_ID,
    *,
    include_plots: bool = False,
    local_dir: Optional[str] = None,
    repo_id: str = HUGGINGFACE_PEPTIDE_DESIGNER_DATA_REPO,
) -> PeptideLandscapeBundle:
    """Load a cached or downloadable aggregate peptide landscape bundle."""

    root = ensure_peptide_landscape_cached(
        landscape_id,
        include_plots=include_plots,
        local_dir=local_dir,
        repo_id=repo_id,
    )
    manifest = json.loads((root / "landscape.json").read_text())
    sampler = json.loads((root / "sampler.json").read_text())
    nodes = pd.read_parquet(root / "nodes.parquet")

    try:
        from safetensors.numpy import load_file
    except ImportError as exc:
        raise PeptideLandscapeError(
            "safetensors is required to load peptide landscape tensor bundles."
        ) from exc

    tensors = load_file(str(root / "landscape.safetensors"))
    _validate_loaded_bundle(landscape_id, manifest, sampler, nodes, tensors)
    return PeptideLandscapeBundle(
        landscape_id=landscape_id,
        root_path=root,
        manifest=manifest,
        sampler=sampler,
        nodes=nodes,
        tensors=tensors,
    )


def resolve_organism(bundle: PeptideLandscapeBundle, organism: str) -> str:
    """Resolve an organism name or slug to the canonical landscape organism."""

    if not organism or str(organism).lower() == "all":
        raise PeptideLandscapeError("A concrete organism is required for activity sampling.")
    requested = _normal_key(organism)
    for candidate in bundle.organisms:
        if requested in _organism_keys(candidate):
            return candidate
    for candidate in bundle.organisms:
        key = _normal_key(candidate)
        if requested in key or key in requested:
            return candidate
    available = ", ".join(bundle.organisms[:16])
    raise PeptideLandscapeError(
        f"Organism '{organism}' is not available in landscape '{bundle.landscape_id}'. "
        f"Available organisms: {available}."
    )


def select_active_landscape_nodes(
    bundle: PeptideLandscapeBundle,
    organism: str,
    *,
    top_n: int = 5,
    min_activity: Optional[float] = None,
    min_observations: Optional[float] = 1.0,
    max_uncertainty: Optional[float] = None,
    require_active_class: bool = True,
    spatial_diversity: bool = True,
) -> pd.DataFrame:
    """Select conservative active nodes for an organism-specific peptide landscape."""

    if top_n <= 0:
        raise PeptideLandscapeError("top_n must be positive.")

    organism_name = resolve_organism(bundle, organism)
    nodes = bundle.nodes[bundle.nodes["organism"].astype(str) == organism_name].copy()
    if nodes.empty:
        raise PeptideLandscapeError(f"No nodes found for organism '{organism_name}'.")

    threshold = (
        float(min_activity)
        if min_activity is not None
        else float(bundle.sampler.get("activity_threshold", 0.5))
    )
    filters = [f"activity_mean >= {threshold:g}"]
    nodes = nodes[pd.to_numeric(nodes["activity_mean"], errors="coerce") >= threshold]

    if require_active_class and "activity_class" in nodes.columns:
        filters.append("activity_class == active_enriched")
        nodes = nodes[nodes["activity_class"].astype(str) == "active_enriched"]

    if min_observations is not None and "n_observations" in nodes.columns:
        filters.append(f"n_observations >= {float(min_observations):g}")
        nodes = nodes[pd.to_numeric(nodes["n_observations"], errors="coerce") >= min_observations]

    if max_uncertainty is not None and "uncertainty" in nodes.columns:
        filters.append(f"uncertainty <= {float(max_uncertainty):g}")
        nodes = nodes[pd.to_numeric(nodes["uncertainty"], errors="coerce") <= max_uncertainty]

    if nodes.empty:
        raise PeptideLandscapeError(
            f"No active nodes matched {organism_name} with filters: {', '.join(filters)}."
        )

    nodes["selection_score"] = _node_selection_score(nodes, bundle.sampler)
    nodes = nodes.sort_values(
        by=["selection_score", "activity_mean", "n_observations", "uncertainty"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    if spatial_diversity and {"x", "y"}.issubset(nodes.columns):
        selected = _greedy_spatially_diverse_nodes(nodes, top_n=top_n)
    else:
        selected = nodes.head(top_n)

    return selected.reset_index(drop=True)


def select_landscape_nodes_by_coordinates(
    bundle: PeptideLandscapeBundle,
    coordinates: Iterable[Sequence[int | float] | dict[str, Any]],
    *,
    organism: Optional[str] = None,
    allow_missing: bool = False,
) -> pd.DataFrame:
    """Resolve integer GTM coordinates to node records."""

    nodes = bundle.nodes.copy()
    organism_name = None
    if organism:
        organism_name = resolve_organism(bundle, organism)
        nodes = nodes[nodes["organism"].astype(str) == organism_name]
    else:
        nodes = nodes.drop_duplicates(subset=["node_id"]).copy()

    selected: list[pd.Series] = []
    missing: list[tuple[int, int]] = []
    for coord in coordinates:
        x, y = _coerce_coordinate(coord)
        matched = nodes[(nodes["x"].astype(int) == x) & (nodes["y"].astype(int) == y)]
        if matched.empty:
            missing.append((x, y))
            continue
        selected.append(matched.iloc[0])

    if missing and not allow_missing:
        raise PeptideLandscapeError(f"No peptide landscape node found for coordinates: {missing}.")
    if not selected:
        raise PeptideLandscapeError("No peptide landscape nodes matched the requested coordinates.")

    result = pd.DataFrame(selected).reset_index(drop=True)
    if organism_name is None and "organism" in result.columns:
        result = result.drop(columns=["organism"])
    if "selection_score" not in result.columns:
        result["selection_score"] = _node_selection_score(result, bundle.sampler)
    return result


def sample_latents_from_nodes(
    bundle: PeptideLandscapeBundle,
    selected_nodes: pd.DataFrame,
    *,
    n_samples: int,
    local_noise_scale: float = 0.25,
    random_state: Optional[int] = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Sample WAE latent vectors from aggregate GTM node coordinates."""

    if n_samples <= 0:
        raise PeptideLandscapeError("n_samples must be positive.")
    if selected_nodes.empty:
        raise PeptideLandscapeError("selected_nodes cannot be empty.")

    node_ids = selected_nodes["node_id"].astype(int).to_numpy()
    centers_scaled = latent_centers_for_nodes(bundle, node_ids)
    rng = np.random.default_rng(random_state)

    weights = None
    if "selection_score" in selected_nodes.columns:
        raw_weights = selected_nodes["selection_score"].astype(float).to_numpy()
        raw_weights = raw_weights - np.nanmin(raw_weights)
        raw_weights = np.where(np.isfinite(raw_weights), raw_weights, 0.0) + 1e-6
        weights = raw_weights / raw_weights.sum()

    chosen = rng.choice(np.arange(len(selected_nodes)), size=n_samples, replace=True, p=weights)
    noise = rng.normal(0.0, local_noise_scale, size=(n_samples, centers_scaled.shape[1]))
    z_scaled = centers_scaled[chosen] + noise

    mean = np.asarray(bundle.tensors["scaler.mean"], dtype=float)
    scale = np.asarray(bundle.tensors["scaler.scale"], dtype=float)
    latents = (z_scaled * scale) + mean

    assignments = selected_nodes.iloc[chosen].reset_index(drop=True).copy()
    assignments.insert(0, "sample_index", np.arange(n_samples))
    return latents.astype(float), assignments


def latent_centers_for_nodes(
    bundle: PeptideLandscapeBundle,
    node_ids: Sequence[int],
) -> np.ndarray:
    """Return scaled WAE latent centers for 1-based GTM node identifiers."""

    phi = np.asarray(bundle.tensors["gtm.phi"], dtype=float)
    weights = np.asarray(bundle.tensors["gtm.weights"], dtype=float)
    centers_scaled = phi @ weights
    indices = np.asarray(node_ids, dtype=int) - 1
    if np.any(indices < 0) or np.any(indices >= centers_scaled.shape[0]):
        raise PeptideLandscapeError(
            f"Node identifiers are outside the 1..{centers_scaled.shape[0]} range."
        )
    return centers_scaled[indices]


def _download_allow_patterns(landscape_id: str, *, include_plots: bool) -> list[str]:
    root = f"{LANDSCAPE_RELATIVE_ROOT}/{landscape_id}"
    patterns = [f"{root}/{name}" for name in REQUIRED_LANDSCAPE_FILES]
    if include_plots:
        patterns.append(f"{root}/plots/**")
    return patterns


def _validate_loaded_bundle(
    landscape_id: str,
    manifest: dict[str, Any],
    sampler: dict[str, Any],
    nodes: pd.DataFrame,
    tensors: dict[str, np.ndarray],
) -> None:
    if manifest.get("landscape_id") and manifest["landscape_id"] != landscape_id:
        raise PeptideLandscapeError(
            f"Manifest landscape_id {manifest['landscape_id']!r} does not match "
            f"requested landscape {landscape_id!r}."
        )
    required_columns = {
        "x",
        "y",
        "node_id",
        "organism",
        "density",
        "activity_mean",
        "activity_class",
        "uncertainty",
        "n_observations",
    }
    missing_cols = sorted(required_columns - set(nodes.columns))
    if missing_cols:
        raise PeptideLandscapeError(
            f"Peptide landscape nodes.parquet is missing columns: {missing_cols}."
        )
    missing_tensors = sorted(name for name in REQUIRED_TENSORS if name not in tensors)
    if missing_tensors:
        raise PeptideLandscapeError(
            f"Peptide landscape tensors are missing required arrays: {missing_tensors}."
        )
    if not isinstance(sampler, dict):
        raise PeptideLandscapeError("sampler.json must contain a JSON object.")


def _node_selection_score(nodes: pd.DataFrame, sampler: dict[str, Any]) -> pd.Series:
    weights = sampler.get("objective_weights") or {}
    activity_weight = float(weights.get("activity", 1.0))
    uncertainty_penalty = float(weights.get("uncertainty_penalty", 0.2))
    density_penalty = float(weights.get("density_penalty", 0.2))

    activity = pd.to_numeric(nodes.get("activity_mean"), errors="coerce").fillna(0.0)
    uncertainty = _minmax(pd.to_numeric(nodes.get("uncertainty"), errors="coerce"))
    density = _minmax(pd.to_numeric(nodes.get("density"), errors="coerce"))
    support = np.log1p(
        pd.to_numeric(nodes.get("n_observations"), errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    support = _minmax(support)
    return (
        (activity_weight * activity)
        + (0.1 * support)
        - (uncertainty_penalty * uncertainty)
        - (density_penalty * density)
    )


def _minmax(series: pd.Series) -> pd.Series:
    series = series.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    span = float(series.max() - series.min())
    if span <= 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - float(series.min())) / span


def _greedy_spatially_diverse_nodes(nodes: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    if len(nodes) <= top_n:
        return nodes.copy()

    remaining = nodes.copy()
    selected_rows = [remaining.iloc[0]]
    remaining = remaining.iloc[1:].copy()
    x_span = max(float(nodes["x"].max() - nodes["x"].min()), 1.0)
    y_span = max(float(nodes["y"].max() - nodes["y"].min()), 1.0)
    diagonal = float(np.hypot(x_span, y_span))

    while len(selected_rows) < top_n and not remaining.empty:
        selected_xy = np.array([[row["x"], row["y"]] for row in selected_rows], dtype=float)
        remaining_xy = remaining[["x", "y"]].to_numpy(dtype=float)
        distances = np.sqrt(((remaining_xy[:, None, :] - selected_xy[None, :, :]) ** 2).sum(axis=2))
        diversity = distances.min(axis=1) / diagonal
        combined = remaining["selection_score"].to_numpy(dtype=float) + 0.1 * diversity
        best_pos = int(np.nanargmax(combined))
        selected_rows.append(remaining.iloc[best_pos])
        remaining = remaining.drop(remaining.index[best_pos]).reset_index(drop=True)

    return pd.DataFrame(selected_rows)


def _coerce_coordinate(coord: Sequence[int | float] | dict[str, Any]) -> tuple[int, int]:
    if isinstance(coord, dict):
        if "x" not in coord or "y" not in coord:
            raise PeptideLandscapeError(f"Coordinate dict must contain x and y: {coord!r}.")
        raw_x, raw_y = coord["x"], coord["y"]
    else:
        values = list(coord)
        if len(values) != 2:
            raise PeptideLandscapeError(f"Coordinate must contain exactly two values: {coord!r}.")
        raw_x, raw_y = values
    return int(round(float(raw_x))), int(round(float(raw_y)))


def _normal_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _organism_keys(value: Any) -> set[str]:
    text = str(value)
    keys = {_normal_key(text)}
    words = [word for word in re.split(r"[^A-Za-z0-9]+", text) if word]
    if len(words) >= 2:
        keys.add(_normal_key(words[0][0] + words[-1]))
    return keys
