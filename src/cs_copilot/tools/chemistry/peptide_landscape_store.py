"""Local aggregate peptide landscapes for prospective revision experiments.

Selection and inverse-scaler sampling adapted from the author's existing branch
114 at ee6156fdc9ee15b8169af8441c150e16b31ecf99. This module uses an explicit
pinned local bundle, never downloads data or deserializes its runtime pickle.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

REQUIRED_TENSORS = ("gtm.phi", "gtm.weights", "scaler.mean", "scaler.scale")
BUNDLE_FILES = ("landscape.json", "landscape.safetensors", "nodes.parquet", "sampler.json")


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

    if not np.isfinite(local_noise_scale) or local_noise_scale < 0:
        raise PeptideLandscapeError("local_noise_scale must be finite and non-negative.")
    node_ids = selected_nodes["node_id"].astype(int).to_numpy()
    centers_scaled = latent_centers_for_nodes(bundle, node_ids)
    rng = np.random.default_rng(random_state)

    weights = node_sampling_probabilities(selected_nodes)

    chosen = rng.choice(np.arange(len(selected_nodes)), size=n_samples, replace=True, p=weights)
    noise = rng.normal(0.0, local_noise_scale, size=(n_samples, centers_scaled.shape[1]))
    z_scaled = centers_scaled[chosen] + noise

    mean = np.asarray(bundle.tensors["scaler.mean"], dtype=float)
    scale = np.asarray(bundle.tensors["scaler.scale"], dtype=float)
    latents = (z_scaled * scale) + mean

    assignments = selected_nodes.iloc[chosen].reset_index(drop=True).copy()
    assignments.insert(0, "sample_index", np.arange(n_samples))
    assignments["node_selection_probability"] = (
        weights[chosen] if weights is not None else 1.0 / len(selected_nodes)
    )
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


def _normal_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _organism_keys(value: Any) -> set[str]:
    text = str(value)
    keys = {_normal_key(text)}
    words = [word for word in re.split(r"[^A-Za-z0-9]+", text) if word]
    if len(words) >= 2:
        keys.add(_normal_key(words[0][0] + words[-1]))
    return keys


def load_local_peptide_landscape(bundle_path: str) -> PeptideLandscapeBundle:
    """Load explicit local safe tensors and aggregate tables, with dimension checks."""
    from safetensors.numpy import load_file

    root = Path(bundle_path).expanduser().resolve()
    for name in BUNDLE_FILES:
        if not (root / name).is_file():
            raise PeptideLandscapeError(f"Missing local landscape input: {root / name}")
    manifest = json.loads((root / "landscape.json").read_text())
    sampler = json.loads((root / "sampler.json").read_text())
    nodes = pd.read_parquet(root / "nodes.parquet")
    tensors = load_file(str(root / "landscape.safetensors"))
    landscape_id = str(manifest["landscape_id"])
    _validate_loaded_bundle(landscape_id, manifest, sampler, nodes, tensors)
    phi, weights, mean, scale = [np.asarray(tensors[key]) for key in REQUIRED_TENSORS]
    dim = manifest.get("latent_dim")
    if (
        phi.ndim != 2
        or weights.ndim != 2
        or phi.shape[1] != weights.shape[0]
        or weights.shape[1] != dim
        or mean.shape != (dim,)
        or scale.shape != (dim,)
        or any(not np.isfinite(a).all() for a in (phi, weights, mean, scale))
        or np.any(scale <= 0)
    ):
        raise PeptideLandscapeError("Invalid GTM/scaler dimensions or nonfinite tensors")
    node_ids = pd.to_numeric(nodes["node_id"], errors="coerce").to_numpy()
    if (
        not np.isfinite(node_ids).all()
        or np.any(node_ids != np.floor(node_ids))
        or np.any(node_ids < 1)
        or np.any(node_ids > phi.shape[0])
        or nodes.duplicated(["organism", "node_id"]).any()
    ):
        raise PeptideLandscapeError("Invalid or duplicate 1-based organism/node identifiers")
    return PeptideLandscapeBundle(landscape_id, root, manifest, sampler, nodes, tensors)


def node_sampling_probabilities(selected_nodes: pd.DataFrame) -> np.ndarray:
    """Return the same normalized node probabilities used by the sampler."""
    if selected_nodes.empty:
        raise PeptideLandscapeError("selected_nodes cannot be empty")
    if "selection_score" not in selected_nodes:
        return np.full(len(selected_nodes), 1.0 / len(selected_nodes))
    raw = selected_nodes["selection_score"].astype(float).to_numpy()
    if not np.isfinite(raw).all():
        raise PeptideLandscapeError("Nonfinite node selection scores")
    raw = raw - raw.min() + 1e-6
    return raw / raw.sum()
