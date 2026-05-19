#!/usr/bin/env python
# coding: utf-8
"""Shared structured working memory for a ChemSpace Copilot session."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from agno.tools.toolkit import Toolkit

from cs_copilot.storage import S3, OutputOperation, operation_rel_path

SESSION_OBJECTS_KEY = "session_objects"
SESSION_MEMORY_SUMMARY_KEY = "session_memory_summary"
MAX_REGISTERED_COMPOUNDS_PER_RESULT = 50
MAX_SUMMARY_ITEMS_PER_TYPE = 6
MAX_CANDIDATE_PREVIEW_ITEMS = 5
CANDIDATE_ARTIFACT_DIR = "candidate_sets"
CANDIDATE_ARTIFACT_FORMAT = "json"
CANDIDATE_DATASET_FORMAT = "csv"
LOADABLE_CSV_SUFFIXES = (".csv", ".csv.gz", ".tsv", ".tab", ".txt")
LOADABLE_SESSION_PATH_PRIORITY = {
    "clean_dataset_path": 0,
    "landscape_data_csv": 1,
    "activity_csv": 2,
    "density_csv": 3,
    "primary_data_csv": 4,
    "dataset_path": 5,
    "raw_dataset_path": 6,
    "filtered_dataset_path": 7,
    "csv_path": 8,
    "path": 9,
}

_OBJECT_TYPES: Dict[str, Tuple[str, str]] = {
    "compound": ("compounds", "cmp"),
    "candidate_set": ("candidate_sets", "cset"),
    "map": ("maps", "map"),
    "zone": ("zones", "zone"),
    "node": ("nodes", "node"),
    "dataset": ("datasets", "ds"),
    "analysis": ("analyses", "ana"),
    "figure": ("figures", "fig"),
    "route": ("routes", "route"),
    "report": ("reports", "rep"),
}

_PLURAL_TO_SINGULAR = {collection: obj_type for obj_type, (collection, _) in _OBJECT_TYPES.items()}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_object_type(object_type: str) -> str:
    normalized = str(object_type).strip().lower().replace(" ", "_")
    if normalized in _OBJECT_TYPES:
        return normalized
    if normalized in _PLURAL_TO_SINGULAR:
        return _PLURAL_TO_SINGULAR[normalized]
    raise ValueError(
        f"Unsupported session object type '{object_type}'. "
        f"Use one of: {', '.join(sorted(_OBJECT_TYPES))}."
    )


def _json_safe(value: Any, *, depth: int = 0, max_items: int = 20) -> Any:
    """Return a compact JSON-serializable representation."""
    if depth > 4:
        return str(value)
    if isinstance(value, float):
        return value if isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, dict):
        out = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= max_items:
                out["_truncated"] = True
                break
            out[str(key)] = _json_safe(item, depth=depth + 1, max_items=max_items)
        return out
    if isinstance(value, (list, tuple, set)):
        values = list(value)
        out = [
            _json_safe(item, depth=depth + 1, max_items=max_items) for item in values[:max_items]
        ]
        if len(values) > max_items:
            out.append({"_truncated": len(values) - max_items})
        return out
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _short_text(value: Any, *, max_chars: int = 160) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def _candidate_smiles(candidate: Any) -> Optional[str]:
    if isinstance(candidate, str):
        return candidate
    if isinstance(candidate, dict):
        smiles = (
            candidate.get("smiles") or candidate.get("canonical_smiles") or candidate.get("smi")
        )
        return str(smiles) if smiles else None
    return None


def _candidate_score(candidate: Any) -> Optional[Any]:
    if not isinstance(candidate, dict):
        return None
    for key in ("score", "ranking_score"):
        if candidate.get(key) is not None:
            return candidate[key]
    properties = candidate.get("properties") or {}
    if isinstance(properties, dict):
        for key in ("seed_tanimoto", "qed"):
            if properties.get(key) is not None:
                return properties[key]
    return None


def compact_candidate_preview(
    candidates: Sequence[Any],
    *,
    limit: int = MAX_CANDIDATE_PREVIEW_ITEMS,
) -> List[Dict[str, Any]]:
    """Return a small LLM-visible candidate preview."""
    preview: List[Dict[str, Any]] = []
    for candidate in list(candidates or [])[:limit]:
        item: Dict[str, Any] = {}
        smiles = _candidate_smiles(candidate)
        if smiles:
            item["smiles"] = smiles
        if isinstance(candidate, dict):
            if candidate.get("valid") is not None:
                item["valid"] = bool(candidate["valid"])
            if candidate.get("error"):
                item["error"] = str(candidate["error"])
        score = _candidate_score(candidate)
        if score is not None:
            item["score"] = score
        if item:
            preview.append(_json_safe(item))
    return preview


def _compact_related(related: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not related:
        return {}
    allowed = (
        "session_key",
        "candidate_set_id",
        "seed_smiles",
        "seed_compound_id",
        "generation_mode",
    )
    return {key: related[key] for key in allowed if related.get(key) is not None}


def _candidate_artifact_rel_path(
    candidate_set_id: str,
    *,
    session_state: Optional[Dict[str, Any]] = None,
) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(candidate_set_id))
    return operation_rel_path(
        OutputOperation.ANALOG_GENERATION,
        CANDIDATE_ARTIFACT_DIR,
        safe_id,
        f"candidates.{CANDIDATE_ARTIFACT_FORMAT}",
        session_state=session_state,
        workflow_slug="analog_generation",
    )


def _candidate_dataset_rel_path(
    candidate_set_id: str,
    top_n: Optional[int] = None,
    *,
    session_state: Optional[Dict[str, Any]] = None,
) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(candidate_set_id))
    filename = f"top_{top_n}.{CANDIDATE_DATASET_FORMAT}" if top_n is not None else "candidates.csv"
    return operation_rel_path(
        OutputOperation.ANALOG_GENERATION,
        CANDIDATE_ARTIFACT_DIR,
        safe_id,
        filename,
        session_state=session_state,
        workflow_slug="analog_generation",
    )


def _candidate_row_from_payload(
    candidate_set_id: str,
    rank: int,
    candidate: Any,
    *,
    source: str,
) -> Optional[Dict[str, Any]]:
    smiles = _candidate_smiles(candidate)
    if not smiles:
        return None

    row: Dict[str, Any] = {
        "smi": smiles,
        "rank": rank,
        "candidate_set_id": candidate_set_id,
        "source": source,
    }
    if isinstance(candidate, dict):
        for key in ("valid", "score", "ranking_score"):
            if candidate.get(key) is not None:
                row[key] = candidate[key]
        properties = candidate.get("properties") or {}
        if isinstance(properties, dict):
            for key in ("seed_tanimoto", "qed"):
                if properties.get(key) is not None:
                    row[key] = properties[key]
    else:
        score = _candidate_score(candidate)
        if score is not None:
            row["score"] = score
    return _json_safe(row, max_items=1000)


def _candidate_dataset_rows_from_payloads(
    candidate_set_id: str,
    candidates: Sequence[Any],
    *,
    source: str,
    top_n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows = []
    for rank, candidate in enumerate(list(candidates or []), start=1):
        if top_n is not None and rank > top_n:
            break
        row = _candidate_row_from_payload(candidate_set_id, rank, candidate, source=source)
        if row is not None:
            rows.append(row)
    return rows


def _candidate_dataset_rows_from_compounds(
    memory: Dict[str, Any],
    candidate_set: Dict[str, Any],
    *,
    top_n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows = []
    candidate_set_id = str(candidate_set.get("id") or "")
    source = str(
        candidate_set.get("generation_engine")
        or candidate_set.get("origin_agent")
        or candidate_set.get("source_tool")
        or "generated_candidates"
    )
    compound_ids = list(candidate_set.get("compound_ids") or [])
    for rank, compound_id in enumerate(compound_ids, start=1):
        if top_n is not None and rank > top_n:
            break
        record = _find_object(memory, compound_id)
        if not isinstance(record, dict):
            continue
        smiles = record.get("smiles")
        if not smiles:
            continue
        row: Dict[str, Any] = {
            "smi": str(smiles),
            "rank": rank,
            "candidate_set_id": candidate_set_id,
            "source": source,
        }
        if record.get("candidate_set_rank") is not None:
            row["candidate_set_rank"] = record["candidate_set_rank"]
        if record.get("generation_engine") is not None:
            row["generation_engine"] = record["generation_engine"]
        properties = record.get("properties") or {}
        if isinstance(properties, dict):
            for key in ("seed_tanimoto", "qed"):
                if properties.get(key) is not None:
                    row[key] = properties[key]
        rows.append(_json_safe(row, max_items=1000))
    return rows


def save_candidate_set_dataset(
    candidate_set_id: str,
    rows: Sequence[Dict[str, Any]],
    *,
    top_n: Optional[int] = None,
    session_state: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist a compact candidate-set CSV dataset for downstream tools."""
    row_list = list(rows or [])
    rel_path = _candidate_dataset_rel_path(
        candidate_set_id,
        top_n=top_n,
        session_state=session_state,
    )
    table = pd.DataFrame(row_list)
    if table.empty:
        table = pd.DataFrame(columns=["smi", "rank", "candidate_set_id", "source"])
    with S3.open(rel_path, "w") as handle:
        table.to_csv(handle, index=False)
    return S3.path(rel_path)


def save_candidate_set_artifact(
    candidate_set_id: str,
    candidates: Sequence[Any],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    session_state: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist full candidate payload to session-scoped storage and return its path."""
    candidate_list = list(candidates or [])
    max_items = max(len(candidate_list), 1)
    payload = {
        "version": 1,
        "candidate_set_id": candidate_set_id,
        "created_at": _now_iso(),
        "count": len(candidate_list),
        "metadata": _json_safe(metadata or {}, max_items=1000),
        "candidates": _json_safe(candidate_list, max_items=max(max_items, 1000)),
    }
    rel_path = _candidate_artifact_rel_path(candidate_set_id, session_state=session_state)
    with S3.open(rel_path, "w") as handle:
        json.dump(payload, handle, sort_keys=True)
    return S3.path(rel_path)


def load_candidate_artifact(artifact_path: str) -> Dict[str, Any]:
    """Load a generated candidate artifact from local/S3 storage."""
    if (
        isinstance(artifact_path, str)
        and not artifact_path.startswith("s3://")
        and (artifact_path.startswith("data/") or Path(artifact_path).is_absolute())
    ):
        with Path(artifact_path).open("r") as handle:
            payload = json.load(handle)
    else:
        with S3.open(artifact_path, "r") as handle:
            payload = json.load(handle)
    if isinstance(payload, list):
        return {"version": 1, "count": len(payload), "candidates": payload}
    if not isinstance(payload, dict):
        raise TypeError(f"Candidate artifact must contain a JSON object: {artifact_path}")
    payload.setdefault("count", len(payload.get("candidates") or []))
    return payload


def ensure_session_objects(session_state: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure the shared session object registry exists and return it."""
    memory = session_state.get(SESSION_OBJECTS_KEY)
    if not isinstance(memory, dict):
        memory = {}
        session_state[SESSION_OBJECTS_KEY] = memory

    memory.setdefault("version", 1)
    memory.setdefault("counters", {})
    memory.setdefault("current", {})
    for collection, _prefix in _OBJECT_TYPES.values():
        memory.setdefault(collection, {})
    return memory


def _next_id(memory: Dict[str, Any], object_type: str) -> str:
    _collection, prefix = _OBJECT_TYPES[object_type]
    counters = memory.setdefault("counters", {})
    counters[object_type] = int(counters.get(object_type, 0)) + 1
    return f"{prefix}_{counters[object_type]:03d}"


def _node_object_id(data: Dict[str, Any]) -> Optional[str]:
    map_id = data.get("map_id")
    node_index = data.get("node_index")
    if map_id is None or node_index is None:
        return None
    safe_map = re.sub(r"[^A-Za-z0-9_]+", "_", str(map_id))
    safe_node = re.sub(r"[^A-Za-z0-9_]+", "_", str(node_index))
    return f"node_{safe_map}_{safe_node}"


def refresh_session_memory_summary(session_state: Dict[str, Any]) -> str:
    """Refresh and return the compact memory summary stored in session_state."""
    summary = summarize_session_memory(session_state)
    session_state[SESSION_MEMORY_SUMMARY_KEY] = summary
    return summary


def register_session_object(
    session_state: Dict[str, Any],
    object_type: str,
    data: Dict[str, Any],
    *,
    object_id: Optional[str] = None,
    label: Optional[str] = None,
    source_agent: Optional[str] = None,
    source_tool: Optional[str] = None,
    set_current: bool = True,
    current_role: Optional[str] = None,
) -> str:
    """Register or update a compact session object and return its ID."""
    normalized_type = _normalize_object_type(object_type)
    memory = ensure_session_objects(session_state)
    collection_name, _prefix = _OBJECT_TYPES[normalized_type]
    collection = memory[collection_name]

    payload = _json_safe(dict(data or {}))
    if object_id is None and normalized_type == "node":
        object_id = _node_object_id(payload)
    if object_id is None:
        object_id = _next_id(memory, normalized_type)

    now = _now_iso()
    existing = collection.get(object_id, {})
    record = dict(existing)
    record.update(payload)
    record["id"] = object_id
    record["object_type"] = normalized_type
    record.setdefault("created_at", now)
    record["updated_at"] = now
    if label is not None:
        record["label"] = label
    else:
        record.setdefault("label", object_id)
    if source_agent is not None:
        record["source_agent"] = source_agent
    if source_tool is not None:
        record["source_tool"] = source_tool

    collection[object_id] = record

    if set_current:
        role = current_role or normalized_type
        memory.setdefault("current", {})[role] = object_id

    refresh_session_memory_summary(session_state)
    return object_id


def update_session_object(
    session_state: Dict[str, Any],
    object_id: str,
    updates: Dict[str, Any],
    *,
    set_current: bool = False,
    current_role: Optional[str] = None,
) -> Dict[str, Any]:
    """Update an existing session object and return the updated record."""
    memory = ensure_session_objects(session_state)
    record = _find_object(memory, object_id)
    if record is None:
        raise KeyError(f"Session object not found: {object_id}")

    record.update(_json_safe(updates or {}))
    record["updated_at"] = _now_iso()
    if set_current:
        role = current_role or record["object_type"]
        memory.setdefault("current", {})[role] = object_id
    refresh_session_memory_summary(session_state)
    return record


def select_session_object(
    session_state: Dict[str, Any],
    object_id: str,
    *,
    role: Optional[str] = None,
) -> Dict[str, Any]:
    """Set an existing object as the current object for its type or role."""
    memory = ensure_session_objects(session_state)
    record = _find_object(memory, object_id)
    if record is None:
        raise KeyError(f"Session object not found: {object_id}")

    current_role = role or record["object_type"]
    memory.setdefault("current", {})[current_role] = object_id
    refresh_session_memory_summary(session_state)
    return record


def get_session_object(session_state: Dict[str, Any], object_id: str) -> Optional[Dict[str, Any]]:
    """Return a session object by ID."""
    return _find_object(ensure_session_objects(session_state), object_id)


def _is_loadable_csv_path(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    path = value.strip().split("?", 1)[0].lower()
    return path.endswith(LOADABLE_CSV_SUFFIXES)


def _loadable_entry(session_key: str, value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, pd.DataFrame):
        return {
            "session_key": session_key,
            "kind": "dataframe",
            "shape": tuple(int(part) for part in value.shape),
            "columns": [str(column) for column in value.columns[:20]],
        }
    if _is_loadable_csv_path(value):
        return {
            "session_key": session_key,
            "kind": "csv_path",
            "path": value,
        }
    return None


def _iter_loadable_session_data(
    value: Any,
    prefix: str,
    output: List[Dict[str, Any]],
    *,
    depth: int = 0,
    max_depth: int = 5,
    max_items: int = 100,
    seen: Optional[set[int]] = None,
) -> None:
    if len(output) >= max_items:
        return
    seen = seen or set()
    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)

    entry = _loadable_entry(prefix, value)
    if entry is not None:
        output.append(entry)
        return
    if depth >= max_depth:
        return

    if isinstance(value, dict):
        for key, item in value.items():
            key_str = str(key)
            if key_str.startswith("_"):
                continue
            child_prefix = f"{prefix}.{key_str}" if prefix else key_str
            _iter_loadable_session_data(
                item,
                child_prefix,
                output,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                seen=seen,
            )
            if len(output) >= max_items:
                return
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            _iter_loadable_session_data(
                item,
                child_prefix,
                output,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                seen=seen,
            )
            if len(output) >= max_items:
                return


def list_loadable_session_data(
    session_state: Dict[str, Any],
    *,
    max_items: int = 100,
) -> List[Dict[str, Any]]:
    """List session DataFrames and CSV-like paths that dataframe tools can load."""
    if not isinstance(session_state, dict):
        return []
    output: List[Dict[str, Any]] = []
    for key, value in session_state.items():
        key_str = str(key)
        if key_str.startswith("_"):
            continue
        _iter_loadable_session_data(value, key_str, output, max_items=max_items)
        if len(output) >= max_items:
            break
    return output


def _resolve_dotted_session_key(
    session_state: Dict[str, Any], session_key: str
) -> Tuple[bool, Any]:
    if session_key in session_state:
        return True, session_state[session_key]

    current: Any = session_state
    for part in str(session_key).split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, (list, tuple)) and part.isdigit():
            index = int(part)
            if 0 <= index < len(current):
                current = current[index]
                continue
        return False, None
    return True, current


def _loadable_sort_key(entry: Dict[str, Any]) -> tuple[int, str]:
    if entry.get("kind") == "dataframe":
        return (-1, str(entry.get("session_key", "")))
    final_key = str(entry.get("session_key", "")).rsplit(".", 1)[-1]
    return (
        LOADABLE_SESSION_PATH_PRIORITY.get(final_key, 100),
        str(entry.get("session_key", "")),
    )


def resolve_loadable_session_data(
    session_state: Dict[str, Any],
    session_key: str,
) -> Dict[str, Any]:
    """Resolve a top-level, dotted, or container session key to a loadable object."""
    if not isinstance(session_state, dict):
        raise ValueError("session_state must be a dictionary")

    found, value = _resolve_dotted_session_key(session_state, session_key)
    if not found:
        available = [entry["session_key"] for entry in list_loadable_session_data(session_state)]
        raise KeyError(
            f"Session key '{session_key}' not found. Loadable session data keys: {available}"
        )

    entry = _loadable_entry(session_key, value)
    if entry is not None:
        entry["value"] = value
        return entry

    nested_entries: List[Dict[str, Any]] = []
    _iter_loadable_session_data(value, session_key, nested_entries)
    if not nested_entries:
        raise TypeError(
            f"Session key '{session_key}' is not a DataFrame or CSV path and contains no "
            "loadable nested DataFrames/CSV paths."
        )

    selected = sorted(nested_entries, key=_loadable_sort_key)[0]
    found_nested, nested_value = _resolve_dotted_session_key(session_state, selected["session_key"])
    if not found_nested:
        raise KeyError(f"Resolved nested session key vanished: {selected['session_key']}")
    selected["value"] = nested_value
    return selected


def list_session_objects(
    session_state: Dict[str, Any],
    object_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List compact session objects, optionally filtered by type."""
    memory = ensure_session_objects(session_state)
    if object_type:
        normalized = _normalize_object_type(object_type)
        collection_name, _prefix = _OBJECT_TYPES[normalized]
        return list(memory[collection_name].values())

    records: List[Dict[str, Any]] = []
    for collection_name, _prefix in _OBJECT_TYPES.values():
        records.extend(memory[collection_name].values())
    return records


def resolve_session_reference(
    session_state: Dict[str, Any],
    reference: str,
    object_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve natural references like 'that compound', IDs, labels, or SMILES."""
    memory = ensure_session_objects(session_state)
    reference_text = str(reference or "").strip()
    reference_lower = reference_text.lower()
    candidates = _candidate_records(memory, object_type)

    current_words = {"", "that", "it", "current", "last", "previous", "latest"}
    if reference_lower in current_words or any(
        reference_lower.startswith(f"{word} ") for word in current_words if word
    ):
        current = _resolve_current(memory, object_type)
        if current is not None:
            return {"status": "resolved", "object": current}
        return {"status": "not_found", "message": "No current session object is selected."}

    direct = _find_object(memory, reference_text)
    if direct is not None and _record_matches_type(direct, object_type):
        return {"status": "resolved", "object": direct}

    numeric = _resolve_numbered_reference(reference_lower, candidates)
    if numeric is not None:
        return {"status": "resolved", "object": numeric}

    matches = []
    for record in candidates:
        haystacks = [
            str(record.get("id", "")),
            str(record.get("label", "")),
            str(record.get("smiles", "")),
            str(record.get("original_smiles", "")),
            str(record.get("node_index", "")),
        ]
        if any(reference_lower == item.lower() for item in haystacks if item):
            matches.append(record)
            continue
        if any(reference_lower in item.lower() for item in haystacks if item):
            matches.append(record)

    if len(matches) == 1:
        return {"status": "resolved", "object": matches[0]}
    if len(matches) > 1:
        return {
            "status": "ambiguous",
            "matches": [_compact_record_for_summary(item) for item in matches[:10]],
            "message": "Reference matched multiple session objects. Ask the user to choose an ID.",
        }
    return {"status": "not_found", "message": f"No session object matched '{reference_text}'."}


def register_compounds_from_candidates(
    session_state: Dict[str, Any],
    candidates: Sequence[Any],
    *,
    source_agent: Optional[str],
    source_tool: str,
    label_prefix: str,
    related: Optional[Dict[str, Any]] = None,
    provenance: Optional[Dict[str, Any]] = None,
    set_current_first: bool = True,
    limit: int = MAX_REGISTERED_COMPOUNDS_PER_RESULT,
) -> List[str]:
    """Register valid compound-like candidates and return their IDs."""
    ids: List[str] = []
    for idx, raw in enumerate(candidates[:limit], start=1):
        if isinstance(raw, str):
            candidate = {"smiles": raw, "valid": True}
        elif isinstance(raw, dict):
            candidate = dict(raw)
        else:
            continue

        if candidate.get("valid") is False:
            continue
        smiles = (
            candidate.get("smiles") or candidate.get("canonical_smiles") or candidate.get("smi")
        )
        if not smiles:
            continue

        score = candidate.get("score")
        if score is None:
            score = candidate.get("ranking_score")
        payload = {
            "smiles": smiles,
            "rank": idx,
            "score": score,
            "related": _compact_related(related),
        }
        payload.update(provenance or {})
        object_id = register_session_object(
            session_state,
            "compound",
            payload,
            label=f"{label_prefix} {idx}",
            source_agent=source_agent,
            source_tool=source_tool,
            set_current=set_current_first and len(ids) == 0,
        )
        ids.append(object_id)
    return ids


def register_generated_candidate_set(
    session_state: Dict[str, Any],
    compound_ids: Sequence[str],
    *,
    source_agent: Optional[str],
    source_tool: str,
    origin_agent: str,
    generation_engine: str,
    session_key: str,
    label: str,
    generation_mode: Optional[str] = None,
    seed_smiles: Optional[str] = None,
    seed_compound_id: Optional[str] = None,
    goal: Optional[str] = None,
    count_attempted: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
    candidates: Optional[Sequence[Any]] = None,
) -> str:
    """Register an ordered generated candidate set and link compounds back to it."""
    ordered_ids = [compound_id for compound_id in compound_ids if compound_id]
    artifact_path = None
    artifact_count = None
    csv_path = None
    csv_count = None
    preview: List[Dict[str, Any]] = []
    candidate_set_id = register_session_object(
        session_state,
        "candidate_set",
        {
            "candidate_set_type": "generated",
            "origin_type": "generated",
            "origin_agent": origin_agent,
            "generation_engine": generation_engine,
            "generation_mode": generation_mode,
            "source_tool": source_tool,
            "session_key": session_key,
            "compound_ids": ordered_ids,
            "seed_smiles": seed_smiles,
            "seed_compound_id": seed_compound_id,
            "goal": _short_text(goal),
            "ranked": True,
            "count_attempted": count_attempted,
            "count_returned": len(ordered_ids),
            "metadata_keys": sorted(str(key) for key in (metadata or {}).keys()),
        },
        label=label,
        source_agent=source_agent,
        source_tool=source_tool,
        set_current=True,
        current_role="candidate_set",
    )
    memory = ensure_session_objects(session_state)
    memory.setdefault("current", {})["generated_compounds"] = candidate_set_id

    if candidates is not None:
        candidate_list = list(candidates or [])
        artifact_metadata = {
            "origin_agent": origin_agent,
            "generation_engine": generation_engine,
            "generation_mode": generation_mode,
            "source_tool": source_tool,
            "session_key": session_key,
            "label": label,
            "seed_smiles": seed_smiles,
            "seed_compound_id": seed_compound_id,
            "goal": goal,
            "count_attempted": count_attempted,
            "metadata": metadata or {},
        }
        artifact_path = save_candidate_set_artifact(
            candidate_set_id,
            candidate_list,
            metadata=artifact_metadata,
            session_state=session_state,
        )
        artifact_rel_path = _candidate_artifact_rel_path(
            candidate_set_id,
            session_state=session_state,
        )
        artifact_count = len(candidate_list)
        preview = compact_candidate_preview(candidate_list)
        session_state[session_key] = {
            "candidate_set_id": candidate_set_id,
            "artifact_path": artifact_path,
            "artifact_rel_path": artifact_rel_path,
            "artifact_format": CANDIDATE_ARTIFACT_FORMAT,
            "count": artifact_count,
            "preview": preview,
        }
        update_session_object(
            session_state,
            candidate_set_id,
            {
                "artifact_path": artifact_path,
                "artifact_rel_path": artifact_rel_path,
                "artifact_format": CANDIDATE_ARTIFACT_FORMAT,
                "artifact_count": artifact_count,
                "preview": preview,
            },
        )

    candidate_set_record = _find_object(memory, candidate_set_id) or {}
    csv_source = str(generation_engine or origin_agent or source_tool or "generated_candidates")
    if candidates is not None:
        csv_rows = _candidate_dataset_rows_from_payloads(
            candidate_set_id,
            list(candidates or []),
            source=csv_source,
        )
    else:
        csv_rows = _candidate_dataset_rows_from_compounds(memory, candidate_set_record)

    if candidates is not None and (csv_rows or ordered_ids):
        csv_path = save_candidate_set_dataset(
            candidate_set_id,
            csv_rows,
            session_state=session_state,
        )
        csv_count = len(csv_rows)
        csv_rel_path = _candidate_dataset_rel_path(candidate_set_id, session_state=session_state)
        pointer = session_state.get(session_key)
        if not isinstance(pointer, dict):
            pointer = {
                "candidate_set_id": candidate_set_id,
                "count": csv_count,
                "preview": [
                    {"smiles": row["smi"]} for row in csv_rows[:MAX_CANDIDATE_PREVIEW_ITEMS]
                ],
            }
            session_state[session_key] = pointer
        pointer.update(
            {
                "candidate_set_id": candidate_set_id,
                "csv_path": csv_path,
                "csv_rel_path": csv_rel_path,
                "csv_format": CANDIDATE_DATASET_FORMAT,
                "csv_count": csv_count,
            }
        )
        update_session_object(
            session_state,
            candidate_set_id,
            {
                "csv_path": csv_path,
                "csv_rel_path": csv_rel_path,
                "csv_format": CANDIDATE_DATASET_FORMAT,
                "csv_count": csv_count,
            },
        )

    for rank, compound_id in enumerate(ordered_ids, start=1):
        record = _find_object(memory, compound_id)
        if record is None:
            continue
        related = dict(record.get("related") or {})
        related["candidate_set_id"] = candidate_set_id
        updates = {
            "origin_type": "generated",
            "origin_agent": origin_agent,
            "generation_engine": generation_engine,
            "candidate_set_id": candidate_set_id,
            "candidate_set_rank": rank,
            "related": related,
        }
        if seed_smiles is not None:
            updates["seed_smiles"] = seed_smiles
        if seed_compound_id is not None:
            updates["seed_compound_id"] = seed_compound_id
        update_session_object(session_state, compound_id, updates)

    refresh_session_memory_summary(session_state)
    return candidate_set_id


def resolve_candidate_set(
    session_state: Dict[str, Any],
    reference: str = "top candidates",
    *,
    top_n: Optional[int] = None,
) -> Dict[str, Any]:
    """Resolve phrases like 'top candidates' to the latest generated candidate set."""
    memory = ensure_session_objects(session_state)
    reference_text = str(reference or "").strip()
    reference_lower = reference_text.lower()
    parsed_top_n = top_n or _parse_top_n(reference_lower)

    candidate_set = _resolve_candidate_set_record(memory, reference_lower)
    if candidate_set is None:
        return {
            "status": "not_found",
            "message": "No generated candidate set is available in session memory.",
        }

    compounds = _candidate_set_compounds(memory, candidate_set, top_n=parsed_top_n)
    return {
        "status": "resolved",
        "candidate_set": candidate_set,
        "compounds": compounds,
        "count": len(compounds),
    }


def load_candidate_set_artifact(
    session_state: Dict[str, Any],
    reference: str = "top candidates",
    *,
    include_candidates: bool = True,
) -> Dict[str, Any]:
    """Load a generated candidate-set artifact by ID, session key, path, or reference."""
    if not isinstance(session_state, dict):
        return {"status": "not_found", "message": "No session state is available."}

    reference_text = str(reference or "").strip()
    candidate_set = None
    artifact_path = reference_text if reference_text.endswith(".json") else None

    if not artifact_path and reference_text in session_state:
        pointer = session_state.get(reference_text)
        if isinstance(pointer, dict):
            artifact_path = pointer.get("artifact_rel_path") or pointer.get("artifact_path")
            candidate_set_id = pointer.get("candidate_set_id")
            if candidate_set_id:
                candidate_set = get_session_object(session_state, str(candidate_set_id))

    if not artifact_path:
        resolved = resolve_candidate_set(session_state, reference_text)
        if resolved.get("status") != "resolved":
            return resolved
        candidate_set = resolved.get("candidate_set")
        if isinstance(candidate_set, dict):
            artifact_path = candidate_set.get("artifact_rel_path") or candidate_set.get(
                "artifact_path"
            )

    if not artifact_path:
        return {
            "status": "not_found",
            "message": f"No candidate artifact path found for '{reference_text}'.",
        }

    payload = load_candidate_artifact(artifact_path)
    candidates = payload.get("candidates") or []
    result = {
        "status": "loaded",
        "artifact_path": payload.get("artifact_path")
        or (
            candidate_set.get("artifact_path") if isinstance(candidate_set, dict) else artifact_path
        ),
        "count": int(payload.get("count") or len(candidates)),
        "preview": compact_candidate_preview(candidates),
    }
    if isinstance(candidate_set, dict):
        result["candidate_set"] = _compact_record_for_summary(candidate_set)
    if include_candidates:
        result["candidates"] = candidates
    return result


def materialize_candidate_set_dataset(
    session_state: Dict[str, Any],
    reference: str = "generated compounds",
    *,
    top_n: Optional[int] = None,
) -> Dict[str, Any]:
    """Resolve a generated candidate set and return a compact CSV path for tools."""
    if not isinstance(session_state, dict):
        return {"status": "not_found", "message": "No session state is available."}
    if top_n is not None and top_n <= 0:
        return {"status": "error", "message": "top_n must be positive when provided."}

    reference_text = str(reference or "").strip() or "generated compounds"
    if reference_text.endswith(LOADABLE_CSV_SUFFIXES):
        return {
            "status": "materialized",
            "candidate_set_id": None,
            "csv_path": reference_text,
            "count": None,
        }

    candidate_set: Optional[Dict[str, Any]] = None
    pointer = session_state.get(reference_text)
    if isinstance(pointer, dict) and pointer.get("candidate_set_id"):
        candidate_set = get_session_object(session_state, str(pointer["candidate_set_id"]))

    if candidate_set is None and reference_text.endswith(f".{CANDIDATE_ARTIFACT_FORMAT}"):
        payload = load_candidate_artifact(reference_text)
        candidate_set_id = str(payload.get("candidate_set_id") or Path(reference_text).stem)
        candidates = payload.get("candidates") or []
        rows = _candidate_dataset_rows_from_payloads(
            candidate_set_id,
            candidates,
            source=str((payload.get("metadata") or {}).get("generation_engine") or "generated"),
            top_n=top_n,
        )
        csv_path = save_candidate_set_dataset(
            candidate_set_id,
            rows,
            top_n=top_n,
            session_state=session_state,
        )
        return {
            "status": "materialized",
            "candidate_set_id": candidate_set_id,
            "csv_path": csv_path,
            "count": len(rows),
        }

    if candidate_set is None:
        resolved = resolve_candidate_set(session_state, reference_text, top_n=top_n)
        if resolved.get("status") != "resolved":
            return resolved
        candidate_set = resolved.get("candidate_set")

    if not isinstance(candidate_set, dict):
        return {
            "status": "not_found",
            "message": f"No generated candidate set found for '{reference_text}'.",
        }

    candidate_set_id = str(candidate_set.get("id") or "")
    if not candidate_set_id:
        return {"status": "not_found", "message": "Resolved candidate set has no ID."}

    if top_n is None and candidate_set.get("csv_path"):
        return {
            "status": "materialized",
            "candidate_set_id": candidate_set_id,
            "csv_path": candidate_set["csv_path"],
            "count": int(
                candidate_set.get("csv_count") or candidate_set.get("count_returned") or 0
            ),
        }

    candidates = None
    artifact_path = candidate_set.get("artifact_rel_path") or candidate_set.get("artifact_path")
    if artifact_path:
        try:
            payload = load_candidate_artifact(str(artifact_path))
            candidates = payload.get("candidates") or []
        except Exception:
            candidates = None

    if candidates:
        rows = _candidate_dataset_rows_from_payloads(
            candidate_set_id,
            candidates,
            source=str(
                candidate_set.get("generation_engine")
                or candidate_set.get("origin_agent")
                or "generated"
            ),
            top_n=top_n,
        )
    else:
        rows = _candidate_dataset_rows_from_compounds(
            ensure_session_objects(session_state),
            candidate_set,
            top_n=top_n,
        )

    if not rows:
        return {
            "status": "not_found",
            "message": f"Candidate set '{candidate_set_id}' has no materializable SMILES rows.",
            "candidate_set_id": candidate_set_id,
        }

    csv_path = save_candidate_set_dataset(
        candidate_set_id,
        rows,
        top_n=top_n,
        session_state=session_state,
    )
    result = {
        "status": "materialized",
        "candidate_set_id": candidate_set_id,
        "csv_path": csv_path,
        "count": len(rows),
    }

    if top_n is None:
        csv_rel_path = _candidate_dataset_rel_path(candidate_set_id, session_state=session_state)
        update_session_object(
            session_state,
            candidate_set_id,
            {
                "csv_path": csv_path,
                "csv_rel_path": csv_rel_path,
                "csv_format": CANDIDATE_DATASET_FORMAT,
                "csv_count": len(rows),
            },
        )
        pointer = session_state.get(str(candidate_set.get("session_key") or ""))
        if isinstance(pointer, dict):
            pointer.update(
                {
                    "csv_path": csv_path,
                    "csv_rel_path": csv_rel_path,
                    "csv_format": CANDIDATE_DATASET_FORMAT,
                    "csv_count": len(rows),
                }
            )
        refresh_session_memory_summary(session_state)

    return result


class SessionStore:
    """Typed facade over Agno's mutable ``session_state`` dictionary."""

    def __init__(self, session_state: Dict[str, Any]):
        if not isinstance(session_state, dict):
            raise TypeError("session_state must be a dictionary")
        self.session_state = session_state

    @property
    def objects(self) -> Dict[str, Any]:
        return ensure_session_objects(self.session_state)

    def register_object(self, object_type: str, data: Dict[str, Any], **kwargs) -> str:
        return register_session_object(self.session_state, object_type, data, **kwargs)

    def get_object(self, object_id: str) -> Optional[Dict[str, Any]]:
        return get_session_object(self.session_state, object_id)

    def resolve_candidate_set(
        self,
        reference: str = "top candidates",
        *,
        top_n: Optional[int] = None,
    ) -> Dict[str, Any]:
        return resolve_candidate_set(self.session_state, reference, top_n=top_n)

    def load_candidate_set_artifact(
        self,
        reference: str = "top candidates",
        *,
        include_candidates: bool = True,
    ) -> Dict[str, Any]:
        return load_candidate_set_artifact(
            self.session_state,
            reference,
            include_candidates=include_candidates,
        )

    def materialize_candidate_set_dataset(
        self,
        reference: str = "generated compounds",
        *,
        top_n: Optional[int] = None,
    ) -> Dict[str, Any]:
        return materialize_candidate_set_dataset(self.session_state, reference, top_n=top_n)

    def refresh_summary(self) -> str:
        return refresh_session_memory_summary(self.session_state)


def update_state_targets(
    agent: Optional[Any],
    session_state: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return unique mutable state dictionaries for shared and direct-agent contexts."""
    targets: List[Dict[str, Any]] = []
    if session_state is not None:
        targets.append(session_state)
    if agent is not None:
        if getattr(agent, "session_state", None) is None:
            agent.session_state = {}
        if not any(agent.session_state is target for target in targets):
            targets.append(agent.session_state)
    return targets


def summarize_session_memory(
    session_state: Dict[str, Any],
    *,
    max_items_per_type: int = MAX_SUMMARY_ITEMS_PER_TYPE,
) -> str:
    """Build a compact human-readable summary for prompt context."""
    memory = ensure_session_objects(session_state)
    current = memory.get("current", {})
    lines = ["Session working memory:"]
    if current:
        current_bits = [f"{role}={object_id}" for role, object_id in sorted(current.items())]
        lines.append(f"- current: {', '.join(current_bits)}")

    for object_type, (collection_name, _prefix) in _OBJECT_TYPES.items():
        records = list(memory.get(collection_name, {}).values())
        if not records:
            continue
        lines.append(f"- {collection_name}: {len(records)}")
        for record in records[-max_items_per_type:]:
            lines.append(f"  - {_compact_record_line(record, object_type)}")
    return "\n".join(lines)


def _candidate_records(
    memory: Dict[str, Any],
    object_type: Optional[str],
) -> List[Dict[str, Any]]:
    if object_type:
        normalized = _normalize_object_type(object_type)
        collection_name, _prefix = _OBJECT_TYPES[normalized]
        return list(memory.get(collection_name, {}).values())

    records: List[Dict[str, Any]] = []
    for collection_name, _prefix in _OBJECT_TYPES.values():
        records.extend(memory.get(collection_name, {}).values())
    return records


def _resolve_current(
    memory: Dict[str, Any],
    object_type: Optional[str],
) -> Optional[Dict[str, Any]]:
    current = memory.get("current", {})
    if object_type:
        normalized = _normalize_object_type(object_type)
        object_id = current.get(normalized)
        return _find_object(memory, object_id) if object_id else None

    for role in (
        "compound",
        "candidate_set",
        "generated_compounds",
        "map",
        "zone",
        "node",
        "dataset",
        "analysis",
        "figure",
        "route",
        "report",
    ):
        object_id = current.get(role)
        if object_id:
            record = _find_object(memory, object_id)
            if record is not None:
                return record
    return None


def _find_object(memory: Dict[str, Any], object_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not object_id:
        return None
    for collection_name, _prefix in _OBJECT_TYPES.values():
        record = memory.get(collection_name, {}).get(object_id)
        if record is not None:
            return record
    return None


def _record_matches_type(record: Dict[str, Any], object_type: Optional[str]) -> bool:
    return object_type is None or record.get("object_type") == _normalize_object_type(object_type)


def _resolve_numbered_reference(
    reference_lower: str,
    candidates: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    match = re.search(
        r"\b(?:compound|candidate set|candidate|map|zone|node|dataset|analysis|figure|route|report)\s+(\d+)\b",
        reference_lower,
    )
    if not match:
        return None
    index = int(match.group(1))
    if index <= 0 or index > len(candidates):
        return None
    return candidates[index - 1]


def _parse_top_n(reference_lower: str) -> Optional[int]:
    match = re.search(r"\btop\s+(\d+)\b", reference_lower)
    if not match:
        return None
    value = int(match.group(1))
    return value if value > 0 else None


def _resolve_candidate_set_record(
    memory: Dict[str, Any],
    reference_lower: str,
) -> Optional[Dict[str, Any]]:
    requested_engine = None
    if "llm" in reference_lower:
        requested_engine = "llm"
    elif "autoencoder" in reference_lower:
        requested_engine = "autoencoder"

    current = memory.get("current", {})
    for role in ("candidate_set", "generated_compounds"):
        object_id = current.get(role)
        record = _find_object(memory, object_id)
        if (
            record is not None
            and record.get("object_type") == "candidate_set"
            and (
                requested_engine is None
                or str(record.get("generation_engine", "")).lower() == requested_engine
            )
        ):
            return record

    direct = _find_object(memory, reference_lower)
    if direct is not None and direct.get("object_type") == "candidate_set":
        return direct

    generated_words = {
        "top candidates",
        "candidates",
        "generated compounds",
        "generated molecules",
        "analogs",
        "analogues",
        "latest designs",
        "design candidates",
        "molecular designer candidates",
        "llm candidates",
        "autoencoder candidates",
    }
    if any(word in reference_lower for word in generated_words):
        candidate_sets = [
            record
            for record in memory.get("candidate_sets", {}).values()
            if requested_engine is None
            or str(record.get("generation_engine", "")).lower() == requested_engine
        ]
        return candidate_sets[-1] if candidate_sets else None

    matches = []
    for record in memory.get("candidate_sets", {}).values():
        haystacks = [
            str(record.get("id", "")),
            str(record.get("label", "")),
            str(record.get("generation_engine", "")),
            str(record.get("generation_mode", "")),
            str(record.get("source_tool", "")),
        ]
        if any(reference_lower == item.lower() for item in haystacks if item):
            matches.append(record)
            continue
        if any(reference_lower in item.lower() for item in haystacks if item):
            matches.append(record)
    return matches[0] if len(matches) == 1 else None


def _candidate_set_compounds(
    memory: Dict[str, Any],
    candidate_set: Dict[str, Any],
    *,
    top_n: Optional[int],
) -> List[Dict[str, Any]]:
    compound_ids = list(candidate_set.get("compound_ids") or [])
    if top_n is not None:
        compound_ids = compound_ids[:top_n]

    compounds = []
    for rank, compound_id in enumerate(compound_ids, start=1):
        record = _find_object(memory, compound_id)
        if record is None:
            continue
        compact = _compact_record_for_summary(record)
        compact.update(
            {
                "rank": record.get("candidate_set_rank") or rank,
                "score": record.get("score"),
                "source_tool": record.get("source_tool"),
                "origin_agent": record.get("origin_agent"),
                "generation_engine": record.get("generation_engine"),
                "candidate_set_id": candidate_set.get("id"),
            }
        )
        compounds.append(compact)
    return compounds


def _compact_record_for_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": record.get("id"),
        "object_type": record.get("object_type"),
        "label": record.get("label"),
        "smiles": record.get("smiles"),
        "candidate_set_id": record.get("candidate_set_id"),
        "node_index": record.get("node_index"),
        "map_id": record.get("map_id"),
        "dataset_path": record.get("dataset_path"),
        "clean_dataset_path": record.get("clean_dataset_path"),
        "raw_dataset_path": record.get("raw_dataset_path"),
        "filtered_dataset_path": record.get("filtered_dataset_path"),
        "descriptor_parquet_path": record.get("descriptor_parquet_path"),
        "figure_kind": record.get("figure_kind"),
        "renderer": record.get("renderer"),
        "report_role": record.get("report_role"),
    }


def _compact_record_line(record: Dict[str, Any], object_type: str) -> str:
    bits = [str(record.get("id")), str(record.get("label", ""))]
    if object_type == "compound" and record.get("smiles"):
        bits.append(str(record["smiles"]))
        if record.get("origin_type") == "generated":
            bits.append(
                f"generated_by={record.get('origin_agent')}/{record.get('generation_engine')}"
            )
    if object_type == "candidate_set":
        bits.append(f"engine={record.get('generation_engine')}")
        bits.append(f"mode={record.get('generation_mode')}")
        bits.append(f"compounds={len(record.get('compound_ids') or [])}")
        if record.get("session_key"):
            bits.append(f"session_key={record['session_key']}")
        if record.get("artifact_path"):
            bits.append("artifact=json")
    if object_type == "map" and record.get("dataset_path"):
        bits.append(f"dataset={record['dataset_path']}")
    if object_type == "dataset":
        if record.get("clean_dataset_path"):
            bits.append(f"clean={record['clean_dataset_path']}")
        elif record.get("dataset_path"):
            bits.append(f"dataset={record['dataset_path']}")
        if record.get("raw_dataset_path"):
            bits.append(f"raw={record['raw_dataset_path']}")
        if record.get("filtered_dataset_path"):
            bits.append(f"filtered={record['filtered_dataset_path']}")
        if record.get("descriptor_parquet_path"):
            bits.append(f"descriptors={record['descriptor_parquet_path']}")
    if object_type == "zone" and record.get("node_ids"):
        bits.append(f"nodes={record['node_ids']}")
    if object_type == "node" and record.get("node_index") is not None:
        bits.append(f"node={record['node_index']}")
    if object_type == "route" and record.get("target_smiles"):
        bits.append(f"target={record['target_smiles']}")
    if object_type == "figure":
        if record.get("figure_kind"):
            bits.append(f"kind={record['figure_kind']}")
        if record.get("renderer"):
            bits.append(f"renderer={record['renderer']}")
        if record.get("report_role"):
            bits.append(f"role={record['report_role']}")
        paths = record.get("paths") or {}
        if isinstance(paths, dict):
            if paths.get("png_path"):
                bits.append(f"png={paths['png_path']}")
            elif paths.get("html_path"):
                bits.append(f"html={paths['html_path']}")
    return " | ".join(bit for bit in bits if bit)


class SessionMemoryToolkit(Toolkit):
    """Tools for inspecting and selecting structured session working memory."""

    def __init__(self):
        super().__init__("session_memory")
        self.register(self.list_session_objects)
        self.register(self.list_loadable_session_data)
        self.register(self.get_session_object)
        self.register(self.select_session_object)
        self.register(self.resolve_session_reference)
        self.register(self.resolve_candidate_set)
        self.register(self.load_candidate_set_artifact)
        self.register(self.materialize_candidate_set_dataset)
        self.register(self.summarize_session_memory)

    def list_session_objects(
        self,
        object_type: Optional[str] = None,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        List important objects remembered in this session.

        Args:
            object_type: Optional type filter such as compound, candidate_set, map,
                zone, node, dataset, analysis, route, or report.
            session_state: Shared session state injected by Agno.
        """
        if session_state is None:
            return []
        return list_session_objects(session_state, object_type)

    def list_loadable_session_data(
        self,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        List DataFrames and CSV paths available through supported session keys.

        Args:
            session_state: Shared session state injected by Agno.
        """
        if session_state is None:
            return []
        return list_loadable_session_data(session_state)

    def get_session_object(
        self,
        object_id: str,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get one remembered object by stable ID.

        Args:
            object_id: Stable object ID such as cmp_001 or map_001.
            session_state: Shared session state injected by Agno.
        """
        if session_state is None:
            return {"status": "not_found", "message": "No session state is available."}
        record = get_session_object(session_state, object_id)
        if record is None:
            return {"status": "not_found", "message": f"No object found for {object_id}."}
        return {"status": "found", "object": record}

    def select_session_object(
        self,
        object_id: str,
        role: Optional[str] = None,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Mark a remembered object as the current object for follow-up references.

        Args:
            object_id: Stable object ID.
            role: Optional current role override, such as compound, map, or zone.
            session_state: Shared session state injected by Agno.
        """
        if session_state is None:
            return {"status": "error", "message": "No session state is available."}
        try:
            record = select_session_object(session_state, object_id, role=role)
        except KeyError as exc:
            return {"status": "not_found", "message": str(exc)}
        return {"status": "selected", "object": record}

    def resolve_session_reference(
        self,
        reference: str,
        object_type: Optional[str] = None,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Resolve natural references like 'that compound', 'the active zone', or an ID.

        Args:
            reference: User reference text or stable object ID.
            object_type: Optional expected object type.
            session_state: Shared session state injected by Agno.
        """
        if session_state is None:
            return {"status": "not_found", "message": "No session state is available."}
        return resolve_session_reference(session_state, reference, object_type)

    def resolve_candidate_set(
        self,
        reference: str = "top candidates",
        top_n: Optional[int] = None,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Resolve generated candidate references such as 'top candidates'.

        Args:
            reference: User reference text, e.g. 'top 3 candidates' or 'LLM candidates'.
            top_n: Optional limit on returned ranked compounds.
            session_state: Shared session state injected by Agno.
        """
        if session_state is None:
            return {"status": "not_found", "message": "No session state is available."}
        return resolve_candidate_set(session_state, reference, top_n=top_n)

    def load_candidate_set_artifact(
        self,
        reference: str = "top candidates",
        include_candidates: bool = False,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Load a generated candidate-set artifact by ID, session key, path, or reference.

        Args:
            reference: Candidate-set ID, session key, artifact path, or phrase.
            include_candidates: Include the full artifact payload. Keep False for summaries.
            session_state: Shared session state injected by Agno.
        """
        if session_state is None:
            return {"status": "not_found", "message": "No session state is available."}
        return load_candidate_set_artifact(
            session_state,
            reference,
            include_candidates=include_candidates,
        )

    def materialize_candidate_set_dataset(
        self,
        reference: str = "generated compounds",
        top_n: Optional[int] = None,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Resolve generated candidates and return a compact CSV path for downstream tools.

        Args:
            reference: Candidate-set ID, session key, CSV/JSON artifact path, or phrase.
            top_n: Optional limit for a ranked subset CSV.
            session_state: Shared session state injected by Agno.
        """
        if session_state is None:
            return {"status": "not_found", "message": "No session state is available."}
        return materialize_candidate_set_dataset(session_state, reference, top_n=top_n)

    def summarize_session_memory(
        self,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Summarize important compounds, candidate sets, maps, zones, nodes,
        analyses, figures, and routes.

        Args:
            session_state: Shared session state injected by Agno.
        """
        if session_state is None:
            return "No session state is available."
        return refresh_session_memory_summary(session_state)
