#!/usr/bin/env python
# coding: utf-8
"""Shared structured working memory for a ChemSpace Copilot session."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from math import isfinite
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agno.tools.toolkit import Toolkit

SESSION_OBJECTS_KEY = "session_objects"
SESSION_MEMORY_SUMMARY_KEY = "session_memory_summary"
MAX_REGISTERED_COMPOUNDS_PER_RESULT = 50
MAX_SUMMARY_ITEMS_PER_TYPE = 6

_OBJECT_TYPES: Dict[str, Tuple[str, str]] = {
    "compound": ("compounds", "cmp"),
    "candidate_set": ("candidate_sets", "cset"),
    "map": ("maps", "map"),
    "zone": ("zones", "zone"),
    "node": ("nodes", "node"),
    "dataset": ("datasets", "ds"),
    "analysis": ("analyses", "ana"),
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

        payload = {
            "smiles": smiles,
            "original_smiles": candidate.get("original_smiles"),
            "rank": idx,
            "score": candidate.get("score"),
            "properties": candidate.get("properties", {}),
            "rationale": candidate.get("rationale"),
            "activity": candidate.get("activity"),
            "related": related or {},
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
) -> str:
    """Register an ordered generated candidate set and link compounds back to it."""
    ordered_ids = [compound_id for compound_id in compound_ids if compound_id]
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
            "goal": goal,
            "ranked": True,
            "count_attempted": count_attempted,
            "count_returned": len(ordered_ids),
            "metadata": metadata or {},
        },
        label=label,
        source_agent=source_agent,
        source_tool=source_tool,
        set_current=True,
        current_role="candidate_set",
    )
    memory = ensure_session_objects(session_state)
    memory.setdefault("current", {})["generated_compounds"] = candidate_set_id

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
        r"\b(?:compound|candidate set|candidate|map|zone|node|dataset|analysis|route|report)\s+(\d+)\b",
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
        "descriptor_parquet_path": record.get("descriptor_parquet_path"),
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
    if object_type == "map" and record.get("dataset_path"):
        bits.append(f"dataset={record['dataset_path']}")
    if object_type == "dataset":
        if record.get("clean_dataset_path"):
            bits.append(f"clean={record['clean_dataset_path']}")
        elif record.get("dataset_path"):
            bits.append(f"dataset={record['dataset_path']}")
        if record.get("raw_dataset_path"):
            bits.append(f"raw={record['raw_dataset_path']}")
        if record.get("descriptor_parquet_path"):
            bits.append(f"descriptors={record['descriptor_parquet_path']}")
    if object_type == "zone" and record.get("node_ids"):
        bits.append(f"nodes={record['node_ids']}")
    if object_type == "node" and record.get("node_index") is not None:
        bits.append(f"node={record['node_index']}")
    if object_type == "route" and record.get("target_smiles"):
        bits.append(f"target={record['target_smiles']}")
    return " | ".join(bit for bit in bits if bit)


class SessionMemoryToolkit(Toolkit):
    """Tools for inspecting and selecting structured session working memory."""

    def __init__(self):
        super().__init__("session_memory")
        self.register(self.list_session_objects)
        self.register(self.get_session_object)
        self.register(self.select_session_object)
        self.register(self.resolve_session_reference)
        self.register(self.resolve_candidate_set)
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

    def summarize_session_memory(
        self,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Summarize important compounds, candidate sets, maps, zones, nodes, analyses, and routes.

        Args:
            session_state: Shared session state injected by Agno.
        """
        if session_state is None:
            return "No session state is available."
        return refresh_session_memory_summary(session_state)
