"""Lossless observed-output capture for offline molecular-generation evaluation.

An observed model output is not proof of the number of internal decoding attempts.
Keep that denominator unknown unless the backend explicitly provides it.
"""

import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def capture_generation_audit(
    outputs: Sequence[Any], *, source: str, requested_count: int, raw_outputs_available: bool
) -> Dict[str, Any]:
    return {
        "schema": "chemspacecopilot.generation_audit.v1",
        "capture_stage": source,
        "raw_outputs_available": raw_outputs_available,
        "requested_count": requested_count,
        "observed_output_count": len(outputs),
        "backend_attempt_count": None,
        "rng_seed": None,
        "rng_state_recorded": False,
        "raw_outputs": list(outputs),
    }


def generation_audit_summary(audit: Dict[str, Any]) -> Dict[str, Any]:
    """Keep all raw strings in the artifact, out of compact tool summaries."""
    return {key: value for key, value in audit.items() if key != "raw_outputs"}


def save_generation_audit(
    audit: Dict[str, Any],
    *,
    session_state: Optional[Dict[str, Any]] = None,
    audit_path: Optional[str] = None,
) -> Optional[str]:
    """Save explicitly to a local path, or to the session artifact store."""
    if audit_path is not None:
        path = Path(audit_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
        return str(path)
    if session_state is None:
        return None
    from cs_copilot.storage import S3
    from cs_copilot.storage.layout import OutputOperation, operation_rel_path

    rel_path = operation_rel_path(
        OutputOperation.ANALOG_GENERATION,
        "generation_audits",
        f"{uuid.uuid4().hex}.json",
        session_state=session_state,
        workflow_slug="analog_generation",
    )
    with S3.open(rel_path, "w") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    return S3.path(rel_path)
