"""Deterministic ChEMBL retrieval preflight policy.

This gate is for external MCP reasoners. The connected client is the reasoning
engine, so this module does **not** parse free text or try to recognize targets
itself — that would only duplicate (worse) what the LLM already does. Instead the
caller supplies the structured retrieval dimensions it has decided on, and this
module enforces the completeness checklist and the "ask the user, don't infer"
contract, returning the canonical clarifying questions for anything still missing.

It is intentionally independent of Agno, ChEMBL backends, and MCP.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

TargetType = Literal["protein", "organism", "unknown"]

# ChEMBL assay_type codes. Natural-language words the caller might pass are
# mapped to the canonical code; this is a small fixed enum, not target parsing.
_ASSAY_WORD_TO_CODE = {
    "binding": "B",
    "functional": "F",
    "admet": "A",
    "adme": "A",
    "toxicity": "T",
    "physicochemical": "P",
    "unassigned": "U",
}
_VALID_ASSAY_CODES = {"B", "F", "A", "T", "P", "U"}

# Mechanism values that mean "no mechanism filter" rather than a specific MoA.
_ANY_MECHANISM_VALUES = {
    "any",
    "all",
    "unspecified",
    "none",
    "no preference",
    "no_preference",
    "no mechanism preference",
}

# Requirements the caller must decide (with the user) rather than infer.
DO_NOT_INFER = ["target_specificity", "organism", "assay_types", "mechanism"]


@dataclass(frozen=True)
class ChemblRetrievalDecision:
    """Structured ChEMBL preflight result for external MCP reasoners."""

    can_proceed: bool
    needs_clarification: bool
    missing_requirements: list[str] = field(default_factory=list)
    clarifying_questions: list[str] = field(default_factory=list)
    target: str | None = None
    target_type: TargetType = "unknown"
    organism: str | None = None
    assay_types: list[str] | None = None
    mechanism: str | None = None
    mechanism_preference: str | None = None
    recommended_next_tool: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def prepare_chembl_retrieval(
    target: str | None = None,
    target_type: str | None = None,
    organism: str | None = None,
    assay_types: list[str] | str | None = None,
    mechanism: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Validate a prospective ChEMBL retrieval from caller-supplied fields.

    The connected LLM extracts these from the user request; this gate only
    checks completeness and returns canonical clarifying questions for any
    missing dimension.

    Parameters
    ----------
    target:
        The specific target the user confirmed — a gene symbol (e.g. ``EPHX2``),
        protein name (e.g. ``soluble epoxide hydrolase``), ChEMBL target id, or an
        organism-level target. Leave empty/None if the user has not named a
        specific target yet; do not infer one.
    target_type:
        ``"protein"`` (default when a target is given) or ``"organism"``. Organism
        targets do not require a separate organism filter.
    organism:
        Organism filter (e.g. ``Homo sapiens``, ``Mus musculus``, ``all species``).
    assay_types:
        Assay types to include — ChEMBL codes (``B``/``F``/``A``/...) or words
        (``binding``/``functional``/``admet``).
    mechanism:
        A specific mechanism of action, or ``"any"``/``"unspecified"`` for no
        mechanism filter.
    notes:
        Optional free-text context. Recorded for the caller; never parsed for the
        decision.
    """

    target_clean = _clean(target) or None
    resolved_type = _normalize_target_type(target_type, target_clean)
    organism_clean = _clean(organism) or None
    codes = _normalize_assay_types(assay_types)
    mechanism_value, mechanism_preference = _normalize_mechanism(mechanism)

    missing: list[str] = []
    questions: list[str] = []

    if not target_clean:
        missing.append("target_specificity")
        questions.append(
            "Which specific target should ChEMBL search — a gene symbol, protein name, "
            "ChEMBL target id, or organism-level target?"
        )

    if resolved_type != "organism" and not organism_clean:
        missing.append("organism")
        questions.append(
            "Which organism should be used for the ChEMBL target or assay filter "
            "(for example, Homo sapiens, Mus musculus, or all species)?"
        )

    if not codes:
        missing.append("assay_types")
        questions.append(
            "Which assay types should be included: binding, functional, ADMET, or a combination?"
        )

    if mechanism_preference is None:
        missing.append("mechanism")
        questions.append(
            "Should assays be filtered to a specific mechanism of action, or should mechanism "
            "be unspecified/any?"
        )

    can_proceed = not missing
    return ChemblRetrievalDecision(
        can_proceed=can_proceed,
        needs_clarification=not can_proceed,
        missing_requirements=missing,
        clarifying_questions=questions,
        target=target_clean,
        target_type=resolved_type,
        organism=organism_clean,
        assay_types=codes,
        mechanism=mechanism_value,
        mechanism_preference=mechanism_preference,
        recommended_next_tool="chembl_convert_to_chembl_query" if can_proceed else None,
    ).as_dict()


def _clean(value: str | None) -> str:
    return " ".join(str(value or "").split())


def _normalize_target_type(target_type: str | None, target: str | None) -> TargetType:
    value = (target_type or "").strip().lower()
    if value in ("protein", "organism"):
        return value  # type: ignore[return-value]
    return "protein" if target else "unknown"


def _normalize_assay_types(value: list[str] | str | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [part.strip() for part in value.replace(";", ",").split(",")]
    else:
        items = [str(part).strip() for part in value]

    codes: list[str] = []
    for item in items:
        if not item:
            continue
        lowered = item.lower()
        if lowered in _ASSAY_WORD_TO_CODE:
            code = _ASSAY_WORD_TO_CODE[lowered]
        elif item.upper() in _VALID_ASSAY_CODES:
            code = item.upper()
        else:
            # Unknown token: pass it through unchanged so the caller sees what it
            # supplied rather than having it silently dropped.
            code = item
        if code not in codes:
            codes.append(code)
    return codes or None


def _normalize_mechanism(mechanism: str | None) -> tuple[str | None, str | None]:
    value = _clean(mechanism)
    if not value:
        return None, None
    if value.lower() in _ANY_MECHANISM_VALUES:
        return None, "any"
    return value, value
