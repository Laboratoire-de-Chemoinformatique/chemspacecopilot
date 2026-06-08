"""Deterministic ChEMBL retrieval preflight policy.

This module is intentionally independent of Agno, ChEMBL backends, and MCP.
It gives external MCP reasoners a structured gate before calling mutating
retrieval tools.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

TargetType = Literal["protein", "organism", "unknown"]

_KNOWN_ABBREVIATIONS = {
    "ABL1": "Abelson tyrosine-protein kinase 1",
    "BRAF": "B-Raf proto-oncogene serine/threonine-protein kinase",
    "CDK2": "cyclin-dependent kinase 2",
    "CYP3A4": "cytochrome P450 3A4",
    "DPP4": "dipeptidyl peptidase 4",
    "EGFR": "epidermal growth factor receptor",
    "JAK2": "Janus kinase 2",
    "MTOR": "mechanistic target of rapamycin",
    "PDE4": "phosphodiesterase 4 family",
    "PDE4A": "phosphodiesterase 4A",
    "PPARG": "peroxisome proliferator-activated receptor gamma",
    "PTP1B": "protein tyrosine phosphatase 1B",
}

_FULL_TARGET_PATTERNS = (
    (re.compile(r"\bcyclin[- ]dependent kinase 2\b", re.I), "cyclin-dependent kinase 2"),
    (re.compile(r"\bepidermal growth factor receptor\b", re.I), "epidermal growth factor receptor"),
    (re.compile(r"\bjanus kinase 2\b", re.I), "Janus kinase 2"),
    (re.compile(r"\bprotein tyrosine phosphatase 1b\b", re.I), "protein tyrosine phosphatase 1B"),
    (re.compile(r"\bcytochrome p450 3a4\b", re.I), "cytochrome P450 3A4"),
    (re.compile(r"\bhiv[- ]?1 reverse transcriptase\b", re.I), "HIV-1 reverse transcriptase"),
)

_ORGANISM_TARGET_PATTERNS = (
    (re.compile(r"\bhiv[- ]?1\b", re.I), "HIV-1"),
    (re.compile(r"\binfluenza(?:\s+[ab])?\b", re.I), "Influenza"),
    (re.compile(r"\bescherichia coli\b|\be\.?\s*coli\b", re.I), "Escherichia coli"),
)

_ORGANISM_FILTER_PATTERNS = (
    (re.compile(r"\bhomo sapiens\b|\bhuman\b|\bin humans?\b", re.I), "Homo sapiens"),
    (re.compile(r"\bmus musculus\b|\bmouse\b|\bmice\b", re.I), "Mus musculus"),
    (re.compile(r"\brattus norvegicus\b|\brat\b", re.I), "Rattus norvegicus"),
    (re.compile(r"\ball species\b|\ball organisms\b|\bany organism\b", re.I), "all species"),
    (re.compile(r"\bescherichia coli\b|\be\.?\s*coli\b", re.I), "Escherichia coli"),
)

_ASSAY_TYPE_PATTERNS = (
    ("B", re.compile(r"\bbinding\b|\bki\b|\bkd\b", re.I)),
    ("F", re.compile(r"\bfunctional\b", re.I)),
    ("A", re.compile(r"\badmet\b|\btoxicity\b|\babsorption\b|\bmetabolism\b", re.I)),
)

_ANY_MECHANISM_RE = re.compile(
    r"\b(?:unspecified|no preference|no mechanism preference|i don't care)\b"
    r"|\b(?:any|all)\s+mechanisms?\b"
    r"|\bmechanisms?\s+(?:any|all|unspecified|no preference)\b"
    r"|\bmechanism\s*[:=]\s*(?:any|all|none|unspecified)\b",
    re.I,
)
_SPECIFIC_MECHANISM_RE = re.compile(
    r"\b(?:agonist|antagonist|inverse agonist|partial agonist|allosteric modulator|"
    r"allosteric|covalent|atp[- ]competitive|modulator)\b",
    re.I,
)

_GENERIC_TARGET_PATTERNS = (
    re.compile(r"\b(?:protein\s+)?kinases?\s*(?:\d+|alpha|beta|gamma|ii|iii)?\b", re.I),
    re.compile(r"\breceptors?\s*(?:\d+|alpha|beta|gamma|ii|iii)?\b", re.I),
    re.compile(r"\b(?:protein|phosphatase|phosphodiesterase)\s*\d*\b", re.I),
    re.compile(r"\b(?:gpcr|nuclear receptor|ion channel|transporter)s?\b", re.I),
)


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
    user_request: str,
    session_summary: str | None = None,
) -> dict[str, Any]:
    """Return a structured decision for a prospective ChEMBL retrieval."""

    request = _clean(user_request)
    context = _clean("\n".join(value for value in (user_request, session_summary) if value))
    if not request:
        return ChemblRetrievalDecision(
            can_proceed=False,
            needs_clarification=True,
            missing_requirements=["user_request"],
            clarifying_questions=["What ChEMBL target or organism should be retrieved?"],
        ).as_dict()

    target, target_type = _extract_target(request)
    organism = _extract_organism_filter(context)
    assay_types = _extract_assay_types(context)
    mechanism, mechanism_preference = _extract_mechanism(context)

    missing: list[str] = []
    questions: list[str] = []

    if _is_generic_target_request(request, target):
        missing.append("target_specificity")
        questions.append(
            "Which specific gene symbol, protein name, ChEMBL target id, or organism-level "
            "target should be retrieved?"
        )
        target = None
        target_type = "unknown"
    elif target is None:
        missing.append("target_specificity")
        questions.append(
            "Which specific biological target or organism-level target should ChEMBL search?"
        )

    if target and _requires_abbreviation_confirmation(
        request, target, organism, assay_types, mechanism_preference
    ):
        missing.append("target_confirmation")
        full_name = _KNOWN_ABBREVIATIONS.get(target.upper(), target)
        questions.append(f"{target} usually refers to {full_name}. Is that the target you mean?")

    if target_type == "protein" and not organism:
        missing.append("organism")
        questions.append(
            "Which organism should be used for the ChEMBL target or assay filter "
            "(for example, Homo sapiens, Mus musculus, or all species)?"
        )

    if not assay_types:
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
        target=target,
        target_type=target_type,
        organism=organism,
        assay_types=assay_types,
        mechanism=mechanism,
        mechanism_preference=mechanism_preference,
        recommended_next_tool="chembl_convert_to_chembl_query" if can_proceed else None,
    ).as_dict()


def _clean(value: str | None) -> str:
    return " ".join(str(value or "").split())


def _extract_target(request: str) -> tuple[str | None, TargetType]:
    for pattern, organism in _ORGANISM_TARGET_PATTERNS:
        if pattern.search(request):
            return organism, "organism"

    for symbol in sorted(_KNOWN_ABBREVIATIONS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(symbol)}\b", request, re.I):
            return symbol, "protein"

    target_id = re.search(r"\bCHEMBL\d+\b", request, re.I)
    if target_id:
        return target_id.group(0).upper(), "protein"

    for pattern, target in _FULL_TARGET_PATTERNS:
        if pattern.search(request):
            return target, "protein"

    return None, "unknown"


def _extract_organism_filter(context: str) -> str | None:
    for pattern, organism in _ORGANISM_FILTER_PATTERNS:
        if pattern.search(context):
            return organism
    return None


def _extract_assay_types(context: str) -> list[str] | None:
    codes = [code for code, pattern in _ASSAY_TYPE_PATTERNS if pattern.search(context)]
    return codes or None


def _extract_mechanism(context: str) -> tuple[str | None, str | None]:
    if _ANY_MECHANISM_RE.search(context):
        return None, "any"
    match = _SPECIFIC_MECHANISM_RE.search(context)
    if match:
        mechanism = match.group(0)
        return mechanism, mechanism
    return None, None


def _is_generic_target_request(request: str, target: str | None) -> bool:
    if target:
        return False
    return any(pattern.search(request) for pattern in _GENERIC_TARGET_PATTERNS)


def _requires_abbreviation_confirmation(
    request: str,
    target: str,
    organism: str | None,
    assay_types: list[str] | None,
    mechanism_preference: str | None,
) -> bool:
    if target.upper() not in _KNOWN_ABBREVIATIONS:
        return False
    if re.search(r"\bfull name\b|\bconfirmed\b|\btarget id\b|\bCHEMBL\d+\b", request, re.I):
        return False
    # Fully scoped MCP requests are allowed to proceed; sparse abbreviation-only
    # requests still ask for confirmation plus the missing retrieval dimensions.
    return not (organism and assay_types and mechanism_preference is not None)
