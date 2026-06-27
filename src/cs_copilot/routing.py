"""Single source of truth for request → workflow / skill / agent routing.

Capability-selection *vocabulary* lives in the catalog: each workflow / skill
declares its trigger words in `workflow.yaml` / `skill.yaml` `keywords:`. This
module owns only the cross-cutting *policy* that can't be expressed as per-entry
data: the concept→agent mapping used to render the Agno team's routing prose,
the read-only preflight intent triggers, and the handful of explicit tie-breaks.

Both consumers — the deterministic MCP bootstrap facade and the LLM-facing team
description — route through this one module, so they cannot drift from each other
or from the catalog.

Import-safety invariant: this module may import only the standard library and the
pure-Python ``cs_copilot.workflows`` / ``cs_copilot.skills`` registries. It must
never import the Agno team, the MCP server, ``model_config``, or ``chainlit_app``
(enforced by ``tests/unit/test_routing_drift.py``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from cs_copilot.skills import get_skill, list_skills, search_skills
from cs_copilot.workflows import WorkflowSpec, get_workflow, search_workflows


@dataclass(frozen=True)
class RoutingResult:
    """Outcome of routing a single request."""

    workflow: WorkflowSpec | None = None
    workflow_error: str | None = None
    skills: tuple[str, ...] = ()
    preflight_tools: tuple[str, ...] = ()
    tie_breaks_applied: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Policy tables (the only place routing *opinions* live; vocabulary stays in
# the catalog and is read from it by slug).
# ---------------------------------------------------------------------------

# Some workflows only make sense as part of a projection pipeline. Suppress them
# unless the request actually mentions GTM/projection.
_PROJECTION_TERMS = ("gtm", "project", "projection", "map", "landscape")
_PROJECTION_GATED_WORKFLOWS = ("candidate-design-to-gtm",)

_GTM_DENSITY_WORKFLOW = "gtm-density-landscape"
_GTM_ACTIVITY_WORKFLOW = "gtm-activity-landscape"
_GTM_DENSITY_TERMS = (
    "density",
    "density map",
    "density landscape",
    "compound distribution",
    "densest nodes",
    "dense nodes",
)
_GTM_ACTIVITY_TERMS = (
    "activity landscape",
    "activity map",
    "sar landscape",
    "active regions",
)

# Slug tokens too generic to signal relevance on their own (these caused the old
# "for"-overlap mis-match in bootstrap).
_SLUG_STOPWORDS = frozenset({"for", "to", "and", "the", "of", "a", "an", "on", "in"})


@dataclass(frozen=True)
class _PreflightDomain:
    """A read-only preflight gate to run before mutating tools in a domain."""

    tool: str
    # Intent verbs that signal the domain without selecting a specific
    # capability (e.g. "compound", "fetch"). These deliberately do NOT live in
    # the catalog: matching "analyze compounds" must trigger chemspace preflight
    # without recommending any workflow or skill.
    triggers: tuple[str, ...]


_PREFLIGHT_DOMAINS = (
    _PreflightDomain(
        tool="chembl_prepare_retrieval",
        triggers=(
            "chembl",
            "bioactivity",
            "assay",
            "activity data",
            "fetch",
            "retrieve",
            "download",
        ),
    ),
    _PreflightDomain(
        tool="chemspace_plan_analysis",
        triggers=(
            "chemical space",
            "compound",
            "gtm",
            "map",
            "landscape",
            "density",
            "project",
            "scaffold",
            "sar",
            "report",
        ),
    ),
)


@dataclass(frozen=True)
class _RoutingDomain:
    """Concept → agent mapping for the team's routing prose and skill selection.

    The trigger vocabulary is read from ``anchor_skill``'s catalog ``keywords:``
    so the generated prose can never diverge from the deterministic routing.
    """

    label: str
    agent: str
    anchor_skill: str
    note: str = ""
    # Skills that must NOT be co-selected when this domain wins (mutual
    # exclusion). Ordering in ``_ROUTING_DOMAINS`` defines precedence.
    exclusive_with: tuple[str, ...] = field(default_factory=tuple)


# Order matters: peptide is evaluated before molecule so an antimicrobial
# peptide request never also selects the small-molecule skill.
_ROUTING_DOMAINS = (
    _RoutingDomain(
        label="peptide",
        agent="Peptide Designer",
        anchor_skill="peptide-design",
        note="(incl. DBAASP antimicrobial landscapes)",
    ),
    _RoutingDomain(
        label="molecule",
        agent="Molecular Designer",
        anchor_skill="molecular-design",
        note='(also the default for an unqualified "generate")',
        exclusive_with=("peptide-design",),
    ),
    _RoutingDomain(
        label="report",
        agent="Report Generator",
        anchor_skill="report-generation",
    ),
    _RoutingDomain(
        label="retrosynthesis",
        agent="SynPlanner",
        anchor_skill="retrosynthesis-planning",
    ),
)


# ---------------------------------------------------------------------------
# Matching primitives
# ---------------------------------------------------------------------------


def _phrase_in(text_lc: str, phrase: str) -> bool:
    """Whole-word, plural-tolerant phrase match against lowercased text.

    Mirrors the registry's keyword scorer: whole-word (so "amp" never matches
    "example") but tolerant of a trailing plural ("candidate" → "candidates").
    """

    p = phrase.strip().lower()
    if not p:
        return False
    return re.search(rf"\b{re.escape(p)}(?:s|es)?\b", text_lc) is not None


def _matches_any(text_lc: str, phrases: tuple[str, ...]) -> bool:
    return any(_phrase_in(text_lc, phrase) for phrase in phrases)


def _dedupe(values) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


@lru_cache(maxsize=1)
def _skill_slugs() -> frozenset[str]:
    return frozenset(spec.slug for spec in list_skills())


def _is_skill(slug: str) -> bool:
    return slug in _skill_slugs()


def _get_workflow_or_none(slug: str) -> WorkflowSpec | None:
    try:
        return get_workflow(slug)
    except KeyError:
        return None


# ---------------------------------------------------------------------------
# Selection steps
# ---------------------------------------------------------------------------


def _workflow_is_relevant(text_lc: str, spec: WorkflowSpec) -> bool:
    """Reject incidental matches (e.g. "compounds" hitting a tool name in text).

    A workflow is relevant only on a real signal: a catalog keyword, a tag, or a
    non-stopword slug token. Stopword slug tokens are excluded so generic words
    like "for" cannot create a spurious match.
    """

    if _matches_any(text_lc, spec.keywords) or _matches_any(text_lc, spec.tags):
        return True
    slug_terms = tuple(
        token for token in spec.slug.replace("-", " ").split() if token not in _SLUG_STOPWORDS
    )
    return _matches_any(text_lc, slug_terms)


def _select_workflow(
    text: str, workflow_slug: str | None, tie_breaks: list[str]
) -> tuple[WorkflowSpec | None, str | None]:
    if workflow_slug:
        try:
            return get_workflow(workflow_slug), None
        except KeyError as exc:
            return None, str(exc)

    text_lc = text.lower()
    if not text_lc:
        return None, None

    results = search_workflows(text, limit=1)
    if not results:
        return None, None
    spec = results[0]
    density_hit = _matches_any(text_lc, _GTM_DENSITY_TERMS)
    activity_hit = _matches_any(text_lc, _GTM_ACTIVITY_TERMS)
    if density_hit and not activity_hit:
        spec = _get_workflow_or_none(_GTM_DENSITY_WORKFLOW) or spec
    elif activity_hit and not density_hit:
        spec = _get_workflow_or_none(_GTM_ACTIVITY_WORKFLOW) or spec
    if spec.slug in _PROJECTION_GATED_WORKFLOWS and not _matches_any(text_lc, _PROJECTION_TERMS):
        tie_breaks.append(f"suppressed {spec.slug} (no projection terms)")
        return None, None
    if not _workflow_is_relevant(text_lc, spec):
        tie_breaks.append(f"rejected weak workflow match {spec.slug}")
        return None, None
    return spec, None


def _select_preflight(text: str, workflow: WorkflowSpec | None) -> list[str]:
    text_lc = text.lower()
    tools = list(workflow.preflight_tools) if workflow else []
    for domain in _PREFLIGHT_DOMAINS:
        if _matches_any(text_lc, domain.triggers):
            tools.append(domain.tool)
    return _dedupe(tools)


def _select_skills(
    text: str,
    workflow: WorkflowSpec | None,
    preflight_present: bool,
) -> list[str]:
    text_lc = text.lower()
    slugs: list[str] = []

    # The matched workflow's skill twin, when one exists, leads the list.
    if workflow and _is_skill(workflow.slug):
        slugs.append(workflow.slug)

    # Density and activity GTM landscapes are separate catalog entries. When the
    # request explicitly asks for both, fetch both skill procedures.
    if _matches_any(text_lc, _GTM_DENSITY_TERMS) and _is_skill(_GTM_DENSITY_WORKFLOW):
        slugs.append(_GTM_DENSITY_WORKFLOW)
    if _matches_any(text_lc, _GTM_ACTIVITY_TERMS) and _is_skill(_GTM_ACTIVITY_WORKFLOW):
        slugs.append(_GTM_ACTIVITY_WORKFLOW)

    # Concept domains (peptide XOR molecule, then report, then retrosynthesis),
    # with vocabulary read from each anchor skill's catalog keywords.
    for domain in _ROUTING_DOMAINS:
        if any(excluded in slugs for excluded in domain.exclusive_with):
            continue
        if _matches_any(text_lc, get_skill(domain.anchor_skill).keywords):
            slugs.append(domain.anchor_skill)

    # Fallback free-text search only when nothing else matched and no preflight
    # gate already framed the request.
    if not preflight_present and not slugs and text.strip():
        slugs.extend(spec.slug for spec in search_skills(text, limit=2))

    return _dedupe(slugs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def match_request(text: str, *, workflow_slug: str | None = None) -> RoutingResult:
    """Route a request to a workflow, skills, and preflight gates.

    Pure and side-effect free: it reads the catalog and applies the policy tables
    above. ``workflow_slug`` forces an explicit workflow (returning
    ``workflow_error`` on an unknown slug).
    """

    text = " ".join(str(text or "").split())
    tie_breaks: list[str] = []

    workflow, workflow_error = _select_workflow(text, workflow_slug, tie_breaks)
    preflight = _select_preflight(text, workflow)
    skills = _select_skills(text, workflow, bool(preflight))

    return RoutingResult(
        workflow=workflow,
        workflow_error=workflow_error,
        skills=tuple(skills),
        preflight_tools=tuple(preflight),
        tie_breaks_applied=tuple(tie_breaks),
    )


def render_routing_rules() -> str:
    """Render the team's molecule/peptide routing prose from the catalog.

    Generated from :data:`_ROUTING_DOMAINS` plus each domain's anchor-skill
    keywords, so the LLM-facing prose stays in lockstep with deterministic
    routing. Used by ``cs_copilot.agents.teams``.
    """

    lines = ["**Molecule vs Peptide Routing**:"]
    for domain in _ROUTING_DOMAINS:
        vocab = ", ".join(f"'{kw}'" for kw in get_skill(domain.anchor_skill).keywords)
        note = f" {domain.note}" if domain.note else ""
        lines.append(f"  - {vocab} → {domain.agent} agent{note}")
    return "\n".join(lines)


def routing_domains() -> tuple[dict, ...]:
    """Expose the concept→agent domains as plain dicts (for tests / tooling)."""

    return tuple(
        {
            "label": d.label,
            "agent": d.agent,
            "anchor_skill": d.anchor_skill,
            "keywords": list(get_skill(d.anchor_skill).keywords),
        }
        for d in _ROUTING_DOMAINS
    )
