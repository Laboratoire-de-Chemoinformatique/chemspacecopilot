"""Tests for the single-source request router (``cs_copilot.routing``)."""

from __future__ import annotations

from cs_copilot.routing import match_request, routing_domains
from cs_copilot.skills import list_skills, search_skills
from cs_copilot.workflows import list_workflows, search_workflows


def _wf(result):
    return result.workflow.slug if result.workflow else None


# --- explicit workflow slug ------------------------------------------------


def test_explicit_workflow_slug_wins():
    result = match_request("use this data", workflow_slug="dataset-normalization")
    assert _wf(result) == "dataset-normalization"
    assert result.workflow_error is None
    # The matched workflow's skill twin leads the skill list.
    assert result.skills[0] == "dataset-normalization"


def test_unknown_workflow_slug_returns_error():
    result = match_request("use this data", workflow_slug="no-such-workflow")
    assert result.workflow is None
    assert result.workflow_error
    # Falls back to free-text skill search when no workflow resolved.
    assert result.skills


# --- domain routing --------------------------------------------------------


def test_chembl_request_selects_target_retrieval_and_preflight():
    result = match_request("Fetch ChEMBL EGFR binding data, any mechanism")
    assert _wf(result) == "chembl-target-retrieval"
    assert result.preflight_tools == ("chembl_prepare_retrieval",)


def test_chembl_plus_gtm_report_selects_combined_workflow():
    result = match_request("retrieve chembl data, build a gtm map and write a report")
    assert _wf(result) == "chembl-to-gtm-report"
    assert "chembl_prepare_retrieval" in result.preflight_tools
    assert "chemspace_plan_analysis" in result.preflight_tools


def test_gtm_request_selects_landscape_and_chemspace_preflight():
    result = match_request("Create a GTM activity landscape for the current clean dataset")
    assert _wf(result) == "gtm-activity-landscape"
    assert result.preflight_tools == ("chemspace_plan_analysis",)


def test_gtm_density_request_selects_density_landscape_and_preflight():
    result = match_request("Show the GTM density map for the current clean dataset")
    assert _wf(result) == "gtm-density-landscape"
    assert result.skills == ("gtm-density-landscape",)
    assert result.preflight_tools == ("chemspace_plan_analysis",)


def test_gtm_density_and_activity_request_fetches_both_skills():
    result = match_request("Show the density map and create an activity landscape")
    assert _wf(result) == "gtm-density-landscape"
    assert result.skills == ("gtm-density-landscape", "gtm-activity-landscape")
    assert result.preflight_tools == ("chemspace_plan_analysis",)


# --- tie-breaks ------------------------------------------------------------


def test_peptide_design_suppresses_projection_workflow():
    result = match_request("Design antimicrobial peptide candidates and rank them")
    assert result.workflow is None  # candidate-design-to-gtm needs projection terms
    assert "peptide-design" in result.skills
    assert any("candidate-design-to-gtm" in tb for tb in result.tie_breaks_applied)


def test_peptide_excludes_molecule_skill():
    result = match_request("design peptides from amino acid building blocks")
    assert "peptide-design" in result.skills
    assert "molecular-design" not in result.skills  # mutual exclusion


def test_unqualified_generate_routes_to_molecule_not_peptide():
    result = match_request("generate some new structures and rank them")
    assert "molecular-design" in result.skills
    assert "peptide-design" not in result.skills


def test_design_with_projection_reaches_candidate_workflow():
    result = match_request("design candidates and project them onto the gtm map")
    assert _wf(result) == "candidate-design-to-gtm"


def test_weak_match_is_rejected():
    # "compounds" only incidentally appears inside the tool name
    # chembl_fetch_compounds; it must not select a workflow or skill.
    result = match_request("help me analyze compounds")
    assert result.workflow is None
    assert result.skills == ()
    assert result.preflight_tools == ("chemspace_plan_analysis",)
    assert any("rejected weak" in tb for tb in result.tie_breaks_applied)


def test_for_stopword_does_not_mismatch_retrosynthesis():
    # Regression: the old slug-overlap gate matched "for" against
    # "retrosynthesis-for-candidates". Whole-word, signal-based matching must not.
    result = match_request("generate analogs for this molecule")
    assert _wf(result) != "retrosynthesis-for-candidates"
    assert "molecular-design" in result.skills


def test_preflight_present_suppresses_fallback_skill_search():
    # A chemspace preflight gate frames the request, so we do not also run a
    # speculative free-text skill search.
    result = match_request("help me analyze compounds")
    assert result.preflight_tools  # chemspace gate fired
    assert result.skills == ()


def test_empty_request_routes_nowhere():
    result = match_request("")
    assert result.workflow is None
    assert result.skills == ()
    assert result.preflight_tools == ()


# --- catalog reachability --------------------------------------------------


def test_every_workflow_keyword_selects_its_workflow():
    workflows = list_workflows()
    limit = len(workflows)
    for spec in workflows:
        for keyword in spec.keywords:
            hits = {w.slug for w in search_workflows(keyword, limit=limit)}
            assert spec.slug in hits, f"{spec.slug} not selected by its keyword {keyword!r}"


def test_every_skill_keyword_selects_its_skill():
    skills = list_skills()
    limit = len(skills)
    for spec in skills:
        for keyword in spec.keywords:
            hits = {s.slug for s in search_skills(keyword, limit=limit)}
            assert spec.slug in hits, f"{spec.slug} not selected by its keyword {keyword!r}"


def test_multiword_keyword_matches_as_phrase():
    # "amino acid" is a multi-word keyword the per-token scorer cannot match.
    hits = {s.slug for s in search_skills("work with amino acids", limit=len(list_skills()))}
    assert "peptide-design" in hits


def test_routing_domains_anchor_real_skills():
    skill_slugs = {s.slug for s in list_skills()}
    for domain in routing_domains():
        assert domain["anchor_skill"] in skill_slugs
        assert domain["keywords"], f"{domain['anchor_skill']} has no keywords"
