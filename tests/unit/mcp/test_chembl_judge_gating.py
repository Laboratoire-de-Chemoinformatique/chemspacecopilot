"""ChEMBL judge gating tests — confirm in-process LLM calls are skipped.

These tests exercise ``ChemblToolkit._filter_suspicious_short_keyword_rows``
directly with the new ``enable_retrieval_judge`` / ``enable_metadata_judge``
flags. We bypass the network/database fetch path by calling the filter
method on hand-crafted DataFrames.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from cs_copilot.tools.databases.chembl import ChemblToolkit


@pytest.fixture
def toolkit():
    # Avoid the real backend probe by skipping __init__ side effects.
    return ChemblToolkit.__new__(ChemblToolkit)


def _suspicious_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_keywords": "CDK",
                "target_pref_name": "Cyclin-dependent kinase 2",
                "target_organism": "Homo sapiens",
                "assay_description": "Inhibition of CDK2",
            }
        ]
    )


def test_retrieval_judge_disabled_status(toolkit):
    with patch.object(toolkit, "_run_chembl_retrieval_judge") as judge:
        result = toolkit._filter_suspicious_short_keyword_rows(
            _suspicious_df(),
            keywords=["CDK"],
            target_query="CDK",
            organism_filter="Homo sapiens",
            query_slug="cdk",
            agent=None,
            session_state=None,
            enable_retrieval_judge=False,
            enable_metadata_judge=False,
        )
    assert judge.call_count == 0
    assert result.summary["judge_status"] == "disabled"
    assert result.summary["metadata_judge_status"] in {"disabled", "not_needed"}


def test_metadata_judge_disabled_status(toolkit):
    df = pd.DataFrame(
        [
            {
                "query_keywords": "long_query_keyword",
                "target_pref_name": "Cyclin-dependent kinase 2",
                "target_organism": "Homo sapiens",
                "assay_description": "Inhibition of CDK2",
            }
        ]
    )
    with patch.object(toolkit, "_run_chembl_metadata_judge") as judge:
        result = toolkit._filter_suspicious_short_keyword_rows(
            df,
            keywords=["long_query_keyword"],
            target_query="CDK",
            organism_filter="Homo sapiens",
            query_slug="cdk",
            agent=None,
            session_state=None,
            enable_retrieval_judge=True,
            enable_metadata_judge=False,
        )
    assert judge.call_count == 0
    assert result.summary["metadata_judge_status"] == "disabled"


def test_default_behavior_invokes_retrieval_judge(toolkit):
    with (
        patch.object(toolkit, "_run_chembl_retrieval_judge", return_value={}) as r,
        patch.object(toolkit, "_run_chembl_metadata_judge", return_value={}) as m,
    ):
        toolkit._filter_suspicious_short_keyword_rows(
            _suspicious_df(),
            keywords=["CDK"],
            target_query="CDK",
            organism_filter="Homo sapiens",
            query_slug="cdk",
            agent=None,
            session_state=None,
        )
    # Suspicious short-keyword rows reach the retrieval judge by default.
    assert r.call_count == 1
    # The metadata judge runs only on rows that survive the retrieval pass; in
    # this fixture all rows are filtered out, so the metadata judge is skipped.
    # That is the correct behavior — see chembl.py:_filter_suspicious_short_keyword_rows.
    assert m.call_count == 0


def test_default_behavior_invokes_metadata_judge(toolkit):
    df = pd.DataFrame(
        [
            {
                "query_keywords": "long_query_keyword",
                "target_pref_name": "Cyclin-dependent kinase 2",
                "target_organism": "Homo sapiens",
                "assay_description": "Inhibition of CDK2",
            }
        ]
    )
    with patch.object(toolkit, "_run_chembl_metadata_judge", return_value={}) as m:
        toolkit._filter_suspicious_short_keyword_rows(
            df,
            keywords=["long_query_keyword"],
            target_query="CDK",
            organism_filter="Homo sapiens",
            query_slug="cdk",
            agent=None,
            session_state=None,
        )
    # Non-suspicious row with populated metadata reaches the metadata judge.
    assert m.call_count == 1
