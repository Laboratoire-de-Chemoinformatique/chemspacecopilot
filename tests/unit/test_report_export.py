#!/usr/bin/env python
# coding: utf-8
"""Unit tests for save_markdown_report."""

import re

import pytest

from cs_copilot.storage import S3
from cs_copilot.tools.io.report_export import save_markdown_report


@pytest.fixture
def clean_storage_env(monkeypatch):
    """Clear storage-related environment variables so writes go to local storage."""
    for key in (
        "USE_S3",
        "S3_ENDPOINT_URL",
        "MINIO_ENDPOINT",
        "MINIO_ENDPOINT_URL",
        "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "ASSETS_BUCKET",
        "S3_BUCKET_NAME",
        "AWS_REGION",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def fixed_session_prefix():
    """Force a stable session prefix so expected paths stay deterministic."""
    original_prefix = S3.prefix
    S3.prefix = "sessions/test-report"
    try:
        yield
    finally:
        S3.prefix = original_prefix


@pytest.fixture
def local_session_root(clean_storage_env, fixed_session_prefix, monkeypatch, tmp_path):
    """Redirect local writes into tmp_path/data/sessions/test-report/."""
    monkeypatch.chdir(tmp_path)
    return tmp_path / "data" / "sessions" / "test-report"


def test_save_markdown_report_with_explicit_filename(local_session_root):
    """Explicit filename should be honoured under reports/ in the session."""
    content = (
        "# EGFR chemotype report\n\n"
        "Top scaffolds: quinazoline, anilinoquinazoline, pyrrolopyrimidine.\n"
    )
    result = save_markdown_report(
        content=content,
        filename="egfr_scaffolds.md",
        report_type="chemotype",
    )

    expected_path = local_session_root / "reports" / "egfr_scaffolds.md"
    assert expected_path.exists()
    assert expected_path.read_text() == content
    assert result.startswith("Markdown report saved to S3: `")
    assert result.endswith("/reports/egfr_scaffolds.md`")


def test_save_markdown_report_autogenerates_filename(local_session_root):
    """When no filename is given, derive '<report_type>_<UTC_timestamp>.md'."""
    content = "# BRAF activity landscape\n\nKey hotspot: V600E neighbourhood.\n"
    result = save_markdown_report(content=content, report_type="gtm_activity")

    assert re.search(r"reports/gtm_activity_\d{8}_\d{6}\.md`$", result)

    written = list((local_session_root / "reports").glob("gtm_activity_*.md"))
    assert len(written) == 1
    assert written[0].read_text() == content


def test_save_markdown_report_enforces_md_extension(local_session_root):
    """Missing or wrong extensions must be normalised to .md (no double extensions)."""
    save_markdown_report(content="# JAK2", filename="jak2_report", report_type="chemotype")
    save_markdown_report(content="# PDE4A", filename="pde4a_report.txt", report_type="chemotype")
    save_markdown_report(
        content="# PPARG", filename="ppar_gamma_report.md", report_type="chemotype"
    )

    reports_dir = local_session_root / "reports"
    files = sorted(p.name for p in reports_dir.iterdir())
    assert files == ["jak2_report.md", "pde4a_report.md", "ppar_gamma_report.md"]


@pytest.mark.parametrize("bad_content", ["", "   ", "\n\n", "\t  \n"])
def test_save_markdown_report_rejects_empty_content(local_session_root, bad_content):
    """Empty or whitespace-only content must raise."""
    with pytest.raises(ValueError, match="content cannot be empty"):
        save_markdown_report(content=bad_content, filename="x.md", report_type="custom")

    # Confirm nothing was written.
    reports_dir = local_session_root / "reports"
    assert not reports_dir.exists() or not list(reports_dir.iterdir())


def test_save_markdown_report_strips_directory_components(local_session_root):
    """Filenames containing path separators must be reduced to their basename."""
    content = "# Sneaky report\n"
    result = save_markdown_report(
        content=content,
        filename="../../etc/passwd.md",
        report_type="custom",
    )

    safe_path = local_session_root / "reports" / "passwd.md"
    assert safe_path.exists()
    assert safe_path.read_text() == content
    # The unsafe target must NOT have been touched.
    assert not (local_session_root.parent.parent.parent / "etc" / "passwd.md").exists()
    assert result.endswith("/reports/passwd.md`")


def test_save_markdown_report_roundtrip_content(local_session_root):
    """The bytes round-trip exactly through save → read."""
    content = (
        "# Combined GTM density + activity report\n"
        "*Generated 2026-04-15*\n\n"
        "## Targets covered\n"
        "- 5-HT2A receptor\n"
        "- PPARG (peroxisome proliferator-activated receptor gamma)\n\n"
        "## Notes\n"
        "Density and activity overlays agree across the lower-right quadrant.\n"
    )
    result = save_markdown_report(content=content, report_type="combined")

    assert "/reports/combined_" in result
    written = list((local_session_root / "reports").glob("combined_*.md"))
    assert len(written) == 1
    assert written[0].read_text() == content
