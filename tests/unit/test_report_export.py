#!/usr/bin/env python
# coding: utf-8
"""Unit tests for report export tools."""

import base64
import re

import pytest

from cs_copilot.storage import S3
from cs_copilot.tools.io.report_export import save_markdown_report, save_rich_report

_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR42mP8z8BQDwAFgwJ"
    "/lHV2qwAAAABJRU5ErkJggg=="
)


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


@pytest.fixture
def stored_png():
    """Write a tiny PNG through the storage layer and return its display path."""
    rel_path = "figures/tiny_map.png"
    with S3.open(rel_path, "wb") as fh:
        fh.write(_TINY_PNG)
    return S3.path(rel_path)


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


def test_save_rich_report_defaults_to_html_and_pdf(local_session_root, stored_png):
    """Rich reports should save readable HTML plus an image-bearing PDF by default."""
    result = save_rich_report(
        title="EGFR GTM Density Report",
        summary=["Dense region appears in the north-east quadrant."],
        sections=[
            {
                "heading": "Density Map",
                "paragraphs": [
                    "The static map below shows projected compounds over the GTM density field."
                ],
                "figures": [
                    {
                        "image_path": stored_png,
                        "name": "GTM density landscape for EGFR compounds",
                        "caption": (
                            "This figure shows the GTM density landscape for EGFR compounds, "
                            "with projected molecules overlaid on dense and sparse regions."
                        ),
                        "artifact_path": "s3://bucket/sessions/test/map.html",
                    }
                ],
            }
        ],
        report_type="gtm_density",
    )

    assert "- HTML: `" in result
    assert "- PDF: `" in result
    assert "- Markdown: `" not in result
    assert re.search(r"reports/gtm_density_\d{8}_\d{6}\.html`", result)
    assert re.search(r"reports/gtm_density_\d{8}_\d{6}\.pdf`", result)

    html_files = list((local_session_root / "reports").glob("gtm_density_*.html"))
    pdf_files = list((local_session_root / "reports").glob("gtm_density_*.pdf"))
    md_files = list((local_session_root / "reports").glob("gtm_density_*.md"))
    assert len(html_files) == 1
    assert len(pdf_files) == 1
    assert not md_files

    html_content = html_files[0].read_text()
    assert "EGFR GTM Density Report" in html_content
    assert "The static map below shows projected compounds" in html_content
    assert "data:image/png;base64," in html_content
    assert "Figure 1. GTM density landscape for EGFR compounds" in html_content
    assert "This figure shows the GTM density landscape for EGFR compounds" in html_content
    assert "s3://bucket/sessions/test/map.html" in html_content
    assert pdf_files[0].read_bytes().startswith(b"%PDF")


def test_save_rich_report_can_emit_markdown_companion(local_session_root, stored_png):
    """When requested, rich reports should also save a Markdown companion."""
    result = save_rich_report(
        title="BRAF Activity Landscape",
        summary=["Activity is concentrated around the lower-left map region."],
        figures=[
            {
                "image_path": stored_png,
                "name": "Figure 9. Activity landscape for BRAF analog generation",
                "caption": (
                    "The Altair activity landscape highlights the map regions used to "
                    "compare generated BRAF analogs against active reference compounds."
                ),
                "artifact_path": "s3://bucket/sessions/test/landscape_plotly.html",
            }
        ],
        filename="../../activity_report.txt",
        report_type="gtm_activity",
        formats=["html", "pdf", "md"],
    )

    reports_dir = local_session_root / "reports"
    assert (reports_dir / "activity_report.html").exists()
    assert (reports_dir / "activity_report.pdf").exists()
    assert (reports_dir / "activity_report.md").exists()
    assert not (local_session_root.parent.parent.parent / "activity_report.txt").exists()
    assert result.endswith("/reports/activity_report.md`")

    markdown_content = (reports_dir / "activity_report.md").read_text()
    assert "### Figure 1. Activity landscape for BRAF analog generation" in markdown_content
    assert "![Figure 1. Activity landscape for BRAF analog generation]" in markdown_content
    assert "The Altair activity landscape highlights" in markdown_content
    assert stored_png in markdown_content
    assert "s3://bucket/sessions/test/landscape_plotly.html" in markdown_content


def test_save_rich_report_numbers_section_and_top_level_figures(local_session_root, stored_png):
    """Figure numbering should be global across section-local and top-level figures."""
    save_rich_report(
        title="Combined GTM Report",
        sections=[
            {
                "heading": "Density",
                "paragraphs": ["Density interpretation."],
                "figures": [
                    {
                        "image_path": stored_png,
                        "caption": "GTM density landscape with projected compounds.",
                    }
                ],
            }
        ],
        figures=[
            {
                "image_path": stored_png,
                "caption": "GTM activity landscape for the same compounds.",
            }
        ],
        filename="combined_report",
        report_type="combined",
        formats=["html", "md"],
    )

    html_content = (local_session_root / "reports" / "combined_report.html").read_text()
    markdown_content = (local_session_root / "reports" / "combined_report.md").read_text()
    assert "Figure 1. GTM density landscape with projected compounds" in html_content
    assert "Figure 2. GTM activity landscape for the same compounds" in html_content
    assert "### Figure 1. GTM density landscape with projected compounds" in markdown_content
    assert "### Figure 2. GTM activity landscape for the same compounds" in markdown_content


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"title": "", "summary": ["x"]}, "title cannot be empty"),
        ({"title": "Empty"}, "report content cannot be empty"),
        ({"title": "Bad format", "summary": ["x"], "formats": ["docx"]}, "unsupported"),
        (
            {"title": "No caption", "figures": [{"image_path": "some_plot.png"}]},
            "figure 1 caption cannot be empty",
        ),
    ],
)
def test_save_rich_report_rejects_invalid_inputs(local_session_root, kwargs, message):
    """Invalid rich-report requests should fail before writing files."""
    with pytest.raises(ValueError, match=message):
        save_rich_report(**kwargs)

    reports_dir = local_session_root / "reports"
    assert not reports_dir.exists() or not list(reports_dir.iterdir())
