#!/usr/bin/env python
# coding: utf-8
"""Unit tests for report export tools."""

import base64
import re

import pytest

from cs_copilot.storage import S3
from cs_copilot.tools.io.report_export import (
    _pdf_inline_markup,
    save_markdown_report,
    save_rich_report,
)

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


def _report_dir(local_session_root, report_type: str):
    matches = list(local_session_root.glob(f"workflows/*/reports/{report_type}"))
    assert len(matches) == 1
    return matches[0]


def _report_files(local_session_root, report_type: str, pattern: str):
    return list(local_session_root.glob(f"workflows/*/reports/{report_type}/{pattern}"))


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

    expected_path = _report_dir(local_session_root, "chemotype") / "egfr_scaffolds.md"
    assert expected_path.exists()
    assert expected_path.read_text() == content
    assert result.startswith("Markdown report saved to S3: `")
    assert result.endswith("/reports/chemotype/egfr_scaffolds.md`")


def test_save_markdown_report_autogenerates_filename(local_session_root):
    """When no filename is given, derive '<report_type>_<UTC_timestamp>.md'."""
    content = "# BRAF activity landscape\n\nKey hotspot: V600E neighbourhood.\n"
    result = save_markdown_report(content=content, report_type="gtm_activity")

    assert re.search(r"reports/gtm_activity/gtm_activity_\d{8}_\d{6}\.md`$", result)

    written = _report_files(local_session_root, "gtm_activity", "gtm_activity_*.md")
    assert len(written) == 1
    assert written[0].read_text() == content


def test_save_markdown_report_enforces_md_extension(local_session_root):
    """Missing or wrong extensions must be normalised to .md (no double extensions)."""
    save_markdown_report(content="# JAK2", filename="jak2_report", report_type="chemotype")
    save_markdown_report(content="# PDE4A", filename="pde4a_report.txt", report_type="chemotype")
    save_markdown_report(
        content="# PPARG", filename="ppar_gamma_report.md", report_type="chemotype"
    )

    reports_dir = _report_dir(local_session_root, "chemotype")
    files = sorted(p.name for p in reports_dir.iterdir())
    assert files == ["jak2_report.md", "pde4a_report.md", "ppar_gamma_report.md"]


@pytest.mark.parametrize("bad_content", ["", "   ", "\n\n", "\t  \n"])
def test_save_markdown_report_rejects_empty_content(local_session_root, bad_content):
    """Empty or whitespace-only content must raise."""
    with pytest.raises(ValueError, match="content cannot be empty"):
        save_markdown_report(content=bad_content, filename="x.md", report_type="custom")

    # Confirm nothing was written.
    assert not _report_files(local_session_root, "custom", "*")


def test_save_markdown_report_strips_directory_components(local_session_root):
    """Filenames containing path separators must be reduced to their basename."""
    content = "# Sneaky report\n"
    result = save_markdown_report(
        content=content,
        filename="../../etc/passwd.md",
        report_type="custom",
    )

    safe_path = _report_dir(local_session_root, "custom") / "passwd.md"
    assert safe_path.exists()
    assert safe_path.read_text() == content
    # The unsafe target must NOT have been touched.
    assert not (local_session_root.parent.parent.parent / "etc" / "passwd.md").exists()
    assert result.endswith("/reports/custom/passwd.md`")


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

    assert "/reports/combined/combined_" in result
    written = _report_files(local_session_root, "combined", "combined_*.md")
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
    assert re.search(r"reports/gtm_density/gtm_density_\d{8}_\d{6}\.html`", result)
    assert re.search(r"reports/gtm_density/gtm_density_\d{8}_\d{6}\.pdf`", result)

    html_files = _report_files(local_session_root, "gtm_density", "gtm_density_*.html")
    pdf_files = _report_files(local_session_root, "gtm_density", "gtm_density_*.pdf")
    md_files = _report_files(local_session_root, "gtm_density", "gtm_density_*.md")
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


def test_save_rich_report_renders_markdown_bold_in_rich_outputs(local_session_root, stored_png):
    """Markdown-style bold should render as rich text instead of literal asterisks."""
    assert _pdf_inline_markup("Potency is **high** & selective") == (
        "Potency is <b>high</b> &amp; selective"
    )

    save_rich_report(
        title="Bold **SAR** Report",
        summary=["**High activity** is concentrated in node 340."],
        sections=[
            {
                "heading": "Activity **Highlights**",
                "paragraphs": [
                    "The **top potency** analog is separated from the dense region.",
                ],
                "figures": [
                    {
                        "image_path": stored_png,
                        "caption": "The map highlights **high-activity** nodes.",
                    }
                ],
                "tables": [
                    {
                        "title": "Bold **Structure** Table",
                        "columns": ["Name", "Description"],
                        "rows": [
                            {
                                "Name": "**Molecule-1**",
                                "Description": "Representative **high-potency** analog.",
                            }
                        ],
                    }
                ],
            }
        ],
        filename="bold_report",
        report_type="chemotype",
        formats=["html", "pdf", "md"],
    )

    reports_dir = _report_dir(local_session_root, "chemotype")
    html_content = (reports_dir / "bold_report.html").read_text()
    assert "<strong>High activity</strong>" in html_content
    assert "<strong>top potency</strong>" in html_content
    assert "high-activity" in html_content
    assert "<strong>Molecule-1</strong>" in html_content
    assert "**High activity**" not in html_content
    assert "**top potency**" not in html_content
    assert (reports_dir / "bold_report.pdf").read_bytes().startswith(b"%PDF")
    assert "**High activity** is concentrated" in (reports_dir / "bold_report.md").read_text()


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

    reports_dir = _report_dir(local_session_root, "gtm_activity")
    assert (reports_dir / "activity_report.html").exists()
    assert (reports_dir / "activity_report.pdf").exists()
    assert (reports_dir / "activity_report.md").exists()
    assert not (local_session_root.parent.parent.parent / "activity_report.txt").exists()
    assert result.endswith("/reports/gtm_activity/activity_report.md`")

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

    reports_dir = _report_dir(local_session_root, "combined")
    html_content = (reports_dir / "combined_report.html").read_text()
    markdown_content = (reports_dir / "combined_report.md").read_text()
    assert "Figure 1. GTM density landscape with projected compounds" in html_content
    assert "Figure 2. GTM activity landscape for the same compounds" in html_content
    assert "### Figure 1. GTM density landscape with projected compounds" in markdown_content
    assert "### Figure 2. GTM activity landscape for the same compounds" in markdown_content


def test_save_rich_report_generates_section_structure_figures(local_session_root):
    """Structure SMILES figures should become report-local PNG assets."""
    save_rich_report(
        title="sEH Scaffold SAR",
        sections=[
            {
                "heading": "Scaffold SAR",
                "paragraphs": [
                    "The benzene scaffold is a compact example used to explain SAR context."
                ],
                "figures": [
                    {
                        "structure_type": "scaffold",
                        "structure_id": "Scaffold-1",
                        "structure_name": "Benzene scaffold",
                        "structure_smiles": "c1ccccc1",
                        "caption": (
                            "Representative scaffold structure used as a compact SAR "
                            "example in the report."
                        ),
                    }
                ],
            }
        ],
        filename="scaffold_sar",
        report_type="chemotype",
        formats=["html", "pdf", "md"],
    )

    reports_dir = _report_dir(local_session_root, "chemotype")
    structure_files = list((reports_dir / "assets" / "structures").glob("*.png"))
    assert len(structure_files) == 1
    assert structure_files[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (reports_dir / "scaffold_sar.pdf").read_bytes().startswith(b"%PDF")

    html_content = (reports_dir / "scaffold_sar.html").read_text()
    markdown_content = (reports_dir / "scaffold_sar.md").read_text()
    expected_fragment = f"/reports/chemotype/assets/structures/{structure_files[0].name}"
    assert "Figure 1. Scaffold-1: Benzene scaffold" in html_content
    assert "data:image/png;base64," in html_content
    assert "### Figure 1. Scaffold-1: Benzene scaffold" in markdown_content
    assert expected_fragment in markdown_content


def test_save_rich_report_places_structure_figures_and_tables(local_session_root):
    """Structure figures should support IDs, names, nodes, placement, and inventory tables."""
    save_rich_report(
        title="Named Structure Report",
        sections=[
            {
                "heading": "SAR Highlights",
                "paragraphs": [
                    "Scaffold-1 anchors the dense GTM region around node 221.",
                    "Molecule-1 is the top potency analog in node 340.",
                ],
                "figures": [
                    {
                        "structure_type": "scaffold",
                        "structure_id": "Scaffold-1",
                        "structure_name": "Piperidine urea phenyl scaffold",
                        "structure_smiles": "O=C(Nc1ccccc1)NC1CCNCC1",
                        "node": "221",
                        "description": "Dominant scaffold in the dense map region.",
                        "caption": "Scaffold-1 is the dominant piperidine urea phenyl scaffold.",
                        "after_paragraph_index": 0,
                    },
                    {
                        "structure_type": "molecule",
                        "structure_id": "Molecule-1",
                        "structure_name": "Top potency piperidine urea analog",
                        "structure_smiles": "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(OC(F)(F)F)cc3)cc2)ccn1",
                        "node": "340",
                        "description": "Representative high-potency molecule.",
                        "caption": "Molecule-1 is the representative top potency analog.",
                        "after_paragraph_index": 1,
                    },
                ],
                "tables": [
                    {
                        "title": "Scaffold Inventory",
                        "columns": [
                            "Scaffold ID",
                            "Scaffold",
                            "SMILES",
                            "Name",
                            "Node",
                            "Description",
                        ],
                        "rows": [
                            {
                                "Scaffold ID": "Scaffold-1",
                                "Scaffold": "Piperidine urea phenyl scaffold",
                                "SMILES": "O=C(Nc1ccccc1)NC1CCNCC1",
                                "Name": "Piperidine urea phenyl scaffold",
                                "Node": "221",
                                "Description": "Dominant scaffold in the dense map region.",
                            }
                        ],
                    },
                    {
                        "title": "Molecule Inventory",
                        "columns": [
                            "Molecule ID",
                            "Molecule",
                            "SMILES",
                            "Name",
                            "Node",
                            "Description",
                        ],
                        "rows": [
                            {
                                "Molecule ID": "Molecule-1",
                                "Molecule": "Top potency piperidine urea analog",
                                "SMILES": "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(OC(F)(F)F)cc3)cc2)ccn1",
                                "Name": "Top potency piperidine urea analog",
                                "Node": "340",
                                "Description": "Representative high-potency molecule.",
                            }
                        ],
                    },
                ],
            }
        ],
        filename="named_structures",
        report_type="chemotype",
        formats=["html", "pdf", "md"],
    )

    reports_dir = _report_dir(local_session_root, "chemotype")
    assert len(list((reports_dir / "assets" / "structures").glob("*.png"))) == 2
    assert (reports_dir / "named_structures.pdf").read_bytes().startswith(b"%PDF")

    html_content = (reports_dir / "named_structures.html").read_text()
    markdown_content = (reports_dir / "named_structures.md").read_text()
    scaffold_paragraph_index = html_content.index(
        "Scaffold-1 anchors the dense GTM region around node 221."
    )
    scaffold_figure_index = html_content.index(
        "Figure 1. Scaffold-1: Piperidine urea phenyl scaffold"
    )
    molecule_paragraph_index = html_content.index("Molecule-1 is the top potency analog")
    molecule_figure_index = html_content.index(
        "Figure 2. Molecule-1: Top potency piperidine urea analog"
    )
    table_index = html_content.index("Scaffold Inventory")
    assert scaffold_paragraph_index < scaffold_figure_index < molecule_paragraph_index
    assert molecule_paragraph_index < molecule_figure_index < table_index
    assert "<th>Scaffold ID</th>" in html_content
    assert "<td>Scaffold-1</td>" in html_content
    assert "<th>Molecule ID</th>" in html_content
    assert "<td>Molecule-1</td>" in html_content
    assert "| Scaffold ID | Scaffold | SMILES | Name | Node | Description |" in markdown_content
    assert "| Molecule ID | Molecule | SMILES | Name | Node | Description |" in markdown_content
    assert "### Figure 1. Scaffold-1: Piperidine urea phenyl scaffold" in markdown_content
    assert "### Figure 2. Molecule-1: Top potency piperidine urea analog" in markdown_content


def test_save_rich_report_generates_structure_figures_from_smiles_tags(local_session_root):
    """SMILES tags in report paragraphs should create compound image figures."""
    smiles = "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(OC(F)(F)F)cc3)cc2)ccn1"
    save_rich_report(
        title="Tagged Compound Report",
        sections=[
            {
                "heading": "Representative Compound",
                "paragraphs": [
                    (
                        "The report highlights <smiles>"
                        f"{smiles}"
                        "</smiles> as a representative compound."
                    )
                ],
            }
        ],
        filename="tagged_compound",
        report_type="chemotype",
        formats=["html", "pdf", "md"],
    )

    reports_dir = _report_dir(local_session_root, "chemotype")
    structure_files = list((reports_dir / "assets" / "structures").glob("*.png"))
    assert len(structure_files) == 1
    assert structure_files[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (reports_dir / "tagged_compound.pdf").read_bytes().startswith(b"%PDF")

    html_content = (reports_dir / "tagged_compound.html").read_text()
    markdown_content = (reports_dir / "tagged_compound.md").read_text()
    expected_fragment = f"/reports/chemotype/assets/structures/{structure_files[0].name}"
    assert "&lt;smiles&gt;" not in html_content
    assert f"The report highlights {smiles} as a representative compound." in html_content
    assert "Figure 1. Molecule-1: Reported compound structure" in html_content
    assert "data:image/png;base64," in html_content
    assert "<smiles>" not in markdown_content
    assert f"The report highlights {smiles} as a representative compound." in markdown_content
    assert expected_fragment in markdown_content


def test_save_rich_report_places_multiple_smiles_tag_figures_after_first_mentions(
    local_session_root,
):
    """Auto-generated SMILES figures should appear after the paragraph that introduced them."""
    first_smiles = "CCO"
    second_smiles = "c1ccccc1"
    save_rich_report(
        title="Tagged Molecule Placement",
        sections=[
            {
                "heading": "Compound Examples",
                "paragraphs": [
                    f"First mention introduces <smiles>{first_smiles}</smiles>.",
                    f"Second mention introduces <smiles>{second_smiles}</smiles>.",
                ],
            }
        ],
        filename="tagged_placement",
        report_type="chemotype",
        formats=["html", "md"],
    )

    reports_dir = _report_dir(local_session_root, "chemotype")
    assert len(list((reports_dir / "assets" / "structures").glob("*.png"))) == 2

    html_content = (reports_dir / "tagged_placement.html").read_text()
    markdown_content = (reports_dir / "tagged_placement.md").read_text()
    first_paragraph_index = html_content.index(f"First mention introduces {first_smiles}.")
    first_figure_index = html_content.index("Figure 1. Molecule-1: Reported compound structure")
    second_paragraph_index = html_content.index(f"Second mention introduces {second_smiles}.")
    second_figure_index = html_content.index("Figure 2. Molecule-2: Reported compound structure")
    assert first_paragraph_index < first_figure_index < second_paragraph_index
    assert second_paragraph_index < second_figure_index
    assert markdown_content.index(
        f"First mention introduces {first_smiles}."
    ) < markdown_content.index("### Figure 1. Molecule-1: Reported compound structure")
    assert markdown_content.index(
        f"Second mention introduces {second_smiles}."
    ) < markdown_content.index("### Figure 2. Molecule-2: Reported compound structure")


def test_save_rich_report_rejects_invalid_structure_smiles(local_session_root):
    """Invalid structure SMILES should fail before report files are written."""
    with pytest.raises(ValueError, match="figure 1 has invalid structure SMILES"):
        save_rich_report(
            title="Invalid Structure Report",
            sections=[
                {
                    "heading": "Scaffolds",
                    "paragraphs": ["Invalid structure example."],
                    "figures": [
                        {
                            "structure_smiles": "not a smiles",
                            "caption": "This invalid structure should not be rendered.",
                        }
                    ],
                }
            ],
            report_type="chemotype",
            formats=["html"],
        )

    assert not _report_files(local_session_root, "chemotype", "*")


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

    assert not _report_files(local_session_root, "report", "*")
