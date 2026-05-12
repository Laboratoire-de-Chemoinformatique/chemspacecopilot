#!/usr/bin/env python
# coding: utf-8
"""Unit tests for report export tools."""

import base64
import re

import pytest

from cs_copilot.storage import S3
from cs_copilot.tools.io.report_export import (
    _html_inline_markup,
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
    assert "s3://bucket/sessions/test/map.html" not in html_content
    assert pdf_files[0].read_bytes().startswith(b"%PDF")


def test_save_rich_report_renders_markdown_bold_in_rich_outputs(local_session_root, stored_png):
    """Markdown-style bold should render as rich text instead of literal asterisks."""
    assert _pdf_inline_markup("Potency is **high** & selective") == (
        "Potency is <b>high</b> &amp; selective"
    )
    assert _html_inline_markup(
        "<strong>High activity</strong> and <string>selectivity</string> <script>x</script>"
    ) == (
        "<strong>High activity</strong> and <strong>selectivity</strong> "
        "&lt;script&gt;x&lt;/script&gt;"
    )
    assert (
        _pdf_inline_markup("<strong>High activity</strong> and <string>selectivity</string>")
        == "<b>High activity</b> and <b>selectivity</b>"
    )

    save_rich_report(
        title="Bold **SAR** Report",
        summary=["**High activity** is concentrated in node 340."],
        sections=[
            {
                "heading": "Activity **Highlights**",
                "paragraphs": [
                    "The **top potency** analog is separated from the dense region. "
                    "<strong>Node 340</strong> and <string>Node 217</string> are highlighted.",
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
                                "Name": "**Molecule_1**",
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
    assert "<strong>Node 340</strong>" in html_content
    assert "<strong>Node 217</strong>" in html_content
    assert "&lt;strong&gt;Node 340&lt;/strong&gt;" not in html_content
    assert "&lt;string&gt;Node 217&lt;/string&gt;" not in html_content
    assert "high-activity" in html_content
    assert "<strong>Molecule_1</strong>" in html_content
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
    assert "s3://bucket/sessions/test/landscape_plotly.html" not in markdown_content


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


def test_save_rich_report_moves_gtm_landscapes_into_matching_sections(
    local_session_root, stored_png
):
    """Top-level GTM landscapes should render beside density/activity analysis when possible."""
    save_rich_report(
        title="Section-Local GTM Report",
        sections=[
            {
                "heading": "GTM Density Analysis",
                "paragraphs": ["Density analysis identifies Node 221 as the main hotspot."],
            },
            {
                "heading": "GTM Activity Analysis",
                "paragraphs": [
                    "Activity analysis identifies Node 340 as the highest-potency area."
                ],
            },
        ],
        figures=[
            {
                "image_path": stored_png,
                "caption": "GTM density landscape with Node 221 labeled.",
                "name": "GTM density landscape for source compounds",
            },
            {
                "image_path": stored_png,
                "caption": "GTM activity landscape with Node 340 labeled.",
                "name": "GTM activity landscape for source compounds",
            },
        ],
        filename="section_local_gtm",
        report_type="combined",
        formats=["html", "md"],
    )

    reports_dir = _report_dir(local_session_root, "combined")
    html_content = (reports_dir / "section_local_gtm.html").read_text()
    markdown_content = (reports_dir / "section_local_gtm.md").read_text()
    density_section_index = html_content.index("<h2>GTM Density Analysis</h2>")
    density_figure_index = html_content.index("Figure 1. GTM density landscape")
    activity_section_index = html_content.index("<h2>GTM Activity Analysis</h2>")
    activity_figure_index = html_content.index("Figure 2. GTM activity landscape")
    assert density_section_index < density_figure_index < activity_section_index
    assert activity_section_index < activity_figure_index
    assert "<h2>Visualizations</h2>" not in html_content
    assert "## Visualizations" not in markdown_content


def test_save_rich_report_omits_plotly_gtm_outputs(local_session_root, stored_png):
    """Report figures should keep static PNGs and omit Plotly images/artifact links."""
    save_rich_report(
        title="PNG Only GTM Report",
        sections=[
            {
                "heading": "GTM Activity Analysis",
                "paragraphs": ["Activity analysis identifies Node 340 as the key area."],
            }
        ],
        figures=[
            {
                "image_path": stored_png,
                "caption": (
                    "GTM activity landscape with Node 340 labeled. "
                    "The interactive Plotly version supports drill-down."
                ),
                "name": "GTM activity landscape Altair PNG",
                "artifact_path": "s3://bucket/plots/activity_plotly_regression.html",
            },
            {
                "image_path": "s3://bucket/plots/activity_plotly_regression.png",
                "caption": "Plotly smooth GTM activity landscape.",
                "name": "Plotly activity landscape",
            },
        ],
        filename="png_only_gtm",
        report_type="combined",
        formats=["html", "md"],
    )

    reports_dir = _report_dir(local_session_root, "combined")
    html_content = (reports_dir / "png_only_gtm.html").read_text()
    markdown_content = (reports_dir / "png_only_gtm.md").read_text()
    assert "Figure 1. GTM activity landscape Altair PNG" in html_content
    assert "data:image/png;base64," in html_content
    assert "activity_plotly_regression.html" not in html_content
    assert "activity_plotly_regression.png" not in html_content
    assert "interactive Plotly version" not in html_content
    assert "Plotly activity landscape" not in html_content
    assert "activity_plotly_regression.html" not in markdown_content
    assert "activity_plotly_regression.png" not in markdown_content


def test_save_rich_report_corrects_density_caption_colors(local_session_root, stored_png):
    """Density captions should not claim a red/orange-vs-blue scale for grayscale plots."""
    save_rich_report(
        title="Density Caption Report",
        sections=[
            {
                "heading": "GTM Density Analysis",
                "paragraphs": ["Density analysis identifies Node 221 as the densest node."],
                "figures": [
                    {
                        "image_path": stored_png,
                        "caption": (
                            "Color intensity represents compound density per node, with "
                            "red/orange indicating highly populated regions and blue indicating "
                            "sparse regions."
                        ),
                        "name": "GTM density landscape",
                    }
                ],
            }
        ],
        filename="density_caption",
        report_type="gtm_density",
        formats=["html", "md"],
    )

    reports_dir = _report_dir(local_session_root, "gtm_density")
    html_content = (reports_dir / "density_caption.html").read_text()
    markdown_content = (reports_dir / "density_caption.md").read_text()
    assert "red/orange" not in html_content
    assert "blue indicating sparse" not in html_content
    assert "grayscale legend" in html_content
    assert "darker cells indicate higher compound density" in html_content
    assert "red/orange" not in markdown_content
    assert "grayscale legend" in markdown_content


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
                        "structure_id": "Scaffold_1",
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
    assert "Figure 1. Scaffold_1: Benzene scaffold" in html_content
    assert "data:image/png;base64," in html_content
    assert "### Figure 1. Scaffold_1: Benzene scaffold" in markdown_content
    assert expected_fragment in markdown_content


def test_save_rich_report_references_top_level_structure_figures(local_session_root):
    """Top-level structure figures should also get text references in HTML and Markdown."""
    save_rich_report(
        title="Top-Level Structure Report",
        summary=["CHEMBL999999 is the reference molecule for comparison."],
        figures=[
            {
                "structure_type": "molecule",
                "structure_id": "CHEMBL999999",
                "structure_name": "Reference ChEMBL molecule",
                "structure_smiles": "CCOc1ccc(NC(=O)NCC)cc1",
                "caption": "Reference compound structure with an existing ChEMBL identifier.",
            }
        ],
        filename="top_level_structure",
        report_type="chemotype",
        formats=["html", "md"],
    )

    reports_dir = _report_dir(local_session_root, "chemotype")
    html_content = (reports_dir / "top_level_structure.html").read_text()
    markdown_content = (reports_dir / "top_level_structure.md").read_text()
    assert "Figure 1. CHEMBL999999: Reference ChEMBL molecule" in html_content
    assert "CHEMBL999999 (Reference ChEMBL molecule) is shown in Figure 1." in html_content
    assert "### Figure 1. CHEMBL999999: Reference ChEMBL molecule" in markdown_content
    assert "CHEMBL999999 (Reference ChEMBL molecule) is shown in Figure 1." in markdown_content


def test_save_rich_report_places_structure_figures_and_tables(local_session_root):
    """Structure figures should support IDs, names, nodes, placement, and inventory tables."""
    save_rich_report(
        title="Named Structure Report",
        sections=[
            {
                "heading": "SAR Highlights",
                "paragraphs": [
                    "Scaffold_1 anchors the dense GTM region around node 221.",
                    "Molecule_1 is the top potency analog in node 340.",
                ],
                "figures": [
                    {
                        "structure_type": "scaffold",
                        "structure_id": "Scaffold_1",
                        "structure_name": "Piperidine urea phenyl scaffold",
                        "structure_smiles": "O=C(Nc1ccccc1)NC1CCNCC1",
                        "node": "221",
                        "description": "Dominant scaffold in the dense map region.",
                        "caption": "Scaffold_1 is the dominant piperidine urea phenyl scaffold.",
                        "after_paragraph_index": 0,
                    },
                    {
                        "structure_type": "molecule",
                        "structure_id": "Molecule_1",
                        "structure_name": "Top potency piperidine urea analog",
                        "structure_smiles": "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(OC(F)(F)F)cc3)cc2)ccn1",
                        "node": "340",
                        "description": "Representative high-potency molecule.",
                        "caption": "Molecule_1 is the representative top potency analog.",
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
                                "Scaffold ID": "Scaffold_1",
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
                                "Molecule ID": "Molecule_1",
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
        "Scaffold_1 anchors the dense GTM region around node 221."
    )
    scaffold_figure_index = html_content.index(
        "Figure 1. Scaffold_1: Piperidine urea phenyl scaffold"
    )
    molecule_paragraph_index = html_content.index("Molecule_1 is the top potency analog")
    molecule_figure_index = html_content.index(
        "Figure 2. Molecule_1: Top potency piperidine urea analog"
    )
    table_index = html_content.index("<h3>Scaffold Inventory</h3>")
    assert scaffold_paragraph_index < scaffold_figure_index < molecule_paragraph_index
    assert molecule_paragraph_index < molecule_figure_index < table_index
    assert "<th>Scaffold ID</th>" in html_content
    assert "<td>Scaffold_1</td>" in html_content
    assert "<th>Molecule ID</th>" in html_content
    assert "<td>Molecule_1</td>" in html_content
    assert "| Scaffold ID | Scaffold | SMILES | Name | Node | Description |" in markdown_content
    assert "| Molecule ID | Molecule | SMILES | Name | Node | Description |" in markdown_content
    assert "Scaffold_1 (Piperidine urea phenyl scaffold) is shown in Figure 1." in html_content
    assert "Molecule_1 (Top potency piperidine urea analog) is shown in Figure 2." in html_content
    assert "Scaffold_1 (Piperidine urea phenyl scaffold) is shown in Figure 1." in markdown_content
    assert (
        "Molecule_1 (Top potency piperidine urea analog) is shown in Figure 2." in markdown_content
    )
    assert "### Figure 1. Scaffold_1: Piperidine urea phenyl scaffold" in markdown_content
    assert "### Figure 2. Molecule_1: Top potency piperidine urea analog" in markdown_content


def test_save_rich_report_visualizes_scaffold_inventory_rows(local_session_root):
    """Scaffold inventory rows should generate scaffold images without explicit figures."""
    save_rich_report(
        title="Scaffold Inventory Report",
        sections=[
            {
                "heading": "Scaffold Summary",
                "paragraphs": [
                    "Scaffold_1 is a representative adamantane urea phenyl scaffold in node 321.",
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
                                "Scaffold ID": "Scaffold_1",
                                "Scaffold": "Adamantane urea phenyl scaffold",
                                "SMILES": "O=C(Nc1ccccc1)NC12CC3CC(CC(C3)C1)C2",
                                "Name": "Adamantane urea phenyl scaffold",
                                "Node": "321",
                                "Description": "High-potency peripheral scaffold.",
                            }
                        ],
                    }
                ],
            }
        ],
        filename="scaffold_inventory_only",
        report_type="chemotype",
        formats=["html", "pdf", "md"],
    )

    reports_dir = _report_dir(local_session_root, "chemotype")
    structure_files = list((reports_dir / "assets" / "structures").glob("*.png"))
    assert len(structure_files) == 1
    assert structure_files[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (reports_dir / "scaffold_inventory_only.pdf").read_bytes().startswith(b"%PDF")

    html_content = (reports_dir / "scaffold_inventory_only.html").read_text()
    markdown_content = (reports_dir / "scaffold_inventory_only.md").read_text()
    paragraph_index = html_content.index("Scaffold_1 is a representative")
    figure_index = html_content.index("Figure 1. Scaffold_1: Adamantane urea phenyl scaffold")
    table_index = html_content.index("<h3>Scaffold Inventory</h3>")
    assert paragraph_index < figure_index < table_index
    assert "data:image/png;base64," in html_content
    assert "Scaffold_1 (Adamantane urea phenyl scaffold) is shown in Figure 1." in html_content
    assert "Scaffold_1 (Adamantane urea phenyl scaffold) is shown in Figure 1." in markdown_content
    assert "### Figure 1. Scaffold_1: Adamantane urea phenyl scaffold" in markdown_content


def test_save_rich_report_generates_structure_figures_from_plain_scaffold_smiles(
    local_session_root,
):
    """Discussed untagged scaffold SMILES should still produce structure figures."""
    save_rich_report(
        title="Plain Scaffold SMILES Report",
        sections=[
            {
                "heading": "Scaffold Analysis",
                "paragraphs": [
                    (
                        "- Piperidine urea scaffold (40 compounds): "
                        "O=C(Nc1ccccc1)NC1CCNCC1 - dominant dense-node scaffold."
                    )
                ],
            }
        ],
        filename="plain_scaffold_smiles",
        report_type="chemotype",
        formats=["html", "md"],
    )

    reports_dir = _report_dir(local_session_root, "chemotype")
    structure_files = list((reports_dir / "assets" / "structures").glob("*.png"))
    assert len(structure_files) == 1
    assert structure_files[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    html_content = (reports_dir / "plain_scaffold_smiles.html").read_text()
    markdown_content = (reports_dir / "plain_scaffold_smiles.md").read_text()
    paragraph_index = html_content.index("Piperidine urea scaffold")
    figure_index = html_content.index("Figure 1. Scaffold_1: Piperidine urea scaffold")
    assert paragraph_index < figure_index
    assert "Scaffold_1 (Piperidine urea scaffold) is shown in Figure 1." in html_content
    assert "### Figure 1. Scaffold_1: Piperidine urea scaffold" in markdown_content


def test_save_rich_report_preserves_existing_chembl_ids_for_molecule_figures(
    local_session_root,
):
    """Existing molecule IDs such as ChEMBL IDs should be reused in text and figure labels."""
    save_rich_report(
        title="ChEMBL Molecule Report",
        sections=[
            {
                "heading": "Molecule Summary",
                "paragraphs": [
                    "CHEMBL123456 is the reported high-potency molecule in node 17.",
                ],
                "tables": [
                    {
                        "title": "Molecule Inventory",
                        "columns": [
                            "ChEMBL ID",
                            "Molecule",
                            "SMILES",
                            "Name",
                            "Node",
                            "Description",
                        ],
                        "rows": [
                            {
                                "ChEMBL ID": "CHEMBL123456",
                                "Molecule": "Top potency ChEMBL analog",
                                "SMILES": "CCOc1ccc(NC(=O)NCC)cc1",
                                "Name": "Top potency ChEMBL analog",
                                "Node": "17",
                                "Description": "Existing ChEMBL compound identifier.",
                            }
                        ],
                    }
                ],
            }
        ],
        filename="chembl_ids",
        report_type="chemotype",
        formats=["html", "pdf", "md"],
    )

    reports_dir = _report_dir(local_session_root, "chemotype")
    html_content = (reports_dir / "chembl_ids.html").read_text()
    markdown_content = (reports_dir / "chembl_ids.md").read_text()
    assert len(list((reports_dir / "assets" / "structures").glob("*.png"))) == 1
    assert "Figure 1. CHEMBL123456: Top potency ChEMBL analog" in html_content
    assert "CHEMBL123456 (Top potency ChEMBL analog) is shown in Figure 1." in html_content
    assert "Molecule_1" not in html_content
    assert "### Figure 1. CHEMBL123456: Top potency ChEMBL analog" in markdown_content
    assert "CHEMBL123456 (Top potency ChEMBL analog) is shown in Figure 1." in markdown_content


def test_save_rich_report_uses_generic_dataset_ids_and_names(local_session_root):
    """Dataset-provided IDs and display names should drive structure figure labels."""
    save_rich_report(
        title="Generic Source Structure Report",
        sections=[
            {
                "heading": "Source Molecule Summary",
                "paragraphs": [
                    "SRC-42 is the dataset-supplied kinase analog discussed in node 8.",
                ],
                "tables": [
                    {
                        "title": "Structure Inventory",
                        "columns": [
                            "source_id",
                            "display_name",
                            "smiles",
                            "node_index",
                            "description",
                        ],
                        "rows": [
                            {
                                "source_id": "SRC-42",
                                "display_name": "Dataset supplied kinase analog",
                                "smiles": "CCOc1ccc(NC(=O)NCC)cc1",
                                "node_index": "8",
                                "description": "Generic source identifier from an uploaded dataset.",
                            }
                        ],
                    }
                ],
            }
        ],
        filename="generic_source_ids",
        report_type="chemotype",
        formats=["html", "md"],
    )

    reports_dir = _report_dir(local_session_root, "chemotype")
    html_content = (reports_dir / "generic_source_ids.html").read_text()
    markdown_content = (reports_dir / "generic_source_ids.md").read_text()
    assert len(list((reports_dir / "assets" / "structures").glob("*.png"))) == 1
    assert "Figure 1. SRC-42: Dataset supplied kinase analog" in html_content
    assert "SRC-42 (Dataset supplied kinase analog) is shown in Figure 1." in html_content
    assert "Molecule_1" not in html_content
    assert "### Figure 1. SRC-42: Dataset supplied kinase analog" in markdown_content
    assert "SRC-42 (Dataset supplied kinase analog) is shown in Figure 1." in markdown_content


def test_save_rich_report_uses_generic_scaffold_ids_and_names(local_session_root):
    """Scaffold IDs/names should be source-agnostic across inventory schemas."""
    save_rich_report(
        title="Generic Scaffold Report",
        sections=[
            {
                "heading": "Scaffold Summary",
                "paragraphs": [
                    "SCF-A anchors the recurring bicyclic scaffold family in node 19.",
                ],
                "tables": [
                    {
                        "title": "Uploaded Scaffold Inventory",
                        "columns": [
                            "scaffold_id",
                            "scaffold_name",
                            "smiles",
                            "node",
                            "notes",
                        ],
                        "rows": [
                            {
                                "scaffold_id": "SCF-A",
                                "scaffold_name": "Recurring bicyclic scaffold",
                                "smiles": "c1ccc2ccccc2c1",
                                "node": "19",
                                "notes": "Generic scaffold identifier from a source file.",
                            }
                        ],
                    }
                ],
            }
        ],
        filename="generic_scaffold_ids",
        report_type="chemotype",
        formats=["html", "md"],
    )

    reports_dir = _report_dir(local_session_root, "chemotype")
    html_content = (reports_dir / "generic_scaffold_ids.html").read_text()
    markdown_content = (reports_dir / "generic_scaffold_ids.md").read_text()
    paragraph_index = html_content.index("SCF-A anchors the recurring")
    figure_index = html_content.index("Figure 1. SCF-A: Recurring bicyclic scaffold")
    table_index = html_content.index("<h3>Uploaded Scaffold Inventory</h3>")
    assert paragraph_index < figure_index < table_index
    assert "SCF-A (Recurring bicyclic scaffold) is shown in Figure 1." in html_content
    assert "Scaffold_1" not in html_content
    assert "### Figure 1. SCF-A: Recurring bicyclic scaffold" in markdown_content
    assert "SCF-A (Recurring bicyclic scaffold) is shown in Figure 1." in markdown_content


def test_save_rich_report_applies_table_source_ids_to_smiles_tag_figures(
    local_session_root,
):
    """Source IDs from tables should upgrade matching figures created from SMILES tags."""
    smiles = "CCOc1ccc(NC(=O)NCC)cc1"
    save_rich_report(
        title="Tagged ChEMBL Molecule Report",
        sections=[
            {
                "heading": "Molecule Summary",
                "paragraphs": [
                    f"The source compound is introduced as <smiles>{smiles}</smiles>.",
                ],
                "tables": [
                    {
                        "title": "Molecule Inventory",
                        "columns": [
                            "ChEMBL ID",
                            "Molecule",
                            "SMILES",
                            "Name",
                            "Node",
                            "Description",
                        ],
                        "rows": [
                            {
                                "ChEMBL ID": "CHEMBL777777",
                                "Molecule": "Known ChEMBL analog",
                                "SMILES": smiles,
                                "Name": "Known ChEMBL analog",
                                "Node": "23",
                                "Description": "Table source ID should be reused.",
                            }
                        ],
                    }
                ],
            }
        ],
        filename="tagged_chembl_ids",
        report_type="chemotype",
        formats=["html", "md"],
    )

    reports_dir = _report_dir(local_session_root, "chemotype")
    html_content = (reports_dir / "tagged_chembl_ids.html").read_text()
    markdown_content = (reports_dir / "tagged_chembl_ids.md").read_text()
    assert len(list((reports_dir / "assets" / "structures").glob("*.png"))) == 1
    assert "Figure 1. CHEMBL777777: Known ChEMBL analog" in html_content
    assert "CHEMBL777777 (Known ChEMBL analog) is shown in Figure 1." in html_content
    assert "Molecule_1" not in html_content
    assert "### Figure 1. CHEMBL777777: Known ChEMBL analog" in markdown_content
    assert "CHEMBL777777 (Known ChEMBL analog) is shown in Figure 1." in markdown_content


def test_save_rich_report_adds_generated_ids_to_inventory_tables(local_session_root):
    """Rows without source IDs should get generated IDs reused in text, tables, and figures."""
    save_rich_report(
        title="Generated Structure ID Report",
        sections=[
            {
                "heading": "Generated IDs",
                "paragraphs": [
                    "The adamantane urea phenyl scaffold is explicitly discussed in node 321.",
                    "The representative analog is explicitly discussed in node 17.",
                ],
                "tables": [
                    {
                        "title": "Scaffold Inventory",
                        "columns": ["Scaffold", "SMILES", "Name", "Node", "Description"],
                        "rows": [
                            {
                                "Scaffold": "Adamantane urea phenyl scaffold",
                                "SMILES": "O=C(Nc1ccccc1)NC12CC3CC(CC(C3)C1)C2",
                                "Name": "Adamantane urea phenyl scaffold",
                                "Node": "321",
                                "Description": "No source scaffold ID was available.",
                            }
                        ],
                    },
                    {
                        "title": "Molecule Inventory",
                        "columns": ["Molecule", "SMILES", "Name", "Node", "Description"],
                        "rows": [
                            {
                                "Molecule": "Representative low-activity analog",
                                "SMILES": "CCOc1ccc(NC(=O)NCC)cc1",
                                "Name": "Representative low-activity analog",
                                "Node": "17",
                                "Description": "No source molecule ID was available.",
                            }
                        ],
                    },
                ],
            }
        ],
        filename="generated_structure_ids",
        report_type="chemotype",
        formats=["html", "md"],
    )

    reports_dir = _report_dir(local_session_root, "chemotype")
    html_content = (reports_dir / "generated_structure_ids.html").read_text()
    markdown_content = (reports_dir / "generated_structure_ids.md").read_text()
    assert len(list((reports_dir / "assets" / "structures").glob("*.png"))) == 2
    assert "Figure 1. Scaffold_1: Adamantane urea phenyl scaffold" in html_content
    assert "Figure 2. Molecule_1: Representative low-activity analog" in html_content
    assert "<th>Scaffold ID</th>" in html_content
    assert "<td>Scaffold_1</td>" in html_content
    assert "<th>Molecule ID</th>" in html_content
    assert "<td>Molecule_1</td>" in html_content
    assert "Scaffold_1 (Adamantane urea phenyl scaffold) is shown in Figure 1." in html_content
    assert "Molecule_1 (Representative low-activity analog) is shown in Figure 2." in html_content
    assert "| Scaffold ID | Scaffold | SMILES | Name | Node | Description |" in markdown_content
    assert "| Molecule ID | Molecule | SMILES | Name | Node | Description |" in markdown_content
    assert "| Scaffold_1 | Adamantane urea phenyl scaffold |" in markdown_content
    assert "| Molecule_1 | Representative low-activity analog |" in markdown_content


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
    assert "Figure 1. Molecule_1: Reported compound structure" in html_content
    assert "Molecule_1 (Reported compound structure) is shown in Figure 1." in html_content
    assert "data:image/png;base64," in html_content
    assert "<smiles>" not in markdown_content
    assert f"The report highlights {smiles} as a representative compound." in markdown_content
    assert "Molecule_1 (Reported compound structure) is shown in Figure 1." in markdown_content
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
    first_figure_index = html_content.index("Figure 1. Molecule_1: Reported compound structure")
    second_paragraph_index = html_content.index(f"Second mention introduces {second_smiles}.")
    second_figure_index = html_content.index("Figure 2. Molecule_2: Reported compound structure")
    assert first_paragraph_index < first_figure_index < second_paragraph_index
    assert second_paragraph_index < second_figure_index
    assert markdown_content.index(
        f"First mention introduces {first_smiles}."
    ) < markdown_content.index("### Figure 1. Molecule_1: Reported compound structure")
    assert markdown_content.index(
        f"Second mention introduces {second_smiles}."
    ) < markdown_content.index("### Figure 2. Molecule_2: Reported compound structure")


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
