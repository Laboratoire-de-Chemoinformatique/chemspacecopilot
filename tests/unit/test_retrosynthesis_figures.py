"""Verify the route mask repair fixes clipping while preserving scientific marks."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
from prepare_retrosynthesis_figures import SVG, repair_mask_extents  # noqa: E402


def test_negative_coordinates_are_visible_without_changing_molecular_marks():
    cairosvg = pytest.importorskip("cairosvg")
    from io import BytesIO

    from PIL import Image

    source = f"""<svg xmlns="{SVG}" viewBox="0 -10 20 20" width="200" height="200">
      <defs><mask id="m"><rect x="0" y="-10" width="20" height="20" fill="white"/></mask></defs>
      <g stroke="black" stroke-width="1" mask="url(#m)">
        <line x1="5" y1="-8" x2="15" y2="-5"/>
      </g><text x="3" y="-6">N</text></svg>""".encode()
    repaired, count = repair_mask_extents(source)
    before, after = ET.fromstring(source), ET.fromstring(repaired)
    for tag in ("line", "text"):
        original = before.find(f".//{{{SVG}}}{tag}")
        corrected = after.find(f".//{{{SVG}}}{tag}")
        assert original.attrib == corrected.attrib and original.text == corrected.text
    assert count == 1
    old = Image.open(
        BytesIO(cairosvg.svg2png(bytestring=source, background_color="white"))
    ).convert("L")
    new = Image.open(BytesIO(cairosvg.svg2png(bytestring=repaired))).convert("L")
    # Bond midpoint at (100,35) was outside the default negative-coordinate mask.
    assert old.getpixel((100, 35)) > 240
    assert new.getpixel((100, 35)) < 40


def test_rejects_non_svg_and_invalid_dimensions():
    with pytest.raises(ValueError, match="Expected an SVG"):
        repair_mask_extents(b"<html/>")
    with pytest.raises(ValueError, match="positive dimensions"):
        repair_mask_extents(f'<svg xmlns="{SVG}" viewBox="0 0 0 10"/>'.encode())
