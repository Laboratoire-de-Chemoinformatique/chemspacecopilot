#!/usr/bin/env python3
"""Repair backend SVG mask extents and export scalable per-route supplements.

Molecule and route geometry are preserved byte-for-byte at the attribute level.
Original SVGs are retained; only mask coordinate bounds and a white page background
are changed. CairoSVG is required only for PDF and optional PNG exports.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from _study_common import file_identity, write_json

SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)


def repair_mask_extents(content: bytes) -> tuple[bytes, int]:
    root = ET.fromstring(content)
    if root.tag != f"{{{SVG}}}svg":
        raise ValueError("Expected an SVG document")
    bounds = root.get("viewBox", "").split()
    if len(bounds) != 4 or any(not math.isfinite(float(value)) for value in bounds):
        raise ValueError("SVG requires a finite four-coordinate viewBox")
    if float(bounds[2]) <= 0 or float(bounds[3]) <= 0:
        raise ValueError("SVG viewBox must have positive dimensions")
    count = 0
    for mask in root.iter(f"{{{SVG}}}mask"):
        mask.set("maskUnits", "userSpaceOnUse")
        for name, value in zip(("x", "y", "width", "height"), bounds, strict=True):
            mask.set(name, value)
        count += 1
    background = ET.Element(
        f"{{{SVG}}}rect",
        dict(zip(("x", "y", "width", "height"), bounds, strict=True), fill="white"),
    )
    root.insert(0, background)
    description = ET.Element(f"{{{SVG}}}desc")
    description.text = (
        "Backend-predicted retrosynthesis route. SVG mask bounds corrected to the complete "
        "viewBox and a white background added; molecular geometry, labels and connectivity "
        "are unchanged. This diagram does not establish synthetic feasibility."
    )
    root.insert(0, description)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True), count


def export_figures(source_dir: Path, output_dir: Path, preview_dir: Path | None = None) -> dict:
    import cairosvg

    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("Corrected figures must be separate from original backend SVGs")
    sources = sorted(source_dir.glob("target_*.svg"))
    if not sources:
        raise ValueError("No original target_*.svg route figures found")
    prepared = [(path, *repair_mask_extents(path.read_bytes())) for path in sources]
    output_dir.mkdir(parents=True, exist_ok=True)
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "transformation": (
            "Set each SVG maskUnits=userSpaceOnUse and its x/y/width/height to root viewBox; "
            "add white viewBox-sized background. Preserve every molecular drawing primitive, "
            "route connection and label. Original backend files are not modified."
        ),
        "figures": [],
    }
    for source, content, count in prepared:
        destination = output_dir / source.name
        destination.write_bytes(content)
        pdf = destination.with_suffix(".pdf")
        cairosvg.svg2pdf(bytestring=content, write_to=str(pdf))
        if preview_dir:
            cairosvg.svg2png(
                bytestring=content,
                write_to=str(preview_dir / destination.with_suffix(".png").name),
                output_width=2400,
            )
        manifest["figures"].append(
            {
                "source": file_identity(source),
                "corrected_svg": file_identity(destination),
                "pdf": file_identity(pdf),
                "masks_corrected": count,
            }
        )
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    try:
        manifest = export_figures(args.source_dir, args.output_dir, args.preview_dir)
    except (ValueError, OSError, ET.ParseError) as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
