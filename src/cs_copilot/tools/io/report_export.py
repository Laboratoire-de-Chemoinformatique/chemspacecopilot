#!/usr/bin/env python
# coding: utf-8
"""Report export tools for the Report Generator agent."""

import base64
import datetime
import hashlib
import html
import io
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from cs_copilot.storage import S3, OutputOperation, operation_rel_path

from .figure_metadata import (
    REPORTABLE_FIGURE_ROLES,
    figure_caption_facts,
    normalize_figure_metadata,
    report_image_path,
    session_figure_metadata,
)
from .session_memory import register_session_object
from .utils import get_mime_type

logger = logging.getLogger(__name__)

_REPORTS_DIR = "reports"
_MD_EXTENSION = ".md"
_HTML_EXTENSION = ".html"
_PDF_EXTENSION = ".pdf"
_PNG_EXTENSION = ".png"
_DEFAULT_REPORT_TYPE = "report"
_SLUG_RX = re.compile(r"[^A-Za-z0-9_-]+")
_FIGURE_NAME_RX = re.compile(r"^\s*Figure\s+\d+\s*[\.:]\s*(?P<title>.*)$", re.IGNORECASE)
_STRUCTURE_ID_RX = re.compile(r"^(?P<type>Scaffold|Molecule)[_-](?P<number>\d+)$", re.IGNORECASE)
_INLINE_EMPHASIS_RX = re.compile(
    r"\*\*(?P<markdown_text>.+?)\*\*"
    r"|<\s*(?P<tag>strong|b|string)\s*>\s*(?P<tag_text>.*?)\s*</\s*(?P=tag)\s*>",
    re.IGNORECASE | re.DOTALL,
)
_SMILES_TAG_RX = re.compile(
    r"<smiles>\s*(?P<smiles>.*?)\s*</smiles>",
    re.IGNORECASE | re.DOTALL,
)
_HTML_TAG_RX = re.compile(r"<[^>]+>")
_PLAIN_SMILES_RX = re.compile(
    r"(?<![A-Za-z0-9_])(?P<smiles>[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]{5,})(?![A-Za-z0-9_])"
)
_GTM_NODE_MENTION_RX = re.compile(
    r"\b(?:GTM\s+)?nodes?\s*(?:#|number|id)?\s*"
    r"(?P<values>\d+(?:\s*(?:,|/|&|and|to|[-–—→>]+)\s*\d+)*)",
    re.IGNORECASE,
)
_SOURCE_ID_SPLIT_RX = re.compile(r"\s*(?:\||;|,)\s*")
_SUPPORTED_RICH_FORMATS = {"html", "pdf", "md", "markdown"}
_FIGURE_METADATA_KEYS = {
    "figure_kind",
    "renderer",
    "report_role",
    "title_subject",
    "paths",
    "color_encoding",
    "overlays",
    "node_labels",
    "caption_facts",
}
_STRUCTURE_ID_FIELDS = (
    "structure_id",
    "structure id",
    "source_id",
    "source id",
    "external_id",
    "external id",
    "dataset_id",
    "dataset id",
    "record_id",
    "record id",
    "compound_id",
    "compound id",
    "compound_ids",
    "compound ids",
    "molecule_id",
    "molecule id",
    "molecule_ids",
    "molecule ids",
    "scaffold_id",
    "scaffold id",
    "scaffold_ids",
    "scaffold ids",
    "molecule_chembl_id",
    "molecule chembl id",
    "molecule_chembl_ids",
    "molecule chembl ids",
    "chembl_id",
    "chembl id",
    "chembl_ids",
    "chembl ids",
    "ChEMBL ID",
    "ChEMBL IDs",
    "id",
    "ID",
)
_MOLECULE_SOURCE_ID_FIELDS = (
    "structure_id",
    "structure id",
    "source_id",
    "source id",
    "external_id",
    "external id",
    "dataset_id",
    "dataset id",
    "record_id",
    "record id",
    "compound_id",
    "compound id",
    "compound_ids",
    "compound ids",
    "molecule_id",
    "molecule id",
    "molecule_ids",
    "molecule ids",
    "molecule_chembl_id",
    "molecule chembl id",
    "molecule_chembl_ids",
    "molecule chembl ids",
    "parent_molecule_chembl_id",
    "parent molecule chembl id",
    "parent_molecule_chembl_ids",
    "parent molecule chembl ids",
    "chembl_id",
    "chembl id",
    "chembl_ids",
    "chembl ids",
    "ChEMBL ID",
    "ChEMBL IDs",
    "Molecule ChEMBL ID",
    "Molecule ChEMBL IDs",
    "Compound ChEMBL ID",
    "Compound ChEMBL IDs",
    "ID",
    "id",
)
_SCAFFOLD_SOURCE_ID_FIELDS = (
    "scaffold_id",
    "scaffold id",
    "scaffold_ids",
    "scaffold ids",
    "structure_id",
    "structure id",
    "source_id",
    "source id",
    "external_id",
    "external id",
    "dataset_id",
    "dataset id",
    "record_id",
    "record id",
    "ID",
    "id",
)
_MOLECULE_DISPLAY_NAME_FIELDS = (
    "molecule_name",
    "molecule name",
    "compound_name",
    "compound name",
    "molecule",
    "compound",
    "structure_name",
    "structure name",
    "display_name",
    "display name",
    "preferred_name",
    "preferred name",
    "pref_name",
    "pref name",
    "user_name",
    "user name",
    "user_provided_name",
    "user provided name",
    "descriptive_name",
    "descriptive name",
    "common_name",
    "common name",
    "name",
    "label",
)
_SCAFFOLD_DISPLAY_NAME_FIELDS = (
    "scaffold_name",
    "scaffold name",
    "scaffold",
    "structure_name",
    "structure name",
    "display_name",
    "display name",
    "preferred_name",
    "preferred name",
    "pref_name",
    "pref name",
    "user_name",
    "user name",
    "user_provided_name",
    "user provided name",
    "descriptive_name",
    "descriptive name",
    "common_name",
    "common name",
    "name",
    "label",
)
_MISSING_SOURCE_ID_VALUES = {"", "-", "na", "n/a", "nan", "none", "null"}
_STRUCTURE_CONTEXT_TERMS = (
    "scaffold",
    "molecule",
    "compound",
    "analog",
    "sar",
    "chemotype",
    "structure",
)
_MAX_AUTO_STRUCTURE_FIGURES_PER_SECTION = 8


def _report_slug(report_type: Optional[str]) -> str:
    slug = _SLUG_RX.sub("_", (report_type or _DEFAULT_REPORT_TYPE).strip()).strip("_")
    return slug or _DEFAULT_REPORT_TYPE


def _timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")


def _report_filename(
    filename: Optional[str],
    extension: str,
    report_type: Optional[str],
) -> str:
    if filename:
        # Strip directory components to keep files inside the workflow report folder.
        name = Path(filename).name
        # Force the requested extension, replacing any other suffix.
        return Path(name).with_suffix(extension).name

    return f"{_report_slug(report_type)}_{_timestamp()}{extension}"


def _rich_report_basename(filename: Optional[str], report_type: Optional[str]) -> str:
    if filename:
        name = Path(filename).name
        path = Path(name)
        return path.with_suffix("").name if path.suffix else path.name

    return f"{_report_slug(report_type)}_{_timestamp()}"


def _write_text_report(content: str, rel_path: str) -> str:
    with S3.open(rel_path, "w") as fh:
        fh.write(content)
    return S3.path(rel_path)


def _write_binary_report(content: bytes, rel_path: str) -> str:
    with S3.open(rel_path, "wb") as fh:
        fh.write(content)
    return S3.path(rel_path)


def _report_rel_path(
    filename: str,
    report_type: Optional[str],
    session_state: Optional[Dict[str, Any]],
) -> str:
    return operation_rel_path(
        OutputOperation.REPORTS,
        _report_slug(report_type),
        filename,
        session_state=session_state,
        workflow_slug="reports",
    )


def _as_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _has_text_value(value: Any) -> bool:
    if isinstance(value, (list, tuple, set)):
        return any(_has_text_value(item) for item in value)
    return bool(_clean_text(value))


def _key_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _mapping_raw_value(mapping: dict[str, Any], *keys: str) -> Any:
    lowered = {str(key).lower(): value for key, value in mapping.items()}
    normalized = {_key_token(key): value for key, value in mapping.items()}
    for key in keys:
        value = lowered.get(str(key).lower())
        if _has_text_value(value):
            return value
        value = normalized.get(_key_token(key))
        if _has_text_value(value):
            return value
    return ""


def _mapping_value(mapping: dict[str, Any], *keys: str) -> str:
    return _clean_text(_mapping_raw_value(mapping, *keys))


def _source_id_candidates(structure_type: str) -> tuple[str, ...]:
    if structure_type == "scaffold":
        return _SCAFFOLD_SOURCE_ID_FIELDS
    return _MOLECULE_SOURCE_ID_FIELDS


def _display_name_candidates(structure_type: str) -> tuple[str, ...]:
    if structure_type == "scaffold":
        return _SCAFFOLD_DISPLAY_NAME_FIELDS
    return _MOLECULE_DISPLAY_NAME_FIELDS


def _source_ids_from_value(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        ids: list[str] = []
        items = sorted(value, key=str) if isinstance(value, set) else value
        for item in items:
            ids.extend(_source_ids_from_value(item))
        return list(dict.fromkeys(ids))

    raw = _clean_text(value)
    if not raw:
        return []

    ids = []
    for part in _SOURCE_ID_SPLIT_RX.split(raw):
        source_id = part.strip().strip("[]'\"")
        if source_id.lower() in _MISSING_SOURCE_ID_VALUES:
            continue
        if source_id and source_id not in ids:
            ids.append(source_id)
    return ids


def _mapping_source_id(mapping: dict[str, Any], *keys: str) -> str:
    raw_value = _mapping_raw_value(mapping, *keys)
    return "|".join(_source_ids_from_value(raw_value))


def _resolve_structure_identity(
    mapping: dict[str, Any],
    structure_type: str,
) -> dict[str, str]:
    normalized_type = "scaffold" if structure_type == "scaffold" else "molecule"
    return {
        "structure_id": _mapping_source_id(
            mapping,
            *_source_id_candidates(normalized_type),
        ),
        "structure_name": _mapping_value(
            mapping,
            *_display_name_candidates(normalized_type),
        ),
    }


def _infer_structure_type_from_mapping(mapping: dict[str, Any]) -> str:
    if _mapping_source_id(mapping, "scaffold_id", "scaffold_ids") or _mapping_value(
        mapping,
        "scaffold_name",
        "scaffold name",
        "scaffold",
        "scaffold_smiles",
        "scaffold smiles",
        "scaffold_smi",
    ):
        return "scaffold"
    if _mapping_source_id(
        mapping,
        "compound_id",
        "compound_ids",
        "molecule_id",
        "molecule_ids",
        "molecule_chembl_id",
        "molecule_chembl_ids",
        "chembl_id",
        "chembl_ids",
    ) or _mapping_value(
        mapping,
        "molecule_name",
        "molecule name",
        "compound_name",
        "compound name",
        "molecule",
        "compound",
        "molecule_smiles",
        "molecule smiles",
        "compound_smiles",
        "compound smiles",
    ):
        return "molecule"
    return ""


def _clean_optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _figure_title(text: str) -> str:
    match = _FIGURE_NAME_RX.match(text)
    title = match.group("title") if match else text
    title = " ".join(title.split()).strip(" .:-")
    return title


def _caption_title(caption: str) -> str:
    title = _figure_title(caption)
    if not title:
        return "Visualization"
    sentence_match = re.split(r"(?<=[.!?])\s+", title, maxsplit=1)
    title = sentence_match[0].strip(" .:-")
    if len(title) > 140:
        title = title[:137].rstrip() + "..."
    return title or "Visualization"


def _format_figure_name(name: str, caption: str, index: int) -> str:
    title = _figure_title(name) if name else _caption_title(caption)
    return f"Figure {index}. {title}"


def _looks_like_figure_metadata(value: Any) -> bool:
    return isinstance(value, dict) and any(key in value for key in _FIGURE_METADATA_KEYS)


def _figure_input_paths(figure: dict[str, Any]) -> dict[str, str]:
    paths = figure.get("paths") if isinstance(figure.get("paths"), dict) else {}
    normalized = {str(key): _clean_text(value) for key, value in paths.items() if value}
    png_path = _clean_text(
        figure.get("png_path")
        or figure.get("image_path")
        or figure.get("path")
        or figure.get("src")
    )
    html_path = _clean_text(
        figure.get("html_path") or figure.get("artifact_path") or figure.get("interactive_path")
    )
    if png_path and "png_path" not in normalized:
        normalized["png_path"] = png_path
    if html_path and "html_path" not in normalized:
        normalized["html_path"] = html_path
    return normalized


def _merge_figure_metadata(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if value not in (None, "", [], {}):
            merged[key] = value
    if isinstance(base.get("paths"), dict) or isinstance(override.get("paths"), dict):
        paths = {}
        if isinstance(base.get("paths"), dict):
            paths.update(base["paths"])
        if isinstance(override.get("paths"), dict):
            paths.update({key: value for key, value in override["paths"].items() if value})
        merged["paths"] = paths
    return normalize_figure_metadata(merged)


def _resolve_figure_metadata(
    figure: dict[str, Any],
    session_state: Optional[Dict[str, Any]],
) -> dict[str, Any]:
    figure_id = _clean_text(figure.get("figure_id") or figure.get("session_figure_id"))
    metadata = session_figure_metadata(session_state, figure_id)

    inline_metadata = figure.get("figure_metadata")
    if not _looks_like_figure_metadata(inline_metadata):
        inline_metadata = figure.get("metadata")
    if _looks_like_figure_metadata(inline_metadata):
        metadata = _merge_figure_metadata(metadata, inline_metadata)

    if _looks_like_figure_metadata(figure):
        direct_metadata = {key: figure[key] for key in _FIGURE_METADATA_KEYS if key in figure}
        direct_metadata.setdefault("paths", _figure_input_paths(figure))
        metadata = _merge_figure_metadata(metadata, direct_metadata)

    return metadata


def _metadata_title(metadata: dict[str, Any]) -> str:
    return _clean_text(metadata.get("title_subject") or metadata.get("label"))


def _metadata_caption(metadata: dict[str, Any]) -> str:
    return " ".join(figure_caption_facts(metadata))


def _normalize_structure_type(value: Any, structure_smiles: str) -> str:
    normalized = _clean_text(value).lower()
    if normalized in {"scaffold", "scaffolds"}:
        return "scaffold"
    if normalized in {"molecule", "molecules", "compound", "compounds"}:
        return "molecule"
    return "molecule" if structure_smiles else ""


def _normalize_figure(
    figure: Any,
    index: int,
    session_state: Optional[Dict[str, Any]] = None,
) -> dict[str, Any]:
    figure_metadata: dict[str, Any] = {}
    figure_id = ""
    if isinstance(figure, str):
        image_path = _clean_text(figure)
        name = ""
        caption = ""
        alt_text = ""
        artifact_path = ""
        structure_smiles = ""
        structure_type = ""
        structure_id = ""
        structure_name = ""
        node = ""
        description = ""
        after_paragraph_index = None
    elif isinstance(figure, dict):
        figure_metadata = _resolve_figure_metadata(figure, session_state)
        figure_id = _clean_text(
            figure.get("figure_id") or figure.get("session_figure_id") or figure_metadata.get("id")
        )
        metadata_image_path = report_image_path(figure_metadata)
        metadata_title = _metadata_title(figure_metadata)
        metadata_caption = _metadata_caption(figure_metadata)
        image_path = _clean_text(
            figure.get("image_path")
            or figure.get("png_path")
            or figure.get("path")
            or figure.get("src")
            or metadata_image_path
            or ""
        )
        caption = _clean_text(
            figure.get("caption") or figure.get("description") or metadata_caption
        )
        name = _clean_text(
            figure.get("name") or figure.get("figure_name") or figure.get("title") or metadata_title
        )
        alt_text = _clean_text(figure.get("alt_text") or figure.get("alt") or metadata_title)
        artifact_path = _clean_text(
            figure.get("artifact_path")
            or figure.get("html_path")
            or figure.get("interactive_path")
            or ""
        )
        structure_smiles = _clean_text(
            figure.get("structure_smiles")
            or figure.get("smiles")
            or figure.get("scaffold_smiles")
            or figure.get("scaffold_smi")
        )
        raw_structure_type = figure.get("structure_type")
        if raw_structure_type is None and (
            figure.get("scaffold_smiles") or figure.get("scaffold_smi")
        ):
            raw_structure_type = "scaffold"
        if raw_structure_type is None:
            raw_structure_type = _infer_structure_type_from_mapping(figure)
        structure_type = _normalize_structure_type(raw_structure_type, structure_smiles)
        identity = (
            _resolve_structure_identity(figure, structure_type)
            if structure_type
            else {"structure_id": "", "structure_name": ""}
        )
        structure_id = identity["structure_id"]
        structure_name = identity["structure_name"]
        node = _clean_text(figure.get("node") or figure.get("gtm_node") or "")
        description = _clean_text(figure.get("structure_description") or figure.get("description"))
        after_paragraph_index = _clean_optional_int(figure.get("after_paragraph_index"))
    else:
        image_path = ""
        name = ""
        caption = ""
        alt_text = ""
        artifact_path = ""
        structure_smiles = ""
        structure_type = ""
        structure_id = ""
        structure_name = ""
        node = ""
        description = ""
        after_paragraph_index = None

    if structure_smiles and not caption:
        caption = description

    if image_path or artifact_path or structure_smiles or figure_metadata:
        if not caption:
            raise ValueError(f"figure {index} caption cannot be empty")
        name = _format_figure_name(name, caption, index)
        alt_text = alt_text or name

    return {
        "name": name,
        "image_path": image_path,
        "caption": caption,
        "alt_text": alt_text,
        "artifact_path": artifact_path,
        "structure_smiles": structure_smiles,
        "structure_type": structure_type,
        "structure_id": structure_id,
        "structure_name": structure_name,
        "node": node,
        "description": description,
        "after_paragraph_index": after_paragraph_index,
        "figure_id": figure_id,
        "figure_metadata": figure_metadata,
    }


def _normalize_figures(
    figures: Any,
    start_index: int = 1,
    session_state: Optional[Dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    if not figures:
        return []
    if not isinstance(figures, (list, tuple)):
        figures = [figures]
    return [
        _normalize_figure(figure, index, session_state=session_state)
        for index, figure in enumerate(figures, start=start_index)
    ]


def _normalize_table(table: Any, index: int) -> dict[str, Any]:
    if not isinstance(table, dict):
        return {
            "title": f"Table {index}",
            "columns": ["Value"],
            "rows": [{"Value": _clean_text(table)}],
        }

    title = _clean_text(table.get("title") or table.get("heading") or f"Table {index}")
    rows = table.get("rows") or table.get("data") or []
    if not isinstance(rows, (list, tuple)):
        rows = [rows]

    columns = _as_strings(table.get("columns") or table.get("headers"))
    if not columns:
        inferred_columns = []
        for row in rows:
            if isinstance(row, dict):
                for key in row:
                    if str(key) not in inferred_columns:
                        inferred_columns.append(str(key))
        columns = inferred_columns or ["Value"]

    normalized_rows = []
    for row in rows:
        if isinstance(row, dict):
            normalized_rows.append({column: _clean_text(row.get(column, "")) for column in columns})
        elif isinstance(row, (list, tuple)):
            normalized_rows.append(
                {
                    column: _clean_text(row[position]) if position < len(row) else ""
                    for position, column in enumerate(columns)
                }
            )
        else:
            normalized_rows.append({columns[0]: _clean_text(row)})

    return {"title": title, "columns": columns, "rows": normalized_rows}


def _normalize_tables(tables: Any) -> list[dict[str, Any]]:
    if not tables:
        return []
    if not isinstance(tables, (list, tuple)):
        tables = [tables]
    return [_normalize_table(table, index) for index, table in enumerate(tables, start=1)]


def _normalize_sections(
    sections: Any,
    session_state: Optional[Dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    if not sections:
        return []
    if not isinstance(sections, (list, tuple)):
        sections = [sections]

    normalized = []
    for index, section in enumerate(sections, start=1):
        if isinstance(section, str):
            normalized.append(
                {
                    "heading": f"Section {index}",
                    "paragraphs": [section],
                    "figures": [],
                    "tables": [],
                }
            )
            continue

        if not isinstance(section, dict):
            normalized.append(
                {
                    "heading": f"Section {index}",
                    "paragraphs": [str(section)],
                    "figures": [],
                    "tables": [],
                }
            )
            continue

        paragraphs = _as_strings(
            section.get("paragraphs") or section.get("content") or section.get("text")
        )
        normalized.append(
            {
                "heading": str(
                    section.get("heading") or section.get("title") or f"Section {index}"
                ),
                "paragraphs": paragraphs,
                "figures": _normalize_figures(
                    section.get("figures"),
                    session_state=session_state,
                ),
                "tables": _normalize_tables(section.get("tables")),
            }
        )

    return normalized


def _renumber_report_figures(
    sections: list[dict[str, Any]],
    figures: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    figure_index = 1
    for section in sections:
        section["figures"] = _ordered_section_figures(section)
        for figure in section["figures"]:
            figure["name"] = _format_figure_name(figure["name"], figure["caption"], figure_index)
            figure["alt_text"] = figure["alt_text"] or figure["name"]
            figure_index += 1

    for figure in figures:
        figure["name"] = _format_figure_name(figure["name"], figure["caption"], figure_index)
        figure["alt_text"] = figure["alt_text"] or figure["name"]
        figure_index += 1

    return sections, figures


def _read_binary_path(path: str) -> bytes:
    if not path:
        raise ValueError("path cannot be empty")

    candidate = Path(path)
    if not path.startswith("s3://") and candidate.exists():
        return candidate.read_bytes()

    with S3.open(path, "rb") as fh:
        return fh.read()


def _extract_smiles_tags(text: str) -> list[str]:
    return [
        match.group("smiles").strip()
        for match in _SMILES_TAG_RX.finditer(text or "")
        if match.group("smiles").strip()
    ]


def _strip_smiles_tags(text: str) -> str:
    return _SMILES_TAG_RX.sub(lambda match: match.group("smiles").strip(), text)


def _strip_html_tags(text: str) -> str:
    return _HTML_TAG_RX.sub("", text)


def _looks_like_structure_context(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in _STRUCTURE_CONTEXT_TERMS)


def _is_valid_smiles(smiles: str) -> bool:
    try:
        from rdkit import Chem
    except ImportError:
        return False

    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return False
    return mol is not None and mol.GetNumAtoms() > 0


def _plain_smiles_candidates(text: str) -> list[str]:
    candidates = []
    for match in _PLAIN_SMILES_RX.finditer(text):
        smiles = match.group("smiles").strip(".,;:")
        if smiles and _is_valid_smiles(smiles) and smiles not in candidates:
            candidates.append(smiles)
    return candidates


def _structure_name_from_plain_text(text: str, smiles: str, structure_type: str) -> str:
    before = text.split(smiles, maxsplit=1)[0]
    before = before.rsplit(":", maxsplit=1)[0] if ":" in before else before
    before = re.sub(r"^\s*[-*]\s*", "", before).strip()
    before = re.sub(r"\([^)]*\)\s*$", "", before).strip()
    words = before.split()
    if len(words) > 8:
        before = " ".join(words[-8:])
    before = before.strip(" .:-")
    if before:
        return before
    return "Reported scaffold" if structure_type == "scaffold" else "Reported molecule"


def _append_plain_smiles_figures(
    sections: list[dict[str, Any]],
    figures: list[dict[str, Any]],
) -> None:
    seen_smiles = {
        figure["structure_smiles"] for figure in figures if figure.get("structure_smiles")
    }
    for section in sections:
        seen_smiles.update(
            figure["structure_smiles"]
            for figure in section["figures"]
            if figure.get("structure_smiles")
        )

    for section in sections:
        section_context = _strip_html_tags(_clean_text(section.get("heading", "")))
        generated_count = 0
        for paragraph_index, paragraph in enumerate(section["paragraphs"]):
            if generated_count >= _MAX_AUTO_STRUCTURE_FIGURES_PER_SECTION:
                break
            plain_text = _strip_html_tags(_strip_smiles_tags(paragraph))
            context = f"{section_context} {plain_text}"
            if not _looks_like_structure_context(context):
                continue

            structure_type = "scaffold" if "scaffold" in context.lower() else "molecule"
            for smiles in _plain_smiles_candidates(plain_text):
                if smiles in seen_smiles:
                    continue
                seen_smiles.add(smiles)
                structure_name = _structure_name_from_plain_text(plain_text, smiles, structure_type)
                section["figures"].append(
                    {
                        "name": structure_name,
                        "image_path": "",
                        "caption": f"Chemical structure for {structure_name}: {smiles}",
                        "alt_text": "",
                        "artifact_path": "",
                        "structure_smiles": smiles,
                        "structure_type": structure_type,
                        "structure_id": "",
                        "structure_name": structure_name,
                        "node": "",
                        "description": f"Chemical structure for {structure_name}: {smiles}",
                        "after_paragraph_index": paragraph_index,
                        "_auto_structure_figure": True,
                    }
                )
                generated_count += 1
                if generated_count >= _MAX_AUTO_STRUCTURE_FIGURES_PER_SECTION:
                    break


def _inline_markup(text: Any, *, bold_tag: str) -> str:
    text = _strip_smiles_tags(_clean_text(text))
    rendered = []
    cursor = 0
    for match in _INLINE_EMPHASIS_RX.finditer(text):
        bold_text = match.group("markdown_text") or match.group("tag_text")
        if not bold_text:
            continue
        rendered.append(html.escape(text[cursor : match.start()]))
        rendered.append(f"<{bold_tag}>{html.escape(bold_text)}</{bold_tag}>")
        cursor = match.end()
    rendered.append(html.escape(text[cursor:]))
    return "".join(rendered)


def _html_inline_markup(text: Any) -> str:
    return _inline_markup(text, bold_tag="strong")


def _pdf_inline_markup(text: Any) -> str:
    return _inline_markup(text, bold_tag="b")


def _figure_placement_key(figure: dict[str, Any], position: int) -> tuple[int, int]:
    paragraph_index = figure.get("after_paragraph_index")
    if isinstance(paragraph_index, int):
        return paragraph_index, position
    return 10**9, position


def _ordered_section_figures(section: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        figure
        for _position, figure in sorted(
            enumerate(section["figures"]),
            key=lambda item: _figure_placement_key(item[1], item[0]),
        )
    ]


def _group_section_figures(
    section: dict[str, Any],
) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    placed: dict[int, list[dict[str, Any]]] = {}
    unplaced = []
    paragraph_count = len(section["paragraphs"])
    for figure in section["figures"]:
        paragraph_index = figure.get("after_paragraph_index")
        if isinstance(paragraph_index, int) and paragraph_index < paragraph_count:
            placed.setdefault(paragraph_index, []).append(figure)
        else:
            unplaced.append(figure)
    return placed, unplaced


def _append_smiles_tag_figures(
    sections: list[dict[str, Any]],
    figures: list[dict[str, Any]],
) -> None:
    seen_smiles = {
        figure["structure_smiles"] for figure in figures if figure.get("structure_smiles")
    }
    for section in sections:
        seen_smiles.update(
            figure["structure_smiles"]
            for figure in section["figures"]
            if figure.get("structure_smiles")
        )

    for section in sections:
        for paragraph_index, paragraph in enumerate(section["paragraphs"]):
            for smiles in _extract_smiles_tags(paragraph):
                if smiles in seen_smiles:
                    continue
                seen_smiles.add(smiles)
                section["figures"].append(
                    {
                        "name": "Reported compound structure",
                        "image_path": "",
                        "caption": f"Chemical structure for reported SMILES: {smiles}",
                        "alt_text": "",
                        "artifact_path": "",
                        "structure_smiles": smiles,
                        "structure_type": "molecule",
                        "structure_id": "",
                        "structure_name": "Reported compound structure",
                        "node": "",
                        "description": f"Chemical structure for reported SMILES: {smiles}",
                        "after_paragraph_index": paragraph_index,
                        "_auto_structure_figure": True,
                    }
                )


def _row_value(row: dict[str, str], *keys: str) -> str:
    return _mapping_value(row, *keys)


def _table_has_column(table: dict[str, Any], *keys: str) -> bool:
    columns = {_key_token(column) for column in table["columns"]}
    return any(_key_token(key) in columns for key in keys)


def _table_structure_type(table: dict[str, Any], row: dict[str, str]) -> str:
    title = str(table["title"]).lower()
    if (
        _table_has_column(
            table,
            "Scaffold ID",
            "scaffold_id",
            "scaffold_ids",
            "Scaffold SMILES",
            "scaffold_smiles",
            "Scaffold Name",
            "scaffold_name",
        )
        or "scaffold" in title
        or _row_value(
            row,
            "Scaffold ID",
            "scaffold_id",
            "scaffold_ids",
            "Scaffold SMILES",
            "scaffold_smiles",
            "Scaffold Name",
            "scaffold_name",
        )
    ):
        return "scaffold"
    if (
        _table_has_column(
            table,
            "Molecule ID",
            "molecule_id",
            "molecule_ids",
            "molecule_chembl_id",
            "molecule_chembl_ids",
            "Compound ID",
            "compound_id",
            "compound_ids",
            "Molecule SMILES",
            "molecule_smiles",
            "Compound SMILES",
            "compound_smiles",
            "Molecule Name",
            "molecule_name",
            "Compound Name",
            "compound_name",
        )
        or "molecule" in title
        or "compound" in title
        or _row_value(
            row,
            "Molecule ID",
            "molecule_id",
            "molecule_ids",
            "molecule_chembl_id",
            "molecule_chembl_ids",
            "Compound ID",
            "compound_id",
            "compound_ids",
            "Molecule SMILES",
            "molecule_smiles",
            "Compound SMILES",
            "compound_smiles",
            "Molecule Name",
            "molecule_name",
            "Compound Name",
            "compound_name",
        )
    ):
        return "molecule"
    if _table_has_column(table, "chembl id", "chembl_id", "chembl_ids"):
        return "molecule"
    if _row_value(
        row,
        "Scaffold ID",
        "Scaffold",
        "scaffold_id",
        "scaffold_ids",
        "scaffold_name",
    ):
        return "scaffold"
    if _row_value(
        row,
        "Molecule ID",
        "Molecule",
        "molecule_id",
        "molecule_ids",
        "molecule_chembl_id",
        "molecule_chembl_ids",
        "molecule_name",
        "Compound ID",
        "Compound",
        "compound_id",
        "compound_ids",
        "compound_name",
    ):
        return "molecule"
    if _row_value(row, *_STRUCTURE_ID_FIELDS) and _row_value(row, "SMILES", "Structure SMILES"):
        return "molecule"
    return ""


def _ensure_table_id_column(table: dict[str, Any], structure_type: str) -> str:
    prefix = _structure_id_prefix(structure_type)
    preferred_columns = [f"{prefix} ID", "Structure ID", "ID"]
    for column in preferred_columns:
        for existing in table["columns"]:
            if existing.lower() == column.lower():
                return existing

    id_column = f"{prefix} ID"
    table["columns"].insert(0, id_column)
    for row in table["rows"]:
        row.setdefault(id_column, "")
    return id_column


def _track_structure_id_target(
    figure: dict[str, Any],
    row: dict[str, str],
    column: str,
) -> None:
    if not column:
        return
    figure.setdefault("_source_id_targets", []).append((row, column))


def _sync_structure_id_targets(figure: dict[str, Any]) -> None:
    structure_id = figure.get("structure_id", "")
    if not structure_id:
        return
    for row, column in figure.get("_source_id_targets", []):
        if isinstance(row, dict) and column and not row.get(column):
            row[column] = structure_id


def _should_apply_source_structure_id(
    current_id: str,
    source_id: str,
    structure_type: str,
) -> bool:
    if not source_id:
        return False
    if not current_id:
        return True
    return _structure_id_count(current_id, structure_type) > 0


def _find_first_mention_index(section: dict[str, Any], terms: list[str]) -> Optional[int]:
    mention_index = _find_actual_mention_index(section, terms)
    if mention_index is not None:
        return mention_index
    return len(section["paragraphs"]) - 1 if section["paragraphs"] else None


def _find_actual_mention_index(section: dict[str, Any], terms: list[str]) -> Optional[int]:
    normalized_terms = [term.lower() for term in terms if term]
    for paragraph_index, paragraph in enumerate(section["paragraphs"]):
        normalized_paragraph = _strip_smiles_tags(paragraph).lower()
        if any(term in normalized_paragraph for term in normalized_terms):
            return paragraph_index
    return None


def _section_heading_matches(section: dict[str, Any], terms: tuple[str, ...]) -> bool:
    heading = str(section.get("heading", "")).lower()
    return any(term in heading for term in terms)


def _find_context_section(
    sections: list[dict[str, Any]],
    heading_terms: tuple[str, ...],
    paragraph_terms: list[str],
) -> tuple[Optional[dict[str, Any]], Optional[int]]:
    for section in sections:
        if _section_heading_matches(section, heading_terms):
            return section, _find_first_mention_index(section, paragraph_terms)

    for section in sections:
        paragraph_index = _find_actual_mention_index(section, paragraph_terms)
        if paragraph_index is not None:
            return section, paragraph_index

    return None, None


def _figure_context_text(figure: dict[str, Any]) -> str:
    metadata = normalize_figure_metadata(figure.get("figure_metadata"))
    bits = [
        _clean_text(figure.get(key))
        for key in (
            "name",
            "caption",
            "artifact_path",
            "image_path",
            "structure_id",
            "structure_name",
            "description",
            "figure_id",
        )
    ]
    bits.extend(
        [
            metadata.get("figure_kind", ""),
            metadata.get("renderer", ""),
            metadata.get("title_subject", ""),
            " ".join(str(node) for node in metadata.get("node_labels") or []),
        ]
    )
    return " ".join(_clean_text(bit) for bit in bits).lower()


def _mentioned_gtm_nodes(text: Any) -> set[int]:
    nodes: set[int] = set()
    plain_text = _strip_html_tags(_strip_smiles_tags(_clean_text(text)))
    for match in _GTM_NODE_MENTION_RX.finditer(plain_text):
        nodes.update(int(value) for value in re.findall(r"\d+", match.group("values")))
    return nodes


def _metadata_node_labels(figure: dict[str, Any]) -> set[int]:
    metadata = normalize_figure_metadata(figure.get("figure_metadata"))
    labels = set()
    for label in metadata.get("node_labels") or []:
        try:
            labels.add(int(label))
        except (TypeError, ValueError):
            continue
    return labels


def _is_gtm_node_plot_figure(figure: dict[str, Any]) -> bool:
    if figure.get("structure_smiles"):
        return False

    metadata = normalize_figure_metadata(figure.get("figure_metadata"))
    figure_kind = _clean_text(metadata.get("figure_kind")).lower()
    if figure_kind.startswith("gtm_"):
        return True

    context = _figure_context_text(figure)
    return ("gtm" in context or "landscape" in context) and any(
        term in context for term in ("density", "activity", "map", "potency", "pchembl")
    )


def _validate_gtm_node_labels(sections: list[dict[str, Any]]) -> None:
    for section in sections:
        figures_by_paragraph: dict[int, list[dict[str, Any]]] = {}
        for figure in section["figures"]:
            paragraph_index = figure.get("after_paragraph_index")
            if not isinstance(paragraph_index, int) and len(section["paragraphs"]) == 1:
                paragraph_index = 0
            if isinstance(paragraph_index, int):
                figures_by_paragraph.setdefault(paragraph_index, []).append(figure)

        for paragraph_index, paragraph in enumerate(section["paragraphs"]):
            mentioned_nodes = _mentioned_gtm_nodes(paragraph)
            if not mentioned_nodes:
                continue

            gtm_figures = [
                figure
                for figure in figures_by_paragraph.get(paragraph_index, [])
                if _is_gtm_node_plot_figure(figure)
            ]
            if not gtm_figures:
                continue

            labeled_nodes: set[int] = set()
            for figure in gtm_figures:
                labeled_nodes.update(_metadata_node_labels(figure))

            missing_nodes = sorted(mentioned_nodes - labeled_nodes)
            if missing_nodes:
                missing_text = ", ".join(str(node) for node in missing_nodes)
                expected_nodes = ", ".join(str(node) for node in sorted(mentioned_nodes))
                raise ValueError(
                    "GTM node labels missing for section "
                    f"'{section['heading']}', paragraph {paragraph_index}: "
                    f"{missing_text}. Regenerate the matching GTM plot with "
                    f"mark_nodes=[{expected_nodes}] before saving the report."
                )


def _is_gtm_landscape_context(context: str) -> bool:
    return (
        "gtm" in context or "landscape" in context or "density" in context or "activity" in context
    )


def _remove_sentences_matching(text: str, *terms: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = [
        sentence
        for sentence in sentences
        if not all(term.lower() in sentence.lower() for term in terms)
    ]
    return " ".join(sentence for sentence in kept if sentence).strip()


_VISUAL_ENCODING_TERMS = (
    "color",
    "colour",
    "colorscale",
    "colour scale",
    "color scale",
    "legend",
    "red",
    "orange",
    "blue",
    "green",
    "purple",
    "gray",
    "grey",
    "grayscale",
    "greyscale",
    "dark",
    "light",
    "marker",
)


def _sentence_mentions_visual_encoding(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(term in lowered for term in _VISUAL_ENCODING_TERMS)


def _remove_visual_encoding_sentences(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = [
        sentence
        for sentence in sentences
        if sentence.strip() and not _sentence_mentions_visual_encoding(sentence)
    ]
    return " ".join(kept).strip()


def _append_missing_sentences(text: str, sentences: list[str]) -> str:
    current = _clean_text(text)
    current_lower = current.lower()
    additions = []
    for sentence in sentences:
        cleaned = _clean_text(sentence)
        if cleaned and cleaned.lower() not in current_lower:
            additions.append(cleaned)
    return " ".join(part for part in [current, *additions] if part).strip()


def _apply_figure_metadata_text(figure: dict[str, Any]) -> bool:
    metadata = normalize_figure_metadata(figure.get("figure_metadata"))
    if not metadata:
        return False

    caption_facts = figure_caption_facts(metadata)
    if caption_facts:
        caption = _clean_text(figure.get("caption"))
        if any(_sentence_mentions_visual_encoding(sentence) for sentence in caption_facts):
            caption = _remove_visual_encoding_sentences(caption)
        figure["caption"] = _append_missing_sentences(caption, caption_facts)

    if metadata.get("figure_kind") == "gtm_density":
        name = figure.get("name", "")
        if name:
            figure["name"] = _strip_density_name_color_notes(name)
        alt_text = figure.get("alt_text", "")
        if alt_text and ("blue" in alt_text.lower() or "red" in alt_text.lower()):
            figure["alt_text"] = _strip_density_name_color_notes(alt_text)
    return True


def _strip_density_name_color_notes(name: str) -> str:
    corrected = re.sub(r"\s*\(blue\)", "", name, flags=re.IGNORECASE)

    def _red_replacement(match: re.Match[str]) -> str:
        prefix = corrected[max(0, match.start() - 50) : match.start()].lower()
        return match.group(0) if "projected" in prefix else ""

    corrected = re.sub(r"\s*\(red\)", _red_replacement, corrected, flags=re.IGNORECASE)
    corrected = re.sub(
        r"\b(?:blue|red)\s+(?=nodes?\b)",
        "",
        corrected,
        flags=re.IGNORECASE,
    )
    return " ".join(corrected.split()).strip()


def _drop_plotly_report_outputs(figures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept = []
    for figure in figures:
        image_path = _clean_text(figure.get("image_path"))
        artifact_path = _clean_text(figure.get("artifact_path"))
        context = _figure_context_text(figure)
        metadata = normalize_figure_metadata(figure.get("figure_metadata"))
        renderer = _clean_text(metadata.get("renderer")).lower()
        report_role = _clean_text(metadata.get("report_role"))

        if (
            metadata
            and report_role not in REPORTABLE_FIGURE_ROLES
            and not figure.get("structure_smiles")
        ):
            continue

        if renderer == "plotly" or "plotly" in image_path.lower():
            continue

        if artifact_path and (
            metadata or "plotly" in artifact_path.lower() or _is_gtm_landscape_context(context)
        ):
            figure["artifact_path"] = ""
            figure["caption"] = _remove_sentences_matching(figure.get("caption", ""), "plotly")

        kept.append(figure)
    return kept


def _correct_density_caption_color_scale(figure: dict[str, Any]) -> None:
    if _apply_figure_metadata_text(figure):
        return

    context = _figure_context_text(figure)
    caption = figure.get("caption", "")
    caption_lower = caption.lower()
    name = figure.get("name", "")
    name_lower = name.lower()
    if "density" not in context:
        return

    if "blue" in name_lower or "red" in name_lower:
        original_name = name
        figure["name"] = _strip_density_name_color_notes(name)
        alt_text = figure.get("alt_text", "")
        alt_text_lower = alt_text.lower()
        if alt_text == original_name or "blue" in alt_text_lower or "red" in alt_text_lower:
            figure["alt_text"] = _strip_density_name_color_notes(alt_text)

    if "density" not in caption_lower:
        return

    if "red" in caption_lower and "projected" in caption_lower and "blue" not in caption_lower:
        return

    if "red/orange" in caption_lower or ("red" in caption_lower and "blue" in caption_lower):
        corrected = _remove_sentences_matching(caption, "red", "blue")
        corrected = _remove_sentences_matching(corrected, "red/orange")
        replacement = (
            "The density-cell colors follow the plot legend; higher legend values indicate "
            "more populated nodes and lower legend values indicate sparse or empty nodes."
        )
        figure["caption"] = f"{corrected} {replacement}".strip()


def _prepare_report_figures_for_rendering(
    sections: list[dict[str, Any]],
    figures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for section in sections:
        section["figures"] = _drop_plotly_report_outputs(section["figures"])
        for figure in section["figures"]:
            _correct_density_caption_color_scale(figure)

    figures = _drop_plotly_report_outputs(figures)
    for figure in figures:
        _correct_density_caption_color_scale(figure)
    return figures


def _move_contextual_top_level_figures(
    sections: list[dict[str, Any]],
    figures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not sections or not figures:
        return figures

    remaining = []
    for figure in figures:
        target_section = None
        target_paragraph_index = None
        context = _figure_context_text(figure)

        if figure.get("structure_smiles"):
            structure_terms = [
                figure.get("structure_id", ""),
                figure.get("structure_name", ""),
                figure.get("structure_smiles", ""),
            ]
            for section in sections:
                target_paragraph_index = _find_actual_mention_index(section, structure_terms)
                if target_paragraph_index is not None:
                    target_section = section
                    break
        elif "density" in context and ("gtm" in context or "landscape" in context):
            target_section, target_paragraph_index = _find_context_section(
                sections,
                ("density",),
                ["density", "density map", "density landscape", "gtm map"],
            )
        elif any(term in context for term in ("activity", "potency", "pchembl")) and (
            "gtm" in context or "landscape" in context
        ):
            target_section, target_paragraph_index = _find_context_section(
                sections,
                ("activity", "sar", "potency"),
                ["activity", "potency", "pchembl", "activity landscape"],
            )

        if target_section is None:
            remaining.append(figure)
            continue

        if figure.get("after_paragraph_index") is None:
            figure["after_paragraph_index"] = target_paragraph_index
        target_section["figures"].append(figure)

    return remaining


def _append_structure_table_figures(
    sections: list[dict[str, Any]],
    figures: list[dict[str, Any]],
) -> None:
    structure_figures_by_smiles = {
        figure["structure_smiles"]: figure for figure in figures if figure.get("structure_smiles")
    }
    for section in sections:
        for figure in section["figures"]:
            if figure.get("structure_smiles"):
                structure_figures_by_smiles.setdefault(figure["structure_smiles"], figure)

    for section in sections:
        for table in section["tables"]:
            for row in table["rows"]:
                structure_type = _table_structure_type(table, row)
                if not structure_type:
                    continue
                smiles = _row_value(
                    row,
                    "SMILES",
                    "smiles",
                    "Structure SMILES",
                    "structure_smiles",
                    "Scaffold SMILES",
                    "scaffold_smiles",
                    "Molecule SMILES",
                    "molecule_smiles",
                    "Compound SMILES",
                    "compound_smiles",
                )
                if not smiles:
                    continue

                identity = _resolve_structure_identity(row, structure_type)
                structure_id = identity["structure_id"]
                source_id_column = ""
                if not structure_id:
                    source_id_column = _ensure_table_id_column(table, structure_type)
                structure_name = identity["structure_name"]
                description = _row_value(row, "Description", "Notes", "Comment")
                node = _row_value(row, "Node", "GTM Node", "GTM Node ID", "node", "node_index")
                first_mention_index = _find_first_mention_index(
                    section,
                    [structure_id, structure_name, smiles],
                )
                caption_subject = ": ".join(part for part in (structure_id, structure_name) if part)
                caption = description or f"Chemical structure for {caption_subject or smiles}."

                existing_figure = structure_figures_by_smiles.get(smiles)
                if existing_figure is not None:
                    if not existing_figure.get("structure_type") or existing_figure.get(
                        "_auto_structure_figure"
                    ):
                        existing_figure["structure_type"] = structure_type
                    current_id = existing_figure.get("structure_id", "")
                    if _should_apply_source_structure_id(current_id, structure_id, structure_type):
                        existing_figure["structure_id"] = structure_id
                    current_name = existing_figure.get("structure_name", "")
                    if structure_name and (
                        not current_name
                        or current_name == "Reported compound structure"
                        or existing_figure.get("_auto_structure_figure")
                    ):
                        existing_figure["structure_name"] = structure_name
                    if node and not existing_figure.get("node"):
                        existing_figure["node"] = node
                    if description and not existing_figure.get("description"):
                        existing_figure["description"] = description
                    if (
                        first_mention_index is not None
                        and existing_figure.get("after_paragraph_index") is None
                    ):
                        existing_figure["after_paragraph_index"] = first_mention_index
                    _track_structure_id_target(existing_figure, row, source_id_column)
                    _sync_structure_id_targets(existing_figure)
                    continue

                figure = {
                    "name": structure_name or structure_id or "Reported structure",
                    "image_path": "",
                    "caption": caption,
                    "alt_text": "",
                    "artifact_path": "",
                    "structure_smiles": smiles,
                    "structure_type": structure_type,
                    "structure_id": structure_id,
                    "structure_name": structure_name,
                    "node": node,
                    "description": description,
                    "after_paragraph_index": first_mention_index,
                }
                _track_structure_id_target(figure, row, source_id_column)
                structure_figures_by_smiles[smiles] = figure
                section["figures"].append(figure)


def _structure_id_prefix(structure_type: str) -> str:
    return "Scaffold" if structure_type == "scaffold" else "Molecule"


def _generated_structure_id(structure_type: str, count: int) -> str:
    return f"{_structure_id_prefix(structure_type)}_{count}"


def _structure_id_count(structure_id: str, structure_type: str) -> int:
    match = _STRUCTURE_ID_RX.match(structure_id)
    if not match:
        return 0
    if match.group("type").lower() != _structure_id_prefix(structure_type).lower():
        return 0
    return int(match.group("number"))


def _structure_figure_name(figure: dict[str, Any]) -> str:
    structure_id = figure.get("structure_id", "")
    structure_name = figure.get("structure_name", "")
    if structure_id and structure_name:
        return f"{structure_id}: {structure_name}"
    return structure_id or structure_name or figure.get("name", "")


def _figure_label(figure: dict[str, Any]) -> str:
    match = re.match(r"^(Figure\s+\d+)\.", figure.get("name", ""))
    return match.group(1) if match else "the following figure"


def _structure_reference_text(figure: dict[str, Any]) -> str:
    if not figure.get("structure_smiles"):
        return ""
    structure_id = figure.get("structure_id", "")
    structure_name = figure.get("structure_name", "")
    if not structure_id and not structure_name:
        return ""

    subject = structure_id
    if structure_name and structure_name != structure_id:
        subject = f"{subject} ({structure_name})" if subject else structure_name
    return f"{subject} is shown in {_figure_label(figure)}."


def _assign_structure_labels(
    sections: list[dict[str, Any]],
    figures: list[dict[str, Any]],
) -> None:
    counters = {"scaffold": 0, "molecule": 0}
    all_figures = list(_iter_report_figures(sections, figures))
    for figure in all_figures:
        structure_type = figure.get("structure_type", "")
        if not figure.get("structure_smiles") or structure_type not in counters:
            continue
        counters[structure_type] = max(
            counters[structure_type],
            _structure_id_count(figure.get("structure_id", ""), structure_type),
        )

    for figure in all_figures:
        structure_smiles = figure.get("structure_smiles", "")
        if not structure_smiles:
            continue
        structure_type = figure.get("structure_type") or "molecule"
        if structure_type not in counters:
            structure_type = "molecule"
            figure["structure_type"] = structure_type
        if not figure.get("structure_id"):
            counters[structure_type] += 1
            figure["structure_id"] = _generated_structure_id(
                structure_type, counters[structure_type]
            )
        _sync_structure_id_targets(figure)
        if not figure.get("structure_name"):
            figure["structure_name"] = _figure_title(figure.get("name", "")) or _caption_title(
                figure.get("caption", "")
            )
        figure["name"] = _structure_figure_name(figure)


def _iter_report_figures(
    sections: list[dict[str, Any]],
    figures: list[dict[str, Any]],
):
    for section in sections:
        yield from section["figures"]
    yield from figures


def _structure_image_filename(smiles: str, index: int, basename: str) -> str:
    digest = hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:12]
    return f"{basename}_structure_{index:03d}_{digest}{_PNG_EXTENSION}"


def _smiles_structure_png(smiles: str) -> bytes:
    from .formatting import smiles_to_png_bytes

    try:
        return smiles_to_png_bytes(smiles, size=(320, 240))
    except ValueError as exc:
        raise ValueError(f"invalid structure SMILES: {smiles}") from exc


def _materialize_structure_figures(
    sections: list[dict[str, Any]],
    figures: list[dict[str, str]],
    basename: str,
    report_type: Optional[str],
    session_state: Optional[Dict[str, Any]],
) -> None:
    for index, figure in enumerate(_iter_report_figures(sections, figures), start=1):
        smiles = figure.get("structure_smiles", "")
        if not smiles or figure.get("image_path"):
            continue

        filename = _structure_image_filename(smiles, index, basename)
        rel_path = operation_rel_path(
            OutputOperation.REPORTS,
            _report_slug(report_type),
            "assets",
            "structures",
            filename,
            session_state=session_state,
            workflow_slug="reports",
        )
        try:
            figure["image_path"] = _write_binary_report(_smiles_structure_png(smiles), rel_path)
        except ValueError as exc:
            raise ValueError(f"figure {index} has {exc}") from exc


def _image_data_url(image_path: str) -> str:
    image_bytes = _read_binary_path(image_path)
    if not image_bytes:
        raise ValueError(f"image file is empty: {image_path}")

    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{get_mime_type(image_path)};base64,{encoded}"


def _html_text_block(text: str) -> str:
    text = _strip_smiles_tags(text)
    stripped = text.strip()
    if not stripped:
        return ""
    rendered = _html_inline_markup(stripped)
    lstripped = text.lstrip()
    if "\n" in text and ("|" in text or lstripped.startswith(("- ", "* "))):
        return f"<pre>{rendered}</pre>"
    return f"<p>{rendered.replace(chr(10), '<br>')}</p>"


def _render_html_figure(figure: dict[str, str], embed_images: bool) -> str:
    name = _html_inline_markup(figure["name"])
    image_path = figure["image_path"]
    caption = _html_inline_markup(figure["caption"])
    alt_text = html.escape(figure["alt_text"])
    artifact_path = figure["artifact_path"]

    image_markup = ""
    if image_path:
        try:
            source = _image_data_url(image_path) if embed_images else image_path
        except Exception as exc:
            logger.warning("Could not embed report image %s: %s", image_path, exc)
            source = image_path
        image_markup = (
            f'<img src="{html.escape(source, quote=True)}" alt="{alt_text}" loading="lazy">'
        )

    artifact_markup = ""
    if artifact_path:
        artifact_markup = (
            '<p class="artifact">Interactive artifact: '
            f"<code>{html.escape(artifact_path)}</code></p>"
        )

    if not image_markup and not artifact_markup:
        return ""

    return (
        "<figure>"
        f'<h3 class="figure-title">{name}</h3>'
        f"{image_markup}"
        f"<figcaption>{caption}</figcaption>"
        f"{artifact_markup}"
        "</figure>"
    )


def _render_html_figure_reference(figure: dict[str, Any]) -> str:
    reference = _structure_reference_text(figure)
    if not reference:
        return ""
    return f'<p class="figure-reference">{_html_inline_markup(reference)}</p>'


def _render_html_table(table: dict[str, Any]) -> str:
    header = "".join(f"<th>{_html_inline_markup(column)}</th>" for column in table["columns"])
    rows = []
    for row in table["rows"]:
        cells = "".join(
            f"<td>{_html_inline_markup(row.get(column, ''))}</td>" for column in table["columns"]
        )
        rows.append(f"<tr>{cells}</tr>")

    return (
        '<div class="table-block">'
        f"<h3>{_html_inline_markup(table['title'])}</h3>"
        "<table>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


def _render_html_report(
    title: str,
    summary: list[str],
    sections: list[dict[str, Any]],
    figures: list[dict[str, Any]],
    embed_images: bool,
) -> str:
    escaped_title = html.escape(title)
    rendered_title = _html_inline_markup(title)
    generated = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    summary_markup = ""
    if summary:
        items = "".join(f"<li>{_html_inline_markup(item)}</li>" for item in summary)
        summary_markup = f"<section><h2>Executive Summary</h2><ul>{items}</ul></section>"

    section_markup = []
    for section in sections:
        placed_figures, unplaced_figures = _group_section_figures(section)
        section_blocks = []
        for paragraph_index, paragraph in enumerate(section["paragraphs"]):
            section_blocks.append(_html_text_block(paragraph))
            section_blocks.extend(
                _render_html_figure_reference(figure) + _render_html_figure(figure, embed_images)
                for figure in placed_figures.get(paragraph_index, [])
            )
        section_blocks.extend(_render_html_table(table) for table in section["tables"])
        section_blocks.extend(
            _render_html_figure_reference(figure) + _render_html_figure(figure, embed_images)
            for figure in unplaced_figures
        )
        section_markup.append(
            "<section>"
            f"<h2>{_html_inline_markup(section['heading'])}</h2>"
            f"{''.join(section_blocks)}"
            "</section>"
        )

    figures_markup = ""
    if figures:
        rendered = "".join(
            _render_html_figure_reference(figure) + _render_html_figure(figure, embed_images)
            for figure in figures
        )
        figures_markup = f"<section><h2>Visualizations</h2>{rendered}</section>"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    :root {{
      color: #1f2933;
      background: #f7f9fb;
      font-family: Arial, Helvetica, sans-serif;
    }}
    body {{
      margin: 0;
      padding: 32px;
    }}
    main {{
      max-width: 980px;
      margin: 0 auto;
      background: #ffffff;
      padding: 40px;
      border: 1px solid #d9e2ec;
    }}
    h1, h2 {{
      color: #102a43;
      line-height: 1.25;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
    }}
    h2 {{
      margin-top: 32px;
      font-size: 21px;
    }}
    p, li {{
      font-size: 15px;
      line-height: 1.58;
    }}
    .meta {{
      color: #627d98;
      margin-top: 0;
    }}
    figure {{
      margin: 24px 0;
      padding: 16px;
      border: 1px solid #d9e2ec;
      background: #f8fbff;
    }}
    img {{
      display: block;
      max-width: 100%;
      height: auto;
      margin: 0 auto;
    }}
    figcaption {{
      margin-top: 10px;
      color: #334e68;
      font-size: 14px;
      line-height: 1.45;
      text-align: center;
    }}
    .figure-title {{
      margin: 0 0 12px;
      color: #243b53;
      font-size: 16px;
      line-height: 1.35;
    }}
    code, pre {{
      background: #eef2f7;
      border-radius: 4px;
      padding: 2px 4px;
    }}
    pre {{
      overflow-x: auto;
      padding: 12px;
      white-space: pre-wrap;
    }}
    .artifact {{
      color: #52606d;
      font-size: 13px;
      text-align: center;
    }}
    .figure-reference {{
      color: #334e68;
      font-size: 14px;
      font-weight: 600;
      margin: 14px 0 0;
    }}
    .table-block {{
      margin: 24px 0;
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      line-height: 1.4;
    }}
    th, td {{
      border: 1px solid #d9e2ec;
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #eef2f7;
      color: #243b53;
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>{rendered_title}</h1>
      <p class="meta">Generated: {generated}</p>
    </header>
    {summary_markup}
    {''.join(section_markup)}
    {figures_markup}
  </main>
</body>
</html>
"""


def _markdown_figure(figure: dict[str, str]) -> str:
    lines = []
    reference = _structure_reference_text(figure)
    if reference:
        lines.extend([reference, ""])
    lines.append(f"### {figure['name']}")
    if figure["image_path"]:
        lines.append(f"![{figure['name']}]({figure['image_path']})")
    if figure["caption"]:
        lines.append(f"*{figure['caption']}*")
    if figure["artifact_path"]:
        lines.append(f"Interactive artifact: `{figure['artifact_path']}`")
    return "\n".join(lines)


def _markdown_table_cell(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|").strip()


def _markdown_table(table: dict[str, Any]) -> str:
    lines = [f"### {table['title']}"]
    header = "| " + " | ".join(_markdown_table_cell(column) for column in table["columns"]) + " |"
    divider = "| " + " | ".join("---" for _column in table["columns"]) + " |"
    lines.extend([header, divider])
    for row in table["rows"]:
        lines.append(
            "| "
            + " | ".join(_markdown_table_cell(row.get(column, "")) for column in table["columns"])
            + " |"
        )
    return "\n".join(lines)


def _render_markdown_report(
    title: str,
    summary: list[str],
    sections: list[dict[str, Any]],
    figures: list[dict[str, Any]],
) -> str:
    generated = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# {title}", f"*Generated: {generated}*", ""]

    if summary:
        lines.extend(["## Executive Summary", ""])
        lines.extend(f"- {item}" for item in summary)
        lines.append("")

    for section in sections:
        lines.extend([f"## {section['heading']}", ""])
        placed_figures, unplaced_figures = _group_section_figures(section)
        for paragraph_index, paragraph in enumerate(section["paragraphs"]):
            lines.extend([_strip_smiles_tags(paragraph), ""])
            for figure in placed_figures.get(paragraph_index, []):
                rendered = _markdown_figure(figure)
                if rendered:
                    lines.extend([rendered, ""])
        for table in section["tables"]:
            rendered = _markdown_table(table)
            if rendered:
                lines.extend([rendered, ""])
        for figure in unplaced_figures:
            rendered = _markdown_figure(figure)
            if rendered:
                lines.extend([rendered, ""])

    if figures:
        lines.extend(["## Visualizations", ""])
        for figure in figures:
            rendered = _markdown_figure(figure)
            if rendered:
                lines.extend([rendered, ""])

    return "\n".join(lines).rstrip() + "\n"


def _render_pdf_report(
    title: str,
    summary: list[str],
    sections: list[dict[str, Any]],
    figures: list[dict[str, Any]],
) -> bytes:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.lib.utils import ImageReader
        from reportlab.platypus import (
            Image,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise RuntimeError("PDF report export requires reportlab to be installed") from exc

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.72 * inch,
        leftMargin=0.72 * inch,
        topMargin=0.72 * inch,
        bottomMargin=0.72 * inch,
    )
    styles = getSampleStyleSheet()
    story = []
    generated = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    story.append(Paragraph(_pdf_inline_markup(title), styles["Title"]))
    story.append(Paragraph(f"Generated: {generated}", styles["Normal"]))
    story.append(Spacer(1, 0.18 * inch))

    if summary:
        story.append(Paragraph("Executive Summary", styles["Heading2"]))
        for item in summary:
            story.append(Paragraph(f"- {_pdf_inline_markup(item)}", styles["BodyText"]))
        story.append(Spacer(1, 0.12 * inch))

    def add_figure(figure: dict[str, str]) -> None:
        reference = _structure_reference_text(figure)
        if reference:
            story.append(Paragraph(_pdf_inline_markup(reference), styles["BodyText"]))
            story.append(Spacer(1, 0.04 * inch))
        if figure["name"]:
            story.append(Paragraph(_pdf_inline_markup(figure["name"]), styles["Heading3"]))
        image_path = figure["image_path"]
        if image_path:
            try:
                image_bytes = _read_binary_path(image_path)
                image_buffer = io.BytesIO(image_bytes)
                reader = ImageReader(io.BytesIO(image_bytes))
                width, height = reader.getSize()
                max_width = document.width
                max_height = 4.7 * inch
                scale = min(max_width / width, max_height / height, 1.0)
                story.append(Image(image_buffer, width=width * scale, height=height * scale))
            except Exception as exc:
                logger.warning("Could not add report image %s to PDF: %s", image_path, exc)
                story.append(
                    Paragraph(f"Image unavailable: {html.escape(image_path)}", styles["Italic"])
                )

        if figure["caption"]:
            story.append(Paragraph(_pdf_inline_markup(figure["caption"]), styles["Italic"]))
        if figure["artifact_path"]:
            story.append(
                Paragraph(
                    f"Interactive artifact: {html.escape(figure['artifact_path'])}",
                    styles["Normal"],
                )
            )
        story.append(Spacer(1, 0.16 * inch))

    def add_table(table: dict[str, Any]) -> None:
        story.append(Paragraph(_pdf_inline_markup(table["title"]), styles["Heading3"]))
        table_data = [
            [
                Paragraph(_pdf_inline_markup(column), styles["BodyText"])
                for column in table["columns"]
            ]
        ]
        for row in table["rows"]:
            table_data.append(
                [
                    Paragraph(_pdf_inline_markup(row.get(column, "")), styles["BodyText"])
                    for column in table["columns"]
                ]
            )
        column_width = document.width / max(len(table["columns"]), 1)
        pdf_table = Table(
            table_data,
            colWidths=[column_width] * len(table["columns"]),
            repeatRows=1,
            hAlign="LEFT",
        )
        pdf_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#243b53")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d9e2ec")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(pdf_table)
        story.append(Spacer(1, 0.16 * inch))

    for section in sections:
        story.append(Paragraph(_pdf_inline_markup(section["heading"]), styles["Heading2"]))
        placed_figures, unplaced_figures = _group_section_figures(section)
        for paragraph_index, paragraph in enumerate(section["paragraphs"]):
            rendered = _pdf_inline_markup(paragraph).replace("\n", "<br/>")
            story.append(Paragraph(rendered, styles["BodyText"]))
            story.append(Spacer(1, 0.08 * inch))
            for figure in placed_figures.get(paragraph_index, []):
                add_figure(figure)
        for table in section["tables"]:
            add_table(table)
        for figure in unplaced_figures:
            add_figure(figure)

    if figures:
        story.append(Paragraph("Visualizations", styles["Heading2"]))
        for figure in figures:
            add_figure(figure)

    document.build(story)
    return buffer.getvalue()


def _normalize_formats(formats: Optional[list[str]]) -> list[str]:
    if formats is None:
        return ["html", "pdf"]

    normalized = []
    for value in formats:
        fmt = str(value).strip().lower()
        if fmt not in _SUPPORTED_RICH_FORMATS:
            raise ValueError(
                f"unsupported report format {value!r}; expected one of "
                f"{sorted(_SUPPORTED_RICH_FORMATS)}"
            )
        fmt = "md" if fmt == "markdown" else fmt
        if fmt not in normalized:
            normalized.append(fmt)

    if not normalized:
        raise ValueError("formats cannot be empty")
    return normalized


def save_markdown_report(
    content: str,
    filename: Optional[str] = None,
    report_type: Optional[str] = None,
    session_state: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save a markdown report to the session-scoped storage (S3 or local).

    Use this tool at the end of a Report Generator run to persist the final
    markdown document so the user can download it. The file is written under
    ``workflows/<workflow_id>/reports/<report_type>/<filename>.md`` inside the
    active session prefix.

    Args:
        content: Full markdown text. Must be non-empty (whitespace-only rejected).
        filename: Optional filename. If omitted,
            ``<report_type>_<UTC_YYYYMMDD_HHMMSS>.md`` is generated. Any
            directory components are stripped, and the extension is normalised
            to ``.md``.
        report_type: Short slug (e.g. ``"chemotype"``, ``"gtm_density"``).
            Used for auto-generated filenames only.

    Returns:
        ``"Markdown report saved to S3: `<path>`"`` — same backticked format as
        ``save_gtm_plot``. Wrap the backticked path in ``<file>...</file>`` when
        echoing to the user so Chainlit renders it as a download bubble.

    Raises:
        ValueError: If ``content`` is empty or whitespace-only.
    """
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content cannot be empty")

    name = _report_filename(filename, _MD_EXTENSION, report_type)
    rel_path = _report_rel_path(name, report_type, session_state)

    try:
        full_path = _write_text_report(content, rel_path)
        if session_state is not None:
            register_session_object(
                session_state,
                "report",
                {
                    "report_type": report_type or _DEFAULT_REPORT_TYPE,
                    "paths": {"Markdown": full_path},
                    "format": "markdown",
                },
                label=name,
                source_tool="save_markdown_report",
                set_current=True,
            )
        logger.info(f"Markdown report saved to {full_path}")
        return f"Markdown report saved to S3: `{full_path}`"
    except Exception as e:
        logger.error(f"Error saving markdown report to {rel_path}: {e}")
        raise


def save_rich_report(
    title: str,
    summary: Optional[list[str]] = None,
    sections: Optional[list[dict[str, Any]]] = None,
    figures: Optional[list[dict[str, Any]]] = None,
    filename: Optional[str] = None,
    report_type: Optional[str] = None,
    formats: Optional[list[str]] = None,
    embed_images: bool = True,
    session_state: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save an image-rich report to session-scoped storage.

    Use this tool when a report should place explanatory text and static images
    in the same document. HTML and PDF are the default outputs. HTML embeds
    readable image files as data URLs by default, while PDF embeds the images
    directly into the document. A Markdown companion can be requested with
    ``formats=["html", "pdf", "md"]``.

    Args:
        title: Report title. Must be non-empty.
        summary: Optional executive-summary bullets.
        sections: Optional ordered sections. Each section may include
            ``heading``/``title``, ``paragraphs``/``content``/``text``, and
            ``figures``/``tables``. Section figures use the same shape as
            top-level ``figures``. SMILES wrapped in ``<smiles>...</smiles>``
            inside section paragraphs are automatically rendered as
            section-local molecule figures.
        figures: Optional top-level figure list. Each figure may include
            ``name`` (or ``figure_name``/``title``), ``caption``,
            ``image_path`` (or ``png_path``/``path``), ``alt_text``, and
            ``artifact_path`` (or ``html_path``) for an interactive companion
            file. A figure may alternatively include ``figure_id`` to resolve a
            registered session figure, or ``figure_metadata`` to drive image
            selection, title/caption facts, color semantics, overlays, and
            report inclusion. Structure figures may include ``structure_smiles`` (or
            ``smiles``), ``structure_type``, ``structure_id``,
            ``structure_name``, ``node``, ``description``, and
            ``after_paragraph_index`` to generate and place a molecule/scaffold
            PNG. Captions are required for every image/artifact/structure
            figure; names are normalized to sequential ``Figure N. ...`` labels.
        filename: Optional base filename. Directory components are stripped.
            The requested output extensions are applied automatically.
        report_type: Short slug used for auto-generated filenames.
        formats: Optional output formats. Supported values are ``"html"``,
            ``"pdf"``, ``"md"``, and ``"markdown"``. Defaults to HTML + PDF.
        embed_images: Whether HTML should embed readable images as data URLs.

    Returns:
        A labeled list of backticked saved paths, for example:
        ``"Rich report saved to S3:\\n- HTML: `<path>`\\n- PDF: `<path>`"``.

    Raises:
        ValueError: If the title is empty, formats are invalid, or no meaningful
            content is supplied.
    """
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title cannot be empty")

    normalized_summary = _as_strings(summary)
    normalized_sections = _normalize_sections(sections, session_state=session_state)
    normalized_figures = _normalize_figures(figures, session_state=session_state)
    _append_smiles_tag_figures(normalized_sections, normalized_figures)
    _append_plain_smiles_figures(normalized_sections, normalized_figures)
    _append_structure_table_figures(normalized_sections, normalized_figures)
    normalized_figures = _move_contextual_top_level_figures(normalized_sections, normalized_figures)
    normalized_figures = _prepare_report_figures_for_rendering(
        normalized_sections, normalized_figures
    )
    _validate_gtm_node_labels(normalized_sections)
    _assign_structure_labels(normalized_sections, normalized_figures)
    normalized_sections, normalized_figures = _renumber_report_figures(
        normalized_sections, normalized_figures
    )

    has_section_text = any(section["paragraphs"] for section in normalized_sections)
    has_section_figures = any(section["figures"] for section in normalized_sections)
    has_section_tables = any(section["tables"] for section in normalized_sections)
    has_report_body = (
        has_section_text or has_section_figures or has_section_tables or normalized_figures
    )
    normalized_formats = _normalize_formats(formats)
    if normalized_summary and not has_report_body:
        raise ValueError("report body content cannot be empty")
    if not normalized_summary and not has_report_body:
        raise ValueError("report content cannot be empty")

    basename = _rich_report_basename(filename, report_type)
    _materialize_structure_figures(
        normalized_sections,
        normalized_figures,
        basename,
        report_type,
        session_state,
    )
    saved_paths = []

    try:
        if "html" in normalized_formats:
            html_content = _render_html_report(
                title=title.strip(),
                summary=normalized_summary,
                sections=normalized_sections,
                figures=normalized_figures,
                embed_images=embed_images,
            )
            rel_path = _report_rel_path(
                f"{basename}{_HTML_EXTENSION}",
                report_type,
                session_state,
            )
            saved_paths.append(("HTML", _write_text_report(html_content, rel_path)))

        if "pdf" in normalized_formats:
            pdf_content = _render_pdf_report(
                title=title.strip(),
                summary=normalized_summary,
                sections=normalized_sections,
                figures=normalized_figures,
            )
            rel_path = _report_rel_path(
                f"{basename}{_PDF_EXTENSION}",
                report_type,
                session_state,
            )
            saved_paths.append(("PDF", _write_binary_report(pdf_content, rel_path)))

        if "md" in normalized_formats:
            markdown_content = _render_markdown_report(
                title=title.strip(),
                summary=normalized_summary,
                sections=normalized_sections,
                figures=normalized_figures,
            )
            rel_path = _report_rel_path(
                f"{basename}{_MD_EXTENSION}",
                report_type,
                session_state,
            )
            saved_paths.append(("Markdown", _write_text_report(markdown_content, rel_path)))

        logger.info("Rich report saved with outputs: %s", saved_paths)
        if session_state is not None:
            register_session_object(
                session_state,
                "report",
                {
                    "report_type": report_type or _DEFAULT_REPORT_TYPE,
                    "paths": dict(saved_paths),
                    "formats": [label for label, _path in saved_paths],
                    "figure_count": len(normalized_figures)
                    + sum(len(section["figures"]) for section in normalized_sections),
                },
                label=title.strip(),
                source_tool="save_rich_report",
                set_current=True,
            )
        formatted_paths = "\n".join(f"- {label}: `{path}`" for label, path in saved_paths)
        return f"Rich report saved to S3:\n{formatted_paths}"
    except Exception as e:
        logger.error(f"Error saving rich report {basename}: {e}")
        raise
