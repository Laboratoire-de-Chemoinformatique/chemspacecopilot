#!/usr/bin/env python
# coding: utf-8
"""Markdown report export tool for the Report Generator agent."""

import datetime
import logging
import re
from pathlib import Path
from typing import Optional

from cs_copilot.storage import S3

logger = logging.getLogger(__name__)

_REPORTS_DIR = "reports"
_MD_EXTENSION = ".md"
_DEFAULT_REPORT_TYPE = "report"
_SLUG_RX = re.compile(r"[^A-Za-z0-9_-]+")


def save_markdown_report(
    content: str,
    filename: Optional[str] = None,
    report_type: Optional[str] = None,
) -> str:
    """
    Save a markdown report to the session-scoped storage (S3 or local).

    Use this tool at the end of a Report Generator run to persist the final
    markdown document so the user can download it. The file is written under
    ``reports/<filename>.md`` inside the active session prefix.

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

    slug = _SLUG_RX.sub("_", (report_type or _DEFAULT_REPORT_TYPE).strip()).strip("_")
    if not slug:
        slug = _DEFAULT_REPORT_TYPE

    if filename:
        # Strip directory components to keep files inside reports/.
        name = Path(filename).name
        # Force .md extension, replacing any other suffix (or adding one if missing).
        name = Path(name).with_suffix(_MD_EXTENSION).name
    else:
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        name = f"{slug}_{timestamp}{_MD_EXTENSION}"

    rel_path = f"{_REPORTS_DIR}/{name}"

    try:
        with S3.open(rel_path, "w") as fh:
            fh.write(content)
        full_path = S3.path(rel_path)
        logger.info(f"Markdown report saved to {full_path}")
        return f"Markdown report saved to S3: `{full_path}`"
    except Exception as e:
        logger.error(f"Error saving markdown report to {rel_path}: {e}")
        raise
