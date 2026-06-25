"""Sanity tests for the lazy-import surface.

We don't have a clean way to exercise the missing-extra path inside the same
process (the mcp package is installed in the dev env). Instead, we assert
that the lazy import helper exposes the documented install hint string so
removal of the message can be caught by tests.
"""

from cs_copilot.mcp import lazy


def test_install_hint_mentions_extra():
    assert "uv sync --extra mcp" in lazy._INSTALL_HINT
    assert "cs_copilot[mcp]" in lazy._INSTALL_HINT


def test_require_mcp_does_not_raise_when_extra_present():
    # Should be a no-op in the dev env where mcp is installed.
    lazy.require_mcp()
