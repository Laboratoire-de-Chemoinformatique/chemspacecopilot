"""On-close confinement and versioned output names."""

from __future__ import annotations

from pathlib import Path

import pytest

from cs_copilot.storage import S3


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    S3.set_session_prefix("sessions/versions")
    return Path(S3.path(""))


def _write(path: str, text: str = "x") -> None:
    with S3.open(path, "w") as handle:
        handle.write(text)


def test_on_close_publishes_immediately_and_records_a_receipt(session):
    receipt: dict = {}
    with S3.confine_writes("workflows/r1", publication_receipt=receipt, commit_policy="on_close"):
        _write("workflows/r1/a.csv", "x,y\n1,2\n")
        # Code that reads its own output through other APIs keeps working.
        assert (session / "workflows" / "r1" / "a.csv").read_text().startswith("x,y")
    assert set(receipt) == {"workflows/r1/a.csv"}
    assert receipt["workflows/r1/a.csv"]["size_bytes"] == 8


def test_on_close_rolls_back_published_files_on_failure(session):
    with pytest.raises(RuntimeError):
        with S3.confine_writes("workflows/r1", commit_policy="on_close"):
            _write("workflows/r1/partial.csv")
            raise RuntimeError("tool failed")
    assert not S3.exists("workflows/r1/partial.csv")


def test_on_close_stays_create_only_and_confined(session):
    _write("workflows/r1/existing.csv")
    with S3.confine_writes("workflows/r1", commit_policy="on_close"):
        with pytest.raises(PermissionError, match="create-only"):
            _write("workflows/r1/existing.csv")
        with pytest.raises(PermissionError, match="outside the active boundary"):
            _write("elsewhere.csv")


def test_unknown_commit_policies_are_rejected(session):
    with pytest.raises(ValueError, match="commit policy"):
        with S3.confine_writes("workflows/r1", commit_policy="sometimes"):
            pass


@pytest.mark.parametrize(
    ("existing", "requested", "expected"),
    [
        ((), "workflows/r1/plot.png", "workflows/r1/plot.png"),
        (("workflows/r1/plot.png",), "workflows/r1/plot.png", "workflows/r1/plot-v2.png"),
        (
            ("workflows/r1/plot.png", "workflows/r1/plot-v2.png"),
            "workflows/r1/plot.png",
            "workflows/r1/plot-v3.png",
        ),
        (("workflows/r1/gtm.pkl.gz",), "workflows/r1/gtm.pkl.gz", "workflows/r1/gtm-v2.pkl.gz"),
        (("workflows/r1/README",), "workflows/r1/README", "workflows/r1/README-v2"),
    ],
)
def test_first_free_path_versions_existing_outputs(session, existing, requested, expected):
    for path in existing:
        _write(path)
    assert S3.first_free_path(requested) == expected


def test_first_free_path_keeps_the_callers_spelling(session):
    _write("workflows/r1/table.csv")
    expanded = S3.path("workflows/r1/table.csv")
    assert S3.first_free_path(expanded) == S3.path("workflows/r1/table-v2.csv")
