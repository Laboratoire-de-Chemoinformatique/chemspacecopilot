"""S3.observe_writes records how confined writes would treat each write."""

from __future__ import annotations

from pathlib import Path

import pytest

from cs_copilot.storage import S3


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    S3.set_session_prefix("sessions/observer")
    root = Path(S3.path(""))
    (root / "workflows" / "run-1").mkdir(parents=True)
    return root


def _write(path: str, mode: str = "w", text: str = "x") -> None:
    with S3.open(path, mode) as handle:
        handle.write(text)


def test_writes_are_classified_and_still_performed(session):
    _write("workflows/run-1/existing.csv", text="old")
    _write("outside.csv", text="old")

    with S3.observe_writes(
        "workflows/run-1", protected_paths=["workflows/run-1/registered.csv"]
    ) as observation:
        _write("workflows/run-1/new.csv")
        _write("workflows/run-1/existing.csv")
        _write("workflows/run-1/existing.csv", mode="a")
        _write("workflows/run-1/registered.csv")
        _write("workflows/run-1/manifest.json")
        _write("outside.csv")
        _write(str((session / "workflows" / "run-1" / "absolute.csv").absolute()))
        # S3.path() spellings resolve into the run but are denied by confinement.
        _write(S3.path("workflows/run-1/expanded.csv"))

    verdicts = {(entry["path"], entry["mode"]): entry["verdict"] for entry in observation.records}
    assert verdicts == {
        ("workflows/run-1/new.csv", "w"): "create_new",
        ("workflows/run-1/existing.csv", "w"): "overwrite_existing",
        ("workflows/run-1/existing.csv", "a"): "append_existing",
        ("workflows/run-1/registered.csv", "w"): "protected_artifact",
        ("workflows/run-1/manifest.json", "w"): "runtime_metadata",
        ("outside.csv", "w"): "outside_boundary",
        ("workflows/run-1/absolute.csv", "w"): "absolute_path",
        ("workflows/run-1/expanded.csv", "w"): "outside_boundary",
    }
    # Registration follows where new bytes actually landed inside the run.
    assert observation.created_paths() == [
        "workflows/run-1/new.csv",
        "workflows/run-1/absolute.csv",
        "workflows/run-1/expanded.csv",
    ]
    # Observation never changes the write itself.
    assert (session / "workflows" / "run-1" / "existing.csv").read_text() == "xx"
    assert (session / "outside.csv").read_text() == "x"
    assert (session / "workflows" / "run-1" / "manifest.json").exists()


def test_confined_writes_are_not_observed(session):
    with S3.observe_writes("workflows/run-1") as observation:
        with S3.confine_writes("workflows/run-1"):
            _write("workflows/run-1/confined.csv")
    assert observation.records == []


def test_reads_are_not_observed(session):
    _write("workflows/run-1/existing.csv")
    with S3.observe_writes("workflows/run-1") as observation:
        with S3.open("workflows/run-1/existing.csv", "r") as handle:
            handle.read()
    assert observation.records == []


def test_observation_is_scoped(session):
    with S3.observe_writes("workflows/run-1"):
        pass
    with S3.observe_writes("workflows/run-1") as observation:
        pass
    _write("workflows/run-1/after.csv")
    assert observation.records == []


def test_scoped_session_prefix_leaves_the_fallback_alone(session):
    fallback = S3._fallback_prefix
    with S3.scoped_session_prefix("sessions/other-chat"):
        assert S3.current_prefix() == "sessions/other-chat"
    assert S3.current_prefix() == "sessions/observer"
    assert S3._fallback_prefix == fallback
