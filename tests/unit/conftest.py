"""Keep the unit tests off the developer's real services."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _no_real_chembl_backend():
    """Hide ``CHEMBL_*`` from the unit tests, whatever put it in the environment.

    ChemblToolkit picks its backend from ``CHEMBL_SQLITE_PATH``,
    ``CHEMBL_PG_HOST`` and ``CHEMBL_MYSQL_HOST``. With any of them set it
    builds a SQL fetcher, which is not the REST client the unit tests mock:
    the mocks go unused, assertions fail, and the suite waits on a real
    database it was never meant to touch.

    Anything that loads a ``.env`` can supply those -- it was a module-level
    ``load_dotenv()`` in the robustness harness, pulled in at collection time,
    but a developer's shell does just as well. Clearing them here makes the
    unit tests hermetic regardless, which is also why this did not reproduce
    on CI, where there is no ``.env`` to read.

    Tests that want a backend still set one themselves; ``monkeypatch`` and
    ``patch.dict`` apply after this fixture and are restored per test.
    """
    with pytest.MonkeyPatch.context() as patcher:
        for name in [key for key in os.environ if key.startswith("CHEMBL_")]:
            patcher.delenv(name, raising=False)
        yield
