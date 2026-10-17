"""Which chat text the Chainlit app draws as a molecule.

``chainlit_app`` loads the model config on import, so the regex is read from
its source rather than imported.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

CHAINLIT_APP = Path(__file__).resolve().parents[2] / "chainlit_app.py"


def _smi_rx() -> re.Pattern[str]:
    tree = ast.parse(CHAINLIT_APP.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SMI_RX" for target in node.targets
        ):
            return re.compile(ast.literal_eval(node.value.args[0]), re.IGNORECASE)
    raise AssertionError("SMI_RX not found in chainlit_app.py")


def _detected(text: str) -> list[str]:
    return [(m.group(1) or m.group(2)).strip() for m in _smi_rx().finditer(text)]


@pytest.mark.parametrize("token", ["B", "F", "P", "CNS", "BBB"])
def test_short_backticked_token_is_not_drawn(token):
    # RDKit parses each of these (B is BH3), so a match draws a bogus structure.
    assert _detected(f"Assay type filter: `{token}`") == []


def test_tagged_and_long_backticked_smiles_are_drawn():
    text = "`CC(=O)Oc1ccccc1C(=O)O`, <smiles>C</smiles> and `<smiles>CCO</smiles>`"
    assert _detected(text) == ["CC(=O)Oc1ccccc1C(=O)O", "C", "CCO"]
