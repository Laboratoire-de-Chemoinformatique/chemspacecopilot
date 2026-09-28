"""Regression checks for the SynPlanner 1.7 API and asset boundary."""

from types import SimpleNamespace

import pytest

from cs_copilot.tools.chemistry.synplanner_assets import (
    SYNPLANNER_FILES,
    resolve_synplanner_assets,
)
from cs_copilot.tools.chemistry.synplanner_toolkit import SynPlannerToolkit


def test_explicit_legacy_assets_fail_without_downloading(tmp_path, monkeypatch):
    import huggingface_hub

    def unexpected_download(**kwargs):
        pytest.fail("An explicit incomplete local preset must not trigger a download")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", unexpected_download)
    (tmp_path / "uspto_reaction_rules.pickle").write_bytes(b"legacy")
    with pytest.raises(FileNotFoundError, match="historical CGRtools"):
        resolve_synplanner_assets(str(tmp_path))


def test_complete_local_preset_is_reused(tmp_path):
    for relative in SYNPLANNER_FILES.values():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    paths = resolve_synplanner_assets(str(tmp_path))
    assert all(path.is_file() for path in paths.values())


def test_modern_policy_selection_parameters_change_effective_attributes():
    toolkit = SynPlannerToolkit()
    toolkit._policy_network = SimpleNamespace(config=None, top_rules=50, rule_prob_threshold=0.0)
    toolkit._apply_policy_config({"top_rules": 100, "rule_prob_threshold": 0.01})
    assert toolkit._policy_network.top_rules == 100
    assert toolkit._policy_network.rule_prob_threshold == 0.01


def test_modern_depth_does_not_read_removed_parallel_dict():
    class ModernTree:
        nodes = {1: SimpleNamespace(depth=0), 2: SimpleNamespace(depth=3)}

        @property
        def nodes_depth(self):
            pytest.fail("nodes_depth was removed in SynPlanner 1.5")

    assert SynPlannerToolkit._safe_max_depth(ModernTree()) == 3


def test_modern_loading_does_not_enter_cgrtools_adapter(monkeypatch):
    from cs_copilot.tools.chemistry import synplanner_toolkit as module

    toolkit = SynPlannerToolkit()
    monkeypatch.setattr(toolkit, "_import_synplanner", lambda: SimpleNamespace(__version__="1.7.0"))
    calls = []
    monkeypatch.setattr(
        toolkit, "_load_modern_synplanner_components", lambda: calls.append("modern")
    )
    monkeypatch.setattr(
        module, "_install_cgrtools_miniracer_compatibility", lambda: pytest.fail("Legacy adapter")
    )
    toolkit._load_synplanner_components()
    assert calls == ["modern"]


def test_png_rendering_preserves_atom_label_cutouts(tmp_path, monkeypatch):
    from PIL import Image

    from cs_copilot.storage import S3

    # A black circle in a luminance mask hides the bond underneath an atom.
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 4 2">
      <defs><mask id="atom"><rect width="4" height="2" fill="white"/>
        <circle cx="2" cy="1" r="0.4" fill="black"/>
      </mask></defs>
      <path d="M0 1 H4" stroke="black" stroke-width="0.1" mask="url(#atom)"/>
    </svg>"""
    monkeypatch.setattr(S3, "open", lambda name, mode: (tmp_path / name).open(mode))
    assert SynPlannerToolkit()._export_crisp(svg, "route.png")
    with Image.open(tmp_path / "route.png") as rendered:
        rgb = rendered.convert("RGB")
        assert rgb.size == (400, 200)
        assert rgb.getpixel((200, 100)) == (255, 255, 255)
        assert rgb.getpixel((100, 100)) == (0, 0, 0)
