"""Unit tests for GTM plotting helpers and landscape dispatch."""

import sys
from pathlib import Path
from types import ModuleType

import altair as alt
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

tools_package_name = "cs_copilot.tools"
if tools_package_name not in sys.modules:
    tools_stub = ModuleType(tools_package_name)
    tools_stub.__path__ = [str(SRC_PATH / "cs_copilot" / "tools")]
    sys.modules[tools_package_name] = tools_stub

from cs_copilot.tools.chemography import gtm_operations  # noqa: E402
from cs_copilot.tools.chemography.gtm import GTMToolkit  # noqa: E402


def _local_s3_open_factory(tmp_path: Path):
    """Return an S3.open replacement that writes into the pytest temp dir."""

    def _open(path: str, mode: str = "rb"):
        target = Path(path)
        if not target.is_absolute():
            target = tmp_path / target.name
        target.parent.mkdir(parents=True, exist_ok=True)
        return open(target, mode)

    return _open


def _tool_names(toolkit: GTMToolkit) -> list[str]:
    """Extract registered tool names across agno toolkit implementations."""

    if isinstance(toolkit.functions, dict):
        return list(toolkit.functions.keys())

    names = []
    for function in toolkit.functions:
        names.append(getattr(function, "name", getattr(function, "__name__", None)))
    return [name for name in names if name]


def test_save_gtm_plot_supports_large_point_datasets(monkeypatch, tmp_path):
    """Large GTM point layers should export without Altair's default 5k row failure."""

    n_rows = 5001
    smiles_col = gtm_operations.SMILES_COLUMN
    density_table = pd.DataFrame(
        {
            "x": [1, 1, 2, 2],
            "y": [1, 2, 1, 2],
            "nodes": [0, 1, 2, 3],
            "density": [1.0, 1.0, 1.0, 1.0],
            "filtered_density": [1.0, 1.0, 1.0, 1.0],
        }
    )
    df = pd.DataFrame({smiles_col: [f"C{i}" for i in range(n_rows)]})
    coords = pd.DataFrame(
        {
            "x": np.tile([1.2, 1.8, 2.2, 2.8], n_rows // 4 + 1)[:n_rows],
            "y": np.tile([1.2, 1.8, 2.2, 2.8], n_rows // 4 + 1)[:n_rows],
            "node_index": np.arange(n_rows) % 4,
        }
    )
    vis_info = pd.DataFrame(
        {
            smiles_col: df[smiles_col],
            "source": ["dataset"] * n_rows,
            "image": ["img"] * n_rows,
        }
    )
    responsibilities = np.full((n_rows, 4), 0.25)

    monkeypatch.setattr(gtm_operations, "load_gtm", lambda *_: (density_table, None))
    monkeypatch.setattr(
        gtm_operations, "data_load_and_prep", lambda *_: (None, df.copy(), None, responsibilities)
    )
    monkeypatch.setattr(gtm_operations, "calculate_latent_coords", lambda *_args, **_kwargs: coords)
    monkeypatch.setattr(
        gtm_operations,
        "encode_molecules",
        lambda *_args, **_kwargs: vis_info.copy(),
    )
    monkeypatch.setattr(gtm_operations.S3, "open", _local_s3_open_factory(tmp_path))

    def _fake_save(self, fp, format=None, **kwargs):
        fp.write(b"png-bytes")

    monkeypatch.setattr(gtm_operations.alt.TopLevelMixin, "save", _fake_save)

    with alt.data_transformers.enable("default", max_rows=5000):
        result = gtm_operations.save_gtm_plot("large_projection.csv", "model.pkl.gz")

    html_path = tmp_path / "model_gtm_plot.html"
    png_path = tmp_path / "model_gtm_plot.png"

    assert "GTM plot saved to S3" in result
    assert html_path.exists()
    assert png_path.exists()
    assert "vegaEmbed" in html_path.read_text()
    assert alt.data_transformers.active == "default"


@pytest.mark.parametrize(
    ("landscape_type", "renderer_name", "table", "expected_suffix"),
    [
        (
            "density",
            "altair_discrete_density_landscape",
            pd.DataFrame(
                {
                    "x": [1, 1, 2, 2],
                    "y": [1, 2, 1, 2],
                    "nodes": [0, 1, 2, 3],
                    "density": [1.0, 2.0, 3.0, 4.0],
                    "filtered_density": [1.0, 2.0, 3.0, 4.0],
                }
            ),
            "_altair_density_landscape",
        ),
        (
            "classification",
            "altair_discrete_class_landscape",
            pd.DataFrame(
                {
                    "x": [1, 1, 2, 2],
                    "y": [1, 2, 1, 2],
                    "nodes": [0, 1, 2, 3],
                    "density": [1.0, 2.0, 3.0, 4.0],
                    "first_class_prob": [0.1, 0.2, 0.3, 0.4],
                    "second_class_prob": [0.9, 0.8, 0.7, 0.6],
                    "first_class_density": [1.0, 1.0, 1.0, 1.0],
                    "second_class_density": [2.0, 2.0, 2.0, 2.0],
                }
            ),
            "_altair_classification_landscape",
        ),
        (
            "regression",
            "altair_discrete_regression_landscape",
            pd.DataFrame(
                {
                    "x": [1, 1, 2, 2],
                    "y": [1, 2, 1, 2],
                    "nodes": [0, 1, 2, 3],
                    "density": [1.0, 2.0, 3.0, 4.0],
                    "filtered_reg_density": [5.0, 6.0, 7.0, 8.0],
                }
            ),
            "_altair_regression_landscape",
        ),
        (
            "query",
            "altair_discrete_query_landscape",
            pd.DataFrame(
                {
                    "x": [1, 1, 2, 2],
                    "y": [1, 2, 1, 2],
                    "nodes": [0, 1, 2, 3],
                    "density": [1.0, 2.0, 3.0, 4.0],
                    "criteria_satisfied": ["yes", "no", "yes", "no"],
                }
            ),
            "_altair_query_landscape",
        ),
    ],
)
def test_save_gtm_landscape_plot_dispatches_to_matching_renderer(
    monkeypatch, tmp_path, landscape_type, renderer_name, table, expected_suffix
):
    """Each supported ChemographyKit landscape type should route to the right renderer."""

    landscape_path = tmp_path / f"{landscape_type}.csv"
    table.to_csv(landscape_path, index=False)
    calls = {}

    def _fake_renderer(source_table, title="", **kwargs):
        calls["renderer"] = renderer_name
        calls["table"] = source_table.copy()
        calls["title"] = title
        calls["kwargs"] = kwargs
        return alt.Chart(pd.DataFrame({"x": [1], "y": [1]})).mark_rect()

    def _fake_write(chart, html_path, png_path):
        calls["html_path"] = html_path
        calls["png_path"] = png_path

    monkeypatch.setattr(gtm_operations, renderer_name, _fake_renderer)
    monkeypatch.setattr(gtm_operations, "_write_chart_outputs", _fake_write)

    result = gtm_operations.save_gtm_landscape_plot(str(landscape_path), landscape_type)

    assert renderer_name == calls["renderer"]
    assert landscape_type in result.lower()
    assert calls["table"].equals(table)
    assert calls["html_path"].endswith(f"{expected_suffix}.html")
    assert calls["png_path"].endswith(f"{expected_suffix}.png")


@pytest.mark.parametrize(
    ("landscape_type", "renderer_name", "table", "expected_suffix"),
    [
        (
            "density",
            "plotly_smooth_density_landscape",
            pd.DataFrame(
                {
                    "x": [1, 1, 2, 2],
                    "y": [1, 2, 1, 2],
                    "nodes": [0, 1, 2, 3],
                    "density": [1.0, 2.0, 3.0, 4.0],
                    "filtered_density": [1.0, 2.0, 3.0, 4.0],
                }
            ),
            "_plotly_density_landscape",
        ),
        (
            "classification",
            "plotly_discrete_class_landscape",
            pd.DataFrame(
                {
                    "x": [1, 1, 2, 2],
                    "y": [1, 2, 1, 2],
                    "nodes": [0, 1, 2, 3],
                    "density": [1.0, 2.0, 3.0, 4.0],
                    "1_prob": [0.1, 0.2, 0.3, 0.4],
                    "2_prob": [0.9, 0.8, 0.7, 0.6],
                    "1_density": [1.0, 1.0, 1.0, 1.0],
                    "2_density": [2.0, 2.0, 2.0, 2.0],
                }
            ),
            "_plotly_classification_landscape",
        ),
        (
            "regression",
            "plotly_smooth_regression_landscape",
            pd.DataFrame(
                {
                    "x": [1, 1, 2, 2],
                    "y": [1, 2, 1, 2],
                    "nodes": [0, 1, 2, 3],
                    "density": [1.0, 2.0, 3.0, 4.0],
                    "filtered_reg_density": [5.0, 6.0, 7.0, 8.0],
                }
            ),
            "_plotly_regression_landscape",
        ),
    ],
)
def test_save_gtm_landscape_plot_dispatches_to_plotly_renderers(
    monkeypatch, tmp_path, landscape_type, renderer_name, table, expected_suffix
):
    """Plotly-enabled ChemographyKit landscapes should use the Plotly renderer set."""

    landscape_path = tmp_path / f"plotly_{landscape_type}.csv"
    table.to_csv(landscape_path, index=False)
    calls = {}

    class _DummyFigure:
        def update_layout(self, **kwargs):
            calls["layout"] = kwargs

    def _fake_renderer(source_table, title="", **kwargs):
        calls["renderer"] = renderer_name
        calls["table"] = source_table.copy()
        calls["title"] = title
        calls["kwargs"] = kwargs
        return _DummyFigure()

    def _fake_write(fig, html_path, png_path):
        calls["html_path"] = html_path
        calls["png_path"] = png_path
        return True

    monkeypatch.setattr(gtm_operations, renderer_name, _fake_renderer)
    monkeypatch.setattr(gtm_operations, "_write_plotly_outputs", _fake_write)

    result = gtm_operations.save_gtm_landscape_plot(
        str(landscape_path), landscape_type, renderer="plotly"
    )

    assert renderer_name == calls["renderer"]
    assert landscape_type in result.lower()
    assert calls["table"].equals(table)
    assert calls["layout"] == {"width": 600, "height": 600}
    assert calls["html_path"].endswith(f"{expected_suffix}.html")
    assert calls["png_path"].endswith(f"{expected_suffix}.png")

    if landscape_type == "classification":
        assert calls["kwargs"]["first_class_prob_column_name"] == "1_prob"
        assert calls["kwargs"]["second_class_prob_column_name"] == "2_prob"
        assert calls["kwargs"]["first_class_density_column_name"] == "1_density"
        assert calls["kwargs"]["second_class_density_column_name"] == "2_density"


def test_save_gtm_landscape_plot_plotly_query_is_rejected(tmp_path):
    """Plotly query landscapes are unsupported because ChemographyKit does not expose one."""

    path = tmp_path / "query.csv"
    pd.DataFrame(
        {
            "x": [1],
            "y": [1],
            "nodes": [1],
            "density": [1.0],
            "criteria_satisfied": ["yes"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Plotly landscapes are only available"):
        gtm_operations.save_gtm_landscape_plot(str(path), "query", renderer="plotly")


def test_save_gtm_landscape_plot_plotly_reports_html_only_when_png_backend_missing(
    monkeypatch, tmp_path
):
    """Plotly landscape exports should still succeed when static image export is unavailable."""

    path = tmp_path / "density.csv"
    pd.DataFrame(
        {
            "x": [1, 1, 2, 2],
            "y": [1, 2, 1, 2],
            "nodes": [0, 1, 2, 3],
            "density": [1.0, 2.0, 3.0, 4.0],
            "filtered_density": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_csv(path, index=False)

    class _DummyFigure:
        def update_layout(self, **kwargs):
            return None

    monkeypatch.setattr(
        gtm_operations, "plotly_smooth_density_landscape", lambda *_args, **_kwargs: _DummyFigure()
    )
    monkeypatch.setattr(gtm_operations, "_write_plotly_outputs", lambda *_args, **_kwargs: False)

    result = gtm_operations.save_gtm_landscape_plot(str(path), "density", renderer="plotly")

    assert "PNG export was skipped" in result


def test_save_gtm_landscape_plot_altair_classification_accepts_chemographykit_default_columns(
    monkeypatch, tmp_path
):
    """Altair classification rendering should accept ChemographyKit's default *_prob/*_density columns."""

    path = tmp_path / "classification.csv"
    table = pd.DataFrame(
        {
            "x": [1, 1, 2, 2],
            "y": [1, 2, 1, 2],
            "nodes": [0, 1, 2, 3],
            "density": [1.0, 2.0, 3.0, 4.0],
            "1_prob": [0.1, 0.2, 0.3, 0.4],
            "2_prob": [0.9, 0.8, 0.7, 0.6],
            "1_density": [1.0, 1.0, 1.0, 1.0],
            "2_density": [2.0, 2.0, 2.0, 2.0],
        }
    )
    table.to_csv(path, index=False)
    calls = {}

    def _fake_renderer(source_table, title="", **kwargs):
        calls["kwargs"] = kwargs
        return alt.Chart(pd.DataFrame({"x": [1], "y": [1]})).mark_rect()

    monkeypatch.setattr(gtm_operations, "altair_discrete_class_landscape", _fake_renderer)
    monkeypatch.setattr(gtm_operations, "_write_chart_outputs", lambda *_args, **_kwargs: None)

    gtm_operations.save_gtm_landscape_plot(str(path), "classification")

    assert calls["kwargs"]["first_class_prob_column_name"] == "1_prob"
    assert calls["kwargs"]["second_class_prob_column_name"] == "2_prob"


def test_save_gtm_landscape_plot_validates_required_columns(tmp_path):
    """Landscape CSVs missing ChemographyKit-required columns should fail clearly."""

    invalid_path = tmp_path / "invalid_classification.csv"
    pd.DataFrame(
        {
            "x": [1],
            "y": [1],
            "nodes": [1],
            "density": [1.0],
        }
    ).to_csv(invalid_path, index=False)

    with pytest.raises(ValueError, match="Classification landscape table must include"):
        gtm_operations.save_gtm_landscape_plot(str(invalid_path), "classification")


def test_gtm_toolkit_registers_save_gtm_landscape_plot():
    """The GTM toolkit should expose the generic landscape plotting helper."""

    toolkit = GTMToolkit()
    assert "save_gtm_landscape_plot" in _tool_names(toolkit)
