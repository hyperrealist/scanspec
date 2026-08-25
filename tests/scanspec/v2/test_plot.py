"""Smoke tests for scanspec.v2.plot — matplotlib output isn't meaningfully
verifiable through assertions alone, so these check the plot runs without
raising and produces the expected shape (axes/lines/legend/patches), not
exact pixel output.
"""

from __future__ import annotations

from unittest.mock import patch

from matplotlib.figure import Figure

from scanspec.v2.core import (
    DetectorGroup,
    MonitorStream,
    TriggerChild,
    TriggerRepeat,
    TriggerSequence,
)
from scanspec.v2.plot import plot_scan, plot_spec
from scanspec.v2.specs import (
    Acquire,
    Ellipse,
    Linspace,
    Polygon,
    Product,
    Repeat,
    Spiral,
    Static,
)


def _flagship_multi_stream_spec() -> Repeat[str, str, str]:
    """Same pattern as test_use_cases.py::test_flagship_multi_stream_concat.

    200 -> 3 reduced to keep the test fast; shape (step + 2 fly legs on a
    shared axis, two named streams) is what matters here.
    """
    diff_det = DetectorGroup(1, 1, 0.01, 0.001, ["diffraction"])
    spec_det = DetectorGroup(1, 1, 0.003, 0.001, ["spectroscopy"])
    diff_acq: Acquire[str, str, str] = Acquire(
        Static("e", 7.0), detectors=[diff_det], stream_name="diff"
    )
    spec_fwd: Acquire[str, str, str] = Acquire(
        Linspace("e", 7.0, 7.1, 20), fly=True, detectors=[spec_det], stream_name="spec"
    )
    spec_rev: Acquire[str, str, str] = Acquire(
        Linspace("e", 7.1, 7.0, 20), fly=True, detectors=[spec_det], stream_name="spec"
    )
    return Repeat(diff_acq.concat(spec_fwd).concat(spec_rev), num=3)


def test_plot_scan_step_2d_returns_figure_with_lines():
    scan = (Linspace("y", 0, 5, 3) * ~Linspace("x", 0, 10, 4)).compile()
    fig = plot_scan(scan, fig=Figure())
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) > 0
    assert fig.axes[0].get_xlabel() == "x"
    assert fig.axes[0].get_ylabel() == "y"


def test_plot_scan_fly_3d():
    spec = Linspace("z", 1, 3, 3) * Acquire(Spiral("x", 0, 5, 2, "y", 10, 5), fly=True)
    fig = plot_spec(spec, fig=Figure())
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) > 0


def test_plot_spec_ellipse_boundary_overlay():
    spec = Ellipse("x", 1, 1.8, 0.2, "y", 2)
    fig = plot_spec(spec, fig=Figure())
    assert len(fig.axes[0].patches) >= 1


def test_plot_spec_polygon_boundary_overlay():
    spec = Polygon("x", "y", [(0, 0), (1, 0), (1, 1), (0, 1)], 0.1)
    fig = plot_spec(spec, fig=Figure())
    assert len(fig.axes[0].patches) >= 1


def test_plot_scan_has_no_boundary_overlay_without_spec():
    """plot_scan (no spec passed) can't walk a spec tree for regions."""
    spec = Ellipse("x", 1, 1.8, 0.2, "y", 2)
    scan = spec.compile()
    fig = plot_scan(scan, fig=Figure())
    assert len(fig.axes[0].patches) == 0


def test_plot_flagship_multi_stream_legend_and_lines():
    spec = _flagship_multi_stream_spec()
    fig = plot_spec(spec, fig=Figure(), max_trigger_markers=100000)
    axes = fig.axes[0]
    assert len(axes.lines) > 0
    legend = axes.get_legend()
    assert legend is not None
    labels = {t.get_text() for t in legend.get_texts()}
    assert labels == {"diff", "spec"}


def test_plot_maximal_multirate_trigger_markers_placed():
    """Parent + nested TriggerChild markers should be resolvable to real coords."""
    spec = Acquire(
        Product(Linspace("y", 0, 5, 3), ~Linspace("x", 0, 10, 4)),
        fly=True,
        detectors=[
            DetectorGroup(1, 1, 0.003, 0.001, ["saxs", "waxs"]),
            DetectorGroup(10, 1, 0.000299992, 8e-9, ["timestamp", "x_enc", "y_enc"]),
        ],
        trigger_sequence=TriggerSequence(
            detectors=frozenset({"saxs", "waxs"}),
            trigger_repeat=TriggerRepeat(num=4, livetime=0.003, deadtime=0.001),
            children=[
                TriggerChild(
                    detectors=frozenset({"timestamp", "x_enc", "y_enc"}),
                    repeats=[
                        TriggerRepeat(num=10, livetime=0.000299992, deadtime=8e-9)
                    ],
                ),
            ],
        ),
        monitors=[MonitorStream("temperature", "tc1")],
    )
    # 3 windows (fly rows) x (4 parent + 4*10 child) = 132 markers -- well
    # under the default cap, so they should actually be drawn as extra lines
    # on top of the path itself.
    fig = plot_spec(spec, fig=Figure())
    assert len(fig.axes[0].lines) > 3  # path segments + marker scatter(s)


def test_plot_trigger_markers_skipped_above_cap():
    spec = _flagship_multi_stream_spec()
    fig_capped = plot_spec(spec, fig=Figure(), max_trigger_markers=0)
    fig_uncapped = plot_spec(spec, fig=Figure(), max_trigger_markers=100000)
    assert len(fig_capped.axes[0].lines) < len(fig_uncapped.axes[0].lines)


def test_fig_reuse_matches_issue_189():
    """plot_spec(..., fig=existing_figure) adds axes to that figure (#189)."""
    fig = Figure()
    plot_spec(Linspace("x", 0, 1, 5), fig=fig)
    assert len(fig.axes) == 1
    plot_spec(Linspace("y", 0, 1, 3), fig=fig)
    assert len(fig.axes) == 2


def test_plot_spec_without_fig_creates_and_shows_one():
    with patch("scanspec.v2.plot.plt.show") as mock_show:
        fig = plot_spec(Linspace("x", 0, 1, 5))
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    mock_show.assert_called_once()
