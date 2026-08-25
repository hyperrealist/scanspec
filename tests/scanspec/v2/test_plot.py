"""Smoke tests for scanspec.v2.plot — matplotlib output isn't meaningfully
verifiable through assertions alone, so these check the plot runs without
raising and produces the expected shape (axes/lines/legend/patches), not
exact pixel output.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
from matplotlib import patches as mpl_patches
from matplotlib.figure import Figure

from scanspec.v2.core import (
    DetectorGroup,
    MonitorStream,
    TriggerChild,
    TriggerRepeat,
    TriggerSequence,
)
from scanspec.v2.plot import plot_path, plot_scan, plot_spec, plot_timeline
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


def _maximal_multirate_spec() -> Acquire[str, str, str]:
    return Acquire(
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


# ---------------------------------------------------------------------------
# plot_path (no detectors -> single-panel plot_scan/plot_spec too)
# ---------------------------------------------------------------------------


def test_plot_scan_step_2d_returns_figure_with_lines():
    scan = (Linspace("y", 0, 5, 3) * ~Linspace("x", 0, 10, 4)).compile()
    fig = plot_scan(scan, fig=Figure())
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1  # no detectors -> no timeline panel
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
    boundary_patches = [
        p for p in fig.axes[0].patches if isinstance(p, mpl_patches.Ellipse)
    ]
    assert len(boundary_patches) >= 1


def test_plot_spec_polygon_boundary_overlay():
    spec = Polygon("x", "y", [(0, 0), (1, 0), (1, 1), (0, 1)], 0.1)
    fig = plot_spec(spec, fig=Figure())
    boundary_patches = [
        p for p in fig.axes[0].patches if isinstance(p, mpl_patches.Polygon)
    ]
    assert len(boundary_patches) >= 1


def test_plot_scan_has_no_boundary_overlay_without_spec():
    """plot_scan (no spec passed) can't walk a spec tree for regions.

    Turnaround arrows are also Patches now (FancyArrowPatch), so check
    specifically for boundary-shape patches rather than patch count.
    """
    spec = Ellipse("x", 1, 1.8, 0.2, "y", 2)
    scan = spec.compile()
    fig = plot_scan(scan, fig=Figure())
    boundary_patches = [
        p
        for p in fig.axes[0].patches
        if isinstance(p, mpl_patches.Ellipse | mpl_patches.Polygon)
    ]
    assert len(boundary_patches) == 0


def test_plot_path_standalone_matches_scan_path_panel():
    spec = Linspace("y", 0, 5, 3) * ~Linspace("x", 0, 10, 4)
    scan = spec.compile()
    fig = plot_path(scan, fig=Figure())
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) > 0


# ---------------------------------------------------------------------------
# Two-panel plot_scan/plot_spec (detectors present -> path + timeline)
# ---------------------------------------------------------------------------


def test_plot_flagship_multi_stream_two_panels_legend_and_lines():
    spec = _flagship_multi_stream_spec()
    fig = plot_spec(spec, fig=Figure(), max_trigger_markers=100000)
    assert len(fig.axes) == 2  # path + timeline
    path_axes, timeline_axes = fig.axes
    assert len(path_axes.lines) > 0
    legend = path_axes.get_legend()
    assert legend is not None
    labels = {t.get_text() for t in legend.get_texts()}
    assert labels == {"diff", "spec"}
    # timeline: one row per stream ("diff", "spec")
    assert {t.get_text() for t in timeline_axes.get_yticklabels()} == {"diff", "spec"}


def test_plot_flagship_multi_stream_stays_within_physical_range():
    """Regression: a reversing fly-forward/fly-reverse run must not overshoot.

    Drawing this as one global parametric spline across the whole run (an
    earlier version of this module did) is numerically fragile once the
    forward and reverse legs meet at floating-point-identical boundaries
    many times in a row -- scipy's chord-length parameterisation degenerates
    and the fitted curve overshoots far outside the real [7.0, 7.1] range.
    """
    spec = _flagship_multi_stream_spec()
    fig = plot_spec(spec, fig=Figure(), max_trigger_markers=0)
    xdata = [np.asarray(line.get_xdata()) for line in fig.axes[0].lines]
    all_x = np.concatenate([x for x in xdata if x.size])
    # True range (with fence/post half-step boundaries) is
    # [6.99737, 7.10263]; give it a little headroom but stay far tighter
    # than the ~7.115 overshoot the old global-spline bug produced.
    assert all_x.min() >= 6.99
    assert all_x.max() <= 7.11


def test_plot_maximal_multirate_trigger_markers_placed():
    """Parent + nested TriggerChild markers should be resolvable to real coords."""
    # 3 windows (fly rows) x (4 parent + 4*10 child) = 132 markers -- well
    # under the default cap, so they should actually be scattered.
    fig = plot_spec(_maximal_multirate_spec(), fig=Figure())
    path_axes = fig.axes[0]
    assert len(path_axes.lines) >= 3  # path segments
    assert len(path_axes.collections) >= 1  # trigger-marker scatter(s)


def test_plot_trigger_markers_skipped_above_cap():
    spec = _flagship_multi_stream_spec()
    fig_capped = plot_spec(spec, fig=Figure(), max_trigger_markers=0)
    fig_uncapped = plot_spec(spec, fig=Figure(), max_trigger_markers=100000)
    assert len(fig_capped.axes[0].collections) < len(fig_uncapped.axes[0].collections)


def test_plot_scan_without_detectors_has_no_timeline_panel():
    scan = Linspace("x", 0, 1, 5).compile()
    fig = plot_scan(scan, fig=Figure())
    assert len(fig.axes) == 1


# ---------------------------------------------------------------------------
# plot_timeline standalone
# ---------------------------------------------------------------------------


def test_plot_timeline_rows_nest_children_under_parent():
    fig = plot_timeline(_maximal_multirate_spec().compile(), fig=Figure())
    assert len(fig.axes) == 1
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert "primary" in labels
    child_rows = [label for label in labels if label.startswith("primary └")]
    assert len(child_rows) == 1
    # broken_barh calls land as PolyCollections on the axes.
    assert len(fig.axes[0].collections) >= 2  # full-period + livetime, per row


def test_plot_timeline_empty_state_for_pure_motion_spec():
    scan = Linspace("x", 0, 1, 5).compile()
    fig = plot_timeline(scan, fig=Figure())
    assert fig.axes[0].get_yticks().size == 0


# ---------------------------------------------------------------------------
# fig embedding (#189) and default-figure behaviour
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# theme="light"/"dark"
# ---------------------------------------------------------------------------


def test_dark_theme_uses_a_dark_figure_background():
    light = plot_spec(Linspace("x", 0, 1, 5), fig=Figure(), theme="light")
    dark = plot_spec(Linspace("x", 0, 1, 5), fig=Figure(), theme="dark")
    # Luminance check rather than an exact colour match: light theme's
    # background should be much brighter than dark theme's.
    light_luminance = np.asarray(light.get_facecolor())[:3].sum()
    dark_luminance = np.asarray(dark.get_facecolor())[:3].sum()
    assert light_luminance > dark_luminance


def test_dark_theme_uses_a_different_stream_palette():
    spec = _flagship_multi_stream_spec()
    light = plot_spec(spec, fig=Figure(), theme="light", max_trigger_markers=0)
    dark = plot_spec(spec, fig=Figure(), theme="dark", max_trigger_markers=0)
    light_legend = light.axes[0].get_legend()
    dark_legend = dark.axes[0].get_legend()
    assert light_legend is not None
    assert dark_legend is not None
    light_colours = {t.get_color() for t in light_legend.get_texts()}
    dark_colours = {t.get_color() for t in dark_legend.get_texts()}
    assert light_colours != dark_colours


def test_dark_theme_3d_panes_are_dark_not_default_grey():
    """Axes3D panes/axis lines are a separate colour API from 2D facecolor/spines.

    Matplotlib's own default pane fill is a light grey that clashes badly
    against a dark figure background if left untouched.
    """
    spec3d = Linspace("z", 1, 3, 3) * Spiral("x", 0, 5, 2, "y", 10, 5)
    fig = plot_path(spec3d.compile(), fig=Figure(), theme="dark")
    axes = fig.axes[0]
    pane_luminance = np.asarray(
        axes.xaxis.pane.get_facecolor()  # type: ignore[reportAttributeAccessIssue]
    )[:3].sum()
    assert pane_luminance < 1.0  # matplotlib's default light-grey pane sums to ~2.85
    axis_line_colour = axes.xaxis.line.get_color()  # type: ignore[reportAttributeAccessIssue]
    assert axis_line_colour != "black" and axis_line_colour != (0.0, 0.0, 0.0, 1.0)
