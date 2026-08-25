"""`plot_scan`/`plot_spec`/`plot_path`/`plot_timeline` — visualize a scanspec.v2 scan.

Two complementary views, since neither alone tells the whole story:

- **Path** (``plot_path``): *where* the scan goes — the motion trajectory,
  coloured by which detector stream is active, with turnarounds and
  individual trigger instants (sized by livetime).
- **Timeline** (``plot_timeline``): *when* things happen — a Gantt-style
  chart, one row per detector stream (with fast multi-rate children nested
  under their parent row), livetime blocks over the scan's real duration.
  This is the more informative view whenever the motion itself doesn't
  spread out spatially (e.g. a scan on a single axis that sweeps back and
  forth) — the interesting structure is in time, not space.

``plot_scan``/``plot_spec`` combine both into one two-panel figure — the
main entry points. ``plot_path``/``plot_timeline`` are the same two panels
as standalone single-panel figures, for when only one story is wanted.

Built directly on ``Window``/``Scan`` rather than ported from 1.x's
plot.py: no precomputed position arrays exist in v2 (see ``Scan``'s
docstring), and the path is drawn window-by-window with plain line
segments rather than one global spline fit — a run that reverses direction
many times (e.g. repeated fly-forward/fly-reverse legs meeting at
floating-point-identical boundaries) makes a single parametric spline fit
numerically fragile, silently overshooting far outside the real data range
rather than raising.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import cycle
from math import isclose
from typing import Any

import numpy as np
import numpy.typing as npt
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import proj3d  # type: ignore

from .core import (
    ConcatSource,
    Scan,
    TriggerRepeat,
    TriggerSequence,
    Window,
    WindowGenerator,
)
from .specs import Ellipse, Polygon, Spec

__all__ = ["plot_scan", "plot_spec", "plot_path", "plot_timeline"]

# Above this many resolved trigger instants across the whole scan, individual
# trigger markers on the *path* view are skipped (stream colouring is kept)
# -- otherwise a dense fly scan with fast child detectors can ask for
# hundreds of thousands of markers on one static figure. The timeline view
# has no such cap: broken_barh renders large numbers of blocks efficiently
# as a single collection, unlike per-point markers.
DEFAULT_MAX_TRIGGER_MARKERS = 2000

_FLY_SAMPLE_POINTS = 20  # samples along a non_linear fly window's true curve
_MARKER_SIZE_RANGE = (5.0, 15.0)  # trigger-marker point size range, by livetime

# A vivid qualitative palette used for streams, in preference to
# matplotlib's default colour cycle -- picked for strong contrast against a
# light background rather than print-muted tones.
_PALETTE = [
    "#3B82F6",  # blue
    "#F97316",  # orange
    "#10B981",  # emerald
    "#EF4444",  # red
    "#8B5CF6",  # violet
    "#EC4899",  # pink
    "#14B8A6",  # teal
    "#F59E0B",  # amber
    "#6366F1",  # indigo
    "#84CC16",  # lime
]

_INK = "#1f1f27"  # primary text/spine colour
_MUTED_INK = "#63636f"  # secondary text colour
_GRID = "#e4e4ea"
_FIG_BG = "#ffffff"
_AXES_BG = "#fbfbfe"
_ROW_BAND = "#f1f1f6"
_NO_STREAM_COLOUR = "#3f3f4a"  # path colour when no detector stream applies
_TURNAROUND_COLOUR = "#b7b7c2"  # de-emphasised connector between runs
_BOUNDARY_COLOUR = "#8B5CF6"  # Ellipse/Polygon region-boundary overlay


# ---------------------------------------------------------------------------
# matplotlib plumbing
# ---------------------------------------------------------------------------


def _plot_arrays(axes: Axes, arrays: list[npt.NDArray[np.float64]], **kwargs: Any):
    if len(arrays) > 2:
        axes.plot3D(arrays[2], arrays[1], arrays[0], **kwargs)  # type: ignore
    elif len(arrays) == 2:
        axes.plot(arrays[1], arrays[0], **kwargs)  # type: ignore
    else:
        axes.plot(arrays[0], np.zeros(len(arrays[0])), **kwargs)  # type: ignore


# https://stackoverflow.com/a/11156353
class Arrow3D(patches.FancyArrowPatch):
    def __init__(
        self,
        xs: npt.NDArray[np.float64],
        ys: npt.NDArray[np.float64],
        zs: npt.NDArray[np.float64],
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__((0, 0), (0, 0), *args, **kwargs)  # type: ignore
        self._verts3d = xs, ys, zs

    # Added here because of https://github.com/matplotlib/matplotlib/issues/21688
    def do_3d_projection(self, _renderer: Any = None):  # type: ignore
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)  # type: ignore
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))  # type: ignore
        return np.min(zs)  # type: ignore


def _add_3d_turnaround_arrow(axes: Axes, arrays: list[npt.NDArray[np.float64]]) -> None:
    arrows = [a[-2:] for a in reversed(arrays)]
    a = Arrow3D(
        *arrows[:3], mutation_scale=10, arrowstyle="-|>", color=_TURNAROUND_COLOUR
    )
    axes.add_artist(a)


def _get_boundaries(spec: Spec[Any, Any, Any]) -> Iterator[patches.Patch]:
    """Region-boundary overlays: a soft fill wash plus a solid coloured edge.

    An outline alone (1.x's ``fill=False``) reads as just a black circle
    when the enclosed path has no detector stream of its own to colour it
    (e.g. a bare ``Ellipse``/``Polygon`` with no ``Acquire`` wrapping it) --
    the fill gives the region presence even then.
    """
    patch_kwargs: dict[str, Any] = {
        "fill": True,
        "facecolor": _BOUNDARY_COLOUR,
        "alpha": 0.1,
        "edgecolor": _BOUNDARY_COLOUR,
        "linewidth": 1.8,
    }
    if isinstance(spec, Ellipse):
        xy = spec.x_centre, spec.y_centre
        y_diam = (
            spec.y_diameter if spec.y_diameter is not None else abs(spec.x_diameter)
        )
        yield patches.Ellipse(xy, spec.x_diameter, y_diam, **patch_kwargs)
    elif isinstance(spec, Polygon):
        yield patches.Polygon(spec.vertices, **patch_kwargs)
    else:
        for name in type(spec).model_fields:
            s = getattr(spec, name)
            if isinstance(s, Spec):
                yield from _get_boundaries(s)  # type: ignore[reportUnknownArgumentType]


# ---------------------------------------------------------------------------
# v2-specific: axes, stream/colour resolution, trigger timing
# ---------------------------------------------------------------------------


def _flatten_axes(scan: Scan[Any, Any, Any]) -> list[Any]:
    """Outer -> inner axis order, matching 1.x's ``spec.axes()``.

    ``Concat``/``Repeat`` generators carry no axes of their own (``axes=[]``)
    — the real axes live on the leaf generators nested inside their
    ``ConcatSource``, so those have to be walked too.
    """
    axes: list[Any] = []

    def _collect(gen: WindowGenerator[Any]) -> None:
        for ax in gen.axes:
            if ax not in axes:
                axes.append(ax)
        if isinstance(gen.source, ConcatSource):
            for child in gen.source.children:
                _collect(child)

    for gen in scan.generators:
        _collect(gen)
    return axes


def _detector_stream_map(scan: Scan[Any, Any, Any]) -> dict[Any, str]:
    mapping: dict[Any, str] = {}
    for stream in scan.windowed_streams:
        for dg in stream.detector_groups:
            for d in dg.detectors:
                mapping[d] = stream.name
    for cstream in scan.continuous_streams:
        for dg in cstream.detector_groups:
            for d in dg.detectors:
                mapping[d] = cstream.name
    return mapping


def _stream_colours(scan: Scan[Any, Any, Any]) -> dict[str, str]:
    names: list[str] = [s.name for s in scan.windowed_streams] + [
        s.name for s in scan.continuous_streams
    ]
    palette = cycle(_PALETTE)
    seen: dict[str, str] = {}
    for name in names:
        if name not in seen:
            seen[name] = next(palette)
    return seen


def _window_streams(
    window: Window[Any, Any], detector_to_stream: dict[Any, str]
) -> frozenset[str]:
    streams: set[str] = set()
    for ts in window.trigger_sequences:
        for d in ts.detectors:
            if d in detector_to_stream:
                streams.add(detector_to_stream[d])
        for child in ts.children:
            for d in child.detectors:
                if d in detector_to_stream:
                    streams.add(detector_to_stream[d])
    return frozenset(streams)


def _window_colour(streams: frozenset[str], stream_colours: dict[str, str]) -> str:
    """One representative colour for a window's active stream set.

    Multi-stream windows (simultaneous streams) currently fall back to the
    first (alphabetically, for stability) stream's colour rather than a
    blended/hatched treatment — a deliberate first-cut simplification, not a
    limitation of the data available.
    """
    if not streams:
        return _NO_STREAM_COLOUR
    return stream_colours[sorted(streams)[0]]


def _marker_size(livetime: float) -> float:
    """Trigger-marker point size, growing (sub-linearly) with livetime."""
    lo, hi = _MARKER_SIZE_RANGE
    if livetime <= 0:
        return lo
    return float(np.clip(lo + 26 * livetime**0.25, lo, hi))


def _repeat_times(
    repeats: list[TriggerRepeat], t0: float
) -> tuple[list[tuple[float, float]], float]:
    """Centred-livetime (time, livetime) trigger midpoints for a repeats list.

    Returns (times from t0, end time) so callers can chain/nest.
    """
    times: list[tuple[float, float]] = []
    t = t0
    for r in repeats:
        if r.livetime is None or r.deadtime is None:
            continue
        period = r.livetime + r.deadtime
        for k in range(r.num):
            times.append((t + k * period + period / 2, r.livetime))
        t += r.num * period
    return times, t


def _trigger_marker_times(ts: TriggerSequence[Any]) -> list[tuple[float, Any, float]]:
    """(time, representative_detector, livetime) triples.

    Covers the parent repeat and every nested child. A child's own repeats
    run in full during *every* parent repeat (per
    ``TriggerSequence``'s docstring), so each child repeat-block is offset
    into its parent repeat's own span before applying the same centred-
    livetime placement recursively.
    """
    result: list[tuple[float, Any, float]] = []
    parent_times, _ = _repeat_times([ts.trigger_repeat], 0.0)
    parent_detector = next(iter(ts.detectors), None)
    result += [(t, parent_detector, lt) for t, lt in parent_times]

    if ts.trigger_repeat.livetime is None or ts.trigger_repeat.deadtime is None:
        return result
    parent_period = ts.trigger_repeat.livetime + ts.trigger_repeat.deadtime
    for child in ts.children:
        child_detector = next(iter(child.detectors), None)
        for p in range(ts.trigger_repeat.num):
            child_times, _ = _repeat_times(child.repeats, p * parent_period)
            result += [(t, child_detector, lt) for t, lt in child_times]
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_scan(
    scan: Scan[Any, Any, Any],
    *,
    fig: Figure | None = None,
    title: str | None = None,
    spec: Spec[Any, Any, Any] | None = None,
    max_trigger_markers: int = DEFAULT_MAX_TRIGGER_MARKERS,
) -> Figure:
    """Plot a compiled Scan: path (top) + trigger timeline (bottom).

    If *scan* has no detectors at all (no windowed/continuous streams), the
    timeline panel is omitted and this degrades to a path-only figure.

    *fig*: draw into this Figure instead of creating one (embedding, e.g. in
    a Qt widget's ``FigureCanvas`` — https://github.com/bluesky/scanspec/issues/189).
    If not given, a new Figure is created and shown (``plt.show()``) before
    returning; if given, the caller owns display.

    *spec*: when given, overlays ``Ellipse``/``Polygon`` region boundaries
    found in the spec tree onto the path panel (meaningless for a bare Scan,
    which retains no spec tree).
    """
    axis_labels = _flatten_axes(scan)
    ndims = len(axis_labels)
    detector_to_stream = _detector_stream_map(scan)
    stream_colours = _stream_colours(scan)
    has_timeline = bool(scan.windowed_streams or scan.continuous_streams)

    owns_figure = fig is None
    if fig is None:
        fig = plt.figure(  # type: ignore
            figsize=(7, 8) if has_timeline else (7, 6), layout="constrained"
        )

    if has_timeline:
        # hspace only when there's no layout engine to manage spacing itself
        # (a caller-supplied `fig=` for embedding, #189, may have none) --
        # constrained layout already spaces subplots sensibly on its own,
        # and stacking an explicit hspace on top of it over-spaces them.
        gridspec_kwargs: dict[str, Any] = (
            {} if fig.get_layout_engine() is not None else {"hspace": 0.3}
        )
        gridspec = fig.add_gridspec(2, 1, height_ratios=[3, 1], **gridspec_kwargs)  # type: ignore
        path_axes = _make_path_axes(fig, ndims, gridspec[0])
        timeline_axes = fig.add_subplot(gridspec[1])
    else:
        path_axes = _make_path_axes(fig, ndims)
        timeline_axes = None

    _style_path_axes(path_axes, ndims, axis_labels)
    _set_title(fig, title or _default_title(axis_labels))

    if spec is not None and ndims <= 2:
        for patch in _get_boundaries(spec):
            path_axes.add_patch(patch)

    _draw_path(
        path_axes,
        scan,
        axis_labels,
        detector_to_stream,
        stream_colours,
        max_trigger_markers,
    )
    if timeline_axes is not None:
        _draw_timeline(fig, timeline_axes, scan, detector_to_stream, stream_colours)

    _apply_modern_style(fig)
    if owns_figure:
        plt.show()  # type: ignore
    return fig


def plot_spec(
    spec: Spec[Any, Any, Any],
    *,
    fig: Figure | None = None,
    title: str | None = None,
    max_trigger_markers: int = DEFAULT_MAX_TRIGGER_MARKERS,
) -> Figure:
    """Compile *spec* and plot it (path + timeline) with region boundaries.

    See ``plot_scan``.
    """
    scan = spec.compile()
    return plot_scan(
        scan,
        fig=fig,
        title=title,
        spec=spec,
        max_trigger_markers=max_trigger_markers,
    )


def plot_path(
    scan: Scan[Any, Any, Any],
    *,
    fig: Figure | None = None,
    title: str | None = None,
    spec: Spec[Any, Any, Any] | None = None,
    max_trigger_markers: int = DEFAULT_MAX_TRIGGER_MARKERS,
) -> Figure:
    """Plot only the motion path — see ``plot_scan``'s path panel."""
    axis_labels = _flatten_axes(scan)
    ndims = len(axis_labels)
    detector_to_stream = _detector_stream_map(scan)
    stream_colours = _stream_colours(scan)

    owns_figure = fig is None
    if fig is None:
        fig = plt.figure(  # type: ignore
            figsize=(6, 6) if ndims else (6, 2), layout="constrained"
        )
    axes = _make_path_axes(fig, ndims)
    _style_path_axes(axes, ndims, axis_labels)
    _set_title(fig, title or _default_title(axis_labels))

    if spec is not None and ndims <= 2:
        for patch in _get_boundaries(spec):
            axes.add_patch(patch)

    _draw_path(
        axes,
        scan,
        axis_labels,
        detector_to_stream,
        stream_colours,
        max_trigger_markers,
    )
    _apply_modern_style(fig)
    if owns_figure:
        plt.show()  # type: ignore
    return fig


def plot_timeline(
    scan: Scan[Any, Any, Any],
    *,
    fig: Figure | None = None,
    title: str | None = None,
) -> Figure:
    """Plot only the trigger timeline (Gantt-style, one row per stream).

    See ``plot_scan``.
    """
    detector_to_stream = _detector_stream_map(scan)
    stream_colours = _stream_colours(scan)

    owns_figure = fig is None
    if fig is None:
        fig = plt.figure(figsize=(8, 3), layout="constrained")  # type: ignore
    axes = fig.add_subplot()
    _set_title(fig, title or "Trigger timeline")

    _draw_timeline(fig, axes, scan, detector_to_stream, stream_colours)
    _apply_modern_style(fig)
    if owns_figure:
        plt.show()  # type: ignore
    return fig


def _default_title(axis_labels: list[Any]) -> str:
    return f"Scan[{', '.join(str(a) for a in axis_labels)}]"


def _set_title(fig: Figure, text: str) -> None:
    fig.suptitle(text, fontsize=13, fontweight="bold", color=_INK)  # type: ignore


def _apply_modern_style(fig: Figure) -> None:
    """Light, uncluttered chrome: no top/right spines, soft grid, muted ink."""
    fig.patch.set_facecolor(_FIG_BG)
    for axes in fig.axes:
        axes.set_facecolor(_AXES_BG)
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axes.spines[side].set_color(_MUTED_INK)
            axes.spines[side].set_linewidth(0.8)
        axes.tick_params(colors=_MUTED_INK, labelsize=9)  # type: ignore
        axes.title.set_color(_INK)
        axes.xaxis.label.set_color(_MUTED_INK)
        axes.yaxis.label.set_color(_MUTED_INK)
        axes.set_axisbelow(True)
        axes.grid(True, color=_GRID, linewidth=0.8)  # type: ignore
        legend = axes.get_legend()
        if legend is not None:
            legend.get_frame().set_alpha(0.9)
            legend.get_frame().set_edgecolor(_GRID)
            legend.get_frame().set_linewidth(0.8)


# ---------------------------------------------------------------------------
# Path panel
# ---------------------------------------------------------------------------


def _make_path_axes(fig: Figure, ndims: int, subplot_spec: Any = None) -> Axes:
    projection = "3d" if ndims > 2 else None
    if subplot_spec is not None:
        return fig.add_subplot(subplot_spec, projection=projection)
    return fig.add_subplot(projection=projection)


def _style_path_axes(axes: Axes, ndims: int, axis_labels: list[Any]) -> None:
    if ndims > 2:
        axes.grid(False)  # type: ignore
        axes.set_zlabel(str(axis_labels[-3]))  # type: ignore
        axes.set_ylabel(str(axis_labels[-2]))  # type: ignore
        axes.view_init(elev=15)  # type: ignore
    elif ndims == 2:
        axes.set_ylabel(str(axis_labels[-2]))  # type: ignore
    else:
        axes.yaxis.set_visible(False)
    if axis_labels:
        axes.set_xlabel(str(axis_labels[-1]))  # type: ignore


def _window_points(
    window: Window[Any, Any], axis_labels: list[Any], last: dict[Any, float]
) -> list[dict[Any, float]]:
    """One or more (axis -> position) points representing *window*'s path.

    Step windows contribute a single point (their only physical position).
    Fly windows contribute their boundary points, plus intermediate samples
    via ``positions()`` when the trajectory is non-linear.
    """
    if window.moving_axes:
        if window.non_linear:
            times = np.linspace(0.0, window.duration, _FLY_SAMPLE_POINTS)
            sampled = window.positions(times)
            pts: list[dict[Any, float]] = []
            for i in range(len(times)):
                pt = dict(last)
                for ax in axis_labels:
                    if ax in sampled:
                        pt[ax] = float(sampled[ax][i])
                    elif ax in window.static_axes:
                        pt[ax] = window.static_axes[ax]
                pts.append(pt)
            return pts
        start_pt = dict(last)
        end_pt = dict(last)
        for ax in axis_labels:
            if ax in window.moving_axes:
                start_pt[ax] = window.moving_axes[ax].start_position
                end_pt[ax] = window.moving_axes[ax].end_position
            elif ax in window.static_axes:
                start_pt[ax] = end_pt[ax] = window.static_axes[ax]
        return [start_pt, end_pt]
    point = dict(last)
    for ax in axis_labels:
        if ax in window.static_axes:
            point[ax] = window.static_axes[ax]
    return [point]


def _draw_path(
    axes: Axes,
    scan: Scan[Any, Any, Any],
    axis_labels: list[Any],
    detector_to_stream: dict[Any, str],
    stream_colours: dict[str, str],
    max_trigger_markers: int,
) -> None:
    """Draw the motion path, turnarounds, trigger markers and legend onto *axes*."""
    trigger_markers = _draw_path_and_streams(
        axes, scan, axis_labels, detector_to_stream, stream_colours
    )
    if len(trigger_markers) <= max_trigger_markers:
        _draw_trigger_markers(axes, trigger_markers)
    _draw_stream_legend(axes, stream_colours)


def _draw_path_and_streams(
    axes: Axes,
    scan: Scan[Any, Any, Any],
    axis_labels: list[Any],
    detector_to_stream: dict[Any, str],
    stream_colours: dict[str, str],
) -> list[tuple[dict[Any, float], str, float]]:
    """Draw the motion path window by window; collect trigger markers.

    Each window's own points are joined by a straight line (or drawn as a
    single marker, for a step window's one physical point); consecutive
    windows are bridged by a plain line when contiguous, or a dashed
    turnaround arrow when their positions jump. Deliberately *not* one
    global spline fit across the whole scan (see module docstring).

    Returns a list of (position, colour, livetime) for every trigger
    instant, deferred to the caller so the total count can be checked
    against the marker cap before actually plotting them.
    """
    last: dict[Any, float] = {}
    trigger_markers: list[tuple[dict[Any, float], str, float]] = []
    first_window = True

    for window in scan:
        window_points = _window_points(window, axis_labels, last)
        start_pos = window_points[0]
        colour = _window_colour(
            _window_streams(window, detector_to_stream), stream_colours
        )

        if not first_window:
            gap = any(
                ax in last
                and ax in start_pos
                and not isclose(last[ax], start_pos[ax], rel_tol=1e-9, abs_tol=1e-9)
                for ax in axis_labels
            )
            if gap:
                _draw_turnaround(axes, axis_labels, last, start_pos)
            else:
                _draw_segment(axes, axis_labels, [last, start_pos], colour)

        _draw_segment(axes, axis_labels, window_points, colour)

        for ts in window.trigger_sequences:
            for t, det, livetime in _trigger_marker_times(ts):
                stream = detector_to_stream.get(det)
                marker_colour = (
                    stream_colours.get(stream, "black") if stream else "black"
                )
                if window.moving_axes and not window.non_linear:
                    frac = 0.0 if window.duration == 0 else t / window.duration
                    pos = {
                        ax: (
                            window.moving_axes[ax].start_position
                            + frac
                            * (
                                window.moving_axes[ax].end_position
                                - window.moving_axes[ax].start_position
                            )
                            if ax in window.moving_axes
                            else window_points[0].get(ax, last.get(ax, 0.0))
                        )
                        for ax in axis_labels
                    }
                elif window.moving_axes:
                    times = np.array([min(max(t, 0.0), window.duration)])
                    sampled = window.positions(times)
                    pos = {
                        ax: (
                            float(sampled[ax][0])
                            if ax in sampled
                            else window_points[0].get(ax, last.get(ax, 0.0))
                        )
                        for ax in axis_labels
                    }
                else:
                    pos = dict(window_points[0])
                trigger_markers.append((pos, marker_colour, livetime))

        last = dict(window_points[-1])
        first_window = False

    return trigger_markers


def _draw_segment(
    axes: Axes,
    axis_labels: list[Any],
    points: list[dict[Any, float]],
    colour: str,
) -> None:
    """Draw a straight line through *points* (or a single marker for one point)."""
    if len(points) == 1:
        arrays = [np.array([points[0].get(ax, 0.0)]) for ax in axis_labels] or [
            np.zeros(1)
        ]
        _plot_arrays(
            axes,
            arrays,
            marker="o",
            markersize=6,
            color=colour,
            markeredgecolor="white",
            markeredgewidth=0.6,
        )
        return
    arrays = [np.array([p.get(ax, 0.0) for p in points]) for ax in axis_labels] or [
        np.zeros(len(points))
    ]
    _plot_arrays(
        axes, arrays, color=colour, linewidth=2.4, alpha=0.95, solid_capstyle="round"
    )


def _draw_turnaround(
    axes: Axes,
    axis_labels: list[Any],
    from_pos: dict[Any, float],
    to_pos: dict[Any, float],
) -> None:
    """Bridge a position discontinuity between runs.

    A curved dashed arrow (2D/1D), or a straight dashed line + 3D arrowhead
    (3D, where curved patches don't project reliably through Axes3D).
    """
    ndims = len(axis_labels)
    if ndims <= 2:
        x_ax = axis_labels[-1] if axis_labels else None
        y_ax = axis_labels[-2] if ndims == 2 else None
        xy_from = (
            from_pos.get(x_ax, 0.0) if x_ax is not None else 0.0,
            from_pos.get(y_ax, 0.0) if y_ax is not None else 0.0,
        )
        xy_to = (
            to_pos.get(x_ax, 0.0) if x_ax is not None else 0.0,
            to_pos.get(y_ax, 0.0) if y_ax is not None else 0.0,
        )
        arrow = patches.FancyArrowPatch(
            xy_from,
            xy_to,
            connectionstyle="arc3,rad=0.25",
            arrowstyle="-|>",
            mutation_scale=10,
            color=_TURNAROUND_COLOUR,
            linestyle="--",
            linewidth=1.2,
        )
        axes.add_patch(arrow)
        return
    arrays = [
        np.array(
            [
                from_pos.get(ax, to_pos.get(ax, 0.0)),
                to_pos.get(ax, from_pos.get(ax, 0.0)),
            ]
        )
        for ax in axis_labels
    ]
    _plot_arrays(axes, arrays, color=_TURNAROUND_COLOUR, linestyle="--")
    _add_3d_turnaround_arrow(axes, arrays)


def _draw_trigger_markers(
    axes: Axes, trigger_markers: list[tuple[dict[Any, float], str, float]]
) -> None:
    """Scatter every trigger instant, sized by livetime and coloured by stream."""
    if not trigger_markers:
        return
    axis_labels = list(trigger_markers[0][0].keys())
    per_axis = [
        np.array([m[0].get(ax, 0.0) for m in trigger_markers]) for ax in axis_labels
    ] or [np.zeros(len(trigger_markers))]
    face_colours = [m[1] for m in trigger_markers]
    sizes = np.array([_marker_size(m[2]) for m in trigger_markers]) ** 2
    # A thin white edge lifts markers off the path line/grid behind them
    # instead of blending into it -- a cheap "halo" for visual pop.
    kwargs: dict[str, Any] = {
        "c": face_colours,
        "s": sizes,
        "alpha": 0.9,
        "edgecolors": "white",
        "linewidths": 0.6,
    }
    if len(per_axis) > 2:
        axes.scatter3D(per_axis[2], per_axis[1], per_axis[0], **kwargs)  # type: ignore
    elif len(per_axis) == 2:
        axes.scatter(per_axis[1], per_axis[0], **kwargs)  # type: ignore
    else:
        axes.scatter(per_axis[0], np.zeros(len(per_axis[0])), **kwargs)  # type: ignore


def _draw_stream_legend(axes: Axes, stream_colours: dict[str, str]) -> None:
    if not stream_colours:
        return
    handles = [
        plt.Line2D(  # type: ignore
            [0],
            [0],
            color=colour,
            marker="o",
            markersize=7,
            linewidth=2.4,
            label=name,
        )
        for name, colour in stream_colours.items()
    ]
    axes.legend(handles=handles, loc="best", fontsize="small")  # type: ignore


# ---------------------------------------------------------------------------
# Timeline panel
# ---------------------------------------------------------------------------


def _collect_timeline_rows(
    scan: Scan[Any, Any, Any],
    detector_to_stream: dict[Any, str],
    stream_colours: dict[str, str],
) -> tuple[
    list[str],
    dict[str, str],
    dict[str, bool],
    dict[str, list[tuple[float, float]]],
    dict[str, list[tuple[float, float]]],
]:
    """Walk the scan once, building per-row (start, duration) interval lists.

    Returns (row order, row colour, row is_parent, full-period intervals,
    livetime-only intervals) — full-period blocks are drawn as a faint
    background (deadtime included) with the livetime portion solid on top,
    so a slow parent and a fast nested child both show duty cycle at a
    glance. Rows: one per stream (parent), plus one per distinct child
    detector-set nested under its parent, ordered by first appearance.
    """
    row_order: list[str] = []
    row_colour: dict[str, str] = {}
    row_is_parent: dict[str, bool] = {}
    full_by_row: dict[str, list[tuple[float, float]]] = {}
    live_by_row: dict[str, list[tuple[float, float]]] = {}

    def _ensure_row(key: str, colour: str, is_parent: bool) -> None:
        if key not in row_colour:
            row_order.append(key)
            row_colour[key] = colour
            row_is_parent[key] = is_parent
            full_by_row[key] = []
            live_by_row[key] = []

    elapsed = 0.0
    for window in scan:
        for ts in window.trigger_sequences:
            parent_stream = detector_to_stream.get(next(iter(ts.detectors), None))
            parent_key = parent_stream or "?"
            parent_colour = stream_colours.get(parent_key, "black")
            _ensure_row(parent_key, parent_colour, True)

            parent_period = 0.0
            if (
                ts.trigger_repeat.livetime is not None
                and ts.trigger_repeat.deadtime is not None
            ):
                parent_period = ts.trigger_repeat.livetime + ts.trigger_repeat.deadtime
                for k in range(ts.trigger_repeat.num):
                    block_start = elapsed + k * parent_period
                    full_by_row[parent_key].append((block_start, parent_period))
                    live_by_row[parent_key].append(
                        (
                            block_start + ts.trigger_repeat.deadtime / 2,
                            ts.trigger_repeat.livetime,
                        )
                    )

            for child in ts.children:
                child_label = ",".join(sorted(str(d) for d in child.detectors))
                child_key = f"{parent_key} └ {child_label}"
                child_stream = detector_to_stream.get(
                    next(iter(child.detectors), None), parent_key
                )
                _ensure_row(
                    child_key, stream_colours.get(child_stream, parent_colour), False
                )
                for p in range(ts.trigger_repeat.num):
                    t = elapsed + p * parent_period
                    for r in child.repeats:
                        if r.livetime is None or r.deadtime is None:
                            continue
                        cperiod = r.livetime + r.deadtime
                        for k in range(r.num):
                            block_start = t + k * cperiod
                            full_by_row[child_key].append((block_start, cperiod))
                            live_by_row[child_key].append(
                                (block_start + r.deadtime / 2, r.livetime)
                            )
                        t += r.num * cperiod
        elapsed += window.duration

    return row_order, row_colour, row_is_parent, full_by_row, live_by_row


def _draw_timeline(
    fig: Figure,
    axes: Axes,
    scan: Scan[Any, Any, Any],
    detector_to_stream: dict[Any, str],
    stream_colours: dict[str, str],
) -> bool:
    """Draw a Gantt-style trigger timeline onto *axes*.

    Returns False (and draws nothing but an empty-state message) if there's
    nothing to show.
    """
    row_order, row_colour, row_is_parent, full_by_row, live_by_row = (
        _collect_timeline_rows(scan, detector_to_stream, stream_colours)
    )
    if not row_order:
        axes.text(  # type: ignore
            0.5,
            0.5,
            "no detector triggering",
            ha="center",
            va="center",
            color="grey",
            transform=axes.transAxes,
        )
        axes.set_xticks([])  # type: ignore
        axes.set_yticks([])  # type: ignore
        return False

    n = len(row_order)
    for i, key in enumerate(row_order):
        y = n - 1 - i
        if i % 2 == 1:
            axes.axhspan(y - 0.5, y + 0.5, color=_ROW_BAND, zorder=0)  # type: ignore
        height = 0.6 if row_is_parent[key] else 0.4
        colour = row_colour[key]
        full = full_by_row[key]
        live = live_by_row[key]
        if full:
            axes.broken_barh(  # type: ignore
                full,
                (y - height / 2, height),
                facecolors=colour,
                alpha=0.18,
                edgecolor="none",
                zorder=1,
            )
        if live:
            axes.broken_barh(  # type: ignore
                live,
                (y - height / 2, height),
                facecolors=colour,
                alpha=0.95,
                edgecolor="none",
                zorder=1,
            )

    axes.set_yticks(range(n))  # type: ignore
    axes.set_yticklabels(list(reversed(row_order)))  # type: ignore
    axes.set_ylim(-0.5, n - 0.5)
    axes.set_xlabel("time (s)")  # type: ignore
    axes.grid(axis="y", visible=False)  # type: ignore

    # Reserve enough left margin for (potentially long) row labels even when
    # this Axes' Figure has no constrained-layout engine of its own -- e.g. a
    # caller-supplied `fig=` for embedding (#189), which owns its own layout.
    if fig.get_layout_engine() is None:
        longest = max(len(label) for label in row_order)
        left = min(0.1 + 0.011 * longest, 0.45)
        fig.subplots_adjust(left=left)  # type: ignore
    return True
