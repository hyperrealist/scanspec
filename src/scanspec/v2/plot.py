"""`plot_scan`/`plot_spec` — visualize a scanspec.v2 scan's motion path and triggering.

Draws the motion path taken through the scan (as 1.x's ``plot_spec`` did),
plus what 1.x had no data model for: which detector stream(s) are active on
each stretch of path, and where individual detector triggers fall — down to
nested multi-rate children (``TriggerChild``) within a parent repeat.

Built directly on ``Window``/``Scan`` rather than fully materialized position
arrays, since v2 deliberately avoids allocating those (see ``Scan``'s
docstring). ``Scan`` is safe to iterate more than once (``__iter__`` is a
plain generator method, not one-shot state) — used here for a cheap
range-finding pass before drawing.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import cycle
from math import isclose
from typing import Any

import numpy as np
import numpy.typing as npt
from matplotlib import colors, patches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import proj3d  # type: ignore
from scipy import interpolate  # type: ignore

from .core import (
    ConcatSource,
    Scan,
    TriggerRepeat,
    TriggerSequence,
    Window,
    WindowGenerator,
)
from .specs import Ellipse, Polygon, Spec

__all__ = ["plot_scan", "plot_spec"]

# Above this many resolved trigger instants across the whole scan, individual
# trigger markers are skipped (path/stream coloring is kept) — otherwise a
# dense fly scan with fast child detectors can ask for hundreds of thousands
# of markers on one static figure.
DEFAULT_MAX_TRIGGER_MARKERS = 2000

_FLY_SAMPLE_POINTS = 20  # samples along a non_linear fly window's true curve


# ---------------------------------------------------------------------------
# matplotlib plumbing (data-model-agnostic; adapted from 1.x's plot.py)
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


def _plot_arrow(axes: Axes, arrays: list[npt.NDArray[np.float64]]):
    if len(arrays) == 1:
        arrays = [np.array([0, 0])] + arrays
    if len(arrays) == 2:
        head = [a[-1] for a in reversed(arrays)]
        tail = [a[-1] - (a[-1] - a[-2]) * 0.1 for a in reversed(arrays)]
        axes.annotate(  # type: ignore
            "",
            tuple(head[:2]),
            tuple(tail[:2]),
            arrowprops={"color": "lightgrey", "arrowstyle": "-|>"},
        )
    elif len(arrays) == 3:
        arrows = [a[-2:] for a in reversed(arrays)]
        a = Arrow3D(*arrows[:3], mutation_scale=10, arrowstyle="-|>", color="lightgrey")
        axes.add_artist(a)


def _plot_spline(
    axes: Axes,
    ranges: list[float],
    arrays: list[npt.NDArray[np.float64]],
    index_colours: dict[int, str],
) -> list[list[npt.NDArray[np.float64]]]:
    """Fit and draw a parametric spline through *arrays*, coloured in pieces.

    ``index_colours`` maps a starting index in ``arrays`` to the colour of
    the piece beginning there (piece extends to the next key, or the end).
    """
    scaled_arrays = [a / r for a, r in zip(arrays, ranges, strict=False)]
    t = np.zeros(len(arrays[0]))
    t[1:] = np.sqrt(sum((arr[1:] - arr[:-1]) ** 2 for arr in scaled_arrays))
    t = np.cumsum(t)
    if t[-1] == 0:
        return []
    for s, r in zip(scaled_arrays, ranges, strict=False):
        if s[0] == s[-1]:
            s += np.linspace(0, r * 1e-7, len(s))
    t /= t[-1]
    k = min(2, len(arrays[0]) - 1)
    tck, _ = interpolate.splprep(scaled_arrays, k=k, s=0)  # type: ignore
    starts = sorted(index_colours)
    stops = starts[1:] + [len(arrays[0]) - 1]
    pieces: list[list[npt.NDArray[np.float64]]] = []
    for start, stop in zip(starts, stops, strict=False):
        start_value: float = t[start]
        stop_value: float = t[stop]
        tnew = np.linspace(start_value, stop_value, num=1001)
        spline: npt.NDArray[np.float64] = interpolate.splev(tnew, tck)  # type: ignore
        unscaled = [a * r for a, r in zip(spline, ranges, strict=False)]
        _plot_arrays(axes, list(unscaled), color=index_colours[start])  # type: ignore
        pieces.append(unscaled)  # type: ignore
    return pieces


def _get_boundaries(spec: Spec[Any, Any, Any]) -> Iterator[patches.Patch]:
    if isinstance(spec, Ellipse):
        xy = spec.x_centre, spec.y_centre
        y_diam = (
            spec.y_diameter if spec.y_diameter is not None else abs(spec.x_diameter)
        )
        yield patches.Ellipse(xy, spec.x_diameter, y_diam, fill=False)
    elif isinstance(spec, Polygon):
        yield patches.Polygon(spec.vertices, fill=False)
    else:
        for name in type(spec).model_fields:
            s = getattr(spec, name)
            if isinstance(s, Spec):
                yield from _get_boundaries(s)  # type: ignore[reportUnknownArgumentType]


# ---------------------------------------------------------------------------
# v2-specific: axes, ranges, stream/colour resolution, trigger timing
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


def _axis_ranges(scan: Scan[Any, Any, Any], axes: list[Any]) -> dict[Any, float]:
    """Per-axis (max - min) over every window, without storing position arrays."""
    lo = dict.fromkeys(axes, float("inf"))
    hi = dict.fromkeys(axes, float("-inf"))
    last: dict[Any, float] = {}
    for window in scan:
        for ax in axes:
            for pos in _window_axis_bounds(window, ax, last):
                lo[ax] = min(lo[ax], pos)
                hi[ax] = max(hi[ax], pos)
        _advance(window, axes, last)
    return {ax: max(hi[ax] - lo[ax], 1e-4) for ax in axes}


def _window_axis_bounds(window: Window[Any, Any], ax: Any, last: dict[Any, float]):
    if ax in window.moving_axes:
        am = window.moving_axes[ax]
        yield am.start_position
        yield am.end_position
    elif ax in window.static_axes:
        yield window.static_axes[ax]
    elif ax in last:
        yield last[ax]


def _advance(window: Window[Any, Any], axes: list[Any], last: dict[Any, float]) -> None:
    """Update *last* (last-known position per axis) after visiting *window*."""
    for ax in axes:
        if ax in window.moving_axes:
            last[ax] = window.moving_axes[ax].end_position
        elif ax in window.static_axes:
            last[ax] = window.static_axes[ax]


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
    palette = cycle(colors.TABLEAU_COLORS)
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
        return "lightgrey"
    return stream_colours[sorted(streams)[0]]


def _repeat_times(repeats: list[TriggerRepeat], t0: float) -> tuple[list[float], float]:
    """Centred-livetime trigger midpoints for a sequential repeats list.

    Returns (absolute times from t0, end time) so callers can chain/nest.
    """
    times: list[float] = []
    t = t0
    for r in repeats:
        if r.livetime is None or r.deadtime is None:
            continue
        period = r.livetime + r.deadtime
        for k in range(r.num):
            times.append(t + k * period + period / 2)
        t += r.num * period
    return times, t


def _trigger_marker_times(ts: TriggerSequence[Any]) -> list[tuple[float, Any]]:
    """(time, representative_detector) for the parent repeat and every nested child.

    A child's own repeats run in full during *every* parent repeat (per
    ``TriggerSequence``'s docstring), so each child repeat-block is offset
    into its parent repeat's own span before applying the same centred-
    livetime placement recursively.
    """
    result: list[tuple[float, Any]] = []
    parent_times, _ = _repeat_times([ts.trigger_repeat], 0.0)
    parent_detector = next(iter(ts.detectors), None)
    result += [(t, parent_detector) for t in parent_times]

    if ts.trigger_repeat.livetime is None or ts.trigger_repeat.deadtime is None:
        return result
    parent_period = ts.trigger_repeat.livetime + ts.trigger_repeat.deadtime
    for child in ts.children:
        child_detector = next(iter(child.detectors), None)
        for p in range(ts.trigger_repeat.num):
            child_times, _ = _repeat_times(child.repeats, p * parent_period)
            result += [(t, child_detector) for t in child_times]
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
    """Plot a compiled Scan: motion path, turnarounds, and detector triggering.

    Colours each stretch of path by which detector stream(s) are active
    during it, and (below ``max_trigger_markers`` total instants) marks
    individual trigger times — including nested multi-rate ``TriggerChild``
    instants — placed at their true position via ``Window.positions()`` on
    fly windows.

    *fig*: draw into this Figure instead of creating one (embedding, e.g. in
    a Qt widget's ``FigureCanvas`` — https://github.com/bluesky/scanspec/issues/189).
    If not given, a new Figure is created and shown (``plt.show()``) before
    returning, matching 1.x's ``plot_spec`` behaviour; if given, the caller
    owns display.

    *spec*: when given, overlays ``Ellipse``/``Polygon`` region boundaries
    found in the spec tree (meaningless for a bare Scan, which retains no
    spec tree — this is why ``plot_spec`` passes it through and ``plot_scan``
    does not derive it itself).
    """
    axis_labels = _flatten_axes(scan)
    ndims = len(axis_labels)
    ranges = _axis_ranges(scan, axis_labels)
    detector_to_stream = _detector_stream_map(scan)
    stream_colours = _stream_colours(scan)

    owns_figure = fig is None
    if fig is None:
        fig = plt.figure(figsize=(6, 6) if ndims else (6, 2))  # type: ignore
    axes = _make_axes(fig, ndims, axis_labels)

    title = title or f"Scan[{', '.join(str(a) for a in axis_labels)}]"
    axes.set_title(title)  # type: ignore

    if spec is not None and ndims <= 2:
        for patch in _get_boundaries(spec):
            axes.add_patch(patch)

    trigger_markers = _draw_path_and_streams(
        axes, scan, axis_labels, ranges, detector_to_stream, stream_colours
    )
    if len(trigger_markers) <= max_trigger_markers:
        _draw_trigger_markers(axes, trigger_markers)

    _draw_legend(axes, stream_colours)

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
    """Compile *spec* and plot it, with region boundaries — see ``plot_scan``."""
    scan = spec.compile()
    return plot_scan(
        scan,
        fig=fig,
        title=title,
        spec=spec,
        max_trigger_markers=max_trigger_markers,
    )


# ---------------------------------------------------------------------------
# Drawing internals
# ---------------------------------------------------------------------------


def _make_axes(fig: Figure, ndims: int, axis_labels: list[Any]) -> Axes:
    if ndims > 2:
        axes = fig.add_subplot(projection="3d")
        axes.grid(False)  # type: ignore
        axes.set_zlabel(str(axis_labels[-3]))  # type: ignore
        axes.set_ylabel(str(axis_labels[-2]))  # type: ignore
        axes.view_init(elev=15)  # type: ignore
    elif ndims == 2:
        axes = fig.add_subplot()
        axes.set_ylabel(str(axis_labels[-2]))  # type: ignore
    else:
        axes = fig.add_subplot()
        axes.yaxis.set_visible(False)
    if axis_labels:
        axes.set_xlabel(str(axis_labels[-1]))  # type: ignore
    return axes


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


def _draw_path_and_streams(
    axes: Axes,
    scan: Scan[Any, Any, Any],
    axis_labels: list[Any],
    ranges: dict[Any, float],
    detector_to_stream: dict[Any, str],
    stream_colours: dict[str, str],
) -> list[tuple[dict[Any, float], str]]:
    """Draw contiguous path runs (spline + turnaround arrows); collect trigger markers.

    Returns a list of (position, colour) for every trigger instant, deferred
    to the caller so the total count can be checked against the marker cap
    before actually plotting them.
    """
    last: dict[Any, float] = {}
    run_arrays: dict[Any, list[float]] = {ax: [] for ax in axis_labels}
    run_colours: dict[int, str] = {}
    run_index = 0
    trigger_markers: list[tuple[dict[Any, float], str]] = []
    first_window = True

    def flush_run() -> None:
        nonlocal run_arrays, run_colours, run_index
        if run_index == 0:
            return
        arrays = [np.array(run_arrays[ax]) for ax in axis_labels] or [
            np.zeros(run_index)
        ]
        ranges_list = [ranges[ax] for ax in axis_labels] or [1.0]
        _draw_run(axes, ranges_list, arrays, run_colours)
        run_arrays = {ax: [] for ax in axis_labels}
        run_colours = {}
        run_index = 0

    for window in scan:
        window_points = _window_points(window, axis_labels, last)
        start_pos = window_points[0]

        gap = not first_window and any(
            ax in last
            and ax in start_pos
            and not isclose(last[ax], start_pos[ax], rel_tol=1e-9, abs_tol=1e-9)
            for ax in axis_labels
        )
        if gap:
            turnaround_from = dict(last)
            flush_run()
            _draw_turnaround(axes, axis_labels, turnaround_from, start_pos)

        colour = _window_colour(
            _window_streams(window, detector_to_stream), stream_colours
        )
        run_colours[run_index] = colour
        for pt in window_points:
            for ax in axis_labels:
                run_arrays[ax].append(pt.get(ax, last.get(ax, 0.0)))
            run_index += 1

        for ts in window.trigger_sequences:
            for t, det in _trigger_marker_times(ts):
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
                trigger_markers.append((pos, marker_colour))

        last = dict(window_points[-1])
        first_window = False

    flush_run()
    return trigger_markers


def _draw_run(
    axes: Axes,
    ranges_list: list[float],
    arrays: list[npt.NDArray[np.float64]],
    index_colours: dict[int, str],
) -> None:
    if len(arrays[0]) == 1:
        _plot_arrays(
            axes,
            [np.array([v]) for v in [a[0] for a in arrays]],
            marker=5,
            color=next(iter(index_colours.values()), "lightgrey"),
        )
        return
    _plot_spline(axes, ranges_list, arrays, index_colours)


def _draw_turnaround(
    axes: Axes,
    axis_labels: list[Any],
    from_pos: dict[Any, float],
    to_pos: dict[Any, float],
) -> None:
    """Bridge a position discontinuity between runs: grey dashed line + arrowhead."""
    arrays = [
        np.array(
            [
                from_pos.get(ax, to_pos.get(ax, 0.0)),
                to_pos.get(ax, from_pos.get(ax, 0.0)),
            ]
        )
        for ax in axis_labels
    ] or [np.zeros(2)]
    _plot_arrays(axes, arrays, color="lightgrey", linestyle="--")
    _plot_arrow(axes, arrays)


def _draw_trigger_markers(
    axes: Axes, trigger_markers: list[tuple[dict[Any, float], str]]
) -> None:
    by_colour: dict[str, list[dict[Any, float]]] = {}
    for pos, colour in trigger_markers:
        by_colour.setdefault(colour, []).append(pos)
    for colour, positions in by_colour.items():
        axis_labels = list(positions[0].keys())
        arrays = [np.array([p[ax] for p in positions]) for ax in axis_labels] or [
            np.zeros(len(positions))
        ]
        _plot_arrays(axes, arrays, linestyle="", marker=".", markersize=3, color=colour)


def _draw_legend(axes: Axes, stream_colours: dict[str, str]) -> None:
    if not stream_colours:
        return
    handles = [
        plt.Line2D([0], [0], color=colour, marker=".", label=name)  # type: ignore
        for name, colour in stream_colours.items()
    ]
    axes.legend(handles=handles, loc="best", fontsize="small")  # type: ignore
