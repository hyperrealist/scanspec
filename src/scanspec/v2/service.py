"""FastAPI service exposing scanspec.v2 specs over REST.

Lets an external web GUI request plottable/triggerable data for a spec over
HTTP, without a Python client -- the same job 1.x's ``service.py`` did, but
adapted to v2's windowed, no-precomputed-arrays data model (see ``core.py``'s
``Scan``/``Window`` docstrings) and extended with a ``/triggers`` endpoint
for v2's actual differentiator: multi-stream/multi-rate detector triggering.

Every endpoint operates at one-row-per-window granularity: a v2 ``Window``
is the unit of collection (one physical point for a step window, one
continuous sweep for a fly window), so "frame" here means "window", not
"discretised sample point" the way 1.x's precomputed ``Dimension`` did.
``Scan.number_of_events`` gives the total row count up front without
materialising any windows.

``plot.py`` and this module deliberately do not import from each other --
``pyproject.toml`` declares ``plotting`` (matplotlib/scipy) and ``service``
(fastapi/uvicorn) as independent optional extras, and anything both need
(axis flattening, turnaround detection, detector-stream resolution, trigger
timing) lives in ``core.py``/``specs.py`` instead, which only depend on
numpy/pydantic.
"""

import base64
import json
from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field

from .core import (
    DetectorGroup,
    Window,
    detector_stream_map,
    flatten_axes,
    is_turnaround,
    repeat_times,
)
from .specs import AnySpec

app = FastAPI(version="0.1.0")

#
# Data model
#

Points = str | list[float]


class PointsFormat(str, Enum):
    """Formats in which we can return point arrays."""

    STRING = "STRING"
    FLOAT_LIST = "FLOAT_LIST"
    BASE64_ENCODED = "BASE64_ENCODED"


class ValidResponse(BaseModel):
    """Response model for spec validation."""

    input_spec: AnySpec[str, str, str] = Field(description="The input scanspec")
    valid_spec: AnySpec[str, str, str] = Field(
        description="The canonical (re-serialised) version of the spec"
    )


class SpecRequest(BaseModel):
    """A request for generated per-window scan data."""

    spec: AnySpec[str, str, str] = Field(description="The spec to generate data from")
    max_frames: int | None = Field(
        description="The maximum number of windows to return; evenly "
        "downsampled if the scan has more than this. None returns all.",
        default=100_000,
    )
    format: PointsFormat = Field(
        description="The format in which to output point data",
        default=PointsFormat.FLOAT_LIST,
    )


class GeneratedPointsResponse(BaseModel):
    """Base class for responses that include generated per-window data."""

    total_frames: int = Field(description="Total number of windows in the scan")
    returned_frames: int = Field(
        description="Number of windows in this response, less than "
        "total_frames if downsampled"
    )
    format: PointsFormat = Field(description="Format of returned point data")


class MidpointsResponse(GeneratedPointsResponse):
    """Midpoints of a generated scan."""

    midpoints: Mapping[str, Points] = Field(
        description="The midpoint of each window, for each axis. For a step "
        "window this is its one physical position; for a fly window, the "
        "midpoint between its start and end boundary positions."
    )


class BoundsResponse(BaseModel):
    """Boundary positions of a generated scan.

    Only fly windows have real boundary kinematics in v2 (see
    ``core.AxisMotion``) -- a step window has a single physical position and
    no bounds, so its entries are omitted (``None``) rather than fabricated.
    Always plain nullable float lists: the STRING/BASE64_ENCODED formats
    (see ``PointsFormat``) can't represent a ``None`` entry, so this endpoint
    doesn't offer them.
    """

    total_frames: int
    returned_frames: int
    lower: Mapping[str, list[float | None]] = Field(
        description="Lower (start) bound per window, per axis; None for a "
        "step window's axes"
    )
    upper: Mapping[str, list[float | None]] = Field(
        description="Upper (end) bound per window, per axis; None for a "
        "step window's axes"
    )


class GapResponse(BaseModel):
    """Presence of gaps (turnarounds) between windows of a generated scan."""

    gap: list[bool] = Field(
        description="One entry per window: whether there's a turnaround "
        "between it and the previous window. Always False for the first "
        "window (there is no previous window)."
    )


class SmallestStepResponse(BaseModel):
    """Information about the smallest steps between window midpoints."""

    absolute: float = Field(
        description="Absolute smallest distance between two window "
        "midpoints on a single axis"
    )
    per_axis: Mapping[str, float] = Field(
        description="Smallest distance between two window midpoints, per axis"
    )


class DetectorGroupInfo(BaseModel):
    """One detector group's trigger configuration, for /triggers."""

    detectors: list[str]
    exposures_per_event: int
    livetime: float | None
    deadtime: float | None


class StreamInfo(BaseModel):
    """One detector stream's identity, for /triggers."""

    name: str
    kind: Literal["windowed", "continuous"]
    detector_groups: list[DetectorGroupInfo]


class TriggerEvent(BaseModel):
    """One resolved trigger instant, for /triggers.

    ``time`` is centred-livetime (ADR 0006): the midpoint of this
    detector's exposure, in seconds from scan start.
    """

    time: float
    livetime: float
    detectors: list[str]
    stream: str | None
    parent: bool = Field(
        description="False if this instant came from a TriggerChild "
        "(a faster nested group) rather than the window's root repeat"
    )


class TriggersResponse(BaseModel):
    """Per-stream metadata and every resolved trigger instant in the scan.

    Unlike /midpoints and /bounds, this is not downsampled -- a GUI wanting
    a bounded response for a very-high-repeat-count scan should apply its
    own cap client-side for now (not yet a feature of this endpoint).
    """

    streams: list[StreamInfo]
    events: list[TriggerEvent]


#
# API routes
#

_EXAMPLE_SPEC_JSON = json.loads(
    '{"axis": "y", "start": 0.0, "stop": 10.0, "num": 3, "type": "Linspace"}'
)

# Annotated[AnySpec[...], Body(...)], not a `= Body(...)` default: AnySpec's
# discriminated-union machinery (see specs.py) only survives FastAPI's
# request-parameter resolution via the modern Annotated form -- the
# default-value form loses the Discriminator and silently mis-resolves to
# whichever Union member's fields happen to structurally match first.
SpecBody = Annotated[AnySpec[str, str, str], Body(..., examples=[_EXAMPLE_SPEC_JSON])]


@app.post("/valid", response_model=ValidResponse)
def valid(spec: SpecBody) -> ValidResponse:
    """Validate whether a scanspec can produce a viable scan.

    v2's Spec is already a real pydantic model (a discriminated union via
    AnySpec) -- FastAPI has already fully validated *spec* by the time this
    function runs, so "canonical form" is just the input itself, unlike
    1.x's separate serialize/deserialize round trip.

    Args:
        spec: The scanspec to validate

    Returns:
        ValidResponse: The spec back, twice -- a 422 is returned instead if
            it isn't valid.

    """
    return ValidResponse(input_spec=spec, valid_spec=spec)


@app.post("/midpoints", response_model=MidpointsResponse)
def midpoints(
    request: SpecRequest = Body(...),
) -> MidpointsResponse:
    """Generate per-window midpoints from a scanspec.

    Args:
        request: Scanspec and formatting/downsampling info.

    Returns:
        MidpointsResponse: Midpoints of the scan, one per window.

    """
    scan = request.spec.compile()
    axis_labels = flatten_axes(scan)
    total = scan.number_of_events
    wanted = _wanted_indices(total, request.max_frames)

    mid_columns: dict[str, list[float]] = {ax: [] for ax in axis_labels}
    last: dict[str, float] = {}
    for idx, window in enumerate(scan):
        _, mid, upper = _window_bounds(window, axis_labels, last)
        last = upper
        if idx in wanted:
            for ax in axis_labels:
                mid_columns[ax].append(mid[ax])

    return MidpointsResponse(
        total_frames=total,
        returned_frames=len(wanted),
        format=request.format,
        midpoints=_format_points(mid_columns, request.format),
    )


@app.post("/bounds", response_model=BoundsResponse)
def bounds(request: SpecRequest = Body(...)) -> BoundsResponse:
    """Generate per-window boundary positions from a scanspec.

    Args:
        request: Scanspec and downsampling info (``request.format`` is
            ignored -- see ``BoundsResponse``).

    Returns:
        BoundsResponse: Lower/upper bounds of the scan, one per window;
            None for a step window's axes (see ``BoundsResponse``).

    """
    scan = request.spec.compile()
    axis_labels = flatten_axes(scan)
    total = scan.number_of_events
    wanted = _wanted_indices(total, request.max_frames)

    lower_columns: dict[str, list[float | None]] = {ax: [] for ax in axis_labels}
    upper_columns: dict[str, list[float | None]] = {ax: [] for ax in axis_labels}
    last: dict[str, float] = {}
    for idx, window in enumerate(scan):
        lower, _, upper = _window_bounds(window, axis_labels, last)
        last = upper
        if idx in wanted:
            has_bounds = bool(window.moving_axes)
            for ax in axis_labels:
                lower_columns[ax].append(lower[ax] if has_bounds else None)
                upper_columns[ax].append(upper[ax] if has_bounds else None)

    return BoundsResponse(
        total_frames=total,
        returned_frames=len(wanted),
        lower=lower_columns,
        upper=upper_columns,
    )


@app.post("/gap", response_model=GapResponse)
def gap(spec: SpecBody) -> GapResponse:
    """Generate turnaround (gap) flags from a scanspec.

    v2 has no precomputed gap flag (see ``Window``'s docstring) -- a
    turnaround is detected by comparing each window's start position
    against the previous window's end position.

    Args:
        spec: The scanspec to walk.

    Returns:
        GapResponse: One boolean per window (see ``GapResponse``).

    """
    scan = spec.compile()
    axis_labels = flatten_axes(scan)

    flags: list[bool] = []
    last: dict[str, float] = {}
    first = True
    for window in scan:
        lower, _, upper = _window_bounds(window, axis_labels, last)
        flags.append(False if first else is_turnaround(last, lower, axis_labels))
        last = upper
        first = False

    return GapResponse(gap=flags)


@app.post("/smalleststep", response_model=SmallestStepResponse)
def smallest_step(spec: SpecBody) -> SmallestStepResponse:
    """Calculate the smallest step between window midpoints, absolute and per-axis.

    Ignores any steps of size 0.

    Args:
        spec: The spec of the scan.

    Returns:
        SmallestStepResponse: A description of the smallest steps in the spec.

    """
    scan = spec.compile()
    axis_labels = flatten_axes(scan)

    mid_columns: dict[str, list[float]] = {ax: [] for ax in axis_labels}
    last: dict[str, float] = {}
    for window in scan:
        _, mid, upper = _window_bounds(window, axis_labels, last)
        last = upper
        for ax in axis_labels:
            mid_columns[ax].append(mid[ax])

    arrays = [np.array(mid_columns[ax]) for ax in axis_labels]
    absolute = _calc_smallest_step(arrays)
    per_axis = {
        ax: _calc_smallest_step([np.array(mid_columns[ax])]) for ax in axis_labels
    }

    return SmallestStepResponse(absolute=absolute, per_axis=per_axis)


@app.post("/triggers", response_model=TriggersResponse)
def triggers(spec: SpecBody) -> TriggersResponse:
    """Generate per-stream metadata and every resolved trigger instant.

    v2's differentiator over 1.x: multi-stream/multi-rate detector
    triggering, the same data ``plot_timeline`` visualises (see
    ``plot.py``), exposed here as JSON.

    Args:
        spec: The spec to walk.

    Returns:
        TriggersResponse: Stream metadata plus a flat, time-ordered list of
            trigger instants (see ``TriggersResponse``).

    """
    scan = spec.compile()
    detector_to_stream = detector_stream_map(scan)

    streams = [
        StreamInfo(
            name=s.name,
            kind="windowed",
            detector_groups=[_detector_group_info(dg) for dg in s.detector_groups],
        )
        for s in scan.windowed_streams
    ] + [
        StreamInfo(
            name=s.name,
            kind="continuous",
            detector_groups=[_detector_group_info(dg) for dg in s.detector_groups],
        )
        for s in scan.continuous_streams
    ]

    events: list[TriggerEvent] = []
    elapsed = 0.0
    for window in scan:
        for ts in window.trigger_sequences:
            parent_stream = _stream_for(ts.detectors, detector_to_stream)
            parent_times, _ = repeat_times([ts.trigger_repeat], elapsed)
            events += [
                TriggerEvent(
                    time=t,
                    livetime=lt,
                    detectors=sorted(str(d) for d in ts.detectors),
                    stream=parent_stream,
                    parent=True,
                )
                for t, lt in parent_times
            ]

            if (
                ts.trigger_repeat.livetime is not None
                and ts.trigger_repeat.deadtime is not None
            ):
                parent_period = ts.trigger_repeat.livetime + ts.trigger_repeat.deadtime
                for child in ts.children:
                    child_stream = _stream_for(child.detectors, detector_to_stream)
                    for p in range(ts.trigger_repeat.num):
                        child_times, _ = repeat_times(
                            child.repeats, elapsed + p * parent_period
                        )
                        events += [
                            TriggerEvent(
                                time=t,
                                livetime=lt,
                                detectors=sorted(str(d) for d in child.detectors),
                                stream=child_stream,
                                parent=False,
                            )
                            for t, lt in child_times
                        ]
        elapsed += window.duration

    return TriggersResponse(streams=streams, events=events)


#
# Utility functions
#


def _stream_for(
    detectors: frozenset[str], detector_to_stream: dict[str, str]
) -> str | None:
    """Stream name for one of *detectors* (they all share a stream)."""
    representative = next(iter(detectors), None)
    if representative is None:
        return None
    return detector_to_stream.get(representative)


def _detector_group_info(dg: DetectorGroup[str]) -> DetectorGroupInfo:
    return DetectorGroupInfo(
        detectors=[str(d) for d in dg.detectors],
        exposures_per_event=dg.exposures_per_event,
        livetime=dg.livetime,
        deadtime=dg.deadtime,
    )


def _wanted_indices(total: int, max_frames: int | None) -> set[int]:
    """Evenly-spaced window indices to keep, capped at *max_frames*."""
    if max_frames is None or max_frames >= total:
        return set(range(total))
    count = max(max_frames, 1)
    return {int(i) for i in np.linspace(0, total - 1, count, dtype=np.int64)}


def _window_bounds(
    window: Window[str, str], axis_labels: list[str], last: dict[str, float]
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """(lower, mid, upper) per axis for *window*.

    A step window's axes have lower == mid == upper (its one physical
    position). A fly window's moving axes get real boundary kinematics
    (``AxisMotion.start_position``/``end_position``); an axis untouched by
    this window (not in ``static_axes``/``moving_axes`` -- a Window only
    records *changed* axes, see ``Scan.__iter__``) falls back to *last*'s
    running position for all three.
    """
    lower = dict(last)
    upper = dict(last)
    for ax in axis_labels:
        if ax in window.moving_axes:
            am = window.moving_axes[ax]
            lower[ax] = am.start_position
            upper[ax] = am.end_position
        elif ax in window.static_axes:
            lower[ax] = upper[ax] = window.static_axes[ax]
    mid = {ax: (lower[ax] + upper[ax]) / 2 for ax in axis_labels}
    return lower, mid, upper


def _format_points(
    columns: Mapping[str, list[float]], format: PointsFormat
) -> Mapping[str, Points]:
    """Convert per-axis point lists to a requested format.

    Args:
        columns: The points to convert, per axis.
        format: The target format.

    Returns:
        Mapping[str, Points]: A mapping of axis to formatted points.

    """
    if format is PointsFormat.FLOAT_LIST:
        return dict(columns)
    arrays = {ax: np.array(pts) for ax, pts in columns.items()}
    if format is PointsFormat.STRING:
        return {ax: str(arr) for ax, arr in arrays.items()}
    return {ax: _base64_encode(arr) for ax, arr in arrays.items()}


def _base64_encode(array: npt.NDArray[np.float64]) -> str:
    return base64.b64encode(array.tobytes()).decode()


def _calc_smallest_step(points: list[npt.NDArray[np.float64]]) -> float:
    # Calc abs diffs of all axes, ignoring any zero values
    absolute_diffs = [np.absolute(axis[1:] - axis[:-1]) for axis in points]
    if not absolute_diffs or absolute_diffs[0].size == 0:
        return 0.0
    # Normalize and remove zeros
    norm_diffs = np.linalg.norm(absolute_diffs, axis=0)
    norm_diffs = norm_diffs[norm_diffs > 0.0]
    if norm_diffs.size == 0:
        return 0.0
    # Return the smallest value (aka. smallest step)
    return float(np.amin(norm_diffs))


def run_app(cors: bool = False, port: int = 8080) -> None:
    """Run an application providing the scanspec.v2 service."""
    if cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    import uvicorn

    uvicorn.run(app=app, port=port)


def scanspec_schema_text() -> str:
    """Generate the OpenAPI schema for the service as a string.

    Returns:
        str: The OpenAPI schema

    """
    return json.dumps(
        get_openapi(
            title=app.title,
            version=app.version,
            openapi_version=app.openapi_version,
            description=app.description,
            routes=app.routes,
        ),
        indent=4,
    )
