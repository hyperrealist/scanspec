"""Tests for scanspec.v2.service (the FastAPI REST service)."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from scanspec.v2.core import DetectorGroup
from scanspec.v2.service import PointsFormat, app
from scanspec.v2.specs import Acquire, Linspace, Repeat, Spec, Static


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


def _json(spec: Spec[Any, Any, Any]) -> dict[str, Any]:
    return spec.model_dump(mode="json")


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# VALID SPEC TEST(S) #


def test_valid_spec(client: TestClient):
    spec = Linspace("x", 0, 1, 5)
    response = client.post("/valid", json=_json(spec))
    assert response.status_code == 200
    body = response.json()
    assert body["input_spec"] == _json(spec)
    assert body["valid_spec"] == _json(spec)


def test_valid_invalid_spec(client: TestClient):
    spec = {"type": "Linspace", "axis": "x", "start": 0.0, "num": 10}
    response = client.post("/valid", json=spec)
    assert response.status_code == 422


def test_valid_resolves_the_correct_discriminated_union_member(client: TestClient):
    """A regression check for a real bug hit during development.

    AnySpec[...] used as a bare `= Body(...)` default (rather than
    Annotated[AnySpec[...], Body(...)]) silently lost its Discriminator
    through FastAPI's request-parameter resolution, mis-resolving any
    Acquire/Concat/etc. body to whichever Union member's fields happened to
    structurally match first (Snake, since both have a lone `spec` field).
    """
    det = DetectorGroup(1, 1, 0.01, 0.001, ["eiger"])
    spec: Acquire[str, str, str] = Acquire(
        Linspace("x", 0, 1, 3), detectors=[det], stream_name="diffraction"
    )
    response = client.post("/valid", json=_json(spec))
    assert response.status_code == 200
    assert response.json()["input_spec"]["type"] == "Acquire"


# MIDPOINTS TEST(S) #


@pytest.mark.parametrize(
    "format,expected_midpoints",
    [
        (PointsFormat.FLOAT_LIST, [0.0, 0.25, 0.5, 0.75, 1.0]),
        (PointsFormat.STRING, "[0.   0.25 0.5  0.75 1.  ]"),
        (
            PointsFormat.BASE64_ENCODED,
            "AAAAAAAAAAAAAAAAAADQPwAAAAAAAOA/AAAAAAAA6D8AAAAAAADwPw==",
        ),
    ],
    ids=["float_list", "string", "base64"],
)
def test_midpoints(client: TestClient, format: PointsFormat, expected_midpoints: Any):
    spec = Linspace("x", 0, 1, 5)
    response = client.post(
        "/midpoints", json={"spec": _json(spec), "max_frames": 5, "format": format}
    )
    assert response.status_code == 200
    assert response.json() == {
        "total_frames": 5,
        "returned_frames": 5,
        "format": format.value,
        "midpoints": {"x": expected_midpoints},
    }


def test_midpoints_subsampling(client: TestClient):
    spec = Linspace("x", 0, 10, 5) * Linspace("y", 0, 10, 5)
    response = client.post("/midpoints", json={"spec": _json(spec), "max_frames": 8})
    assert response.status_code == 200
    body = response.json()
    assert body["total_frames"] == 25
    assert body["returned_frames"] == 8
    assert len(body["midpoints"]["x"]) == 8
    assert len(body["midpoints"]["y"]) == 8


def test_midpoints_default_max_frames_returns_everything(client: TestClient):
    spec = Linspace("x", 0, 1, 5)
    response = client.post("/midpoints", json={"spec": _json(spec)})
    assert response.status_code == 200
    assert response.json()["returned_frames"] == 5


# BOUNDS TEST(S) #


def test_bounds_fly_scan_has_real_boundary_positions(client: TestClient):
    spec: Acquire[str, str, str] = Acquire(Linspace("x", 0.0, 1.0, 5), fly=True)
    response = client.post("/bounds", json={"spec": _json(spec)})
    assert response.status_code == 200
    body = response.json()
    assert body["total_frames"] == 1
    assert body["lower"]["x"] == [-0.125]
    assert body["upper"]["x"] == [1.125]


def test_bounds_step_scan_omits_bounds_rather_than_fabricating_them(
    client: TestClient,
):
    spec = Linspace("x", 0.0, 1.0, 5)
    response = client.post("/bounds", json={"spec": _json(spec)})
    assert response.status_code == 200
    body = response.json()
    assert body["total_frames"] == 5
    assert body["lower"]["x"] == [None] * 5
    assert body["upper"]["x"] == [None] * 5


# GAP TEST(S) #


def test_gap_first_window_is_never_a_turnaround(client: TestClient):
    spec = Linspace("x", 0, 1, 5)
    response = client.post("/gap", json=_json(spec))
    assert response.status_code == 200
    gap = response.json()["gap"]
    assert len(gap) == 5
    assert gap[0] is False


def test_gap_within_a_single_fly_window_has_no_turnaround(client: TestClient):
    spec: Acquire[str, str, str] = Acquire(Linspace("x", 0, 1, 5), fly=True)
    response = client.post("/gap", json=_json(spec))
    assert response.status_code == 200
    assert response.json()["gap"] == [False]


# SMALLEST STEP TEST(S) #


def test_smallest_step(client: TestClient):
    spec = Linspace("y", 0.0, 10.0, 3) * Linspace("x", 0.0, 10.0, 5)
    response = client.post("/smalleststep", json=_json(spec))
    assert response.status_code == 200
    assert response.json() == {"absolute": 2.5, "per_axis": {"y": 5.0, "x": 2.5}}


# TRIGGERS TEST(S) #


def test_triggers_single_stream(client: TestClient):
    det = DetectorGroup(1, 1, 0.01, 0.001, ["eiger"])
    spec: Acquire[str, str, str] = Acquire(
        Linspace("x", 0, 1, 3), detectors=[det], stream_name="diffraction"
    )
    response = client.post("/triggers", json=_json(spec))
    assert response.status_code == 200
    body = response.json()
    assert [s["name"] for s in body["streams"]] == ["diffraction"]
    assert body["streams"][0]["detector_groups"] == [
        {
            "detectors": ["eiger"],
            "exposures_per_event": 1,
            "livetime": 0.01,
            "deadtime": 0.001,
        }
    ]
    assert len(body["events"]) == 3
    assert all(e["stream"] == "diffraction" for e in body["events"])
    assert all(e["parent"] is True for e in body["events"])


def test_triggers_flagship_multi_stream(client: TestClient):
    """The flagship pattern: a step stream and a faster fly stream, both

    resolved to their own stream names in the event list.
    """
    response = client.post("/triggers", json=_json(_flagship_multi_stream_spec()))
    assert response.status_code == 200
    body = response.json()
    assert {s["name"] for s in body["streams"]} == {"diff", "spec"}
    # 3 repeats x (1 diff event + 20 fwd + 20 rev spec events) = 123
    assert len(body["events"]) == 3 * (1 + 20 + 20)
    diff_events = [e for e in body["events"] if e["stream"] == "diff"]
    spec_events = [e for e in body["events"] if e["stream"] == "spec"]
    assert len(diff_events) == 3
    assert len(spec_events) == 120
    assert all(e["detectors"] == ["diffraction"] for e in diff_events)
    assert all(e["detectors"] == ["spectroscopy"] for e in spec_events)


def test_triggers_pure_motion_spec_has_no_streams_or_events(client: TestClient):
    response = client.post("/triggers", json=_json(Linspace("x", 0, 1, 5)))
    assert response.status_code == 200
    assert response.json() == {"streams": [], "events": []}
