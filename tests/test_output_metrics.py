import json

import pytest

from referai.metrics import composite_detection_score, trajectory_diagnostics
from referai.output import JSONWriter
from referai.schemas import FrameObservations, TrackedObject


def frame(frame_id, track_id=7):
    return FrameObservations(
        frame_id=frame_id,
        timestamp=frame_id / 25,
        objects=[TrackedObject(track_id, 0, "player", 0.9, (1, 2, 3, 4))],
    )


def test_json_writer_produces_valid_incremental_array(tmp_path):
    path = tmp_path / "observations.json"
    with JSONWriter(path) as writer:
        writer.write(frame(0))
        writer.write(frame(1))
    payload = json.loads(path.read_text())
    assert payload[0]["objects"][0]["class"] == "player"
    assert payload[1]["frame_id"] == 1


def test_composite_detection_score():
    score = composite_detection_score(
        {"player": 1.0, "referee": 0.5, "goalkeeper": 0.5, "ball": 0.25}
    )
    assert score == pytest.approx(0.60)


def test_trajectory_gap_diagnostics():
    diagnostics = trajectory_diagnostics([frame(0), frame(1), frame(4)])
    assert diagnostics.tracks == 1
    assert diagnostics.fragmented_tracks == 1
    assert diagnostics.total_gaps == 2
