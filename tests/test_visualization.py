from pathlib import Path

import cv2
import numpy as np
import pytest

from referai.visualization import read_mot_boxes, visualize_mot_sequences


def make_visualization_sequence(root: Path, split: str, name: str) -> Path:
    sequence = root / split / name
    (sequence / "img1").mkdir(parents=True)
    (sequence / "gt").mkdir()
    for frame_id in (1, 2):
        image = np.full((48, 64, 3), 220, dtype=np.uint8)
        assert cv2.imwrite(str(sequence / "img1" / "{:06d}.jpg".format(frame_id)), image)
    (sequence / "seqinfo.ini").write_text(
        (
            "[Sequence]\nname={}\nimDir=img1\nframeRate=12\nseqLength=2\n"
            "imWidth=64\nimHeight=48\nimExt=.jpg\n"
        ).format(name),
        encoding="utf-8",
    )
    (sequence / "gt" / "gt.txt").write_text(
        "1,7,5,6,20,25,1,1,1\n"
        "2,7,7,6,20,25,1,1,1\n"
        "2,9,40,10,12,20,0,1,1\n",
        encoding="utf-8",
    )
    return sequence


def test_visualize_mot_creates_video_with_track_ids(tmp_path):
    source = tmp_path / "dataset"
    make_visualization_sequence(source, "val", "football-1")

    results = visualize_mot_sequences(
        source,
        tmp_path / "videos",
        num_sequences=1,
        split="val",
        show_boxes=True,
    )

    assert len(results) == 1
    assert results[0].sequence == "football-1"
    assert results[0].frames == 2
    assert results[0].boxes == 2
    assert results[0].fps == 12
    video = Path(results[0].output)
    assert video.is_file() and video.stat().st_size > 0
    capture = cv2.VideoCapture(str(video))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
    capture.release()


def test_visualize_without_boxes_and_explicit_sequence(tmp_path):
    source = tmp_path / "dataset"
    make_visualization_sequence(source, "train", "football-a")
    make_visualization_sequence(source, "train", "football-b")

    results = visualize_mot_sequences(
        source,
        tmp_path / "videos",
        sequence_names=["football-b"],
        show_boxes=False,
        max_frames=1,
    )

    assert [result.sequence for result in results] == ["football-b"]
    assert results[0].boxes == 0
    assert results[0].frames == 1
    assert results[0].output.endswith("football-b_raw.mp4")


def test_mot_reader_ignores_non_valid_ground_truth(tmp_path):
    sequence = make_visualization_sequence(tmp_path, "train", "football-a")
    boxes = read_mot_boxes(sequence)
    assert [box[0] for box in boxes[1]] == [7]
    assert [box[0] for box in boxes[2]] == [7]


def test_visualize_rejects_too_many_sequences(tmp_path):
    source = tmp_path / "dataset"
    make_visualization_sequence(source, "train", "football-a")
    with pytest.raises(ValueError, match="seulement 1"):
        visualize_mot_sequences(source, tmp_path / "videos", num_sequences=2)
