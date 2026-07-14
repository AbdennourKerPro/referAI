import json
from pathlib import Path

import pytest

from referai.data import (
    assert_match_disjoint,
    assign_splits,
    create_match_map_template,
    create_oversampled_dataset,
    discover_sequences,
    prepare_mot_dataset,
)


def make_sequence(root: Path, split: str, name: str, class_id: int = 1):
    sequence = root / split / name
    (sequence / "img1").mkdir(parents=True)
    (sequence / "gt").mkdir()
    (sequence / "img1" / "000001.jpg").write_bytes(b"not-decoded-in-test")
    (sequence / "seqinfo.ini").write_text(
        (
            "[Sequence]\nname={}\nimDir=img1\nframeRate=25\nseqLength=1\n"
            "imWidth=100\nimHeight=50\nimExt=.jpg\n"
        ).format(name)
    )
    (sequence / "gt" / "gt.txt").write_text(
        "1,7,10,5,20,10,1,{},1\n".format(class_id)
    )


def test_prepare_mot_writes_yolo_and_manifest(tmp_path):
    source = tmp_path / "mot"
    make_sequence(source, "train", "seq-a")
    make_sequence(source, "val", "seq-b")
    make_sequence(source, "test", "seq-c")
    output = tmp_path / "yolo"
    stats = prepare_mot_dataset(source, output, link_mode="copy")
    assert stats.images == 3
    assert stats.boxes == 3
    label = output / "labels" / "train" / "seq-a" / "000001.txt"
    assert label.read_text().strip() == "0 0.20000000 0.20000000 0.20000000 0.20000000"
    assert (output / "mot_gt" / "test" / "seq-c" / "gt" / "gt.txt").exists()
    assert (output / "mot_gt" / "test" / "seq-c" / "seqinfo.ini").exists()
    assert "seq-c" in (output / "mot_gt" / "seqmaps" / "test.txt").read_text()


def test_match_split_has_no_leakage(tmp_path):
    source = tmp_path / "mot"
    for name in ("clip-a", "clip-b", "clip-c", "clip-d"):
        make_sequence(source, "unknown", name)
    match_map = tmp_path / "matches.json"
    match_map.write_text(
        json.dumps({"clip-a": "m1", "clip-b": "m1", "clip-c": "m2", "clip-d": "m3"})
    )
    sequences = discover_sequences(source, match_map)
    splits = assign_splits(sequences, "by-match", seed=3)
    manifest = [
        {"match_id": sequence.match_id, "split": splits[sequence.name]} for sequence in sequences
    ]
    assert_match_disjoint(manifest)
    assert splits["clip-a"] == splits["clip-b"]


def test_sequence_list_filters_other_sports(tmp_path):
    source = tmp_path / "mot"
    make_sequence(source, "train", "football-1")
    make_sequence(source, "train", "basketball-1")
    sequence_list = tmp_path / "football.txt"
    sequence_list.write_text("football-1\n")
    sequences = discover_sequences(source, sequence_list=sequence_list)
    assert [sequence.name for sequence in sequences] == ["football-1"]


def test_create_match_map_template_and_require_completed_map(tmp_path):
    source = tmp_path / "mot"
    make_sequence(source, "train", "clip-a")
    mapping = tmp_path / "match_map.csv"
    assert create_match_map_template(source, mapping) == 1
    assert "sequence,match_id,original_split,source_path" in mapping.read_text()
    with pytest.raises(ValueError, match="match_id vide"):
        prepare_mot_dataset(
            source,
            tmp_path / "output",
            split_strategy="by-match",
            match_map=mapping,
        )


def test_missing_match_map_has_actionable_error(tmp_path):
    source = tmp_path / "mot"
    make_sequence(source, "train", "clip-a")
    with pytest.raises(FileNotFoundError, match="create-match-map"):
        prepare_mot_dataset(
            source,
            tmp_path / "output",
            split_strategy="by-match",
            match_map=tmp_path / "missing.csv",
        )


def test_ball_oversampling_repeats_rare_images(tmp_path):
    source = tmp_path / "mot"
    make_sequence(source, "train", "ball-seq", class_id=4)
    make_sequence(source, "val", "val-seq", class_id=1)
    make_sequence(source, "test", "test-seq", class_id=1)
    output = tmp_path / "yolo"
    prepare_mot_dataset(source, output, class_map={1: 0, 4: 3}, link_mode="copy")
    generated = create_oversampled_dataset(output / "data.yaml", class_id=3, factor=4)
    entries = (output / "train_class3_x4.txt").read_text().splitlines()
    assert len(entries) == 4
    assert generated.name == "data_oversampled.yaml"
