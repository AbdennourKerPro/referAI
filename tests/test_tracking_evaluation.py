import json
from pathlib import Path

import pytest

from referai.tracking_evaluation import (
    build_trackeval_command,
    discover_evaluation_sequences,
    evaluate_tracking,
    parse_trackeval_summary,
)


def make_evaluation_dataset(tmp_path: Path):
    root = tmp_path / "dataset"
    sequence = "match-001"
    images = root / "images" / "val" / sequence
    gt = root / "mot_gt" / "val" / sequence / "gt"
    seqinfo = root / "mot_gt" / "val" / sequence / "seqinfo.ini"
    seqmap = root / "mot_gt" / "seqmaps" / "val.txt"
    images.mkdir(parents=True)
    gt.mkdir(parents=True)
    seqmap.parent.mkdir(parents=True)
    (images / "000001.jpg").write_bytes(b"image")
    (gt / "gt.txt").write_text("1,7,10,20,30,40,1,1,1\n", encoding="utf-8")
    seqinfo.write_text("[Sequence]\nseqLength=1\n", encoding="utf-8")
    seqmap.write_text("name\n{}\n".format(sequence), encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "path: {}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: player\n".format(root.as_posix()),
        encoding="utf-8",
    )
    return data_yaml, sequence


def make_trackeval_root(tmp_path: Path) -> Path:
    root = tmp_path / "TrackEval"
    (root / "scripts").mkdir(parents=True)
    (root / "trackeval").mkdir()
    (root / "scripts" / "run_mot_challenge.py").write_text("", encoding="utf-8")
    return root


def test_discover_evaluation_sequences_requires_ground_truth(tmp_path):
    data_yaml, sequence = make_evaluation_dataset(tmp_path)
    _, seqmap, sequences = discover_evaluation_sequences(data_yaml, "val")
    assert seqmap.name == "val.txt"
    assert [item.name for item in sequences] == [sequence]
    assert sequences[0].images[0].name == "000001.jpg"

    sequences[0].ground_truth.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Verite terrain MOT absente"):
        discover_evaluation_sequences(data_yaml, "val")


def test_build_trackeval_command_uses_custom_mot_layout(tmp_path):
    root = make_trackeval_root(tmp_path)
    command = build_trackeval_command(
        root,
        tmp_path / "gt",
        tmp_path / "trackers",
        "referai",
        "val",
        ["match-001", "match-002"],
    )
    assert command[1:3] == ["-m", "referai.trackeval_runner"]
    assert command[3].endswith("run_mot_challenge.py")
    assert command[command.index("--SKIP_SPLIT_FOL") + 1] == "True"
    assert command[command.index("--DO_PREPROC") + 1] == "False"
    assert command[command.index("--OUTPUT_SUB_FOLDER") + 1] == "trackeval"
    assert command[command.index("--SEQ_INFO") + 1 : command.index("--SKIP_SPLIT_FOL")] == [
        "match-001",
        "match-002",
    ]
    assert command[command.index("--METRICS") + 1 : command.index("--USE_PARALLEL")] == [
        "HOTA",
        "CLEAR",
        "Identity",
    ]


def test_parse_trackeval_summary(tmp_path):
    summary = tmp_path / "referai" / "pedestrian_summary.txt"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        "HOTA DetA AssA MOTA IDF1 IDSW Frag\n"
        "61.5 70.0 54.2 82.1 76.4 12 20\n",
        encoding="utf-8",
    )
    path, metrics = parse_trackeval_summary(tmp_path, "referai")
    assert path == summary
    assert metrics["HOTA"] == pytest.approx(61.5)
    assert metrics["IDSW"] == pytest.approx(12)


def test_evaluate_tracking_can_reuse_predictions(tmp_path, monkeypatch):
    data_yaml, sequence = make_evaluation_dataset(tmp_path)
    trackeval_root = make_trackeval_root(tmp_path)
    output = tmp_path / "evaluation"
    prediction = output / "predictions" / "referai" / "data" / "{}.txt".format(sequence)
    prediction.parent.mkdir(parents=True)
    prediction.write_text("1,1,10,20,30,40,0.9,-1,-1,-1\n", encoding="utf-8")

    def fake_run(command, check):
        assert check is True
        result_root = Path(command[command.index("--TRACKERS_FOLDER") + 1])
        summary = result_root / "referai" / "trackeval" / "pedestrian_summary.txt"
        summary.parent.mkdir(parents=True)
        summary.write_text(
            "HOTA DetA AssA MOTA MOTP IDF1 IDP IDR IDSW Frag CLR_Re CLR_Pr\n"
            "61 70 54 82 79 76 78 74 12 20 91 92\n",
            encoding="utf-8",
        )

    monkeypatch.setattr("referai.tracking_evaluation.subprocess.run", fake_run)
    result = evaluate_tracking(
        data_yaml=data_yaml,
        trackeval_root=trackeval_root,
        output=output,
        skip_inference=True,
    )
    assert result["metrics"]["HOTA"] == pytest.approx(61)
    assert result["metrics"]["IDF1"] == pytest.approx(76)
    assert result["inference"] is None
    saved = json.loads((output / "tracking_metrics.json").read_text(encoding="utf-8"))
    assert saved["sequences"] == [sequence]
