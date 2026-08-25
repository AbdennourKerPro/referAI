import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from referai.hardware import RuntimeProfile
from referai.role_classification import (
    TemporalRoleSmoother,
    _crop_path,
    _disable_optimizer_foreach,
    balanced_class_sample_weights,
    classification_metrics,
    macro_f1_from_indices,
    mean_probabilities,
    predict_role_tracks,
    train_role_classifier,
    validate_role_classifier,
)
from referai.role_postprocessing import (
    fit_spatial_prior,
    fuse_with_spatial_prior,
    normalized_image_position,
    normalized_pitch_position,
    observation_quality,
    weighted_mean_probabilities,
)
from referai.soccernet import (
    _safe_extract_zip,
    download_soccernet_gamestate,
    prepare_gamestate_roles,
)


def make_clip(root: Path, split: str = "train", version: str = "1.3") -> Path:
    clip = root / split / "SNGS-001"
    image_root = clip / "img1"
    image_root.mkdir(parents=True)
    for index in (1, 2):
        Image.new("RGB", (100, 80), color=(20 * index, 100, 30)).save(
            image_root / "{:06d}.jpg".format(index)
        )
    images = [
        {
            "image_id": str(index),
            "file_name": "{:06d}.jpg".format(index),
            "width": 100,
            "height": 80,
            "is_labeled": True,
        }
        for index in (1, 2)
    ]
    annotations = []
    annotation_id = 1
    roles = (("player", 1), ("goalkeeper", 2), ("referee", 3), ("other", 4))
    for role, track_id in roles:
        for image_id in ("1", "2"):
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "track_id": track_id,
                    "supercategory": "object",
                    "bbox_image": {"x_center": 50, "y_center": 40, "w": 20, "h": 40},
                    "bbox_pitch": {"x_bottom_middle": 1.0, "y_bottom_middle": 2.0},
                    "attributes": {
                        "role": role,
                        "team": "left" if role in {"player", "goalkeeper"} else None,
                        "jersey": "9" if role == "player" else None,
                    },
                }
            )
            annotation_id += 1
    annotations.append(
        {
            "id": annotation_id,
            "image_id": "1",
            "track_id": 5,
            "supercategory": "object",
            "bbox_image": {"x_center": 70, "y_center": 60, "w": 8, "h": 8},
            "bbox_pitch": {"x_bottom_middle": 3.0, "y_bottom_middle": 4.0},
            "attributes": {"role": "ball", "team": None, "jersey": None},
        }
    )
    payload = {
        "info": {
            "version": version,
            "name": "SNGS-001",
            "im_dir": "img1",
            "seq_length": 2,
        },
        "images": images,
        "annotations": annotations,
        "categories": [],
    }
    labels = clip / "Labels-GameState.json"
    labels.write_text(json.dumps(payload), encoding="utf-8")
    return labels


def test_prepare_gamestate_roles_creates_crops_and_manifests(tmp_path):
    source = tmp_path / "SoccerNetGS"
    make_clip(source, "train")
    make_clip(source, "valid")
    output = tmp_path / "roles"

    stats = prepare_gamestate_roles(
        source,
        output,
        splits=("train", "valid"),
        frame_stride=1,
        max_samples_per_track=1,
    )

    assert stats.clips == 2
    assert stats.crops == 6
    assert stats.ball_annotations == 2
    assert stats.roles == {
        "player": 2,
        "goalkeeper": 2,
        "referee": 2,
    }
    assert len(list((output / "train" / "player").glob("*.jpg"))) == 1
    assert len(list((output / "val" / "player").glob("*.jpg"))) == 1
    manifest = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert {row["split"] for row in manifest} == {"train", "val"}
    assert all("bbox_pitch" in row and "track_id" in row for row in manifest)
    assert all(row["image_width"] == 100 and row["image_height"] == 80 for row in manifest)
    assert all("\\" not in row["crop_path"] for row in manifest)
    assert all("\\" not in row["source_image"] for row in manifest)
    assert {row["role"] for row in manifest} == {"player", "goalkeeper", "referee"}
    assert all(row["crop_ltrb"] == [28, 8, 72, 72] for row in manifest)
    assert _crop_path(output, "train\\player\\crop.jpg") == output / "train/player/crop.jpg"
    dataset_yaml = (output / "dataset.yaml").read_text(encoding="utf-8")
    assert "path: ." in dataset_yaml
    assert "context: 0.3" in dataset_yaml
    assert "- other" not in dataset_yaml
    balls = (output / "ball_annotations.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(balls) == 2


def test_prepare_gamestate_roles_rejects_old_annotations(tmp_path):
    source = tmp_path / "SoccerNetGS"
    make_clip(source, version="1.2")
    with pytest.raises(ValueError, match="trop ancien"):
        prepare_gamestate_roles(source, tmp_path / "roles", splits=("train",))


def test_prepare_gamestate_roles_rejects_other_role(tmp_path):
    source = tmp_path / "SoccerNetGS"
    make_clip(source)
    with pytest.raises(ValueError, match="Roles du classifieur invalides"):
        prepare_gamestate_roles(source, tmp_path / "roles", splits=("train",), roles=("other",))


def test_safe_extract_zip_rejects_path_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("../escape.txt", "no")
    with pytest.raises(ValueError, match="Archive dangereuse"):
        _safe_extract_zip(archive, tmp_path / "output")


def test_download_uses_official_sdk_layout_and_extracts(tmp_path):
    class FakeDownloader:
        def __init__(self, LocalDirectory):
            self.root = Path(LocalDirectory)

        def downloadDataTask(self, task, split):
            archive = self.root / task / "train.zip"
            archive.parent.mkdir(parents=True)
            source = tmp_path / "source"
            labels = make_clip(source)
            image = labels.parent / "img1" / "000001.jpg"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.write(labels, "SNGS-001/Labels-GameState.json")
                zipped.write(image, "SNGS-001/img1/000001.jpg")

    results = download_soccernet_gamestate(
        tmp_path / "download",
        splits=("train",),
        downloader_class=FakeDownloader,
    )
    assert results[0].labels == 1
    assert results[0].image_directories == 1
    assert (tmp_path / "download" / "train" / "SNGS-001" / "Labels-GameState.json").is_file()


def test_temporal_smoothing_and_metrics():
    smoother = TemporalRoleSmoother(alpha=0.25)
    assert smoother.update("clip:1", [0.8, 0.2]) == pytest.approx([0.8, 0.2])
    assert smoother.update("clip:1", [0.0, 1.0]) == pytest.approx([0.6, 0.4])
    metrics = classification_metrics(
        ["player", "player", "referee"],
        ["player", "referee", "referee"],
        ["player", "referee"],
    )
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["per_class"]["player"]["recall"] == pytest.approx(0.5)


def test_balanced_sampling_weights_and_macro_f1():
    weights = balanced_class_sample_weights([0, 0, 0, 0, 1, 1, 2])
    assert weights == pytest.approx([0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 1.0])
    assert macro_f1_from_indices([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0], 3) == pytest.approx(
        (0.5 + 0.8 + 2 / 3) / 3
    )


def test_track_probability_mean_preserves_all_observations():
    assert mean_probabilities([1.4, 0.6], 2) == pytest.approx([0.7, 0.3])
    with pytest.raises(ValueError, match="count"):
        mean_probabilities([1.0, 0.0], 0)


def test_quality_weighted_aggregation_reduces_a_weak_outlier():
    aggregate = weighted_mean_probabilities(
        [[0.1, 0.9], [0.9, 0.1]],
        [0.1, 0.9],
    )
    assert aggregate == pytest.approx([0.82, 0.18])


def test_image_and_pitch_positions_are_normalized():
    image_position = normalized_image_position(
        {"x_center": 50, "y_center": 40, "w": 20, "h": 40}, 100, 80
    )
    assert image_position is not None
    assert image_position["x_bottom"] == pytest.approx(0.5)
    assert image_position["y_bottom"] == pytest.approx(0.75)
    assert image_position["box_area"] == pytest.approx(0.1)

    pitch_position = normalized_pitch_position(
        {"x_bottom_middle": 50.0, "y_bottom_middle": 10.0}
    )
    assert pitch_position is not None
    assert pitch_position["goal_line_distance_m"] == pytest.approx(2.5)
    assert pitch_position["goal_line_distance_normalized"] == pytest.approx(2.5 / 52.5)


def test_spatial_prior_learns_goalkeeper_near_goal_without_class_imbalance():
    samples = [
        ("goalkeeper", (0.02, 0.2)),
        ("goalkeeper", (0.04, 0.2)),
        ("player", (0.75, 0.2)),
        ("player", (0.80, 0.2)),
        ("player", (0.85, 0.2)),
        ("player", (0.90, 0.2)),
    ]
    prior = fit_spatial_prior(
        samples,
        ("player", "goalkeeper"),
        "pitch_oracle",
        bins_x=4,
        bins_y=2,
        smoothing=0.5,
    )
    near_goal = prior.probabilities((0.03, 0.2))
    midfield = prior.probabilities((0.80, 0.2))
    assert near_goal[1] > near_goal[0]
    assert midfield[0] > midfield[1]


def test_quality_scores_sharp_crop_above_flat_crop():
    flat = Image.new("RGB", (64, 64), color=(128, 128, 128))
    checker = Image.new("RGB", (64, 64))
    checker.putdata(
        [(255, 255, 255) if (index + index // 64) % 2 else (0, 0, 0) for index in range(4096)]
    )
    flat_quality = observation_quality(flat, [0.6, 0.3, 0.1])
    sharp_quality = observation_quality(checker, [0.6, 0.3, 0.1])
    assert sharp_quality["sharpness"] > flat_quality["sharpness"]
    assert sharp_quality["weight"] > flat_quality["weight"]


def test_spatial_fusion_increases_role_supported_by_prior():
    fused = fuse_with_spatial_prior([0.55, 0.45], [0.1, 0.9], strength=0.75, quality=0.2)
    assert fused[1] > fused[0]


def test_xpu_optimizer_disables_foreach_for_every_parameter_group():
    optimizer = SimpleNamespace(
        defaults={"foreach": None},
        param_groups=[{"foreach": None}, {}],
    )
    assert _disable_optimizer_foreach(optimizer) is optimizer
    assert optimizer.defaults["foreach"] is False
    assert all(group["foreach"] is False for group in optimizer.param_groups)


def test_train_role_classifier_builds_ultralytics_classification_args(tmp_path, monkeypatch):
    dataset = tmp_path / "roles"
    (dataset / "train" / "player").mkdir(parents=True)
    (dataset / "val" / "player").mkdir(parents=True)
    config = tmp_path / "train_roles.yaml"
    config.write_text(
        "model: yolo11n-cls.pt\n"
        "data: {}\n"
        "train:\n"
        "  epochs: 2\n"
        "  imgsz: 96\n"
        "  project: {}\n"
        "  name: roles\n".format(dataset.as_posix(), (tmp_path / "runs").as_posix()),
        encoding="utf-8",
    )
    captured = {}
    model_paths = []

    class FakeModel:
        def train(self, **kwargs):
            captured.update(kwargs)
            return "trained"

    def make_model(path):
        model_paths.append(path)
        return FakeModel()

    monkeypatch.setattr(
        "referai.role_classification._import_yolo", lambda backend="cpu": make_model
    )
    monkeypatch.setattr(
        "referai.role_classification._role_classification_trainer", lambda backend: "macro_f1_trainer"
    )
    monkeypatch.setattr(
        "referai.role_classification.resolve_profile",
        lambda _: RuntimeProfile(
            name="cpu",
            backend="cpu",
            device_ids=(),
            imgsz=224,
            batch=1,
            amp=False,
            half=False,
            workers=0,
            max_oom_retries=0,
            min_imgsz=224,
        ),
    )
    monkeypatch.setattr(
        "referai.role_classification.inspect_gpus", lambda backend="auto": []
    )
    result = train_role_classifier(config)
    assert result == "trained"
    assert captured["task"] == "classify"
    assert captured["data"] == str(dataset.resolve())
    assert captured["epochs"] == 2
    assert captured["trainer"] == "macro_f1_trainer"
    checkpoint = tmp_path / "runs" / "roles" / "weights" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    train_role_classifier(config, resume=True)
    assert Path(model_paths[-1]) == checkpoint.resolve()
    assert captured["resume"] is True


def test_validate_role_classifier_uses_macro_f1_validator(tmp_path, monkeypatch):
    captured = {}

    class FakeModel:
        def val(self, validator=None, **kwargs):
            captured["validator"] = validator
            captured.update(kwargs)
            return "validated"

    monkeypatch.setattr(
        "referai.role_classification._import_yolo", lambda backend="cpu": lambda _: FakeModel()
    )
    monkeypatch.setattr(
        "referai.role_classification._role_classification_trainer",
        lambda backend: SimpleNamespace(validator_class="macro_f1_validator"),
    )
    monkeypatch.setattr(
        "referai.role_classification.resolve_profile",
        lambda _: RuntimeProfile(
            name="cpu",
            backend="cpu",
            device_ids=(),
            imgsz=224,
            batch=1,
            amp=False,
            half=False,
            workers=0,
            max_oom_retries=0,
            min_imgsz=224,
        ),
    )

    result = validate_role_classifier(tmp_path / "best.pt", tmp_path / "roles")

    assert result == "validated"
    assert captured["validator"] == "macro_f1_validator"
    assert captured["split"] == "val"
    assert "half" not in captured


def test_predict_role_tracks_exports_four_auditable_variants(tmp_path, monkeypatch):
    source = tmp_path / "SoccerNetGS"
    make_clip(source, "train")
    make_clip(source, "valid")
    dataset = tmp_path / "roles"
    prepare_gamestate_roles(
        source,
        dataset,
        splits=("train", "valid"),
        frame_stride=1,
        max_samples_per_track=1,
    )

    class FakeTensor:
        def __init__(self, values):
            self.values = values

        def cpu(self):
            return self

        def tolist(self):
            return self.values

    class FakeModel:
        names = {0: "goalkeeper", 1: "player", 2: "referee"}

        def predict(self, source, **kwargs):
            results = []
            for path in source:
                role = Path(path).parent.name
                probabilities = {
                    "goalkeeper": [0.8, 0.1, 0.1],
                    "player": [0.1, 0.8, 0.1],
                    "referee": [0.1, 0.1, 0.8],
                }[role]
                with Image.open(path) as crop:
                    image = crop.convert("RGB").copy()
                results.append(
                    SimpleNamespace(
                        probs=SimpleNamespace(data=FakeTensor(probabilities)),
                        orig_img=image,
                    )
                )
            return results

    monkeypatch.setattr(
        "referai.role_classification._import_yolo", lambda backend="cpu": lambda _: FakeModel()
    )
    monkeypatch.setattr(
        "referai.role_classification.resolve_profile",
        lambda _: RuntimeProfile(
            name="cpu",
            backend="cpu",
            device_ids=(),
            imgsz=224,
            batch=3,
            amp=False,
            half=False,
            workers=0,
            max_oom_retries=0,
            min_imgsz=224,
        ),
    )

    output = tmp_path / "evaluation"
    summary = predict_role_tracks(
        tmp_path / "best.pt",
        dataset,
        output,
        split="val",
        batch=3,
    )

    assert summary["track_baseline"]["macro_f1"] == pytest.approx(1.0)
    assert summary["track_quality_weighted"]["macro_f1"] == pytest.approx(1.0)
    assert summary["track_image_prior"]["macro_f1"] == pytest.approx(1.0)
    assert summary["track_pitch_oracle"]["macro_f1"] == pytest.approx(1.0)
    assert summary["postprocessing"]["prior_fit_split"] == "train"
    predictions = [
        json.loads(line)
        for line in (output / "role_predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(predictions) == 3
    assert all("quality" in row and "aggregation_contributions" in row for row in predictions)
    assert all(row["image_position"]["available"] for row in predictions)
    tracks = [
        json.loads(line)
        for line in (output / "track_role_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(
        set(row["aggregations"])
        == {"baseline", "quality_weighted", "image_prior", "pitch_oracle"}
        for row in tracks
    )
