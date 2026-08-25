"""Download and preparation utilities for SoccerNet Game State Reconstruction."""

import json
import logging
import math
import os
import re
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

LOGGER = logging.getLogger(__name__)

SOCCERNET_SPLITS = ("train", "valid", "test", "challenge")
HUMAN_ROLES = ("player", "goalkeeper", "referee", "other")
ROLE_CLASSIFIER_ROLES = ("player", "goalkeeper", "referee")
KNOWN_ROLES = HUMAN_ROLES + ("ball",)


@dataclass(frozen=True)
class DownloadedSplit:
    split: str
    archive: Optional[str]
    destination: Optional[str]
    labels: int
    image_directories: int


@dataclass
class RolePreparationStats:
    clips: int = 0
    images: int = 0
    annotations: int = 0
    human_annotations: int = 0
    ball_annotations: int = 0
    crops: int = 0
    skipped_annotations: int = 0
    roles: Dict[str, int] = field(default_factory=dict)
    splits: Dict[str, int] = field(default_factory=dict)
    split_roles: Dict[str, Dict[str, int]] = field(default_factory=dict)


def _validate_splits(splits: Sequence[str]) -> Tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(split).lower() for split in splits))
    invalid = sorted(set(normalized) - set(SOCCERNET_SPLITS))
    if invalid:
        raise ValueError("Splits SoccerNet invalides: {}".format(invalid))
    if not normalized:
        raise ValueError("Au moins un split SoccerNet est requis")
    return normalized


def _import_soccernet_downloader() -> Any:
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError as exc:
        raise RuntimeError(
            "Le SDK SoccerNet est absent. Installez requirements_soccernet.txt."
        ) from exc
    return SoccerNetDownloader


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    """Extract a zip file while rejecting path traversal entries."""
    archive = Path(archive).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    destination_text = str(destination)
    with zipfile.ZipFile(str(archive), "r") as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if os.path.commonpath((destination_text, str(target))) != destination_text:
                raise ValueError("Archive dangereuse, chemin hors destination: {}".format(member.filename))
        zipped.extractall(str(destination))


def _inspect_downloaded_split(root: Path, split: str) -> Tuple[int, int]:
    destination = root / split
    if not destination.is_dir():
        return 0, 0
    labels = sum(1 for _ in destination.rglob("Labels-GameState.json"))
    image_directories = sum(1 for path in destination.rglob("img1") if path.is_dir())
    return labels, image_directories


def download_soccernet_gamestate(
    output: Path,
    task: str = "gamestate-2024",
    splits: Sequence[str] = ("train", "valid", "test"),
    extract: bool = True,
    keep_archives: bool = True,
    password: Optional[str] = None,
    downloader_class: Optional[Any] = None,
) -> List[DownloadedSplit]:
    """Download official GameState archives with the SoccerNet SDK and optionally extract them."""
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    splits = _validate_splits(splits)
    Downloader = downloader_class or _import_soccernet_downloader()
    downloader = Downloader(LocalDirectory=str(output))
    if password:
        downloader.password = password
    LOGGER.info("Telechargement SoccerNet task=%s splits=%s vers %s", task, splits, output)
    downloader.downloadDataTask(task=task, split=list(splits))

    results = []
    for split in splits:
        archive = output / task / "{}.zip".format(split)
        destination = output / split
        if extract:
            if archive.is_file():
                LOGGER.info("Extraction de %s vers %s", archive, destination)
                _safe_extract_zip(archive, destination)
            elif not destination.is_dir():
                raise FileNotFoundError(
                    "Archive SoccerNet introuvable apres telechargement: {}".format(archive)
                )
        labels, image_directories = _inspect_downloaded_split(output, split)
        if extract and split in {"train", "valid"} and not labels:
            raise ValueError(
                "Aucun Labels-GameState.json trouve dans {}. Verifiez le telechargement."
                .format(destination)
            )
        results.append(
            DownloadedSplit(
                split=split,
                archive=str(archive) if archive.is_file() else None,
                destination=str(destination) if destination.is_dir() else None,
                labels=labels,
                image_directories=image_directories,
            )
        )
        if extract and archive.is_file() and not keep_archives:
            archive.unlink()
    (output / "download_manifest.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8"
    )
    return results


def _version_tuple(value: Any) -> Tuple[int, ...]:
    numbers = re.findall(r"\d+", str(value))
    return tuple(int(number) for number in numbers) or (0,)


def _safe_token(value: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return token or "unknown"


def _output_split(source_split: str) -> str:
    return "val" if source_split == "valid" else source_split


def _image_identifier(image: Mapping[str, Any]) -> str:
    identifier = image.get("image_id", image.get("id"))
    if identifier is None:
        raise ValueError("Image SoccerNet sans image_id")
    return str(identifier)


def _bbox_to_ltrb(
    bbox: Mapping[str, Any],
    image_width: int,
    image_height: int,
    context: float,
) -> Optional[Tuple[int, int, int, int]]:
    try:
        width = float(bbox["w"])
        height = float(bbox["h"])
        if "x_center" in bbox and "y_center" in bbox:
            left = float(bbox["x_center"]) - width / 2.0
            top = float(bbox["y_center"]) - height / 2.0
        else:
            left = float(bbox["x"])
            top = float(bbox["y"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    padding = max(width, height) * context
    x1 = max(0, int(math.floor(left - padding)))
    y1 = max(0, int(math.floor(top - padding)))
    x2 = min(image_width, int(math.ceil(left + width + padding)))
    y2 = min(image_height, int(math.ceil(top + height + padding)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _find_image(clip_root: Path, image_dir: str, file_name: str) -> Path:
    candidates = (
        clip_root / image_dir / file_name,
        clip_root / file_name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _write_json_line(stream: Any, payload: Mapping[str, Any]) -> None:
    stream.write(json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":")) + "\n")


def _annotation_record(
    source: Path,
    source_split: str,
    output_split: str,
    sequence: str,
    image_path: Path,
    image_id: str,
    frame_index: int,
    image: Mapping[str, Any],
    annotation: Mapping[str, Any],
) -> Dict[str, Any]:
    attributes = annotation.get("attributes") or {}
    return {
        "split": output_split,
        "source_split": source_split,
        "sequence": sequence,
        "image_id": image_id,
        "frame_index": frame_index,
        "track_id": annotation.get("track_id"),
        "role": str(attributes.get("role", "")).lower(),
        "team": attributes.get("team"),
        "jersey": attributes.get("jersey"),
        "bbox_image": annotation.get("bbox_image"),
        "bbox_pitch": annotation.get("bbox_pitch"),
        "image_width": image.get("width"),
        "image_height": image.get("height"),
        "source_image": image_path.relative_to(source).as_posix(),
    }


def _load_game_state(path: Path, minimum_version: str) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Annotation GameState invalide: {}".format(path))
    version = (payload.get("info") or {}).get("version")
    if _version_tuple(version) < _version_tuple(minimum_version):
        raise ValueError(
            "Dataset GameState trop ancien dans {}: version {}, minimum {}"
            .format(path, version, minimum_version)
        )
    return payload


def prepare_gamestate_roles(
    source: Path,
    output: Path,
    splits: Sequence[str] = ("train", "valid", "test"),
    roles: Sequence[str] = ROLE_CLASSIFIER_ROLES,
    frame_stride: int = 5,
    max_samples_per_track: int = 40,
    context: float = 0.30,
    min_crop_size: int = 12,
    jpeg_quality: int = 95,
    minimum_version: str = "1.3",
    max_clips: Optional[int] = None,
) -> RolePreparationStats:
    """Create an Ultralytics classification dataset and rich manifests from GameState JSON."""
    if frame_stride < 1:
        raise ValueError("--frame-stride doit etre >= 1")
    if max_samples_per_track < 1:
        raise ValueError("--max-samples-per-track doit etre >= 1")
    if context < 0:
        raise ValueError("--context doit etre >= 0")
    if min_crop_size < 1:
        raise ValueError("--min-crop-size doit etre >= 1")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("--jpeg-quality doit etre entre 1 et 100")
    if max_clips is not None and max_clips < 1:
        raise ValueError("--max-clips doit etre >= 1")

    splits = _validate_splits(splits)
    included_roles = tuple(dict.fromkeys(str(role).lower() for role in roles))
    invalid_roles = sorted(set(included_roles) - set(ROLE_CLASSIFIER_ROLES))
    if invalid_roles:
        raise ValueError("Roles du classifieur invalides: {}".format(invalid_roles))
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError("Dataset SoccerNet GameState introuvable: {}".format(source))
    output.mkdir(parents=True, exist_ok=True)
    for split in (_output_split(split) for split in splits):
        for role in included_roles:
            (output / split / role).mkdir(parents=True, exist_ok=True)

    label_files: List[Tuple[str, Path]] = []
    for split in splits:
        split_root = source / split
        if not split_root.is_dir():
            LOGGER.warning("Split SoccerNet absent, ignore: %s", split_root)
            continue
        label_files.extend((split, path) for path in sorted(split_root.rglob("Labels-GameState.json")))
    if max_clips is not None:
        label_files = label_files[:max_clips]
    if not label_files:
        raise ValueError("Aucun Labels-GameState.json trouve dans {}".format(source))

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow est requis pour preparer les crops SoccerNet") from exc

    stats = RolePreparationStats(
        roles={role: 0 for role in included_roles},
        splits={_output_split(split): 0 for split in splits},
        split_roles={
            _output_split(split): {role: 0 for role in included_roles} for split in splits
        },
    )
    occurrence: DefaultDict[Tuple[str, str, str], int] = defaultdict(int)
    saved: DefaultDict[Tuple[str, str, str], int] = defaultdict(int)
    manifest_path = output / "manifest.jsonl"
    annotations_path = output / "annotations.jsonl"
    balls_path = output / "ball_annotations.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest_stream, annotations_path.open(
        "w", encoding="utf-8"
    ) as annotation_stream, balls_path.open("w", encoding="utf-8") as ball_stream:
        for source_split, labels_path in label_files:
            payload = _load_game_state(labels_path, minimum_version)
            clip_root = labels_path.parent
            info = payload.get("info") or {}
            sequence = str(info.get("name") or clip_root.name)
            image_dir = str(info.get("im_dir") or "img1")
            images = payload.get("images") or []
            annotations = payload.get("annotations") or []
            image_map = {_image_identifier(image): image for image in images}
            image_order = {identifier: index for index, identifier in enumerate(image_map)}
            annotations_by_image: DefaultDict[str, List[Mapping[str, Any]]] = defaultdict(list)
            for annotation in annotations:
                if annotation.get("supercategory") != "object":
                    continue
                annotations_by_image[str(annotation.get("image_id"))].append(annotation)

            output_split = _output_split(source_split)
            stats.clips += 1
            stats.splits[output_split] = stats.splits.get(output_split, 0) + 1
            stats.images += len(images)
            for image_id, frame_annotations in sorted(
                annotations_by_image.items(), key=lambda item: image_order.get(item[0], 10**12)
            ):
                image = image_map.get(image_id)
                if image is None:
                    stats.skipped_annotations += len(frame_annotations)
                    continue
                file_name = str(image.get("file_name") or "")
                image_path = _find_image(clip_root, image_dir, file_name)
                frame_index = image_order.get(image_id, 0)
                opened_image = None
                for annotation in sorted(
                    frame_annotations, key=lambda item: str(item.get("track_id"))
                ):
                    stats.annotations += 1
                    record = _annotation_record(
                        source,
                        source_split,
                        output_split,
                        sequence,
                        image_path,
                        image_id,
                        frame_index,
                        image,
                        annotation,
                    )
                    role = record["role"]
                    _write_json_line(annotation_stream, record)
                    if role == "ball":
                        stats.ball_annotations += 1
                        _write_json_line(ball_stream, record)
                        continue
                    if role not in HUMAN_ROLES:
                        stats.skipped_annotations += 1
                        continue
                    stats.human_annotations += 1
                    if role not in included_roles:
                        continue
                    if record["track_id"] is None:
                        stats.skipped_annotations += 1
                        continue
                    track_key = (output_split, sequence, str(record["track_id"]))
                    track_occurrence = occurrence[track_key]
                    occurrence[track_key] += 1
                    if track_occurrence % frame_stride or saved[track_key] >= max_samples_per_track:
                        continue
                    if not image_path.is_file():
                        stats.skipped_annotations += 1
                        continue
                    if opened_image is None:
                        opened_image = Image.open(str(image_path)).convert("RGB")
                    record["image_width"] = opened_image.width
                    record["image_height"] = opened_image.height
                    box = _bbox_to_ltrb(
                        annotation.get("bbox_image") or {},
                        opened_image.width,
                        opened_image.height,
                        context,
                    )
                    if box is None or box[2] - box[0] < min_crop_size or box[3] - box[1] < min_crop_size:
                        stats.skipped_annotations += 1
                        continue
                    name = "{}__f{:06d}__t{}__n{:03d}.jpg".format(
                        _safe_token(sequence),
                        frame_index + 1,
                        _safe_token(record["track_id"]),
                        saved[track_key],
                    )
                    crop_path = output / output_split / role / name
                    opened_image.crop(box).save(str(crop_path), format="JPEG", quality=jpeg_quality)
                    record["crop_path"] = crop_path.relative_to(output).as_posix()
                    record["crop_ltrb"] = list(box)
                    _write_json_line(manifest_stream, record)
                    saved[track_key] += 1
                    stats.crops += 1
                    stats.roles[role] = stats.roles.get(role, 0) + 1
                    stats.split_roles[output_split][role] += 1
                if opened_image is not None:
                    opened_image.close()

    dataset = {
        "task": "classification",
        # Keep the generated metadata portable when the prepared dataset is moved
        # from Windows to a Linux training machine.
        "path": ".",
        "train": "train",
        "val": "val",
        "test": "test",
        "names": list(included_roles),
        "source": str(source),
        "minimum_gamestate_version": minimum_version,
        "frame_stride": frame_stride,
        "max_samples_per_track": max_samples_per_track,
        "context": context,
    }
    (output / "dataset.yaml").write_text(
        yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8"
    )
    (output / "preparation_stats.json").write_text(
        json.dumps(asdict(stats), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return stats


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("JSONL invalide dans {}:{}".format(path, line_number)) from exc
