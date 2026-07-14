"""Creation de videos de controle pour les sequences au format MOTChallenge."""

import configparser
import csv
import logging
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple

from .data import IMAGE_SUFFIXES, SequenceInfo, discover_sequences

LOGGER = logging.getLogger(__name__)

MOTBox = Tuple[int, float, float, float, float]


@dataclass(frozen=True)
class VisualizationResult:
    sequence: str
    split: Optional[str]
    output: str
    frames: int
    boxes: int
    fps: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _frame_id(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(
            "Le nom d'image MOT doit etre numerique, fichier recu: {}".format(path)
        ) from exc


def _read_fps(sequence_root: Path, override: Optional[float]) -> float:
    if override is not None:
        if override <= 0:
            raise ValueError("--fps doit etre strictement positif")
        return float(override)
    info_path = sequence_root / "seqinfo.ini"
    if info_path.is_file():
        parser = configparser.ConfigParser()
        parser.read(str(info_path))
        try:
            value = float(parser["Sequence"]["frameRate"])
            if value > 0:
                return value
        except (KeyError, ValueError):
            LOGGER.warning("frameRate invalide dans %s; utilisation de 25 FPS", info_path)
    return 25.0


def read_mot_boxes(sequence_root: Path) -> DefaultDict[int, List[MOTBox]]:
    """Lit les boites MOT valides, indexees par numero d'image."""
    result: DefaultDict[int, List[MOTBox]] = defaultdict(list)
    gt_path = Path(sequence_root) / "gt" / "gt.txt"
    if not gt_path.is_file():
        return result
    with gt_path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream), start=1):
            if len(row) < 6:
                LOGGER.warning("Annotation ignoree dans %s:%d", gt_path, line_number)
                continue
            try:
                frame = int(float(row[0]))
                track_id = int(float(row[1]))
                left, top, width, height = (float(value) for value in row[2:6])
                confidence = float(row[6]) if len(row) > 6 else 1.0
            except ValueError:
                LOGGER.warning("Annotation non numerique ignoree dans %s:%d", gt_path, line_number)
                continue
            if confidence <= 0 or width <= 0 or height <= 0:
                continue
            result[frame].append((track_id, left, top, width, height))
    return result


def _track_color(track_id: int) -> Tuple[int, int, int]:
    """Retourne une couleur BGR stable et suffisamment lumineuse pour une identite."""
    return (
        64 + (track_id * 37) % 192,
        64 + (track_id * 17) % 192,
        64 + (track_id * 97) % 192,
    )


def _draw_boxes(frame: object, boxes: Sequence[MOTBox]) -> int:
    import cv2

    height, width = frame.shape[:2]  # type: ignore[attr-defined]
    thickness = max(1, round(min(width, height) / 500))
    font_scale = max(0.45, min(width, height) / 1100.0)
    drawn = 0
    for track_id, left, top, box_width, box_height in boxes:
        x1 = max(0, min(width - 1, round(left)))
        y1 = max(0, min(height - 1, round(top)))
        x2 = max(0, min(width - 1, round(left + box_width)))
        y2 = max(0, min(height - 1, round(top + box_height)))
        if x2 <= x1 or y2 <= y1:
            continue
        color = _track_color(track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = "ID {}".format(track_id)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_top = max(0, y1 - text_height - baseline - 4)
        label_right = min(width - 1, x1 + text_width + 6)
        cv2.rectangle(frame, (x1, label_top), (label_right, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 3, max(text_height, y1 - baseline - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
        drawn += 1
    return drawn


def _select_sequences(
    available: Sequence[SequenceInfo],
    count: int,
    names: Optional[Sequence[str]],
    shuffle: bool,
    seed: int,
) -> List[SequenceInfo]:
    if names:
        by_name = {sequence.name: sequence for sequence in available}
        requested = list(dict.fromkeys(names))
        missing = [name for name in requested if name not in by_name]
        if missing:
            raise ValueError("Sequence(s) introuvable(s): {}".format(", ".join(missing)))
        return [by_name[name] for name in requested]
    if count < 1:
        raise ValueError("--num-sequences doit etre >= 1")
    if count > len(available):
        raise ValueError(
            "{} sequence(s) demandee(s), mais seulement {} disponible(s)".format(
                count, len(available)
            )
        )
    candidates = list(available)
    if shuffle:
        random.Random(seed).shuffle(candidates)
    return candidates[:count]


def visualize_mot_sequences(
    source: Path,
    output: Path,
    num_sequences: int = 2,
    sequence_list: Optional[Path] = None,
    split: str = "all",
    sequence_names: Optional[Sequence[str]] = None,
    show_boxes: bool = True,
    shuffle: bool = False,
    seed: int = 42,
    max_frames: Optional[int] = None,
    fps: Optional[float] = None,
) -> List[VisualizationResult]:
    """Genere un MP4 par sequence MOT, avec ou sans boites et identites GT."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless est requis pour la visualisation") from exc

    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if max_frames is not None and max_frames < 1:
        raise ValueError("--max-frames doit etre >= 1")
    sequences = discover_sequences(source, sequence_list=sequence_list)
    if split != "all":
        sequences = [sequence for sequence in sequences if sequence.original_split == split]
        if not sequences:
            raise ValueError("Aucune sequence trouvee dans le split '{}'".format(split))
    selected = _select_sequences(sequences, num_sequences, sequence_names, shuffle, seed)
    output.mkdir(parents=True, exist_ok=True)
    results: List[VisualizationResult] = []

    for sequence in selected:
        images = sorted(
            (
                path
                for path in (sequence.root / "img1").iterdir()
                if path.suffix.lower() in IMAGE_SUFFIXES
            ),
            key=_frame_id,
        )
        if max_frames is not None:
            images = images[:max_frames]
        if not images:
            LOGGER.warning("Sequence sans image ignoree: %s", sequence.root)
            continue
        first = cv2.imread(str(images[0]))
        if first is None:
            raise ValueError("Image illisible: {}".format(images[0]))
        height, width = first.shape[:2]
        sequence_fps = _read_fps(sequence.root, fps)
        suffix = "gt" if show_boxes else "raw"
        video_path = output / "{}_{}.mp4".format(sequence.name, suffix)
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), sequence_fps, (width, height)
        )
        if not writer.isOpened():
            raise ValueError("Video de sortie impossible a creer: {}".format(video_path))
        annotations = read_mot_boxes(sequence.root) if show_boxes else defaultdict(list)
        if show_boxes and not annotations:
            LOGGER.warning("Aucune GT disponible pour %s; video produite sans boites", sequence.name)
        frame_count = 0
        box_count = 0
        try:
            for image_path in images:
                frame = cv2.imread(str(image_path))
                if frame is None:
                    LOGGER.warning("Image illisible ignoree: %s", image_path)
                    continue
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                if show_boxes:
                    box_count += _draw_boxes(frame, annotations.get(_frame_id(image_path), []))
                writer.write(frame)
                frame_count += 1
        finally:
            writer.release()
        results.append(
            VisualizationResult(
                sequence=sequence.name,
                split=sequence.original_split,
                output=str(video_path),
                frames=frame_count,
                boxes=box_count,
                fps=sequence_fps,
            )
        )
        LOGGER.info("Video creee: %s (%d images, %d boites)", video_path, frame_count, box_count)
    return results
