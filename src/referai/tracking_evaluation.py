"""Evaluation multi-objets sur des sequences MOT avec l'implementation TrackEval."""

import csv
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .config import load_yaml
from .data import IMAGE_SUFFIXES
from .hardware import inspect_gpus, profile_summary, resolve_profile
from .output import MOTWriter
from .schemas import FrameObservations
from .tracking import _objects_from_result
from .training import _import_yolo

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackingInferenceStats:
    sequences: int
    frames: int
    tracked_objects: int
    elapsed_seconds: float
    fps: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MOTSequence:
    name: str
    images: Tuple[Path, ...]
    ground_truth: Path
    seqinfo: Path


def _dataset_root(data_yaml: Path, config: Mapping[str, Any]) -> Path:
    root = Path(str(config.get("path", data_yaml.parent))).expanduser()
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root.resolve()


def _read_seqmap(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError("Seqmap introuvable: {}".format(path))
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    names = [row[0].strip() for index, row in enumerate(rows) if index > 0 and row and row[0]]
    if not names:
        raise ValueError("Aucune sequence dans le seqmap {}".format(path))
    return names


def _image_order(path: Path) -> Tuple[int, Any]:
    try:
        return 0, int(path.stem)
    except ValueError:
        return 1, path.name


def discover_evaluation_sequences(
    data_yaml: Path,
    split: str,
    requested: Optional[Sequence[str]] = None,
    max_sequences: Optional[int] = None,
) -> Tuple[Path, Path, List[MOTSequence]]:
    data_yaml = Path(data_yaml).expanduser().resolve()
    config = load_yaml(data_yaml)
    root = _dataset_root(data_yaml, config)
    split_entry = config.get(split)
    if not isinstance(split_entry, str):
        raise ValueError("Le split '{}' doit etre un chemin dans {}".format(split, data_yaml))
    images_root = Path(split_entry).expanduser()
    if not images_root.is_absolute():
        images_root = root / images_root
    images_root = images_root.resolve()
    if not images_root.is_dir():
        raise FileNotFoundError("Dossier d'images introuvable: {}".format(images_root))

    mot_root = root / "mot_gt"
    seqmap = mot_root / "seqmaps" / "{}.txt".format(split)
    available = _read_seqmap(seqmap)
    if requested:
        requested_names = list(dict.fromkeys(requested))
        missing = sorted(set(requested_names) - set(available))
        if missing:
            raise ValueError("Sequence(s) absente(s) du split {}: {}".format(split, missing))
        names = requested_names
    else:
        names = available
    if max_sequences is not None:
        if max_sequences < 1:
            raise ValueError("--max-sequences doit etre >= 1")
        names = names[:max_sequences]

    sequences = []
    for name in names:
        sequence_images = images_root / name
        images = tuple(
            sorted(
                (
                    path
                    for path in sequence_images.iterdir()
                    if path.suffix.lower() in IMAGE_SUFFIXES
                ),
                key=_image_order,
            )
        ) if sequence_images.is_dir() else ()
        if not images:
            raise ValueError("Aucune image pour la sequence {} dans {}".format(name, images_root))
        gt_file = mot_root / split / name / "gt" / "gt.txt"
        seqinfo = mot_root / split / name / "seqinfo.ini"
        if not gt_file.is_file() or gt_file.stat().st_size == 0:
            raise ValueError(
                "Verite terrain MOT absente pour {}. Le split '{}' ne peut pas etre evalue."
                .format(name, split)
            )
        if not seqinfo.is_file():
            raise FileNotFoundError("seqinfo.ini introuvable pour {}: {}".format(name, seqinfo))
        sequences.append(MOTSequence(name, images, gt_file, seqinfo))
    return root, seqmap, sequences


def _reset_tracker_state(model: Any) -> None:
    predictor = getattr(model, "predictor", None)
    trackers = getattr(predictor, "trackers", None)
    reset = False
    if trackers:
        for tracker in trackers:
            reset_method = getattr(tracker, "reset", None)
            if callable(reset_method):
                reset_method()
                reset = True
    if predictor is not None and not reset:
        # Recreer uniquement le predictor conserve les poids mais garantit un nouvel etat temporel.
        model.predictor = None


def generate_mot_predictions(
    sequences: Sequence[MOTSequence],
    weights: Path,
    tracker: Path,
    predictions_dir: Path,
    hardware: Optional[Dict[str, Any]] = None,
    confidence: float = 0.05,
    iou: float = 0.70,
    class_id: int = 0,
) -> TrackingInferenceStats:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless est requis pour evaluer le tracking") from exc
    profile = resolve_profile(hardware)
    LOGGER.info(profile_summary(profile, inspect_gpus()))
    inference_device = profile.device_ids[0] if profile.device_ids else "cpu"
    YOLO = _import_yolo()
    model = YOLO(str(Path(weights).expanduser().resolve()))
    tracker = Path(tracker).expanduser().resolve()
    if not tracker.is_file():
        raise FileNotFoundError("Configuration tracker introuvable: {}".format(tracker))
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    predictions_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    total_objects = 0
    started = time.perf_counter()
    for sequence_index, sequence in enumerate(sequences):
        if sequence_index:
            _reset_tracker_state(model)
        output = predictions_dir / "{}.txt".format(sequence.name)
        with MOTWriter(output) as writer:
            for image_path in sequence.images:
                frame = cv2.imread(str(image_path))
                if frame is None:
                    raise ValueError("Image illisible: {}".format(image_path))
                results = model.track(
                    source=frame,
                    persist=True,
                    tracker=str(tracker),
                    conf=confidence,
                    iou=iou,
                    classes=[class_id],
                    imgsz=profile.imgsz,
                    device=inference_device,
                    half=profile.half,
                    verbose=False,
                )
                try:
                    frame_number = int(image_path.stem)
                except ValueError as exc:
                    raise ValueError(
                        "Le nom d'image MOT doit etre numerique: {}".format(image_path)
                    ) from exc
                objects = [
                    obj
                    for obj in _objects_from_result(results[0])
                    if obj.class_id == class_id and obj.track_id >= 0
                ]
                writer.write(
                    FrameObservations(
                        frame_id=frame_number - 1,
                        timestamp=0.0,
                        objects=objects,
                    )
                )
                total_frames += 1
                total_objects += len(objects)
        LOGGER.info(
            "Predictions MOT: %s (%d images) -> %s",
            sequence.name,
            len(sequence.images),
            output,
        )
    elapsed = time.perf_counter() - started
    return TrackingInferenceStats(
        sequences=len(sequences),
        frames=total_frames,
        tracked_objects=total_objects,
        elapsed_seconds=elapsed,
        fps=total_frames / elapsed if elapsed else 0.0,
    )


def _write_seqmap(path: Path, sequences: Sequence[MOTSequence]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "name\n" + "\n".join(sequence.name for sequence in sequences) + "\n",
        encoding="utf-8",
    )
    return path.resolve()


def build_trackeval_command(
    trackeval_root: Path,
    gt_folder: Path,
    trackers_folder: Path,
    tracker_name: str,
    split: str,
    sequence_names: Sequence[str],
) -> List[str]:
    trackeval_root = Path(trackeval_root).expanduser().resolve()
    script = trackeval_root / "scripts" / "run_mot_challenge.py"
    package = trackeval_root / "trackeval"
    if not script.is_file() or not package.is_dir():
        raise FileNotFoundError(
            "Depot TrackEval invalide: {}. Le script scripts/run_mot_challenge.py est requis."
            .format(trackeval_root)
        )
    return [
        sys.executable,
        "-m",
        "referai.trackeval_runner",
        str(script),
        "--GT_FOLDER",
        str(Path(gt_folder).resolve()),
        "--TRACKERS_FOLDER",
        str(Path(trackers_folder).resolve()),
        "--TRACKERS_TO_EVAL",
        tracker_name,
        "--TRACKER_SUB_FOLDER",
        "data",
        "--OUTPUT_SUB_FOLDER",
        "trackeval",
        "--BENCHMARK",
        "referAI",
        "--SPLIT_TO_EVAL",
        split,
        "--SEQ_INFO",
        *sequence_names,
        "--SKIP_SPLIT_FOL",
        "True",
        "--DO_PREPROC",
        "False",
        "--METRICS",
        "HOTA",
        "CLEAR",
        "Identity",
        "--USE_PARALLEL",
        "False",
        "--PLOT_CURVES",
        "False",
        "--PRINT_CONFIG",
        "False",
    ]


def _parse_value(value: str) -> Any:
    try:
        return float(value)
    except ValueError:
        return value


def parse_trackeval_summary(output_folder: Path, tracker_name: str) -> Tuple[Path, Dict[str, Any]]:
    output_folder = Path(output_folder).expanduser().resolve()
    candidates = [
        path
        for path in output_folder.rglob("*_summary.txt")
        if tracker_name in path.parts
    ]
    if not candidates:
        raise FileNotFoundError(
            "Resume TrackEval introuvable sous {} pour le tracker {}"
            .format(output_folder, tracker_name)
        )
    summary = sorted(candidates)[0]
    lines = [line.strip() for line in summary.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("Resume TrackEval invalide: {}".format(summary))
    headers = lines[0].split()
    values = lines[1].split()
    if len(headers) != len(values):
        raise ValueError(
            "Colonnes TrackEval incoherentes dans {}: {} noms, {} valeurs"
            .format(summary, len(headers), len(values))
        )
    return summary, {header: _parse_value(value) for header, value in zip(headers, values)}


def evaluate_tracking(
    data_yaml: Path,
    trackeval_root: Path,
    output: Path,
    split: str = "val",
    weights: Optional[Path] = None,
    tracker: Path = Path("configs/bytetrack_football.yaml"),
    hardware: Optional[Dict[str, Any]] = None,
    confidence: float = 0.05,
    iou: float = 0.70,
    class_id: int = 0,
    tracker_name: str = "referai",
    requested_sequences: Optional[Sequence[str]] = None,
    max_sequences: Optional[int] = None,
    skip_inference: bool = False,
) -> Dict[str, Any]:
    if Path(tracker_name).name != tracker_name:
        raise ValueError("--tracker-name doit etre un nom simple sans dossier")
    root, _, sequences = discover_evaluation_sequences(
        data_yaml, split, requested_sequences, max_sequences
    )
    output = Path(output).expanduser().resolve()
    tracker_root = output / "predictions"
    predictions_dir = tracker_root / tracker_name / "data"
    inference_stats = None
    if skip_inference:
        missing = [
            sequence.name
            for sequence in sequences
            if not (predictions_dir / "{}.txt".format(sequence.name)).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Predictions manquantes avec --skip-inference: {}".format(missing[:5])
            )
    else:
        if weights is None:
            raise ValueError("--weights est obligatoire sauf avec --skip-inference")
        inference_stats = generate_mot_predictions(
            sequences=sequences,
            weights=weights,
            tracker=tracker,
            predictions_dir=predictions_dir,
            hardware=hardware,
            confidence=confidence,
            iou=iou,
            class_id=class_id,
        )
        (output / "inference_stats.json").write_text(
            json.dumps(inference_stats.to_dict(), indent=2), encoding="utf-8"
        )

    _write_seqmap(output / "seqmap_{}.txt".format(split), sequences)
    command = build_trackeval_command(
        trackeval_root=trackeval_root,
        gt_folder=root / "mot_gt" / split,
        trackers_folder=tracker_root,
        tracker_name=tracker_name,
        split=split,
        sequence_names=[sequence.name for sequence in sequences],
    )
    LOGGER.info("Lancement TrackEval: %s", " ".join(command))
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("TrackEval a echoue avec le code {}".format(exc.returncode)) from exc

    summary_path, all_metrics = parse_trackeval_summary(tracker_root, tracker_name)
    selected_metrics = {
        name: all_metrics[name]
        for name in (
            "HOTA",
            "DetA",
            "AssA",
            "MOTA",
            "MOTP",
            "IDF1",
            "IDP",
            "IDR",
            "IDSW",
            "Frag",
            "CLR_Re",
            "CLR_Pr",
        )
        if name in all_metrics
    }
    result = {
        "split": split,
        "tracker": tracker_name,
        "class_id": class_id,
        "sequences": [sequence.name for sequence in sequences],
        "metrics": selected_metrics,
        "all_metrics": all_metrics,
        "inference": inference_stats.to_dict() if inference_stats else None,
        "predictions": str(predictions_dir),
        "trackeval_summary": str(summary_path),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "tracking_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result
