"""Pipeline video YOLO -> ByteTrack -> sorties structurees et video annotee."""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional

from .hardware import (
    inspect_gpus,
    peak_memory_allocated_mb,
    profile_summary,
    reset_peak_memory_stats,
    resolve_profile,
)
from .output import MOTWriter, make_observation_writer
from .schemas import FrameObservations, RunStatistics, TrackedObject
from .training import _import_yolo

LOGGER = logging.getLogger(__name__)


def _objects_from_result(result: Any) -> List[TrackedObject]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().tolist()
    confidences = boxes.conf.detach().cpu().tolist()
    class_ids = boxes.cls.detach().cpu().tolist()
    track_ids = boxes.id.detach().cpu().tolist() if boxes.id is not None else [-1] * len(xyxy)
    names = result.names
    objects = []
    for bbox, confidence, class_id, track_id in zip(xyxy, confidences, class_ids, track_ids):
        class_index = int(class_id)
        if isinstance(names, dict):
            class_name = str(names.get(class_index, class_index))
        else:
            class_name = str(names[class_index])
        objects.append(
            TrackedObject(
                track_id=int(track_id),
                class_id=class_index,
                class_name=class_name,
                confidence=float(confidence),
                bbox=tuple(float(value) for value in bbox),
            )
        )
    return objects


def _write_trajectories(path: Path, trajectories: Dict[int, List[Dict[str, Any]]]) -> None:
    payload = [
        {"track_id": track_id, "observations": observations}
        for track_id, observations in sorted(trajectories.items())
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def track_video(
    video: Path,
    weights: Path,
    output: Path,
    tracker: Path,
    hardware: Optional[Dict[str, Any]] = None,
    annotated_video: Optional[Path] = None,
    mot_output: Optional[Path] = None,
    trajectories_output: Optional[Path] = None,
    confidence: float = 0.05,
    iou: float = 0.7,
) -> RunStatistics:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless est requis pour traiter une video") from exc
    profile = resolve_profile(hardware)
    LOGGER.info(profile_summary(profile, inspect_gpus(backend=profile.backend)))
    # Un flux temporel est sequentiel: un seul GPU garde l'etat ByteTrack. Les GPU
    # additionnels servent au DDP d'entrainement ou a plusieurs videos en parallele.
    inference_device = profile.primary_device
    YOLO = _import_yolo(profile.backend)
    model = YOLO(str(weights))
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError("Video impossible a ouvrir: {}".format(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = None
    if annotated_video is not None:
        annotated_video = Path(annotated_video)
        annotated_video.parent.mkdir(parents=True, exist_ok=True)
        video_writer = cv2.VideoWriter(
            str(annotated_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not video_writer.isOpened():
            capture.release()
            raise ValueError("Video de sortie impossible a creer: {}".format(annotated_video))
    mot_writer = MOTWriter(mot_output) if mot_output else None
    trajectories: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
    if profile.device_ids:
        try:
            reset_peak_memory_stats(profile.backend, inference_device)
        except (ImportError, RuntimeError, TypeError):
            pass
    frame_id = 0
    started = time.perf_counter()
    writer = make_observation_writer(output)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            results = model.track(
                source=frame,
                persist=True,
                tracker=str(tracker),
                conf=confidence,
                iou=iou,
                imgsz=profile.imgsz,
                device=inference_device,
                half=profile.half,
                verbose=False,
            )
            result = results[0]
            observation = FrameObservations(
                frame_id=frame_id,
                timestamp=frame_id / fps,
                objects=_objects_from_result(result),
            )
            writer.write(observation)
            if mot_writer:
                mot_writer.write(observation)
            if trajectories_output:
                for obj in observation.objects:
                    x1, y1, x2, y2 = obj.bbox
                    trajectories[obj.track_id].append(
                        {
                            "frame_id": frame_id,
                            "timestamp": observation.timestamp,
                            "x": (x1 + x2) / 2.0,
                            "y": (y1 + y2) / 2.0,
                            "w": x2 - x1,
                            "h": y2 - y1,
                        }
                    )
            if video_writer is not None:
                video_writer.write(result.plot())
            frame_id += 1
    finally:
        writer.close()
        if mot_writer:
            mot_writer.close()
        capture.release()
        if video_writer is not None:
            video_writer.release()
    elapsed = time.perf_counter() - started
    if trajectories_output:
        _write_trajectories(trajectories_output, trajectories)
    peak_memory = None
    if profile.device_ids:
        try:
            peak_memory = peak_memory_allocated_mb(profile.backend, inference_device)
        except (ImportError, RuntimeError, TypeError):
            pass
    stats = RunStatistics(
        frames=frame_id,
        elapsed_seconds=elapsed,
        fps=frame_id / elapsed if elapsed else 0.0,
        mean_latency_ms=(elapsed / frame_id * 1000.0) if frame_id else 0.0,
        peak_gpu_memory_mb=peak_memory,
    )
    stats_path = Path(str(output) + ".stats.json")
    stats_path.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
    return stats
