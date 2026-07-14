"""Metriques de selection et diagnostics de trajectoires."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

from .schemas import FrameObservations


COMPOSITE_WEIGHTS = {"player": 0.35, "referee": 0.20, "goalkeeper": 0.15, "ball": 0.30}


def composite_detection_score(ap_by_class: Mapping[str, float]) -> float:
    missing = set(COMPOSITE_WEIGHTS) - set(ap_by_class)
    if missing:
        raise ValueError("AP manquantes pour: {}".format(sorted(missing)))
    return sum(COMPOSITE_WEIGHTS[name] * float(ap_by_class[name]) for name in COMPOSITE_WEIGHTS)


@dataclass(frozen=True)
class TrajectoryDiagnostics:
    tracks: int
    observations: int
    fragmented_tracks: int
    total_gaps: int
    mean_track_length: float


def trajectory_diagnostics(frames: Iterable[FrameObservations]) -> TrajectoryDiagnostics:
    seen: Dict[int, list] = defaultdict(list)
    for frame in frames:
        for obj in frame.objects:
            if obj.track_id >= 0:
                seen[obj.track_id].append(frame.frame_id)
    gaps = []
    lengths = []
    fragmented = 0
    for frame_ids in seen.values():
        ordered = sorted(set(frame_ids))
        track_gaps = sum(max(0, right - left - 1) for left, right in zip(ordered, ordered[1:]))
        gaps.append(track_gaps)
        lengths.append(len(ordered))
        fragmented += int(track_gaps > 0)
    observations = sum(lengths)
    return TrajectoryDiagnostics(
        tracks=len(seen),
        observations=observations,
        fragmented_tracks=fragmented,
        total_gaps=sum(gaps),
        mean_track_length=(observations / len(seen)) if seen else 0.0,
    )


def ball_recall_by_area(
    matches: Sequence[Mapping[str, float]], thresholds: Sequence[float] = (16.0 ** 2, 32.0 ** 2)
) -> Dict[str, float]:
    """Calcule le rappel du ballon par aire depuis des GT deja appariees.

    Chaque entree contient ``area`` et ``matched`` (0/1). Cette fonction reste
    independante du protocole d'appariement choisi par l'evaluateur.
    """
    bins = [
        ("tiny", 0.0, thresholds[0]),
        ("small", thresholds[0], thresholds[1]),
        ("large", thresholds[1], float("inf")),
    ]
    result = {}
    for name, lower, upper in bins:
        selected = [row for row in matches if lower <= float(row["area"]) < upper]
        result[name] = (
            sum(float(row["matched"]) for row in selected) / len(selected) if selected else 0.0
        )
    return result
