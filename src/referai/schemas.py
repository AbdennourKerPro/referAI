"""Objets de sortie stables et independants du moteur de detection."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


CLASS_NAMES = ("player", "goalkeeper", "referee", "ball")


@dataclass(frozen=True)
class TrackedObject:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["class"] = data.pop("class_name")
        data.pop("class_id")
        data["bbox"] = [round(float(value), 3) for value in self.bbox]
        data["confidence"] = round(float(self.confidence), 6)
        return data


@dataclass(frozen=True)
class FrameObservations:
    frame_id: int
    timestamp: float
    objects: List[TrackedObject]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp": round(float(self.timestamp), 6),
            "objects": [obj.to_dict() for obj in self.objects],
        }


@dataclass(frozen=True)
class RunStatistics:
    frames: int
    elapsed_seconds: float
    fps: float
    mean_latency_ms: float
    peak_gpu_memory_mb: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

