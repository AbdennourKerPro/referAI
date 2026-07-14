"""Ecriture progressive des observations sans conserver toute la video en RAM."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

from .schemas import FrameObservations


class ObservationWriter:
    def write(self, frame: FrameObservations) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "ObservationWriter":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class JSONWriter(ObservationWriter):
    def __init__(self, path: Path, lines: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream: TextIO = self.path.open("w", encoding="utf-8")
        self.lines = lines
        self.first = True
        if not lines:
            self.stream.write("[\n")

    def write(self, frame: FrameObservations) -> None:
        encoded = json.dumps(frame.to_dict(), ensure_ascii=False, separators=(",", ":"))
        if self.lines:
            self.stream.write(encoded + "\n")
            return
        if not self.first:
            self.stream.write(",\n")
        self.stream.write(encoded)
        self.first = False

    def close(self) -> None:
        if self.stream.closed:
            return
        if not self.lines:
            self.stream.write("\n]\n")
        self.stream.close()


class ParquetWriter(ObservationWriter):
    def __init__(self, path: Path, batch_rows: int = 5000) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow est requis pour une sortie Parquet") from exc
        self.pa = pa
        self.pq = pq
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.batch_rows = batch_rows
        self.rows: List[Dict[str, Any]] = []
        self.writer: Optional[Any] = None

    def write(self, frame: FrameObservations) -> None:
        for obj in frame.objects:
            x1, y1, x2, y2 = obj.bbox
            self.rows.append(
                {
                    "frame_id": frame.frame_id,
                    "timestamp": frame.timestamp,
                    "track_id": obj.track_id,
                    "class_id": obj.class_id,
                    "class": obj.class_name,
                    "confidence": obj.confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
        if len(self.rows) >= self.batch_rows:
            self._flush()

    def _flush(self) -> None:
        if not self.rows:
            return
        table = self.pa.Table.from_pylist(self.rows)
        if self.writer is None:
            self.writer = self.pq.ParquetWriter(str(self.path), table.schema, compression="zstd")
        self.writer.write_table(table)
        self.rows.clear()

    def close(self) -> None:
        self._flush()
        if self.writer is None:
            # Cree tout de meme un fichier valide pour une video sans detection.
            empty = self.pa.table(
                {
                    "frame_id": self.pa.array([], type=self.pa.int64()),
                    "timestamp": self.pa.array([], type=self.pa.float64()),
                    "track_id": self.pa.array([], type=self.pa.int64()),
                    "class_id": self.pa.array([], type=self.pa.int64()),
                    "class": self.pa.array([], type=self.pa.string()),
                    "confidence": self.pa.array([], type=self.pa.float64()),
                    "x1": self.pa.array([], type=self.pa.float64()),
                    "y1": self.pa.array([], type=self.pa.float64()),
                    "x2": self.pa.array([], type=self.pa.float64()),
                    "y2": self.pa.array([], type=self.pa.float64()),
                }
            )
            self.pq.write_table(empty, str(self.path), compression="zstd")
        else:
            self.writer.close()


class MOTWriter(ObservationWriter):
    """Sortie MOTChallenge a 10 colonnes pour TrackEval/SoccerNet."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("w", encoding="utf-8")

    def write(self, frame: FrameObservations) -> None:
        for obj in frame.objects:
            x1, y1, x2, y2 = obj.bbox
            self.stream.write(
                "{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.6f},-1,-1,-1\n".format(
                    frame.frame_id + 1,
                    obj.track_id,
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1,
                    obj.confidence,
                )
            )

    def close(self) -> None:
        if not self.stream.closed:
            self.stream.close()


def make_observation_writer(path: Path) -> ObservationWriter:
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        return JSONWriter(path)
    if suffix == ".jsonl":
        return JSONWriter(path, lines=True)
    if suffix == ".parquet":
        return ParquetWriter(path)
    raise ValueError("Extension de sortie attendue: .json, .jsonl ou .parquet")

