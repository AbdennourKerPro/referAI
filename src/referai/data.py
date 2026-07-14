"""Conversion des jeux MOT17/SoccerNet/SportsMOT vers le format YOLO."""

import configparser
import csv
import json
import os
import random
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from .schemas import CLASS_NAMES

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_ALIASES = {"train": "train", "val": "val", "valid": "val", "test": "test"}


@dataclass(frozen=True)
class SequenceInfo:
    name: str
    root: Path
    original_split: Optional[str]
    match_id: str


@dataclass
class ConversionStats:
    sequences: int = 0
    images: int = 0
    boxes: int = 0
    ignored_boxes: int = 0
    empty_images: int = 0


def _infer_split(path: Path) -> Optional[str]:
    for part in reversed(path.parts):
        split = SPLIT_ALIASES.get(part.lower())
        if split:
            return split
    return None


def _load_match_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "Fichier de correspondance introuvable: {}. Creez-le avec "
            "'referai-football create-match-map --source <dataset> --output {}'.".format(
                path, path
            )
        )
    if path.suffix.lower() == ".json":
        content = json.loads(path.read_text(encoding="utf-8"))
        return {str(key).strip(): str(value).strip() for key, value in content.items()}
    result = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            result[str(row["sequence"]).strip()] = str(row["match_id"]).strip()
    return result


def create_match_map_template(
    source: Path,
    output: Path,
    sequence_list: Optional[Path] = None,
    force: bool = False,
) -> int:
    """Cree un CSV a completer, sans inventer de regroupements entre clips."""
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists() and not force:
        raise FileExistsError(
            "{} existe deja; utilisez --force uniquement pour le remplacer.".format(output)
        )
    sequences = discover_sequences(source, sequence_list=sequence_list)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("sequence", "match_id", "original_split", "source_path"),
        )
        writer.writeheader()
        for sequence in sequences:
            relative = sequence.root.relative_to(source)
            # Si le dataset contient split/match/sequence, le dossier match est
            # une suggestion fiable. Avec split/sequence, la valeur reste vide.
            middle = [
                part
                for part in relative.parts[:-1]
                if part.lower() not in SPLIT_ALIASES
            ]
            writer.writerow(
                {
                    "sequence": sequence.name,
                    "match_id": "/".join(middle),
                    "original_split": sequence.original_split or "",
                    "source_path": str(relative),
                }
            )
    return len(sequences)


def discover_sequences(
    source: Path,
    match_map: Optional[Path] = None,
    sequence_list: Optional[Path] = None,
) -> List[SequenceInfo]:
    source = Path(source).expanduser().resolve()
    mapping = _load_match_map(match_map)
    allowed = None
    if sequence_list is not None:
        allowed = {
            row.strip()
            for row in Path(sequence_list).read_text(encoding="utf-8").splitlines()
            if row.strip() and row.strip().lower() not in {"name", "sequence"}
        }
    sequences = []
    for image_dir in sorted(path for path in source.rglob("img1") if path.is_dir()):
        root = image_dir.parent
        name = root.name
        if allowed is not None and name not in allowed:
            continue
        sequences.append(
            SequenceInfo(
                name=name,
                root=root,
                original_split=_infer_split(root.relative_to(source)),
                match_id=mapping.get(name, name),
            )
        )
    if not sequences:
        raise ValueError("Aucune sequence MOT contenant un dossier img1 dans {}".format(source))
    duplicate_names = {
        item.name for item in sequences if sum(s.name == item.name for s in sequences) > 1
    }
    if duplicate_names:
        raise ValueError("Noms de sequences dupliques: {}".format(sorted(duplicate_names)))
    return sequences


def assign_splits(
    sequences: Sequence[SequenceInfo],
    strategy: str,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> Dict[str, str]:
    if strategy == "existing":
        missing = [sequence.name for sequence in sequences if sequence.original_split is None]
        if missing:
            raise ValueError("Split absent dans le chemin pour: {}".format(missing[:5]))
        return {sequence.name: str(sequence.original_split) for sequence in sequences}
    if strategy != "by-match":
        raise ValueError("Strategie de split inconnue: {}".format(strategy))
    if abs(sum(ratios) - 1.0) > 1e-6 or any(value < 0 for value in ratios):
        raise ValueError("Les ratios train/val/test doivent etre positifs et totaliser 1")
    match_ids = sorted({sequence.match_id for sequence in sequences})
    random.Random(seed).shuffle(match_ids)
    count = len(match_ids)
    train_end = int(round(count * ratios[0]))
    val_end = train_end + int(round(count * ratios[1]))
    # Garde au moins un match de test si le corpus le permet.
    train_end = min(train_end, count)
    val_end = min(val_end, count)
    match_splits = {}
    for index, match_id in enumerate(match_ids):
        match_splits[match_id] = (
            "train" if index < train_end else "val" if index < val_end else "test"
        )
    return {sequence.name: match_splits[sequence.match_id] for sequence in sequences}


def _read_dimensions(sequence_root: Path, image: Path) -> Tuple[int, int]:
    info_path = sequence_root / "seqinfo.ini"
    if info_path.is_file():
        parser = configparser.ConfigParser()
        parser.read(str(info_path))
        section = parser["Sequence"]
        return int(section["imWidth"]), int(section["imHeight"])
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV est requis si seqinfo.ini est absent") from exc
    frame = cv2.imread(str(image))
    if frame is None:
        raise ValueError("Image illisible: {}".format(image))
    return int(frame.shape[1]), int(frame.shape[0])


def _read_annotations(
    sequence_root: Path,
    class_map: Mapping[int, int],
) -> Tuple[DefaultDict[int, List[Tuple[int, float, float, float, float]]], int]:
    annotations: DefaultDict[int, List[Tuple[int, float, float, float, float]]] = defaultdict(list)
    ignored = 0
    gt_path = sequence_root / "gt" / "gt.txt"
    if not gt_path.is_file():
        return annotations, ignored
    with gt_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            if len(row) < 6:
                ignored += 1
                continue
            frame_id = int(float(row[0]))
            left, top, width, height = (float(value) for value in row[2:6])
            confidence = float(row[6]) if len(row) > 6 else 1.0
            source_class = int(float(row[7])) if len(row) > 7 else -1
            target_class = class_map.get(source_class)
            if target_class is None or confidence <= 0 or width <= 0 or height <= 0:
                ignored += 1
                continue
            annotations[frame_id].append((target_class, left, top, width, height))
    return annotations, ignored


def _yolo_line(
    box: Tuple[int, float, float, float, float], width: int, height: int
) -> Optional[str]:
    class_id, left, top, box_width, box_height = box
    x1 = max(0.0, min(float(width), left))
    y1 = max(0.0, min(float(height), top))
    x2 = max(0.0, min(float(width), left + box_width))
    y2 = max(0.0, min(float(height), top + box_height))
    if x2 <= x1 or y2 <= y1:
        return None
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    normalized_width = (x2 - x1) / width
    normalized_height = (y2 - y1) / height
    return "{} {:.8f} {:.8f} {:.8f} {:.8f}".format(
        class_id, cx, cy, normalized_width, normalized_height
    )


def _materialize(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    if mode == "symlink":
        destination.symlink_to(os.path.relpath(str(source), str(destination.parent)))
    elif mode == "hardlink":
        os.link(str(source), str(destination))
    elif mode == "copy":
        shutil.copy2(str(source), str(destination))
    else:
        raise ValueError("Mode de copie inconnu: {}".format(mode))


def prepare_mot_dataset(
    source: Path,
    output: Path,
    split_strategy: str = "existing",
    match_map: Optional[Path] = None,
    sequence_list: Optional[Path] = None,
    class_map: Optional[Mapping[int, int]] = None,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    link_mode: str = "symlink",
) -> ConversionStats:
    output = Path(output).expanduser().resolve()
    if split_strategy == "by-match" and match_map is None:
        raise ValueError(
            "--split-strategy by-match exige --match-map. Generez un squelette avec "
            "'referai-football create-match-map --source {} --output match_map.csv', "
            "puis renseignez un match_id pour chaque sequence.".format(source)
        )
    sequences = discover_sequences(source, match_map, sequence_list)
    if split_strategy == "by-match":
        missing_matches = [sequence.name for sequence in sequences if not sequence.match_id]
        if missing_matches:
            raise ValueError(
                "match_id vide pour {} sequence(s), par exemple: {}".format(
                    len(missing_matches), ", ".join(missing_matches[:5])
                )
            )
    splits = assign_splits(sequences, split_strategy, ratios, seed)
    class_map = dict(class_map or {-1: 0, 1: 0})
    invalid_targets = set(class_map.values()) - set(range(len(CLASS_NAMES)))
    if invalid_targets:
        raise ValueError("Classes cibles invalides: {}".format(sorted(invalid_targets)))
    stats = ConversionStats(sequences=len(sequences))
    manifest = []
    for sequence in sequences:
        split = splits[sequence.name]
        annotations, ignored = _read_annotations(sequence.root, class_map)
        stats.ignored_boxes += ignored
        images = sorted(
            path
            for path in (sequence.root / "img1").iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            continue
        width, height = _read_dimensions(sequence.root, images[0])
        for image in images:
            try:
                frame_id = int(image.stem)
            except ValueError:
                stats.ignored_boxes += len(annotations.get(0, []))
                continue
            destination = output / "images" / split / sequence.name / image.name
            _materialize(image.resolve(), destination, link_mode)
            label_path = output / "labels" / split / sequence.name / (image.stem + ".txt")
            label_path.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for box in annotations.get(frame_id, []):
                line = _yolo_line(box, width, height)
                if line is None:
                    stats.ignored_boxes += 1
                else:
                    lines.append(line)
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            stats.images += 1
            stats.boxes += len(lines)
            stats.empty_images += int(not lines)
        manifest.append(
            {"sequence": sequence.name, "match_id": sequence.match_id, "split": split}
        )
        gt_source = sequence.root / "gt" / "gt.txt"
        if gt_source.is_file():
            _materialize(
                gt_source.resolve(),
                output / "mot_gt" / split / sequence.name / "gt" / "gt.txt",
                link_mode,
            )
        seqinfo_source = sequence.root / "seqinfo.ini"
        if seqinfo_source.is_file():
            _materialize(
                seqinfo_source.resolve(),
                output / "mot_gt" / split / sequence.name / "seqinfo.ini",
                link_mode,
            )
    dataset = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "data.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "conversion_stats.json").write_text(
        json.dumps(asdict(stats), indent=2), encoding="utf-8"
    )
    seqmap_root = output / "mot_gt" / "seqmaps"
    seqmap_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        names = sorted(row["sequence"] for row in manifest if row["split"] == split)
        (seqmap_root / "{}.txt".format(split)).write_text(
            "name\n" + "\n".join(names) + ("\n" if names else ""), encoding="utf-8"
        )
    return stats


def create_oversampled_dataset(
    data_yaml: Path, class_id: int = 3, factor: int = 4, output_yaml: Optional[Path] = None
) -> Path:
    if factor < 1:
        raise ValueError("Le facteur doit etre >= 1")
    data_yaml = Path(data_yaml).expanduser().resolve()
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(config.get("path", data_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()
    train_entry = Path(config["train"])
    train_root = train_entry if train_entry.is_absolute() else dataset_root / train_entry
    images = sorted(path for path in train_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    lines: List[str] = []
    rare_count = 0
    for image in images:
        try:
            relative = image.relative_to(dataset_root / "images")
            label = dataset_root / "labels" / relative.with_suffix(".txt")
        except ValueError:
            label = Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt")
        contains_class = False
        if label.is_file():
            contains_class = any(
                row.split() and int(row.split()[0]) == class_id
                for row in label.read_text(encoding="utf-8").splitlines()
            )
        repeats = factor if contains_class else 1
        rare_count += int(contains_class)
        lines.extend([str(image.resolve())] * repeats)
    if not images:
        raise ValueError("Aucune image d'entrainement trouvee dans {}".format(train_root))
    list_path = dataset_root / "train_class{}_x{}.txt".format(class_id, factor)
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_yaml = Path(output_yaml or dataset_root / "data_oversampled.yaml").resolve()
    config["train"] = str(list_path)
    config["oversampling"] = {"class_id": class_id, "factor": factor, "source_images": rare_count}
    output_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return output_yaml


def parse_class_map(path: Optional[Path]) -> Dict[int, int]:
    if path is None:
        return {-1: 0, 1: 0}
    content = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    names = {name: index for index, name in enumerate(CLASS_NAMES)}
    result = {}
    for source, target in content.items():
        result[int(source)] = names[target] if isinstance(target, str) else int(target)
    return result


def assert_match_disjoint(manifest: Iterable[Mapping[str, str]]) -> None:
    seen: Dict[str, str] = {}
    for row in manifest:
        previous = seen.setdefault(row["match_id"], row["split"])
        if previous != row["split"]:
            raise AssertionError(
                "Fuite de match: {} est dans {} et {}".format(
                    row["match_id"], previous, row["split"]
                )
            )
