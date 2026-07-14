"""Augmentations photometriques hors ligne propres aux videos de football."""

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from .data import IMAGE_SUFFIXES


@dataclass(frozen=True)
class AugmentationStats:
    source_images: int
    generated_images: int


def _label_for(image: Path, dataset_root: Path) -> Path:
    try:
        relative = image.relative_to(dataset_root / "images")
        return dataset_root / "labels" / relative.with_suffix(".txt")
    except ValueError:
        return Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt")


def _motion_blur(image, cv2, np, rng: random.Random):
    size = rng.choice((3, 5, 7, 9))
    kernel = np.zeros((size, size), dtype=np.float32)
    direction = rng.choice(("horizontal", "vertical", "diagonal"))
    if direction == "horizontal":
        kernel[size // 2, :] = 1.0
    elif direction == "vertical":
        kernel[:, size // 2] = 1.0
    else:
        np.fill_diagonal(kernel, 1.0)
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)


def _augment_image(image, cv2, np, rng: random.Random):
    alpha = rng.uniform(0.75, 1.25)
    beta = rng.uniform(-24.0, 24.0)
    augmented = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    hsv = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= rng.uniform(0.70, 1.30)
    hsv[:, :, 2] *= rng.uniform(0.85, 1.15)
    augmented = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < 0.65:
        augmented = _motion_blur(augmented, cv2, np, rng)
    if rng.random() < 0.55:
        height, width = augmented.shape[:2]
        factor = rng.uniform(0.45, 0.80)
        reduced = cv2.resize(
            augmented,
            (max(1, int(width * factor)), max(1, int(height * factor))),
            interpolation=cv2.INTER_AREA,
        )
        augmented = cv2.resize(reduced, (width, height), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.45:
        height, width = augmented.shape[:2]
        for _ in range(rng.randint(1, 3)):
            box_width = rng.randint(max(2, width // 80), max(3, width // 18))
            box_height = rng.randint(max(2, height // 80), max(3, height // 12))
            left = rng.randint(0, max(0, width - box_width))
            top = rng.randint(0, max(0, height - box_height))
            color = tuple(int(value) for value in augmented[top, left])
            cv2.rectangle(
                augmented,
                (left, top),
                (left + box_width, top + box_height),
                color,
                thickness=-1,
            )
    if rng.random() < 0.75:
        quality = rng.randint(35, 85)
        ok, encoded = cv2.imencode(".jpg", augmented, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            augmented = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return augmented


def augment_training_dataset(
    data_yaml: Path,
    copies: int = 1,
    seed: int = 42,
    max_images: Optional[int] = None,
) -> AugmentationStats:
    if copies < 1:
        raise ValueError("copies doit etre >= 1")
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenCV et NumPy sont requis pour les augmentations") from exc
    data_yaml = Path(data_yaml).expanduser().resolve()
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(config.get("path", data_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()
    train_entry = Path(config["train"])
    if train_entry.suffix.lower() == ".txt":
        raise ValueError(
            "Augmenter le data.yaml original avant de creer la liste surechantillonnee"
        )
    train_root = train_entry if train_entry.is_absolute() else dataset_root / train_entry
    images = sorted(
        path
        for path in train_root.rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES and "__aug" not in path.stem
    )
    if max_images is not None:
        images = images[:max_images]
    rng = random.Random(seed)
    generated = 0
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        source_label = _label_for(image_path, dataset_root)
        for copy_index in range(copies):
            suffix = "__aug{:02d}".format(copy_index + 1)
            destination_image = image_path.with_name(image_path.stem + suffix + ".jpg")
            destination_label = _label_for(destination_image, dataset_root)
            augmented = _augment_image(image, cv2, np, rng)
            destination_image.parent.mkdir(parents=True, exist_ok=True)
            destination_label.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination_image), augmented):
                raise IOError("Impossible d'ecrire {}".format(destination_image))
            if source_label.is_file():
                shutil.copy2(str(source_label), str(destination_label))
            else:
                destination_label.write_text("", encoding="utf-8")
            generated += 1
    return AugmentationStats(source_images=len(images), generated_images=generated)
