from pathlib import Path

import cv2
import numpy as np
import yaml

from referai.augment import augment_training_dataset


def test_offline_augmentation_keeps_label(tmp_path: Path):
    root = tmp_path / "dataset"
    image_path = root / "images" / "train" / "seq" / "000001.jpg"
    label_path = root / "labels" / "train" / "seq" / "000001.txt"
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    cv2.imwrite(str(image_path), np.full((40, 80, 3), 120, dtype=np.uint8))
    label_path.write_text("3 0.5 0.5 0.1 0.1\n")
    data = root / "data.yaml"
    data.write_text(
        yaml.safe_dump({"path": str(root), "train": "images/train", "val": "images/val"})
    )
    stats = augment_training_dataset(data, copies=1, seed=7)
    augmented_image = root / "images" / "train" / "seq" / "000001__aug01.jpg"
    augmented_label = root / "labels" / "train" / "seq" / "000001__aug01.txt"
    assert stats.generated_images == 1
    assert augmented_image.is_file()
    assert augmented_label.read_text() == label_path.read_text()
