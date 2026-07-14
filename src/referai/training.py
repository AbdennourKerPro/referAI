"""Fine-tuning YOLO avec reprise et degradation controlee en cas d'OOM."""

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from .config import load_yaml, resolve_relative_paths
from .hardware import RuntimeProfile, inspect_gpus, is_cuda_oom, profile_summary, resolve_profile

LOGGER = logging.getLogger(__name__)


def _import_yolo() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics absent. Installez requirements_K80.txt ou requirements_3090Ti.txt."
        ) from exc
    return YOLO


def train_detector(
    config_path: Path,
    hardware_override: Optional[Dict[str, Any]] = None,
    resume: bool = False,
) -> Any:
    config = resolve_relative_paths(load_yaml(config_path), config_path)
    hardware_config = dict(config.pop("hardware", {}))
    hardware_config.update(hardware_override or {})
    profile = resolve_profile(hardware_config)
    LOGGER.info(profile_summary(profile, inspect_gpus()))
    model_name = config.pop("model", "yolo11m.pt")
    data = config.pop("data", None)
    if not data:
        raise ValueError("Le champ 'data' est obligatoire dans la configuration d'entrainement")
    train_args = dict(config.pop("train", {}))
    if config:
        raise ValueError("Cles de configuration inconnues: {}".format(sorted(config)))
    train_args.update(
        {
            "data": str(data),
            "device": profile.device,
            "imgsz": profile.imgsz,
            "batch": profile.batch,
            "amp": profile.amp,
            "workers": profile.workers,
            "resume": resume,
        }
    )
    YOLO = _import_yolo()
    model = YOLO(model_name)
    current = profile
    for attempt in range(profile.max_oom_retries + 1):
        train_args.update(
            {
                "device": current.device,
                "imgsz": current.imgsz,
                "batch": current.batch,
                "amp": current.amp,
                "workers": current.workers,
            }
        )
        try:
            LOGGER.info(
                "Entrainement tentative %s/%s: imgsz=%s batch=%s devices=%s amp=%s",
                attempt + 1,
                profile.max_oom_retries + 1,
                current.imgsz,
                current.batch,
                current.device,
                current.amp,
            )
            return model.train(**train_args)
        except (RuntimeError, MemoryError, subprocess.CalledProcessError) as exc:
            # Ultralytics remonte parfois l'OOM DDP comme CalledProcessError sans
            # retranscrire stderr dans l'exception. Dans ce seul cas multi-GPU,
            # une degradation prudente est tentee; l'erreur finale reste propagee.
            opaque_ddp_failure = (
                isinstance(exc, subprocess.CalledProcessError) and len(current.device_ids) > 1
            )
            retries_exhausted = attempt >= profile.max_oom_retries
            if (not is_cuda_oom(exc) and not opaque_ddp_failure) or retries_exhausted:
                raise
            reduced = current.with_lower_memory_pressure()
            if reduced == current:
                raise
            LOGGER.warning(
                "Memoire CUDA insuffisante; nouvel essai avec imgsz=%s batch=%s",
                reduced.imgsz,
                reduced.batch,
            )
            current = reduced
            try:
                import torch

                torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass


def validate_detector(
    weights: Path, data: Path, hardware: Optional[Dict[str, Any]] = None, split: str = "val"
) -> Any:
    profile = resolve_profile(hardware)
    YOLO = _import_yolo()
    model = YOLO(str(weights))
    validation_device = profile.device_ids[0] if profile.device_ids else "cpu"
    validation_batch = max(1, profile.batch // max(1, len(profile.device_ids)))
    return model.val(
        data=str(data),
        split=split,
        device=validation_device,
        imgsz=profile.imgsz,
        batch=validation_batch,
        half=profile.half,
    )
