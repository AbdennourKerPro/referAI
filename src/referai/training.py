"""Fine-tuning YOLO avec reprise et degradation controlee en cas d'OOM."""

import gc
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from .config import load_yaml, resolve_relative_paths
from .hardware import (
    empty_accelerator_cache,
    inspect_gpus,
    is_accelerator_oom,
    profile_summary,
    resolve_profile,
)

LOGGER = logging.getLogger(__name__)
MIN_XPU_ULTRALYTICS = (8, 4, 67)


def _version_tuple(value: str) -> tuple[int, int, int]:
    parts = []
    for component in value.split(".")[:3]:
        digits = "".join(character for character in component if character.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple((parts + [0, 0, 0])[:3])


def _import_yolo(backend: str = "cpu") -> Any:
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics absent. Installez requirements_XPU.txt, requirements_K80.txt "
            "ou requirements_3090Ti.txt selon votre materiel."
        ) from exc
    installed = _version_tuple(str(getattr(ultralytics, "__version__", "0")))
    if backend == "xpu" and installed < MIN_XPU_ULTRALYTICS:
        minimum = ".".join(str(value) for value in MIN_XPU_ULTRALYTICS)
        raise RuntimeError(
            "Le backend XPU requiert Ultralytics >= {} (version installee: {}). "
            "Lancez: python -m pip install -r requirements_XPU.txt".format(
                minimum, getattr(ultralytics, "__version__", "inconnue")
            )
        )
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
    LOGGER.info(profile_summary(profile, inspect_gpus(backend=profile.backend)))
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
    YOLO = _import_yolo(profile.backend)
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
        model = YOLO(model_name)
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
            if (not is_accelerator_oom(exc) and not opaque_ddp_failure) or retries_exhausted:
                raise
            reduced = current.with_lower_memory_pressure()
            if reduced == current:
                raise
            LOGGER.warning(
                "Memoire %s insuffisante; nouvel essai avec imgsz=%s batch=%s",
                current.backend.upper(),
                reduced.imgsz,
                reduced.batch,
            )
            current = reduced
        # Le trainer Ultralytics retient le modele et l'optimiseur. Il doit etre
        # detruit avant empty_cache pour liberer les allocations de la tentative.
        del model
        gc.collect()
        try:
            empty_accelerator_cache(current.backend)
        except (ImportError, RuntimeError):
            pass


def validate_detector(
    weights: Path, data: Path, hardware: Optional[Dict[str, Any]] = None, split: str = "val"
) -> Any:
    profile = resolve_profile(hardware)
    YOLO = _import_yolo(profile.backend)
    model = YOLO(str(weights))
    validation_device = profile.primary_device
    validation_batch = max(1, profile.batch // max(1, len(profile.device_ids)))
    return model.val(
        data=str(data),
        split=split,
        device=validation_device,
        imgsz=profile.imgsz,
        batch=validation_batch,
        half=profile.half,
    )
