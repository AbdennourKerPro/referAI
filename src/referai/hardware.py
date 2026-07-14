"""Selection sure du profil de calcul pour Kepler K80, Ampere et CPU."""

import logging
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    memory_mb: int
    capability: Tuple[int, int]


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    device_ids: Tuple[int, ...]
    imgsz: int
    batch: int
    amp: bool
    half: bool
    workers: int
    max_oom_retries: int = 3
    min_imgsz: int = 640

    @property
    def device(self) -> Any:
        if not self.device_ids:
            return "cpu"
        if len(self.device_ids) == 1:
            return self.device_ids[0]
        return list(self.device_ids)

    def with_lower_memory_pressure(self) -> "RuntimeProfile":
        gpu_count = max(1, len(self.device_ids))
        if self.batch >= gpu_count * 2:
            reduced = max(gpu_count, (self.batch // 2 // gpu_count) * gpu_count)
            return replace(self, batch=reduced)
        if self.imgsz > self.min_imgsz:
            return replace(self, imgsz=max(self.min_imgsz, self.imgsz - 128))
        return self


def inspect_gpus(torch_module: Optional[Any] = None) -> List[GPUInfo]:
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore
        except ImportError:
            return []
    if not torch_module.cuda.is_available():
        return []
    devices = []
    for index in range(torch_module.cuda.device_count()):
        props = torch_module.cuda.get_device_properties(index)
        devices.append(
            GPUInfo(
                index=index,
                name=str(props.name),
                memory_mb=int(props.total_memory // (1024 * 1024)),
                capability=tuple(torch_module.cuda.get_device_capability(index)),
            )
        )
    return devices


def _configured_ids(config: Dict[str, Any], available: Sequence[GPUInfo]) -> Tuple[int, ...]:
    requested = config.get("devices", "auto")
    max_devices = int(config.get("max_devices", len(available) or 1))
    available_ids = {gpu.index for gpu in available}
    if requested == "auto" or requested is None:
        return tuple(gpu.index for gpu in available[:max_devices])
    if isinstance(requested, int):
        requested = [requested]
    if isinstance(requested, str):
        requested = [int(value.strip()) for value in requested.split(",") if value.strip()]
    selected = tuple(int(value) for value in requested[:max_devices])
    missing = set(selected) - available_ids
    if missing:
        raise ValueError("GPU demandes mais indisponibles: {}".format(sorted(missing)))
    return selected


def resolve_profile(
    config: Optional[Dict[str, Any]] = None, torch_module: Any = None
) -> RuntimeProfile:
    config = dict(config or {})
    available = inspect_gpus(torch_module)
    if not available:
        return RuntimeProfile(
            name="cpu",
            device_ids=(),
            imgsz=int(config.get("imgsz", 640)),
            batch=int(config.get("batch", 1)),
            amp=False,
            half=False,
            workers=int(config.get("workers", 2)),
            max_oom_retries=0,
            min_imgsz=int(config.get("min_imgsz", 640)),
        )

    device_ids = _configured_ids(config, available)
    selected = [gpu for gpu in available if gpu.index in device_ids]
    is_k80 = bool(selected) and all("K80" in gpu.name.upper() for gpu in selected)
    is_legacy = any(gpu.capability[0] < 7 for gpu in selected)
    default_name = "k80" if is_k80 else "modern_cuda"
    defaults = {
        "k80": {"imgsz": 960, "batch_per_gpu": 1, "workers": 3, "amp": False},
        "modern_cuda": {"imgsz": 1280, "batch_per_gpu": 8, "workers": 8, "amp": True},
    }[default_name]
    gpu_count = len(device_ids)
    batch = int(
        config.get(
            "batch", int(config.get("batch_per_gpu", defaults["batch_per_gpu"])) * gpu_count
        )
    )
    if gpu_count > 1 and batch % gpu_count:
        adjusted = max(gpu_count, (batch // gpu_count) * gpu_count)
        LOGGER.warning(
            "Batch %s non divisible par %s GPU; ajustement a %s", batch, gpu_count, adjusted
        )
        batch = adjusted
    amp_requested = bool(config.get("amp", defaults["amp"]))
    amp = amp_requested and not is_legacy
    if amp_requested and is_legacy:
        LOGGER.warning(
            "AMP desactive sur GPU Kepler/legacy pour eviter les kernels FP16 incompatibles."
        )
    return RuntimeProfile(
        name=str(config.get("name", default_name)),
        device_ids=device_ids,
        imgsz=int(config.get("imgsz", defaults["imgsz"])),
        batch=batch,
        amp=amp,
        half=amp,
        workers=int(config.get("workers", defaults["workers"])),
        max_oom_retries=int(config.get("max_oom_retries", 3)),
        min_imgsz=int(config.get("min_imgsz", 640)),
    )


def is_cuda_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cudnn_status_alloc_failed" in message


def profile_summary(profile: RuntimeProfile, gpus: Sequence[GPUInfo]) -> str:
    devices = ", ".join(
        "{}:{} ({} MiB, sm_{}{})".format(g.index, g.name, g.memory_mb, *g.capability)
        for g in gpus
        if g.index in profile.device_ids
    ) or "CPU"
    return (
        "profil={} devices=[{}] imgsz={} batch_global={} amp={} workers={}".format(
            profile.name, devices, profile.imgsz, profile.batch, profile.amp, profile.workers
        )
    )
