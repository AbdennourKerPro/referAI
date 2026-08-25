"""Selection sure du profil de calcul pour CUDA, Intel XPU et CPU."""

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
    backend: str = "cuda"


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    backend: str
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
        if self.backend == "xpu":
            indices = ",".join(str(index) for index in self.device_ids)
            return "xpu:{}".format(indices)
        if len(self.device_ids) == 1:
            return self.device_ids[0]
        return list(self.device_ids)

    @property
    def primary_device(self) -> Any:
        """Return one accelerator for sequential inference and validation."""
        if not self.device_ids:
            return "cpu"
        if self.backend == "xpu":
            return "xpu:{}".format(self.device_ids[0])
        return self.device_ids[0]

    def with_lower_memory_pressure(self) -> "RuntimeProfile":
        gpu_count = max(1, len(self.device_ids))
        if self.batch >= gpu_count * 2:
            reduced = max(gpu_count, (self.batch // 2 // gpu_count) * gpu_count)
            return replace(self, batch=reduced)
        if self.imgsz > self.min_imgsz:
            return replace(self, imgsz=max(self.min_imgsz, self.imgsz - 128))
        return self


def inspect_gpus(torch_module: Optional[Any] = None, backend: str = "auto") -> List[GPUInfo]:
    """Inspect CUDA or Intel XPU devices, preferring CUDA in automatic mode."""
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore
        except ImportError:
            return []
    requested = str(backend or "auto").lower()
    if requested not in {"auto", "cuda", "xpu", "cpu"}:
        raise ValueError("Backend materiel invalide: {}".format(requested))
    if requested == "cpu":
        return []
    candidates = ("cuda", "xpu") if requested == "auto" else (requested,)
    for candidate in candidates:
        device_module = getattr(torch_module, candidate, None)
        if device_module is None or not device_module.is_available():
            continue
        devices = []
        for index in range(device_module.device_count()):
            try:
                props = device_module.get_device_properties(index)
            except (AttributeError, RuntimeError):
                props = None
            if candidate == "cuda":
                name = str(props.name)
                capability = tuple(device_module.get_device_capability(index))
            else:
                try:
                    name = str(device_module.get_device_name(index))
                except (AttributeError, RuntimeError):
                    name = str(getattr(props, "name", "Intel XPU {}".format(index)))
                capability = (0, 0)
            total_memory = int(getattr(props, "total_memory", 0) or 0)
            devices.append(
                GPUInfo(
                    index=index,
                    name=name,
                    memory_mb=total_memory // (1024 * 1024),
                    capability=capability,
                    backend=candidate,
                )
            )
        return devices
    return []


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
    requested_backend = str(config.get("backend", "auto")).lower()
    available = inspect_gpus(torch_module, requested_backend)
    if not available:
        if requested_backend not in {"auto", "cpu"}:
            raise ValueError(
                "Backend {} demande mais indisponible dans PyTorch".format(
                    requested_backend.upper()
                )
            )
        return RuntimeProfile(
            name="cpu",
            backend="cpu",
            device_ids=(),
            imgsz=int(config.get("imgsz", 640)),
            batch=int(config.get("batch", 1)),
            amp=False,
            half=False,
            workers=int(config.get("workers", 2)),
            max_oom_retries=0,
            min_imgsz=int(config.get("min_imgsz", 640)),
        )

    backend = available[0].backend
    device_ids = _configured_ids(config, available)
    selected = [gpu for gpu in available if gpu.index in device_ids]
    is_k80 = backend == "cuda" and bool(selected) and all(
        "K80" in gpu.name.upper() for gpu in selected
    )
    is_legacy = backend == "cuda" and any(gpu.capability[0] < 7 for gpu in selected)
    default_name = "intel_xpu" if backend == "xpu" else "k80" if is_k80 else "modern_cuda"
    defaults = {
        "k80": {"imgsz": 960, "batch_per_gpu": 1, "workers": 3, "amp": False},
        "modern_cuda": {"imgsz": 1280, "batch_per_gpu": 8, "workers": 8, "amp": True},
        "intel_xpu": {"imgsz": 640, "batch_per_gpu": 8, "workers": 4, "amp": False},
    }[default_name]
    gpu_count = len(device_ids)
    batch_per_device = int(
        config.get("batch_per_device", config.get("batch_per_gpu", defaults["batch_per_gpu"]))
    )
    batch = int(
        config.get(
            "batch", batch_per_device * gpu_count
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
        backend=backend,
        device_ids=device_ids,
        imgsz=int(config.get("imgsz", defaults["imgsz"])),
        batch=batch,
        amp=amp,
        half=amp,
        workers=int(config.get("workers", defaults["workers"])),
        max_oom_retries=int(config.get("max_oom_retries", 3)),
        min_imgsz=int(config.get("min_imgsz", 640)),
    )


def is_accelerator_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "out of memory",
            "cudnn_status_alloc_failed",
            "xpu out of memory",
            "ur_result_error_out_of_resources",
            "ur_result_error_out_of_device_memory",
            "ze_result_error_out_of_device_memory",
            "ur_result_error_mem_object_allocation_failure",
        )
    )


# Backward-compatible name used by older integrations.
is_cuda_oom = is_accelerator_oom


def empty_accelerator_cache(backend: str, torch_module: Optional[Any] = None) -> None:
    """Release cached accelerator memory when the selected backend supports it."""
    if backend == "cpu":
        return
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore
        except ImportError:
            return
    device_module = getattr(torch_module, backend, None)
    empty_cache = getattr(device_module, "empty_cache", None)
    if empty_cache is not None:
        empty_cache()


def reset_peak_memory_stats(
    backend: str, device: Any, torch_module: Optional[Any] = None
) -> None:
    """Reset peak memory statistics without assuming a CUDA backend."""
    if backend == "cpu":
        return
    if torch_module is None:
        import torch as torch_module  # type: ignore
    reset = getattr(getattr(torch_module, backend, None), "reset_peak_memory_stats", None)
    if reset is not None:
        reset(device)


def peak_memory_allocated_mb(
    backend: str, device: Any, torch_module: Optional[Any] = None
) -> Optional[float]:
    """Read peak allocated memory for CUDA/XPU when the runtime exposes it."""
    if backend == "cpu":
        return None
    if torch_module is None:
        import torch as torch_module  # type: ignore
    maximum = getattr(getattr(torch_module, backend, None), "max_memory_allocated", None)
    if maximum is None:
        return None
    return float(maximum(device)) / (1024 * 1024)


def profile_summary(profile: RuntimeProfile, gpus: Sequence[GPUInfo]) -> str:
    descriptions = []
    for gpu in gpus:
        if gpu.backend != profile.backend or gpu.index not in profile.device_ids:
            continue
        memory = "{} MiB".format(gpu.memory_mb) if gpu.memory_mb else "memoire partagee"
        suffix = ", sm_{}{}".format(*gpu.capability) if gpu.backend == "cuda" else ""
        descriptions.append(
            "{}:{}:{} ({}{})".format(
                gpu.backend.upper(), gpu.index, gpu.name, memory, suffix
            )
        )
    devices = ", ".join(descriptions) or "CPU"
    return (
        "profil={} backend={} devices=[{}] imgsz={} batch_global={} amp={} workers={}".format(
            profile.name,
            profile.backend,
            devices,
            profile.imgsz,
            profile.batch,
            profile.amp,
            profile.workers,
        )
    )
