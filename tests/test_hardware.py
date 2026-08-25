import pytest

from referai.hardware import (
    inspect_gpus,
    is_accelerator_oom,
    profile_summary,
    resolve_profile,
)


class Properties:
    def __init__(self, name, memory_gb):
        self.name = name
        self.total_memory = memory_gb * 1024 ** 3


class FakeCuda:
    def __init__(self, names, capabilities, memories):
        self.names = names
        self.capabilities = capabilities
        self.memories = memories

    def is_available(self):
        return bool(self.names)

    def device_count(self):
        return len(self.names)

    def get_device_properties(self, index):
        return Properties(self.names[index], self.memories[index])

    def get_device_capability(self, index):
        return self.capabilities[index]


class FakeTorch:
    def __init__(self, cuda, xpu=None):
        self.cuda = cuda
        if xpu is not None:
            self.xpu = xpu


class FakeXpu:
    def __init__(self, names, memories):
        self.names = names
        self.memories = memories

    def is_available(self):
        return bool(self.names)

    def device_count(self):
        return len(self.names)

    def get_device_properties(self, index):
        return Properties(self.names[index], self.memories[index])

    def get_device_name(self, index):
        return self.names[index]


def test_k80_uses_three_cards_without_amp():
    torch = FakeTorch(FakeCuda(["Tesla K80"] * 4, [(3, 7)] * 4, [11] * 4))
    profile = resolve_profile({"max_devices": 3, "amp": True}, torch)
    assert profile.device_ids == (0, 1, 2)
    assert profile.batch == 3
    assert profile.amp is False
    assert profile.imgsz == 960


def test_3090ti_uses_amp_and_larger_batch():
    torch = FakeTorch(FakeCuda(["NVIDIA GeForce RTX 3090 Ti"], [(8, 6)], [24]))
    profile = resolve_profile({}, torch)
    assert profile.device == 0
    assert profile.batch == 8
    assert profile.imgsz == 1280
    assert profile.amp is True


def test_intel_xpu_is_selected_when_cuda_is_unavailable():
    torch = FakeTorch(FakeCuda([], [], []), FakeXpu(["Intel Arc Graphics"], [8]))
    profile = resolve_profile(
        {"backend": "xpu", "batch": 16, "imgsz": 224, "workers": 4}, torch
    )
    assert profile.backend == "xpu"
    assert profile.device == "xpu:0"
    assert profile.primary_device == "xpu:0"
    assert profile.batch == 16
    assert profile.imgsz == 224
    assert profile.amp is False
    assert "XPU:0:Intel Arc Graphics" in profile_summary(
        profile, inspect_gpus(torch, "xpu")
    )


def test_explicit_xpu_profile_fails_instead_of_silently_using_cpu():
    torch = FakeTorch(FakeCuda([], [], []), FakeXpu([], []))
    with pytest.raises(ValueError, match="XPU.*indisponible"):
        resolve_profile({"backend": "xpu"}, torch)


def test_level_zero_resource_error_is_treated_as_accelerator_oom():
    error = RuntimeError(
        "level_zero backend failed with error: 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)"
    )
    assert is_accelerator_oom(error)


def test_ddp_oom_reduction_preserves_one_sample_per_gpu():
    torch = FakeTorch(FakeCuda(["Tesla K80"] * 3, [(3, 7)] * 3, [11] * 3))
    profile = resolve_profile({}, torch)
    reduced = profile.with_lower_memory_pressure()
    assert reduced.batch == 3
    assert reduced.imgsz == 832


def test_cpu_fallback():
    profile = resolve_profile({}, FakeTorch(FakeCuda([], [], [])))
    assert profile.backend == "cpu"
    assert profile.device == "cpu"
    assert profile.amp is False

