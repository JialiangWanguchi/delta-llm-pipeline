from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    model_id: str
    repository: str
    checkpoint_gb: float
    assigned_gpus: int
    capabilities: tuple[str, ...]


@dataclass(frozen=True)
class GPUSpec:
    key: str
    label: str
    partition: str
    vram_gb: float
    max_gpus: int
    charge_factor: float


MODEL_SPECS: dict[str, ModelSpec] = {
    "bagel-7b": ModelSpec(
        key="bagel-7b",
        label="BAGEL-7B-MoT",
        model_id="ByteDance-Seed/BAGEL-7B-MoT",
        repository="https://github.com/ByteDance-Seed/BAGEL.git",
        checkpoint_gb=29.6,
        assigned_gpus=1,
        capabilities=("text-to-image", "image-edit", "image-understanding"),
    ),
    "thinkmorph-7b": ModelSpec(
        key="thinkmorph-7b",
        label="ThinkMorph-7B",
        model_id="ThinkMorph/ThinkMorph-7B",
        repository="https://github.com/ThinkMorph/ThinkMorph.git",
        checkpoint_gb=29.6,
        assigned_gpus=2,
        capabilities=(
            "text-to-image",
            "image-edit",
            "image-understanding",
            "interleaved-reasoning",
        ),
    ),
}


GPU_SPECS: dict[str, GPUSpec] = {
    "a40": GPUSpec(
        key="a40",
        label="NVIDIA A40 48 GB",
        partition="gpuA40x4",
        vram_gb=48,
        max_gpus=4,
        charge_factor=0.5,
    )
}


def validate_gpu_count(gpu_count: int) -> tuple[bool, str]:
    if gpu_count not in {3, 4}:
        return False, "双模型服务需要 3 张 A40；可使用第 4 张卡增加余量"
    layout = "BAGEL 1 张 + ThinkMorph 2 张"
    if gpu_count == 4:
        layout += " + 1 张预留卡"
    return True, layout


def estimate_weighted_gpu_hours(gpu_count: int, hours: float) -> float:
    return gpu_count * hours * GPU_SPECS["a40"].charge_factor
