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
        assigned_gpus=1,
        capabilities=(
            "text-to-image",
            "image-edit",
            "image-understanding",
            "interleaved-reasoning",
        ),
    ),
}


GPU_SPECS: dict[str, GPUSpec] = {
    "a100": GPUSpec(
        key="a100",
        label="NVIDIA A100 40 GB",
        partition="gpuA100x4",
        vram_gb=40,
        max_gpus=4,
        charge_factor=1.0,
    ),
    "h200": GPUSpec(
        key="h200",
        label="NVIDIA H200 141 GB",
        partition="gpuH200x8",
        vram_gb=141,
        max_gpus=8,
        charge_factor=3.0,
    ),
}


def validate_gpu_count(gpu_count: int) -> tuple[bool, str]:
    if gpu_count != 4:
        return False, "双模型服务固定使用 4 张 A100"
    return True, "A100-0/1: BAGEL 各1个副本；A100-2/3: ThinkMorph 各1个副本"


def estimate_weighted_gpu_hours(gpu_count: int, hours: float) -> float:
    return gpu_count * hours * GPU_SPECS["a100"].charge_factor
