from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPUSpec:
    key: str
    label: str
    partition: str
    vram_gb: float
    max_gpus: int
    charge_factor: float
    host_memory_gb_per_gpu: int


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    model_id: str
    parameters: str
    weight_format: str
    required_vram_gb: float
    default_max_model_len: int
    notes: str
    extra_args: tuple[str, ...] = ()


GPU_SPECS: dict[str, GPUSpec] = {
    "a40": GPUSpec(
        key="a40",
        label="NVIDIA A40 48 GB (低成本首选)",
        partition="gpuA40x4",
        vram_gb=48,
        max_gpus=4,
        charge_factor=0.5,
        host_memory_gb_per_gpu=56,
    ),
    "a100": GPUSpec(
        key="a100",
        label="NVIDIA A100 40 GB x4 节点",
        partition="gpuA100x4",
        vram_gb=40,
        max_gpus=4,
        charge_factor=1.0,
        host_memory_gb_per_gpu=56,
    ),
    "a100x8": GPUSpec(
        key="a100x8",
        label="NVIDIA A100 40 GB x8 大内存节点",
        partition="gpuA100x8",
        vram_gb=40,
        max_gpus=8,
        charge_factor=1.5,
        host_memory_gb_per_gpu=120,
    ),
    "h200": GPUSpec(
        key="h200",
        label="NVIDIA H200 141 GB (昂贵)",
        partition="gpuH200x8",
        vram_gb=141,
        max_gpus=8,
        charge_factor=3.0,
        host_memory_gb_per_gpu=120,
    ),
}


MODEL_SPECS: dict[str, ModelSpec] = {
    "qwen3-8b": ModelSpec(
        key="qwen3-8b",
        label="Qwen3 8B（入门/吞吐优先）",
        model_id="Qwen/Qwen3-8B",
        parameters="8.2B",
        weight_format="BF16",
        required_vram_gb=24,
        default_max_model_len=32768,
        notes="单张 A40/A100 即可；Apache-2.0。",
    ),
    "qwen2.5-14b": ModelSpec(
        key="qwen2.5-14b",
        label="Qwen2.5 14B Instruct（效果/成本平衡）",
        model_id="Qwen/Qwen2.5-14B-Instruct",
        parameters="14.7B",
        weight_format="BF16",
        required_vram_gb=34,
        default_max_model_len=8192,
        notes="单张 A40/A100；为 KV cache 保守限制到 8K。Apache-2.0。",
    ),
    "qwen3-32b-awq": ModelSpec(
        key="qwen3-32b-awq",
        label="Qwen3 32B AWQ（单卡大模型）",
        model_id="Qwen/Qwen3-32B-AWQ",
        parameters="32.8B, 4-bit",
        weight_format="AWQ INT4",
        required_vram_gb=32,
        default_max_model_len=16384,
        notes="单张 A40/A100；量化模型，显存友好。Apache-2.0。",
        extra_args=("--quantization", "awq"),
    ),
    "deepseek-r1-32b": ModelSpec(
        key="deepseek-r1-32b",
        label="DeepSeek R1 Distill Qwen 32B（推理实验）",
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        parameters="32.8B",
        weight_format="BF16",
        required_vram_gb=68,
        default_max_model_len=16384,
        notes="建议 2x A40/A100，或 1x H200；MIT/上游 Qwen 许可。",
    ),
}


def available_vram_gb(gpu_key: str, gpu_count: int, utilization: float = 0.90) -> float:
    gpu = GPU_SPECS[gpu_key]
    return gpu.vram_gb * gpu_count * utilization


def validate_selection(
    model: ModelSpec,
    gpu: GPUSpec,
    gpu_count: int,
    utilization: float = 0.90,
) -> tuple[bool, str]:
    if gpu_count < 1 or gpu_count > gpu.max_gpus:
        return False, f"{gpu.partition} 单节点仅支持 1-{gpu.max_gpus} 张 GPU"
    usable = available_vram_gb(gpu.key, gpu_count, utilization)
    if usable < model.required_vram_gb:
        return (
            False,
            (
                f"预计可用显存 {usable:.1f} GB，小于模型保守需求 "
                f"{model.required_vram_gb:.1f} GB"
            ),
        )
    return True, f"预计可用显存 {usable:.1f} GB，满足保守需求"


def estimate_weighted_gpu_hours(gpu: GPUSpec, gpu_count: int, hours: float) -> float:
    return gpu_count * hours * gpu.charge_factor
