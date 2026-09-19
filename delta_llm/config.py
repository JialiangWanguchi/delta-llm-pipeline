from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib


@dataclass(frozen=True)
class Config:
    login_host: str = "login.delta.ncsa.illinois.edu"
    account: str = "bhsz-delta-gpu"
    project_root: str = "/projects/bhsz"
    work_root: str = "/work/nvme/bhsz"
    default_hours: float = 47.5
    python_version: str = "3.10"
    torch_version: str = "2.5.1"
    torchvision_version: str = "0.20.1"
    transformers_version: str = "4.49.0"
    flash_attn_version: str = "2.5.8"
    cuda_wheel: str = "cu124"
    vllm_version: str = "0.20.2"
    vllm_package_version: str = "0.20.2+cu129"
    vllm_wheel_url: str = (
        "https://github.com/vllm-project/vllm/releases/download/v0.20.2/"
        "vllm-0.20.2%2Bcu129-cp38-abi3-manylinux_2_31_x86_64.whl"
    )
    vllm_wheel_sha256: str = (
        "2f8c2bf2ac6d3d16f930535e66822abd71065468521884eb5b910225b2abef4b"
    )
    vllm_torch_index_url: str = "https://download.pytorch.org/whl/cu129"
    bagel_commit: str = "a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f"
    thinkmorph_commit: str = "c1a48adfa212259c8ad79dfd9d05d87c27340cef"
    shared_root: str = "/projects/bhsz/delta-llm/shared"
    runtime_root: str = "/work/nvme/bhsz/delta-llm/shared"
    default_exposure: str = "none"
    cloudflared_url: str = (
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    )
    named_public_url: str = ""
    tailscale_version: str = "1.102.4"
    tailscale_url: str = (
        "https://pkgs.tailscale.com/stable/tailscale_1.102.4_amd64.tgz"
    )


def default_config_path() -> Path:
    return Path.cwd() / "config.toml"


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    target = Path(path) if path else default_config_path()
    if not target.exists():
        return Config()

    with target.open("rb") as handle:
        raw = tomllib.load(handle)

    delta = raw.get("delta", {})
    runtime = raw.get("runtime", {})
    exposure = raw.get("exposure", {})
    return Config(
        login_host=str(delta.get("login_host", Config.login_host)),
        account=str(delta.get("account", Config.account)),
        project_root=str(delta.get("project_root", Config.project_root)).rstrip("/"),
        work_root=str(delta.get("work_root", Config.work_root)).rstrip("/"),
        default_hours=float(delta.get("default_hours", Config.default_hours)),
        python_version=str(runtime.get("python_version", Config.python_version)),
        torch_version=str(runtime.get("torch_version", Config.torch_version)),
        torchvision_version=str(runtime.get("torchvision_version", Config.torchvision_version)),
        transformers_version=str(runtime.get("transformers_version", Config.transformers_version)),
        flash_attn_version=str(runtime.get("flash_attn_version", Config.flash_attn_version)),
        cuda_wheel=str(runtime.get("cuda_wheel", Config.cuda_wheel)),
        vllm_version=str(runtime.get("vllm_version", Config.vllm_version)),
        vllm_package_version=str(
            runtime.get("vllm_package_version", Config.vllm_package_version)
        ),
        vllm_wheel_url=str(runtime.get("vllm_wheel_url", Config.vllm_wheel_url)),
        vllm_wheel_sha256=str(
            runtime.get("vllm_wheel_sha256", Config.vllm_wheel_sha256)
        ),
        vllm_torch_index_url=str(
            runtime.get("vllm_torch_index_url", Config.vllm_torch_index_url)
        ),
        bagel_commit=str(runtime.get("bagel_commit", Config.bagel_commit)),
        thinkmorph_commit=str(runtime.get("thinkmorph_commit", Config.thinkmorph_commit)),
        shared_root=str(runtime.get("shared_root", Config.shared_root)).rstrip("/"),
        runtime_root=str(runtime.get("runtime_root", Config.runtime_root)).rstrip("/"),
        default_exposure=str(exposure.get("default_mode", Config.default_exposure)),
        cloudflared_url=str(exposure.get("cloudflared_url", Config.cloudflared_url)),
        named_public_url=str(exposure.get("named_public_url", "")).rstrip("/"),
        tailscale_version=str(
            exposure.get("tailscale_version", Config.tailscale_version)
        ),
        tailscale_url=str(exposure.get("tailscale_url", Config.tailscale_url)),
    )
