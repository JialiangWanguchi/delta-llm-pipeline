from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    login_host: str = "login.delta.ncsa.illinois.edu"
    account: str = "bhsz-delta-gpu"
    project_root: str = "/projects/bhsz"
    work_root: str = "/work/nvme/bhsz"
    default_hours: float = 47.5
    vllm_version: str = "0.20.0"
    cuda_wheel: str = "cu128"
    shared_root: str = "/projects/bhsz/delta-llm/shared"
    gpu_memory_utilization: float = 0.90
    default_exposure: str = "none"
    cloudflared_url: str = (
        "https://github.com/cloudflare/cloudflared/releases/latest/download/"
        "cloudflared-linux-amd64"
    )
    named_public_url: str = ""


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
        vllm_version=str(runtime.get("vllm_version", Config.vllm_version)),
        cuda_wheel=str(runtime.get("cuda_wheel", Config.cuda_wheel)),
        shared_root=str(runtime.get("shared_root", Config.shared_root)).rstrip("/"),
        gpu_memory_utilization=float(
            runtime.get("gpu_memory_utilization", Config.gpu_memory_utilization)
        ),
        default_exposure=str(exposure.get("default_mode", Config.default_exposure)),
        cloudflared_url=str(exposure.get("cloudflared_url", Config.cloudflared_url)),
        named_public_url=str(exposure.get("named_public_url", "")).rstrip("/"),
    )
