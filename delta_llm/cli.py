from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from datetime import datetime, timezone

from .catalog import (
    GPU_SPECS,
    MODEL_SPECS,
    ModelSpec,
    estimate_weighted_gpu_hours,
    validate_selection,
)
from .config import Config, load_config
from .remote import (
    SSHRunner,
    ensure_interactive_terminal,
    save_local_state,
    validate_deployment_id,
    validate_username,
)
from .templates import (
    DeployParams,
    render_deploy_script,
    render_doctor_script,
    render_list_script,
    render_logs_script,
    render_status_script,
    render_stop_script,
)

EXPOSURE_MODES = ("none", "cloudflare-quick", "cloudflare-named")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="delta-llm",
        description="Deploy OpenAI-compatible vLLM endpoints on NCSA Delta.",
    )
    parser.add_argument("--config", help="TOML config path (default: ./config.toml)")
    parser.add_argument("--username", help="NCSA username (or NCSA_USERNAME env var)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("models", help="Show the curated model/GPU catalog")
    sub.add_parser("doctor", help="Check Delta account, partitions, storage, and outbound network")

    deploy = sub.add_parser("deploy", help="Deploy one vLLM service through one SSH+Duo login")
    deploy.add_argument("--model", choices=sorted(MODEL_SPECS))
    deploy.add_argument("--gpu", choices=sorted(GPU_SPECS))
    deploy.add_argument("--gpus", type=int, help="GPU count on one node")
    deploy.add_argument("--hours", type=float, help="Wall time; maximum 48")
    deploy.add_argument("--max-model-len", type=int)
    deploy.add_argument("--exposure", choices=EXPOSURE_MODES)
    deploy.add_argument("--acknowledge-external-tunnel", action="store_true")
    deploy.add_argument("--detach", action="store_true", help="Return after sbatch submission")
    deploy.add_argument("--dry-run", action="store_true", help="Validate and print the plan only")

    sub.add_parser("list", help="List your deployments on Delta")
    status = sub.add_parser("status", help="Show one deployment status")
    status.add_argument("deployment_id")
    logs = sub.add_parser("logs", help="Tail one deployment's logs")
    logs.add_argument("deployment_id")
    logs.add_argument("--lines", type=int, default=120)
    stop = sub.add_parser("stop", help="Cancel a deployment and revoke its remote API key")
    stop.add_argument("deployment_id")
    stop.add_argument("--yes", action="store_true")
    return parser


def get_username(args: argparse.Namespace) -> str:
    username = args.username or os.environ.get("NCSA_USERNAME")
    if not username:
        username = input("NCSA username: ").strip()
    return validate_username(username)


def prompt_choice(title: str, items: list[tuple[str, str]], default_key: str) -> str:
    print(f"\n{title}")
    for index, (key, label) in enumerate(items, start=1):
        marker = " (default)" if key == default_key else ""
        print(f"  {index}. {key}: {label}{marker}")
    raw = input(f"Select [default {default_key}]: ").strip()
    if not raw:
        return default_key
    if raw.isdigit() and 1 <= int(raw) <= len(items):
        return items[int(raw) - 1][0]
    keys = {key for key, _ in items}
    if raw in keys:
        return raw
    raise ValueError(f"Unknown selection: {raw}")


def recommended_gpu(model: ModelSpec) -> tuple[str, int]:
    if model.key == "deepseek-r1-32b":
        return "a40", 2
    return "a40", 1


def make_deployment_id(model_key: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{model_key}-{stamp}-{secrets.token_hex(2)}"


def print_catalog(config: Config) -> None:
    print("Curated models (VRAM numbers include a conservative KV-cache/runtime reserve):\n")
    for model in MODEL_SPECS.values():
        print(f"{model.key:18} {model.parameters:14} {model.weight_format:10}")
        print(f"  HF: {model.model_id}")
        print(f"  Conservative VRAM: {model.required_vram_gb:.0f} GB; {model.notes}")
        compatible = []
        for gpu in GPU_SPECS.values():
            for count in range(1, gpu.max_gpus + 1):
                ok, _ = validate_selection(model, gpu, count, config.gpu_memory_utilization)
                if ok:
                    compatible.append(f"{count}x {gpu.key}")
                    break
        print(f"  Minimum compatible choices: {', '.join(compatible)}\n")

    print("Delta GPU choices:\n")
    for gpu in GPU_SPECS.values():
        print(
            f"{gpu.key:8} {gpu.partition:12} {gpu.vram_gb:.0f} GB/card, "
            f"max {gpu.max_gpus}, charge factor {gpu.charge_factor:g}"
        )


def collect_deploy_params(
    args: argparse.Namespace, config: Config, username: str
) -> DeployParams:
    model_key = args.model
    if not model_key:
        model_key = prompt_choice(
            "Model",
            [(key, spec.label) for key, spec in MODEL_SPECS.items()],
            "qwen3-4b-instruct",
        )
    model = MODEL_SPECS[model_key]
    default_gpu, default_count = recommended_gpu(model)

    gpu_key = args.gpu
    if not gpu_key:
        gpu_key = prompt_choice(
            "GPU type",
            [(key, spec.label) for key, spec in GPU_SPECS.items()],
            default_gpu,
        )
    gpu = GPU_SPECS[gpu_key]

    gpu_count = args.gpus
    if gpu_count is None:
        raw = input(f"GPU count [default {default_count}]: ").strip()
        gpu_count = int(raw) if raw else default_count

    ok, reason = validate_selection(model, gpu, gpu_count, config.gpu_memory_utilization)
    if not ok:
        raise ValueError(f"Unsafe model/GPU selection: {reason}")

    hours = args.hours
    if hours is None:
        raw = input(f"Duration hours [default {config.default_hours:g}]: ").strip()
        hours = float(raw) if raw else config.default_hours
    if not 0 < hours <= 48:
        raise ValueError("Duration must be greater than 0 and no more than 48 hours")

    max_model_len = args.max_model_len or model.default_max_model_len
    if max_model_len < 256:
        raise ValueError("max-model-len must be at least 256")

    exposure = args.exposure
    if not exposure:
        exposure = prompt_choice(
            "Exposure",
            [
                ("none", "Delta internal address only; safest"),
                ("cloudflare-quick", "random public HTTPS URL; prototype only, no SSE"),
                ("cloudflare-named", "team-managed Cloudflare Tunnel and stable URL"),
            ],
            config.default_exposure,
        )

    if exposure == "cloudflare-quick" and not args.acknowledge_external_tunnel:
        print(
            "\nWARNING: Quick Tunnel sends API traffic through Cloudflare, has no SLA, "
            "does not support SSE, and must be approved for your NCSA project."
        )
        if input("Type YES to continue: ").strip() != "YES":
            raise RuntimeError("External tunnel was not acknowledged")

    cf_token = ""
    if exposure == "cloudflare-named":
        cf_token = os.environ.get("DELTA_LLM_CF_TUNNEL_TOKEN", "")
        if not cf_token:
            raise ValueError("Set DELTA_LLM_CF_TUNNEL_TOKEN for cloudflare-named mode")
        if not config.named_public_url.startswith("https://"):
            raise ValueError("Configure exposure.named_public_url with an https:// URL")

    api_key = "sk-delta-" + secrets.token_urlsafe(32)
    return DeployParams(
        username=username,
        deployment_id=make_deployment_id(model.key),
        api_key=api_key,
        model=model,
        gpu=gpu,
        gpu_count=gpu_count,
        hours=hours,
        max_model_len=max_model_len,
        exposure=exposure,
        hf_token=os.environ.get("HF_TOKEN", ""),
        cf_tunnel_token=cf_token,
        detach=bool(args.detach),
    )


def print_plan(params: DeployParams, config: Config) -> None:
    _, reason = validate_selection(
        params.model, params.gpu, params.gpu_count, config.gpu_memory_utilization
    )
    estimate = estimate_weighted_gpu_hours(params.gpu, params.gpu_count, params.hours)
    print("\nDeployment plan")
    print(f"  ID:          {params.deployment_id}")
    print(f"  Model:       {params.model.model_id}")
    print(f"  Partition:   {params.gpu.partition}")
    print(f"  GPUs:        {params.gpu_count} x {params.gpu.label}")
    print(f"  Duration:    {params.hours:g} hours")
    print(f"  Context:     {params.max_model_len} tokens")
    print(f"  Exposure:    {params.exposure}")
    print(f"  VRAM check:  {reason}")
    print(f"  Cost guide:  ~{estimate:.2f} weighted GPU-hours (estimate only)")


def run_deploy(args: argparse.Namespace, config: Config) -> int:
    username = get_username(args)
    params = collect_deploy_params(args, config, username)
    print_plan(params, config)
    if args.dry_run:
        print("\nDry run only: no SSH connection or Slurm job was created.")
        return 0

    ensure_interactive_terminal()
    print("\nYou will enter the NCSA Kerberos password and approve Duo once.")
    result = SSHRunner(username, config.login_host).run_script(
        render_deploy_script(config, params)
    )
    if result is None:
        raise RuntimeError("Remote deployment returned no structured result")

    state = {
        "deployment_id": result.deployment_id,
        "username": username,
        "job_id": result.job_id,
        "state": result.state,
        "endpoint": result.endpoint,
        "expires_at": result.expires_at,
        "model": params.model.model_id,
        "served_model": params.model.key,
        "gpu": params.gpu.key,
        "gpu_count": params.gpu_count,
        "api_key": params.api_key,
        "remote_dir": result.remote_dir,
    }
    state_path = save_local_state(
        result.deployment_id,
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
    )
    print("\nDeployment created")
    print(f"  Deployment: {result.deployment_id}")
    print(f"  Job:        {result.job_id}")
    print(f"  State:      {result.state}")
    print(f"  Base URL:   {result.endpoint}")
    print(f"  API Key:    {params.api_key}")
    print(f"  Expires:    {result.expires_at}")
    print(f"  Local state: {state_path}")
    print("\nThe API key is shown once here and stored in the protected local state file.")
    return 0


def run_remote_command(args: argparse.Namespace, config: Config) -> int:
    username = get_username(args)
    ensure_interactive_terminal()
    runner = SSHRunner(username, config.login_host)
    if args.command == "doctor":
        runner.run_script(render_doctor_script(config))
    elif args.command == "list":
        runner.run_script(render_list_script(config, username))
    elif args.command == "status":
        deployment_id = validate_deployment_id(args.deployment_id)
        runner.run_script(render_status_script(config, username, deployment_id))
    elif args.command == "logs":
        deployment_id = validate_deployment_id(args.deployment_id)
        if not 1 <= args.lines <= 5000:
            raise ValueError("--lines must be between 1 and 5000")
        runner.run_script(render_logs_script(config, username, deployment_id, args.lines))
    elif args.command == "stop":
        deployment_id = validate_deployment_id(args.deployment_id)
        if not args.yes:
            answer = input(f"Cancel {deployment_id} and revoke its API key? [y/N] ").strip()
            if answer.lower() not in {"y", "yes"}:
                print("Cancelled locally; no remote change was made.")
                return 0
        runner.run_script(render_stop_script(config, username, deployment_id))
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "models":
            print_catalog(config)
            code = 0
        elif args.command == "deploy":
            code = run_deploy(args, config)
        else:
            code = run_remote_command(args, config)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        code = 1
    raise SystemExit(code)
