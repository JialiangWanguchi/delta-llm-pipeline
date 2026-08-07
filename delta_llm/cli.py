from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from datetime import datetime, timezone

from .catalog import MODEL_SPECS, estimate_weighted_gpu_hours, validate_gpu_count
from .config import Config, load_config
from .remote import (
    SSHRunner,
    ensure_interactive_terminal,
    mark_local_state_stopped,
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
        prog="delta-multimodal",
        description="Deploy BAGEL-7B and ThinkMorph-7B behind one API key on NCSA Delta.",
    )
    parser.add_argument("--config", help="TOML config path (default: ./config.toml)")
    parser.add_argument("--username", help="NCSA username (or NCSA_USERNAME env var)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models", help="Show the two bundled models and API capabilities")
    sub.add_parser("doctor", help="Check Delta account, A40 queue, storage, and network")

    deploy = sub.add_parser("deploy", help="Deploy both models through one SSH+Duo login")
    deploy.add_argument("--gpus", type=int, choices=(3, 4), default=3)
    deploy.add_argument("--hours", type=float, help="Wall time; maximum 48")
    deploy.add_argument("--exposure", choices=EXPOSURE_MODES)
    deploy.add_argument("--acknowledge-external-tunnel", action="store_true")
    deploy.add_argument("--detach", action="store_true", help="Return after sbatch submission")
    deploy.add_argument("--dry-run", action="store_true", help="Validate without SSH or Slurm")

    sub.add_parser("list", help="List your deployments on Delta")
    status = sub.add_parser("status", help="Show one deployment status")
    status.add_argument("deployment_id")
    logs = sub.add_parser("logs", help="Tail BAGEL, ThinkMorph, gateway, and tunnel logs")
    logs.add_argument("deployment_id")
    logs.add_argument("--lines", type=int, default=120)
    stop = sub.add_parser("stop", help="Stop both models and revoke the shared API key")
    stop.add_argument("deployment_id")
    stop.add_argument("--yes", action="store_true")
    return parser


def get_username(args: argparse.Namespace) -> str:
    username = args.username or os.environ.get("NCSA_USERNAME")
    if not username:
        username = input("NCSA username: ").strip()
    return validate_username(username)


def make_deployment_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"bagel-thinkmorph-{stamp}-{secrets.token_hex(2)}"


def print_catalog() -> None:
    print("This pipeline always deploys both models behind one API key:\n")
    for model in MODEL_SPECS.values():
        print(f"{model.key}: {model.label}")
        print(f"  Hugging Face: {model.model_id}")
        print(f"  Checkpoint: ~{model.checkpoint_gb:g} GB")
        print(f"  Assigned A40 GPUs: {model.assigned_gpus}")
        print(f"  Capabilities: {', '.join(model.capabilities)}\n")
    print("Default layout: BAGEL GPU 0; ThinkMorph GPUs 1-2; one gateway/key.")


def collect_deploy_params(args: argparse.Namespace, config: Config, username: str) -> DeployParams:
    ok, reason = validate_gpu_count(args.gpus)
    if not ok:
        raise ValueError(reason)
    hours = args.hours if args.hours is not None else config.default_hours
    if not 0 < hours <= 48:
        raise ValueError("Duration must be greater than 0 and no more than 48 hours")
    exposure = args.exposure or config.default_exposure
    if exposure == "cloudflare-quick" and not args.acknowledge_external_tunnel:
        print(
            "\nWARNING: Quick Tunnel sends API traffic through Cloudflare, has no SLA, "
            "and must be approved for your NCSA project."
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
    return DeployParams(
        username=username,
        deployment_id=make_deployment_id(),
        api_key="sk-delta-mm-" + secrets.token_urlsafe(32),
        gpu_count=args.gpus,
        hours=hours,
        exposure=exposure,
        hf_token=os.environ.get("HF_TOKEN", ""),
        cf_tunnel_token=cf_token,
        detach=bool(args.detach),
    )


def print_plan(params: DeployParams) -> None:
    _, layout = validate_gpu_count(params.gpu_count)
    estimate = estimate_weighted_gpu_hours(params.gpu_count, params.hours)
    print("\nDual-model deployment plan")
    print(f"  ID:          {params.deployment_id}")
    print("  Models:      bagel-7b, thinkmorph-7b")
    print("  Partition:   gpuA40x4")
    print(f"  GPUs:        {params.gpu_count} x NVIDIA A40 48 GB")
    print(f"  Layout:      {layout}")
    print(f"  Duration:    {params.hours:g} hours")
    print(f"  Exposure:    {params.exposure}")
    print(f"  Cost guide:  ~{estimate:.2f} weighted GPU-hours")


def run_deploy(args: argparse.Namespace, config: Config) -> int:
    username = get_username(args)
    params = collect_deploy_params(args, config, username)
    print_plan(params)
    if args.dry_run:
        print("\nDry run only: no SSH connection or Slurm job was created.")
        return 0

    ensure_interactive_terminal()
    print("\nEnter the NCSA password and approve Duo once.")
    result = SSHRunner(username, config.login_host).run_script(render_deploy_script(config, params))
    if result is None:
        raise RuntimeError("Remote deployment returned no structured result")
    state = {
        "deployment_id": result.deployment_id,
        "username": username,
        "job_id": result.job_id,
        "state": result.state,
        "endpoint": result.endpoint,
        "expires_at": result.expires_at,
        "models": list(MODEL_SPECS),
        "gpu": "a40",
        "gpu_count": params.gpu_count,
        "gpu_layout": {"bagel-7b": [0], "thinkmorph-7b": [1, 2]},
        "api_key": params.api_key,
        "remote_dir": result.remote_dir,
    }
    state_path = save_local_state(
        result.deployment_id,
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
    )
    print("\nDual-model API created")
    print(f"  Deployment: {result.deployment_id}")
    print(f"  Job:        {result.job_id}")
    print(f"  State:      {result.state}")
    print(f"  Base URL:   {result.endpoint}")
    print(f"  API Key:    {params.api_key}")
    print(f"  Models:     {', '.join(MODEL_SPECS)}")
    print(f"  Expires:    {result.expires_at}")
    print(f"  Local state: {state_path}")
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
            answer = input(f"Stop both models in {deployment_id}? [y/N] ").strip()
            if answer.lower() not in {"y", "yes"}:
                print("Cancelled locally; no remote change was made.")
                return 0
        result = runner.run_script(render_stop_script(config, username, deployment_id))
        if result is None or result.state != "STOPPED":
            raise RuntimeError("Remote stop returned no STOPPED confirmation")
        local_path = mark_local_state_stopped(deployment_id)
        if local_path:
            print(f"Updated local state and removed cached API key: {local_path}")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "models":
            print_catalog()
            code = 0
        elif args.command == "deploy":
            code = run_deploy(args, config)
        else:
            code = run_remote_command(args, config)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        code = 1
    raise SystemExit(code)
