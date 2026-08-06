from __future__ import annotations

import base64
import shlex
from dataclasses import dataclass

from .catalog import GPUSpec, ModelSpec
from .config import Config


def q(value: str | float) -> str:
    return shlex.quote(str(value))


def b64(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def hours_to_slurm(hours: float) -> str:
    total_minutes = max(1, round(hours * 60))
    hh, mm = divmod(total_minutes, 60)
    return f"{hh:02d}:{mm:02d}:00"


@dataclass(frozen=True)
class DeployParams:
    username: str
    deployment_id: str
    api_key: str
    model: ModelSpec
    gpu: GPUSpec
    gpu_count: int
    hours: float
    max_model_len: int
    exposure: str
    hf_token: str
    cf_tunnel_token: str
    detach: bool


def render_deploy_script(config: Config, params: DeployParams) -> str:
    user_root = f"{config.project_root}/{params.username}/delta-llm"
    deploy_dir = f"{user_root}/deployments/{params.deployment_id}"
    env_dir = (
        f"{config.shared_root}/envs/vllm-{config.vllm_version}-{config.cuda_wheel}"
    )
    hf_cache = f"{config.shared_root}/cache/huggingface"
    cloudflared = f"{config.shared_root}/bin/cloudflared"
    wheel_url = config.vllm_wheel_url
    cpus = min(64, max(8, params.gpu_count * 16))
    host_mem = params.gpu.host_memory_gb_per_gpu * params.gpu_count
    slurm_time = hours_to_slurm(params.hours)
    listen_host = "0.0.0.0" if params.exposure == "none" else "127.0.0.1"
    extra_args = " ".join(q(value) for value in params.model.extra_args)
    if extra_args:
        extra_args = " \\" + "\n  " + extra_args

    named_url = config.named_public_url if params.exposure == "cloudflare-named" else ""

    return fr"""#!/usr/bin/env bash
set -euo pipefail
umask 007

ACCOUNT={q(config.account)}
PROJECT_ROOT={q(config.project_root)}
SHARED_ROOT={q(config.shared_root)}
ENV_DIR={q(env_dir)}
HF_CACHE={q(hf_cache)}
CLOUDFLARED={q(cloudflared)}
DEPLOY_ID={q(params.deployment_id)}
USER_ROOT={q(user_root)}
DEPLOY_DIR={q(deploy_dir)}
MODEL_ID={q(params.model.model_id)}
SERVED_MODEL={q(params.model.key)}
GPU_PARTITION={q(params.gpu.partition)}
GPU_COUNT={params.gpu_count}
API_KEY_B64={q(b64(params.api_key))}
HF_TOKEN_B64={q(b64(params.hf_token))}
CF_TOKEN_B64={q(b64(params.cf_tunnel_token))}
EXPOSURE={q(params.exposure)}
NAMED_URL={q(named_url)}
VLLM_WHEEL={q(wheel_url)}

echo "[delta-llm] validating allocation and project storage"
accounts | grep -F "$ACCOUNT" >/dev/null || {{
  echo "ERROR: account $ACCOUNT is not available to $USER" >&2
  accounts >&2
  exit 20
}}
sinfo -h -p "$GPU_PARTITION" -o '%T' | grep -Eq '^(idle|mix|alloc|comp|drain)' || {{
  echo "ERROR: partition $GPU_PARTITION is unavailable or unknown" >&2
  sinfo -p "$GPU_PARTITION" >&2 || true
  exit 23
}}
mkdir -p "$USER_ROOT/deployments" "$DEPLOY_DIR" "$SHARED_ROOT/envs" "$HF_CACHE" "$SHARED_ROOT/bin"
chmod 700 "$DEPLOY_DIR"
chmod 2770 "$SHARED_ROOT" "$SHARED_ROOT/envs" "$SHARED_ROOT/bin" || true

mkdir -p "$DEPLOY_DIR/secrets" "$DEPLOY_DIR/logs"
chmod 700 "$DEPLOY_DIR/secrets"

# Revoke credentials left by earlier failed/stopped deployments owned by this user.
for old_dir in "$USER_ROOT/deployments"/*; do
  [[ -d "$old_dir" ]] || continue
  old_state="$(cat "$old_dir/state" 2>/dev/null || true)"
  if [[ "$old_state" == FAILED || "$old_state" == STOPPED ]]; then
    rm -f "$old_dir/secrets/api_key" "$old_dir/secrets/cf_tunnel_token" \
      "$old_dir/endpoint"
  fi
done

printf '%s' "$API_KEY_B64" | base64 -d > "$DEPLOY_DIR/secrets/api_key"
chmod 600 "$DEPLOY_DIR/secrets/api_key"
if [[ -n "$HF_TOKEN_B64" ]]; then
  printf '%s' "$HF_TOKEN_B64" | base64 -d > "$DEPLOY_DIR/secrets/hf_token"
  chmod 600 "$DEPLOY_DIR/secrets/hf_token"
fi
if [[ -n "$CF_TOKEN_B64" ]]; then
  printf '%s' "$CF_TOKEN_B64" | base64 -d > "$DEPLOY_DIR/secrets/cf_tunnel_token"
  chmod 600 "$DEPLOY_DIR/secrets/cf_tunnel_token"
fi

env_ready() {{
  [[ -f "$ENV_DIR/.delta-llm-ready" ]] || return 1
  [[ -x "$ENV_DIR/bin/vllm" ]] || return 1
  [[ "$(head -n 1 "$ENV_DIR/bin/vllm")" == "#!$ENV_DIR/bin/python" ]]
}}

if ! env_ready; then
  echo "[delta-llm] shared vLLM environment is missing; scheduling one-time setup"
  LOCK_DIR="$ENV_DIR.installing"
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    rm -rf "$ENV_DIR"
    cat > "$DEPLOY_DIR/setup.slurm" <<SETUP_SLURM
#!/usr/bin/env bash
#SBATCH --account={config.account}
#SBATCH --partition=gpuA40x4
#SBATCH --job-name=delta-vllm-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32g
#SBATCH --time=01:00:00
#SBATCH --output=$DEPLOY_DIR/logs/setup_%j.out
#SBATCH --error=$DEPLOY_DIR/logs/setup_%j.err
set -euo pipefail
trap 'status=\$?; rm -rf "$LOCK_DIR"; if [[ \$status -ne 0 ]]; then rm -rf "$ENV_DIR"; fi' EXIT
umask 007
module purge
module load miniforge3-python
python -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip
"$ENV_DIR/bin/python" -m pip install "$VLLM_WHEEL" \\
  --extra-index-url https://download.pytorch.org/whl/{config.cuda_wheel}
"$ENV_DIR/bin/vllm" --version
touch "$ENV_DIR/.delta-llm-ready"
chmod -R g+rX "$ENV_DIR" || true
SETUP_SLURM
    SETUP_JOB="$(sbatch --parsable "$DEPLOY_DIR/setup.slurm")"
    echo "[delta-llm] setup job $SETUP_JOB submitted"
    while squeue -h -j "$SETUP_JOB" | grep -q .; do
      squeue -h -j "$SETUP_JOB" -o '[delta-llm] setup %T: %R'
      sleep 15
    done
    if ! env_ready; then
      echo "ERROR: vLLM setup failed" >&2
      cat "$DEPLOY_DIR/logs/setup_${{SETUP_JOB}}.out" 2>/dev/null || true
      cat "$DEPLOY_DIR/logs/setup_${{SETUP_JOB}}.err" 2>/dev/null || true
      exit 21
    fi
  else
    echo "[delta-llm] another user is installing the shared environment; waiting"
    for _ in $(seq 1 240); do
      env_ready && break
      [[ ! -d "$LOCK_DIR" ]] && break
      sleep 15
    done
    env_ready || {{
      echo "ERROR: shared environment installation did not finish" >&2
      exit 22
    }}
  fi
fi

if [[ "$EXPOSURE" == cloudflare-* && ! -x "$CLOUDFLARED" ]]; then
  echo "[delta-llm] downloading cloudflared to shared project storage"
  TMP_CF="$CLOUDFLARED.tmp.$$"
  curl -fsSL {q(config.cloudflared_url)} -o "$TMP_CF"
  chmod 750 "$TMP_CF"
  mv -f "$TMP_CF" "$CLOUDFLARED"
fi

cat > "$DEPLOY_DIR/metadata.env" <<METADATA
DEPLOYMENT_ID=$DEPLOY_ID
MODEL_ID=$MODEL_ID
SERVED_MODEL=$SERVED_MODEL
GPU_PARTITION=$GPU_PARTITION
GPU_COUNT=$GPU_COUNT
EXPOSURE=$EXPOSURE
METADATA
chmod 600 "$DEPLOY_DIR/metadata.env"

cat > "$DEPLOY_DIR/service.slurm" <<'SERVICE_HEADER'
#!/usr/bin/env bash
SERVICE_HEADER
cat >> "$DEPLOY_DIR/service.slurm" <<SERVICE_CONFIG
#SBATCH --account={config.account}
#SBATCH --partition={params.gpu.partition}
#SBATCH --job-name=llm-{params.deployment_id[:20]}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gpus-per-node={params.gpu_count}
#SBATCH --mem={host_mem}g
#SBATCH --time={slurm_time}
#SBATCH --requeue
#SBATCH --output=$DEPLOY_DIR/logs/slurm_%j.out
#SBATCH --error=$DEPLOY_DIR/logs/slurm_%j.err
SERVICE_CONFIG
cat >> "$DEPLOY_DIR/service.slurm" <<'SERVICE_BODY'

set -euo pipefail
umask 077
DEPLOY_DIR={q(deploy_dir)}
source "$DEPLOY_DIR/metadata.env"
ENV_DIR={q(env_dir)}
HF_CACHE={q(hf_cache)}
CLOUDFLARED={q(cloudflared)}
MODEL_ID={q(params.model.model_id)}
SERVED_MODEL={q(params.model.key)}
EXPOSURE={q(params.exposure)}
NAMED_URL={q(named_url)}
mkdir -p "$DEPLOY_DIR/logs"

export HF_HOME="$HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE/hub"
export VLLM_API_KEY="$(< "$DEPLOY_DIR/secrets/api_key")"
if [[ -s "$DEPLOY_DIR/secrets/hf_token" ]]; then
  export HF_TOKEN="$(< "$DEPLOY_DIR/secrets/hf_token")"
fi

echo STARTING > "$DEPLOY_DIR/state"
rm -f "$DEPLOY_DIR/endpoint"
VLLM_PID=""
TUNNEL_PID=""
cleanup() {{
  [[ -n "$TUNNEL_PID" ]] && kill "$TUNNEL_PID" 2>/dev/null || true
  [[ -n "$VLLM_PID" ]] && kill "$VLLM_PID" 2>/dev/null || true
  CURRENT_STATE="$(cat "$DEPLOY_DIR/state" 2>/dev/null || true)"
  [[ "$CURRENT_STATE" == FAILED ]] || echo STOPPED > "$DEPLOY_DIR/state"
}}
trap cleanup EXIT INT TERM

"$ENV_DIR/bin/vllm" serve "$MODEL_ID" \
  --host {listen_host} \
  --port 8000 \
  --served-model-name "$SERVED_MODEL" \
  --tensor-parallel-size {params.gpu_count} \
  --max-model-len {params.max_model_len} \
  --gpu-memory-utilization {config.gpu_memory_utilization:.3f} \
  --generation-config vllm{extra_args} \
  > "$DEPLOY_DIR/logs/vllm.log" 2>&1 &
VLLM_PID=$!

for _ in $(seq 1 360); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo FAILED > "$DEPLOY_DIR/state"
    tail -n 200 "$DEPLOY_DIR/logs/vllm.log" >&2 || true
    exit 31
  fi
  sleep 10
done
curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1 || {{
  echo FAILED > "$DEPLOY_DIR/state"
  echo "vLLM did not become healthy within 60 minutes" >&2
  exit 32
}}

case "$EXPOSURE" in
  none)
    ENDPOINT="http://$(hostname -f):8000/v1"
    ;;
  cloudflare-quick)
    "$CLOUDFLARED" tunnel --url http://127.0.0.1:8000 --no-autoupdate \
      > "$DEPLOY_DIR/logs/cloudflared.log" 2>&1 &
    TUNNEL_PID=$!
    ENDPOINT=""
    for _ in $(seq 1 60); do
      ENDPOINT="$(grep -Eo 'https://[-a-z0-9]+\.trycloudflare\.com' \
        "$DEPLOY_DIR/logs/cloudflared.log" | head -n 1 || true)"
      [[ -n "$ENDPOINT" ]] && break
      kill -0 "$TUNNEL_PID" 2>/dev/null || break
      sleep 2
    done
    [[ -n "$ENDPOINT" ]] || {{
      echo FAILED > "$DEPLOY_DIR/state"
      cat "$DEPLOY_DIR/logs/cloudflared.log" >&2 || true
      exit 33
    }}
    ENDPOINT="$ENDPOINT/v1"
    ;;
  cloudflare-named)
    CF_TOKEN="$(< "$DEPLOY_DIR/secrets/cf_tunnel_token")"
    "$CLOUDFLARED" tunnel --no-autoupdate run --token "$CF_TOKEN" \
      > "$DEPLOY_DIR/logs/cloudflared.log" 2>&1 &
    TUNNEL_PID=$!
    sleep 5
    kill -0 "$TUNNEL_PID" 2>/dev/null || {{
      echo FAILED > "$DEPLOY_DIR/state"
      cat "$DEPLOY_DIR/logs/cloudflared.log" >&2 || true
      exit 34
    }}
    ENDPOINT="$NAMED_URL/v1"
    ;;
  *)
    echo "Unsupported exposure mode: $EXPOSURE" >&2
    exit 35
    ;;
esac

printf '%s\n' "$ENDPOINT" > "$DEPLOY_DIR/endpoint"
echo READY > "$DEPLOY_DIR/state"
echo "[delta-llm] READY: $ENDPOINT"

if [[ -n "$TUNNEL_PID" ]]; then
  wait -n "$VLLM_PID" "$TUNNEL_PID"
else
  wait "$VLLM_PID"
fi
SERVICE_BODY

chmod 700 "$DEPLOY_DIR/service.slurm"
JOB_ID="$(sbatch --parsable "$DEPLOY_DIR/service.slurm")"
printf '%s\n' "$JOB_ID" > "$DEPLOY_DIR/job_id"
echo "[delta-llm] service job $JOB_ID submitted"

if [[ {str(params.detach).lower()} == true ]]; then
  echo "DELTA_LLM_RESULT|$DEPLOY_ID|$JOB_ID|SUBMITTED|-|-|$DEPLOY_DIR"
  exit 0
fi

for _ in $(seq 1 480); do
  STATE="$(cat "$DEPLOY_DIR/state" 2>/dev/null || true)"
  if [[ "$STATE" == READY && -s "$DEPLOY_DIR/endpoint" ]]; then
    ENDPOINT="$(< "$DEPLOY_DIR/endpoint")"
    EXPIRES="$(scontrol show job -o "$JOB_ID" | tr ' ' '\n' | sed -n 's/^EndTime=//p')"
    echo "DELTA_LLM_RESULT|$DEPLOY_ID|$JOB_ID|READY|$ENDPOINT|$EXPIRES|$DEPLOY_DIR"
    exit 0
  fi
  if ! squeue -h -j "$JOB_ID" | grep -q .; then
    FINAL="$(sacct -n -X -j "$JOB_ID" -o State | awk 'NF {{print $1; exit}}')"
    echo "ERROR: service job ended before becoming ready: $FINAL" >&2
    tail -n 200 "$DEPLOY_DIR/logs/slurm_${{JOB_ID}}.err" 2>/dev/null || true
    tail -n 200 "$DEPLOY_DIR/logs/vllm.log" 2>/dev/null || true
    exit 36
  fi
  squeue -h -j "$JOB_ID" -o '[delta-llm] service %T: %R' | head -n 1
  sleep 15
done

echo "ERROR: timed out waiting two hours for service readiness" >&2
exit 37
"""


def render_status_script(config: Config, username: str, deployment_id: str) -> str:
    deploy_dir = f"{config.project_root}/{username}/delta-llm/deployments/{deployment_id}"
    return fr"""#!/usr/bin/env bash
set -euo pipefail
DEPLOY_DIR={q(deploy_dir)}
[[ -d "$DEPLOY_DIR" ]] || {{ echo "Deployment not found: {deployment_id}" >&2; exit 40; }}
JOB_ID="$(< "$DEPLOY_DIR/job_id")"
STATE="$(cat "$DEPLOY_DIR/state" 2>/dev/null || echo UNKNOWN)"
ENDPOINT="$(cat "$DEPLOY_DIR/endpoint" 2>/dev/null || echo -)"
if squeue -h -j "$JOB_ID" | grep -q .; then
  squeue -j "$JOB_ID" -o 'JOBID=%i STATE=%T ELAPSED=%M LIMIT=%l NODE=%R'
  EXPIRES="$(scontrol show job -o "$JOB_ID" | tr ' ' '\n' | sed -n 's/^EndTime=//p')"
else
  sacct -X -j "$JOB_ID" --format=JobID,JobName,State,Elapsed,End
  EXPIRES="-"
fi
echo "endpoint=$ENDPOINT"
echo "DELTA_LLM_RESULT|{deployment_id}|$JOB_ID|$STATE|$ENDPOINT|$EXPIRES|$DEPLOY_DIR"
"""


def render_logs_script(
    config: Config, username: str, deployment_id: str, lines: int
) -> str:
    deploy_dir = f"{config.project_root}/{username}/delta-llm/deployments/{deployment_id}"
    return fr"""#!/usr/bin/env bash
set -euo pipefail
DEPLOY_DIR={q(deploy_dir)}
[[ -d "$DEPLOY_DIR" ]] || {{ echo "Deployment not found: {deployment_id}" >&2; exit 40; }}
echo '=== vLLM ==='
tail -n {int(lines)} "$DEPLOY_DIR/logs/vllm.log" 2>/dev/null || true
echo '=== cloudflared ==='
tail -n {int(lines)} "$DEPLOY_DIR/logs/cloudflared.log" 2>/dev/null || true
echo '=== Slurm stderr ==='
JOB_ID="$(< "$DEPLOY_DIR/job_id")"
tail -n {int(lines)} "$DEPLOY_DIR/logs/slurm_${{JOB_ID}}.err" 2>/dev/null || true
"""


def render_stop_script(config: Config, username: str, deployment_id: str) -> str:
    deploy_dir = f"{config.project_root}/{username}/delta-llm/deployments/{deployment_id}"
    return fr"""#!/usr/bin/env bash
set -euo pipefail
DEPLOY_DIR={q(deploy_dir)}
[[ -d "$DEPLOY_DIR" ]] || {{ echo "Deployment not found: {deployment_id}" >&2; exit 40; }}
JOB_ID="$(< "$DEPLOY_DIR/job_id")"
scancel "$JOB_ID" 2>/dev/null || true
rm -f "$DEPLOY_DIR/secrets/api_key" "$DEPLOY_DIR/endpoint"
echo STOPPED > "$DEPLOY_DIR/state"
echo "Stopped job $JOB_ID and revoked its API key"
echo "DELTA_LLM_RESULT|{deployment_id}|$JOB_ID|STOPPED|-|-|$DEPLOY_DIR"
"""


def render_list_script(config: Config, username: str) -> str:
    root = f"{config.project_root}/{username}/delta-llm/deployments"
    return fr"""#!/usr/bin/env bash
set -euo pipefail
ROOT={q(root)}
printf '%-30s %-12s %-12s %s\n' DEPLOYMENT JOB_ID STATE ENDPOINT
[[ -d "$ROOT" ]] || exit 0
for dir in "$ROOT"/*; do
  [[ -d "$dir" ]] || continue
  id="$(basename "$dir")"
  job="$(cat "$dir/job_id" 2>/dev/null || echo -)"
  state="$(cat "$dir/state" 2>/dev/null || echo UNKNOWN)"
  endpoint="$(cat "$dir/endpoint" 2>/dev/null || echo -)"
  printf '%-30s %-12s %-12s %s\n' "$id" "$job" "$state" "$endpoint"
done
"""


def render_doctor_script(config: Config) -> str:
    return fr"""#!/usr/bin/env bash
set -euo pipefail
echo "user=$USER"
echo "host=$(hostname -f)"
echo '=== accounts ==='
accounts
echo '=== target partitions ==='
sinfo -h -o '%P|%l|%D|%G' | grep -E 'gpu(A40x4|A100x4|A100x8|H200x8)' || true
echo '=== project storage ==='
ls -ld {q(config.project_root)}
quota
echo '=== outbound checks ==='
curl -IsS --max-time 10 https://huggingface.co | head -n 1 || true
curl -IsS --max-time 10 https://github.com | head -n 1 || true
"""
