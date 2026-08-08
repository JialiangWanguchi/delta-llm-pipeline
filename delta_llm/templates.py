from __future__ import annotations

import base64
import shlex
from dataclasses import dataclass
from pathlib import Path

from .config import Config


def q(value: str | float) -> str:
    return shlex.quote(str(value))


def b64(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def runtime_source(name: str) -> str:
    return Path(__file__).with_name(name).read_text(encoding="utf-8")


def hours_to_slurm(hours: float) -> str:
    total_minutes = max(1, round(hours * 60))
    hh, mm = divmod(total_minutes, 60)
    return f"{hh:02d}:{mm:02d}:00"


@dataclass(frozen=True)
class DeployParams:
    username: str
    deployment_id: str
    api_key: str
    gpu_count: int
    hours: float
    exposure: str
    hf_token: str
    cf_tunnel_token: str
    detach: bool
    recover_stalled_setup: bool
    replace_existing_services: bool


def render_deploy_script(config: Config, params: DeployParams) -> str:
    user_root = f"{config.project_root}/{params.username}/delta-llm"
    deploy_dir = f"{user_root}/deployments/{params.deployment_id}"
    env_name = (
        f"multimodal-py{config.python_version}-torch{config.torch_version}-{config.cuda_wheel}"
    )
    env_dir = f"{config.runtime_root}/envs/{env_name}"
    source_root = f"{config.shared_root}/sources"
    bagel_repo = f"{source_root}/bagel"
    thinkmorph_repo = f"{source_root}/thinkmorph"
    model_root = f"{config.shared_root}/models"
    bagel_model = f"{model_root}/BAGEL-7B-MoT"
    thinkmorph_model = f"{model_root}/ThinkMorph-7B"
    cloudflared = f"{config.shared_root}/bin/cloudflared"
    offload_root = f"{config.work_root}/{params.username}/delta-llm/offload/{params.deployment_id}"
    legacy_env_dir = f"{config.shared_root}/envs/{env_name}"
    slurm_time = hours_to_slurm(params.hours)
    thinkmorph_gpu_count = 1 if params.gpu_count == 2 else 2
    service_cpus = 32 if params.gpu_count == 2 else 48
    service_memory = "120g" if params.gpu_count == 2 else "220g"
    named_url = config.named_public_url if params.exposure == "cloudflare-named" else ""
    env_fingerprint = ";".join(
        (
            f"python={config.python_version}",
            f"torch={config.torch_version}",
            f"transformers={config.transformers_version}",
            f"flash-attn={config.flash_attn_version}",
            f"cuda={config.cuda_wheel}",
            f"bagel={config.bagel_commit}",
            f"thinkmorph={config.thinkmorph_commit}",
        )
    )
    worker_b64 = b64(runtime_source("runtime_worker.py"))
    gateway_b64 = b64(runtime_source("runtime_gateway.py"))

    return rf"""#!/usr/bin/env bash
set -euo pipefail
umask 007

ACCOUNT={q(config.account)}
PROJECT_ROOT={q(config.project_root)}
SHARED_ROOT={q(config.shared_root)}
RUNTIME_ROOT={q(config.runtime_root)}
ENV_DIR={q(env_dir)}
SOURCE_ROOT={q(source_root)}
BAGEL_REPO={q(bagel_repo)}
THINKMORPH_REPO={q(thinkmorph_repo)}
MODEL_ROOT={q(model_root)}
BAGEL_MODEL={q(bagel_model)}
THINKMORPH_MODEL={q(thinkmorph_model)}
CLOUDFLARED={q(cloudflared)}
OFFLOAD_ROOT={q(offload_root)}
LEGACY_ENV_DIR={q(legacy_env_dir)}
DEPLOY_ID={q(params.deployment_id)}
USER_ROOT={q(user_root)}
DEPLOY_DIR={q(deploy_dir)}
GPU_COUNT={params.gpu_count}
EXPOSURE={q(params.exposure)}
NAMED_URL={q(named_url)}
API_KEY_B64={q(b64(params.api_key))}
HF_TOKEN_B64={q(b64(params.hf_token))}
CF_TOKEN_B64={q(b64(params.cf_tunnel_token))}
ENV_FINGERPRINT={q(env_fingerprint)}
BAGEL_COMMIT={q(config.bagel_commit)}
THINKMORPH_COMMIT={q(config.thinkmorph_commit)}
RECOVER_STALLED_SETUP={str(params.recover_stalled_setup).lower()}
REPLACE_EXISTING_SERVICES={str(params.replace_existing_services).lower()}

echo "[delta-multimodal] validating allocation, partition, and storage"
accounts | grep -F "$ACCOUNT" >/dev/null || {{
  echo "ERROR: account $ACCOUNT is not available to $USER" >&2
  exit 20
}}
sinfo -h -p gpuA40x4 -o '%T' | grep -Eq '^(idle|mix|alloc|comp|drain)' || {{
  echo "ERROR: gpuA40x4 is unavailable" >&2
  exit 23
}}
mkdir -p "$USER_ROOT/deployments" "$DEPLOY_DIR/logs" "$DEPLOY_DIR/secrets" \
  "$DEPLOY_DIR/runtime" "$RUNTIME_ROOT/envs" "$RUNTIME_ROOT/conda-pkgs" \
  "$RUNTIME_ROOT/pip-cache" "$SOURCE_ROOT" "$MODEL_ROOT" "$SHARED_ROOT/bin" \
  "$OFFLOAD_ROOT"
chmod 700 "$DEPLOY_DIR" "$DEPLOY_DIR/secrets"
chmod 2770 "$SHARED_ROOT" "$RUNTIME_ROOT" "$RUNTIME_ROOT/envs" \
  "$RUNTIME_ROOT/conda-pkgs" "$RUNTIME_ROOT/pip-cache" "$SOURCE_ROOT" \
  "$MODEL_ROOT" "$SHARED_ROOT/bin" || true

if [[ "$REPLACE_EXISTING_SERVICES" == true ]]; then
  echo "[delta-multimodal] cancelling this user's previous dual-model service jobs"
  mapfile -t OLD_SERVICE_JOBS < <(
    squeue -h -u "$USER" -o '%A|%j' | \
      awk -F'|' '$2 ~ /^mm-bagel-thinkmorph-/ {{print $1}}' | sort -u
  )
  if (( ${{#OLD_SERVICE_JOBS[@]}} )); then
    scancel "${{OLD_SERVICE_JOBS[@]}}"
    for _ in $(seq 1 30); do
      squeue -h -j "$(IFS=,; echo "${{OLD_SERVICE_JOBS[*]}}")" | grep -q . || break
      sleep 2
    done
  fi
fi

if [[ "$RECOVER_STALLED_SETUP" == true ]]; then
  echo "[delta-multimodal] cancelling this user's stalled setup jobs"
  mapfile -t STALLED_JOBS < <(squeue -h -u "$USER" -n delta-mm-setup -o '%A' | sort -u)
  if (( ${{#STALLED_JOBS[@]}} )); then
    scancel "${{STALLED_JOBS[@]}}"
    for _ in $(seq 1 30); do
      squeue -h -j "$(IFS=,; echo "${{STALLED_JOBS[*]}}")" | grep -q . || break
      sleep 2
    done
    squeue -h -j "$(IFS=,; echo "${{STALLED_JOBS[*]}}")" | grep -q . && {{
      echo "ERROR: stalled setup jobs did not stop" >&2
      exit 25
    }}
  fi
  [[ "$LEGACY_ENV_DIR" == /projects/bhsz/delta-llm/shared/envs/* ]] || exit 26
  rm -rf "$LEGACY_ENV_DIR.installing" "$LEGACY_ENV_DIR"
fi

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
printf '%s' {q(worker_b64)} | base64 -d > "$DEPLOY_DIR/runtime/worker.py"
printf '%s' {q(gateway_b64)} | base64 -d > "$DEPLOY_DIR/runtime/gateway.py"
chmod 600 "$DEPLOY_DIR/runtime/worker.py" "$DEPLOY_DIR/runtime/gateway.py"

env_ready() {{
  [[ -x "$ENV_DIR/bin/python" ]] || return 1
  [[ -f "$ENV_DIR/.delta-multimodal-ready" ]] || return 1
  [[ "$(< "$ENV_DIR/.delta-multimodal-ready")" == "$ENV_FINGERPRINT" ]] || return 1
  [[ -f "$BAGEL_MODEL/ema.safetensors" ]] || return 1
  [[ -f "$THINKMORPH_MODEL/model.safetensors" ]] || return 1
  [[ "$(git -C "$BAGEL_REPO" rev-parse HEAD 2>/dev/null)" == "$BAGEL_COMMIT" ]] || return 1
  [[ "$(git -C "$THINKMORPH_REPO" rev-parse HEAD 2>/dev/null)" == "$THINKMORPH_COMMIT" ]]
}}

if ! env_ready; then
  echo "[delta-multimodal] shared runtime/models missing; scheduling one-time setup"
  LOCK_DIR="$ENV_DIR.installing"
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    cat > "$DEPLOY_DIR/setup.slurm" <<SETUP_SLURM
#!/usr/bin/env bash
#SBATCH --account={config.account}
#SBATCH --partition=gpuA40x4
#SBATCH --job-name=delta-mm-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=96g
#SBATCH --time=04:00:00
#SBATCH --output=$DEPLOY_DIR/logs/setup_%j.out
#SBATCH --error=$DEPLOY_DIR/logs/setup_%j.err
set -euo pipefail
trap 'status=\$?; rm -rf "$LOCK_DIR"; if [[ \$status -ne 0 ]]; then rm -f "$ENV_DIR/.delta-multimodal-ready"; fi' EXIT
umask 007
module purge
module load miniforge3-python
module load cuda 2>/dev/null || true
export CONDA_PKGS_DIRS="$RUNTIME_ROOT/conda-pkgs"
export PIP_CACHE_DIR="$RUNTIME_ROOT/pip-cache"

sync_repo() {{
  local url="\$1" commit="\$2" target="\$3"
  if [[ ! -d "\$target/.git" ]]; then
    rm -rf "\$target"
    git clone --filter=blob:none --no-checkout "\$url" "\$target"
  fi
  git -C "\$target" fetch --depth 1 origin "\$commit"
  git -C "\$target" sparse-checkout init --no-cone
  git -C "\$target" sparse-checkout set '/data/' '/modeling/' '/inferencer.py'
  git -C "\$target" checkout --detach "\$commit"
}}
sync_repo https://github.com/ByteDance-Seed/BAGEL.git "$BAGEL_COMMIT" "$BAGEL_REPO"
sync_repo https://github.com/ThinkMorph/ThinkMorph.git "$THINKMORPH_COMMIT" "$THINKMORPH_REPO"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  rm -rf "$ENV_DIR"
  conda create -y --solver libmamba -p "$ENV_DIR" \\
    python={config.python_version} pip
fi
"$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel ninja packaging
"$ENV_DIR/bin/python" -m pip install \\
  torch=={config.torch_version} torchvision=={config.torchvision_version} \\
  --index-url https://download.pytorch.org/whl/{config.cuda_wheel}
"$ENV_DIR/bin/python" -m pip install \\
  accelerate==1.2.1 einops==0.8.1 fastapi==0.115.6 \\
  huggingface_hub==0.29.1 numpy==1.26.4 opencv-python-headless==4.10.0.84 \\
  pillow==11.0.0 pyyaml==6.0.2 requests==2.32.3 safetensors==0.4.5 \\
  scipy==1.13.1 sentencepiece==0.2.0 transformers=={config.transformers_version} \\
  uvicorn==0.34.0 python-multipart==0.0.20
if ! "$ENV_DIR/bin/python" -c 'import flash_attn; assert flash_attn.__version__ == "{config.flash_attn_version}"' 2>/dev/null; then
  command -v nvcc >/dev/null || {{ echo "ERROR: nvcc is required for flash-attn" >&2; exit 24; }}
  MAX_JOBS=16 "$ENV_DIR/bin/python" -m pip install \\
    flash-attn=={config.flash_attn_version} --no-build-isolation
fi

HF_TOKEN=""
if [[ -s "$DEPLOY_DIR/secrets/hf_token" ]]; then HF_TOKEN="\$(< "$DEPLOY_DIR/secrets/hf_token")"; fi
export HF_TOKEN
"$ENV_DIR/bin/python" -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="ByteDance-Seed/BAGEL-7B-MoT", local_dir="$BAGEL_MODEL", allow_patterns=["*.json","*.safetensors","*.txt"])'
"$ENV_DIR/bin/python" -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="ThinkMorph/ThinkMorph-7B", local_dir="$THINKMORPH_MODEL", allow_patterns=["*.json","*.safetensors","*.txt"])'
"$ENV_DIR/bin/python" -c 'import torch, transformers, flash_attn; assert torch.cuda.is_available(); print(torch.__version__, transformers.__version__, flash_attn.__version__)'
printf '%s\n' "$ENV_FINGERPRINT" > "$ENV_DIR/.delta-multimodal-ready"
chmod -R g+rX "$ENV_DIR" "$BAGEL_REPO" "$THINKMORPH_REPO" \\
  "$BAGEL_MODEL" "$THINKMORPH_MODEL" || true
SETUP_SLURM
    SETUP_JOB="$(sbatch --parsable "$DEPLOY_DIR/setup.slurm")"
    printf '%s\n' "$SETUP_JOB" > "$LOCK_DIR/job_id"
    printf '%s\n' "$DEPLOY_DIR" > "$LOCK_DIR/deploy_dir"
    echo "[delta-multimodal] setup job $SETUP_JOB submitted"
    while squeue -h -j "$SETUP_JOB" | grep -q .; do
      squeue -h -j "$SETUP_JOB" -o '[delta-multimodal] setup %T: %R'
      sleep 20
    done
    if ! env_ready; then
      echo "ERROR: multimodal setup failed" >&2
      tail -n 200 "$DEPLOY_DIR/logs/setup_${{SETUP_JOB}}.out" 2>/dev/null || true
      tail -n 200 "$DEPLOY_DIR/logs/setup_${{SETUP_JOB}}.err" 2>/dev/null || true
      exit 21
    fi
  else
    echo "[delta-multimodal] another member is installing shared assets; waiting"
    # Covers queue time plus the setup job's four-hour wall time. The SSH
    # keepalive in the local client keeps this one authenticated session alive.
    for attempt in $(seq 1 2160); do
      env_ready && break
      [[ ! -d "$LOCK_DIR" ]] && break
      if [[ -s "$LOCK_DIR/job_id" ]]; then
        ACTIVE_SETUP="$(< "$LOCK_DIR/job_id")"
        if (( attempt % 3 == 0 )); then
          squeue -h -j "$ACTIVE_SETUP" \
            -o '[delta-multimodal] shared setup %T: %R' | head -n 1 || true
        fi
      elif (( attempt % 15 == 0 )); then
        echo "[delta-multimodal] waiting for legacy shared setup lock"
      fi
      sleep 20
    done
    env_ready || {{ echo "ERROR: shared setup did not finish" >&2; exit 22; }}
  fi
fi

if [[ "$EXPOSURE" == cloudflare-* && ! -x "$CLOUDFLARED" ]]; then
  TMP_CF="$CLOUDFLARED.tmp.$$"
  curl -fsSL {q(config.cloudflared_url)} -o "$TMP_CF"
  chmod 750 "$TMP_CF"
  mv -f "$TMP_CF" "$CLOUDFLARED"
fi

cat > "$DEPLOY_DIR/metadata.env" <<METADATA
DEPLOYMENT_ID=$DEPLOY_ID
MODELS=bagel-7b,thinkmorph-7b
GPU_PARTITION=gpuA40x4
GPU_COUNT=$GPU_COUNT
GPU_LAYOUT=bagel-7b:1,thinkmorph-7b:{thinkmorph_gpu_count}
EXPOSURE=$EXPOSURE
METADATA
chmod 600 "$DEPLOY_DIR/metadata.env"

cat > "$DEPLOY_DIR/service.slurm" <<'SERVICE_HEADER'
#!/usr/bin/env bash
SERVICE_HEADER
cat >> "$DEPLOY_DIR/service.slurm" <<SERVICE_CONFIG
#SBATCH --account={config.account}
#SBATCH --partition=gpuA40x4
#SBATCH --job-name=mm-{params.deployment_id[:20]}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={service_cpus}
#SBATCH --gpus-per-node={params.gpu_count}
#SBATCH --mem={service_memory}
#SBATCH --time={slurm_time}
#SBATCH --requeue
#SBATCH --output=$DEPLOY_DIR/logs/slurm_%j.out
#SBATCH --error=$DEPLOY_DIR/logs/slurm_%j.err
SERVICE_CONFIG
cat >> "$DEPLOY_DIR/service.slurm" <<'SERVICE_BODY'

set -euo pipefail
umask 077
DEPLOY_DIR={q(deploy_dir)}
ENV_DIR={q(env_dir)}
BAGEL_REPO={q(bagel_repo)}
THINKMORPH_REPO={q(thinkmorph_repo)}
BAGEL_MODEL={q(bagel_model)}
THINKMORPH_MODEL={q(thinkmorph_model)}
CLOUDFLARED={q(cloudflared)}
OFFLOAD_ROOT={q(offload_root)}
EXPOSURE={q(params.exposure)}
NAMED_URL={q(named_url)}
source "$DEPLOY_DIR/metadata.env"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export DELTA_MULTIMODAL_API_KEY="$(< "$DEPLOY_DIR/secrets/api_key")"

echo STARTING > "$DEPLOY_DIR/state"
rm -f "$DEPLOY_DIR/endpoint"
PIDS=()
cleanup() {{
  for pid in "${{PIDS[@]}}"; do kill "$pid" 2>/dev/null || true; done
  current="$(cat "$DEPLOY_DIR/state" 2>/dev/null || true)"
  [[ "$current" == FAILED ]] || echo STOPPED > "$DEPLOY_DIR/state"
}}
trap cleanup EXIT INT TERM

ALLOCATED_CUDA="${{CUDA_VISIBLE_DEVICES:-0,1}}"
IFS=',' read -r -a CUDA_IDS <<< "$ALLOCATED_CUDA"
if [[ "${{#CUDA_IDS[@]}}" -lt "$GPU_COUNT" ]]; then
  echo FAILED > "$DEPLOY_DIR/state"
  echo "Expected at least $GPU_COUNT allocated GPUs, got $ALLOCATED_CUDA" >&2
  exit 30
fi
BAGEL_CUDA="${{CUDA_IDS[0]}}"
if (( GPU_COUNT >= 3 )); then
  THINKMORPH_CUDA="${{CUDA_IDS[1]}},${{CUDA_IDS[2]}}"
else
  THINKMORPH_CUDA="${{CUDA_IDS[1]}}"
fi

CUDA_VISIBLE_DEVICES="$BAGEL_CUDA" "$ENV_DIR/bin/python" \
  "$DEPLOY_DIR/runtime/worker.py" --model bagel-7b \
  --repo-dir "$BAGEL_REPO" --checkpoint-dir "$BAGEL_MODEL" \
  --offload-dir "$OFFLOAD_ROOT/bagel" --port 8101 \
  > "$DEPLOY_DIR/logs/bagel.log" 2>&1 &
BAGEL_PID=$!
PIDS+=("$BAGEL_PID")

CUDA_VISIBLE_DEVICES="$THINKMORPH_CUDA" "$ENV_DIR/bin/python" \
  "$DEPLOY_DIR/runtime/worker.py" --model thinkmorph-7b \
  --repo-dir "$THINKMORPH_REPO" --checkpoint-dir "$THINKMORPH_MODEL" \
  --offload-dir "$OFFLOAD_ROOT/thinkmorph" --port 8102 \
  > "$DEPLOY_DIR/logs/thinkmorph.log" 2>&1 &
THINKMORPH_PID=$!
PIDS+=("$THINKMORPH_PID")

wait_for_worker() {{
  local name="$1" port="$2" pid="$3" log="$4"
  for _ in $(seq 1 540); do
    curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1 && return 0
    if ! kill -0 "$pid" 2>/dev/null; then
      echo FAILED > "$DEPLOY_DIR/state"
      echo "$name worker exited during startup" >&2
      tail -n 200 "$log" >&2 || true
      return 1
    fi
    sleep 10
  done
  echo FAILED > "$DEPLOY_DIR/state"
  echo "$name worker did not become ready within 90 minutes" >&2
  tail -n 200 "$log" >&2 || true
  return 1
}}
wait_for_worker bagel 8101 "$BAGEL_PID" "$DEPLOY_DIR/logs/bagel.log"
wait_for_worker thinkmorph 8102 "$THINKMORPH_PID" "$DEPLOY_DIR/logs/thinkmorph.log"

"$ENV_DIR/bin/python" "$DEPLOY_DIR/runtime/gateway.py" \
  > "$DEPLOY_DIR/logs/gateway.log" 2>&1 &
GATEWAY_PID=$!
PIDS+=("$GATEWAY_PID")
for _ in $(seq 1 60); do
  curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1 && break
  kill -0 "$GATEWAY_PID" 2>/dev/null || {{
    echo FAILED > "$DEPLOY_DIR/state"
    tail -n 200 "$DEPLOY_DIR/logs/gateway.log" >&2 || true
    exit 31
  }}
  sleep 2
done
curl -fsS http://127.0.0.1:8000/health >/dev/null || {{
  echo FAILED > "$DEPLOY_DIR/state"; exit 32;
}}

case "$EXPOSURE" in
  none)
    ENDPOINT="http://$(hostname -f):8000/v1"
    ;;
  cloudflare-quick)
    "$CLOUDFLARED" tunnel --url http://127.0.0.1:8000 --no-autoupdate \
      > "$DEPLOY_DIR/logs/cloudflared.log" 2>&1 &
    TUNNEL_PID=$!
    PIDS+=("$TUNNEL_PID")
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
    PIDS+=("$TUNNEL_PID")
    sleep 5
    kill -0 "$TUNNEL_PID" 2>/dev/null || {{ echo FAILED > "$DEPLOY_DIR/state"; exit 34; }}
    ENDPOINT="$NAMED_URL/v1"
    ;;
  *) echo "Unsupported exposure: $EXPOSURE" >&2; exit 35 ;;
esac

printf '%s\n' "$ENDPOINT" > "$DEPLOY_DIR/endpoint"
echo READY > "$DEPLOY_DIR/state"
echo "[delta-multimodal] READY: $ENDPOINT"
wait -n "${{PIDS[@]}}"
echo FAILED > "$DEPLOY_DIR/state"
exit 38
SERVICE_BODY

chmod 700 "$DEPLOY_DIR/service.slurm"
JOB_ID="$(sbatch --parsable "$DEPLOY_DIR/service.slurm")"
printf '%s\n' "$JOB_ID" > "$DEPLOY_DIR/job_id"
echo "[delta-multimodal] service job $JOB_ID submitted"

if [[ {str(params.detach).lower()} == true ]]; then
  echo "DELTA_LLM_RESULT|$DEPLOY_ID|$JOB_ID|SUBMITTED|-|-|$DEPLOY_DIR"
  exit 0
fi

for _ in $(seq 1 600); do
  STATE="$(cat "$DEPLOY_DIR/state" 2>/dev/null || true)"
  if [[ "$STATE" == READY && -s "$DEPLOY_DIR/endpoint" ]]; then
    ENDPOINT="$(< "$DEPLOY_DIR/endpoint")"
    EXPIRES="$(scontrol show job -o "$JOB_ID" | tr ' ' '\n' | sed -n 's/^EndTime=//p')"
    echo "DELTA_LLM_RESULT|$DEPLOY_ID|$JOB_ID|READY|$ENDPOINT|$EXPIRES|$DEPLOY_DIR"
    exit 0
  fi
  if ! squeue -h -j "$JOB_ID" | grep -q .; then
    FINAL="$(sacct -n -X -j "$JOB_ID" -o State | awk 'NF {{print $1; exit}}')"
    echo "ERROR: service job ended before ready: $FINAL" >&2
    tail -n 200 "$DEPLOY_DIR/logs/slurm_${{JOB_ID}}.err" 2>/dev/null || true
    tail -n 120 "$DEPLOY_DIR/logs/bagel.log" 2>/dev/null || true
    tail -n 120 "$DEPLOY_DIR/logs/thinkmorph.log" 2>/dev/null || true
    exit 36
  fi
  squeue -h -j "$JOB_ID" -o '[delta-multimodal] service %T: %R' | head -n 1
  sleep 15
done
echo "ERROR: timed out waiting for multimodal service" >&2
exit 37
"""


def render_status_script(config: Config, username: str, deployment_id: str) -> str:
    deploy_dir = f"{config.project_root}/{username}/delta-llm/deployments/{deployment_id}"
    return rf"""#!/usr/bin/env bash
set -euo pipefail
DEPLOY_DIR={q(deploy_dir)}
[[ -d "$DEPLOY_DIR" ]] || {{ echo "Deployment not found: {deployment_id}" >&2; exit 40; }}
JOB_ID="$(< "$DEPLOY_DIR/job_id")"
STATE="$(cat "$DEPLOY_DIR/state" 2>/dev/null || echo UNKNOWN)"
ENDPOINT="$(cat "$DEPLOY_DIR/endpoint" 2>/dev/null || echo -)"
if squeue -h -j "$JOB_ID" | grep -q .; then
  squeue -j "$JOB_ID" -o 'JOBID=%i STATE=%T ELAPSED=%M LIMIT=%l NODE=%R'
else
  sacct -X -j "$JOB_ID" -o JobID,State,Elapsed,Timelimit,NodeList
fi
echo "DEPLOYMENT={deployment_id} STATE=$STATE ENDPOINT=$ENDPOINT"
echo "DELTA_LLM_RESULT|{deployment_id}|$JOB_ID|$STATE|$ENDPOINT|-|$DEPLOY_DIR"
"""


def render_logs_script(config: Config, username: str, deployment_id: str, lines: int) -> str:
    deploy_dir = f"{config.project_root}/{username}/delta-llm/deployments/{deployment_id}"
    return rf"""#!/usr/bin/env bash
set -euo pipefail
DEPLOY_DIR={q(deploy_dir)}
[[ -d "$DEPLOY_DIR" ]] || {{ echo "Deployment not found" >&2; exit 40; }}
for name in bagel thinkmorph gateway cloudflared; do
  file="$DEPLOY_DIR/logs/$name.log"
  [[ -f "$file" ]] || continue
  echo "===== $name ====="
  tail -n {lines} "$file"
done
"""


def render_stop_script(config: Config, username: str, deployment_id: str) -> str:
    deploy_dir = f"{config.project_root}/{username}/delta-llm/deployments/{deployment_id}"
    return rf"""#!/usr/bin/env bash
set -euo pipefail
DEPLOY_DIR={q(deploy_dir)}
[[ -d "$DEPLOY_DIR" ]] || {{ echo "Deployment not found" >&2; exit 40; }}
JOB_ID="$(< "$DEPLOY_DIR/job_id")"
scancel "$JOB_ID" 2>/dev/null || true
rm -f "$DEPLOY_DIR/secrets/api_key" "$DEPLOY_DIR/secrets/cf_tunnel_token" "$DEPLOY_DIR/endpoint"
echo STOPPED > "$DEPLOY_DIR/state"
echo "Stopped job $JOB_ID and revoked its API key"
echo "DELTA_LLM_RESULT|{deployment_id}|$JOB_ID|STOPPED|-|-|$DEPLOY_DIR"
"""


def render_list_script(config: Config, username: str) -> str:
    root = f"{config.project_root}/{username}/delta-llm/deployments"
    return rf"""#!/usr/bin/env bash
set -euo pipefail
ROOT={q(root)}
[[ -d "$ROOT" ]] || {{ echo "No deployments"; exit 0; }}
printf '%-54s %-12s %-12s %s\n' DEPLOYMENT JOB STATE ENDPOINT
for dir in "$ROOT"/*; do
  [[ -d "$dir" && -f "$dir/job_id" ]] || continue
  id="${{dir##*/}}"; job="$(< "$dir/job_id")"
  state="$(cat "$dir/state" 2>/dev/null || echo UNKNOWN)"
  endpoint="$(cat "$dir/endpoint" 2>/dev/null || echo -)"
  printf '%-54s %-12s %-12s %s\n' "$id" "$job" "$state" "$endpoint"
done
"""


def render_doctor_script(config: Config) -> str:
    return r"""#!/usr/bin/env bash
set -euo pipefail
echo '=== ACCOUNT ==='
accounts
echo '=== A40 PARTITION ==='
sinfo -p gpuA40x4 -o '%P %a %l %D %t %G'
echo '=== STORAGE ==='
quota 2>/dev/null || true
echo '=== OUTBOUND ==='
curl -fsSI --max-time 15 https://huggingface.co | head -n 1
echo '=== CUDA MODULES ==='
module -t avail cuda 2>&1 | tail -n 20 || true
"""


def render_setup_status_script(config: Config, username: str) -> str:
    user_root = f"{config.project_root}/{username}/delta-llm"
    return rf"""#!/usr/bin/env bash
set -u
ROOT={q(user_root)}
SHARED={q(config.shared_root)}
RUNTIME={q(config.runtime_root)}
echo '=== TIME / HOST ==='
date -Is
hostname -f
echo '=== ACTIVE JOBS ==='
squeue -u "$USER" -o '%i|%j|%T|%M|%l|%R' || true
echo '=== SETUP ACCOUNTING (TODAY) ==='
sacct -S "$(date +%F)" -u "$USER" --name delta-mm-setup -X -n -P \
  -o JobID,JobName,State,Elapsed,Timelimit,ExitCode,NodeList 2>/dev/null || true
echo '=== SHARED LOCK / READY ==='
ls -ld "$SHARED"/envs/*.installing "$RUNTIME"/envs/*.installing 2>/dev/null || \
  echo 'no installing lock'
find "$SHARED/envs" "$RUNTIME/envs" -maxdepth 2 \
  -name '.delta-multimodal-ready' -type f \
  -print -exec cat {{}} \; 2>/dev/null || true
echo '=== SHARED USAGE ==='
du -sh "$SHARED/envs" "$RUNTIME/envs" "$SHARED/models" "$SHARED/sources" \
  2>/dev/null || true
echo '=== CHECKPOINTS ==='
find "$SHARED/models" -maxdepth 2 -type f \
  \( -name 'ema.safetensors' -o -name 'model.safetensors' -o -name 'ae.safetensors' \) \
  -printf '%TY-%Tm-%Td %TH:%TM | %s bytes | %p\n' 2>/dev/null | sort
echo '=== NEWEST SETUP LOGS ==='
find "$ROOT/deployments" -type f \
  \( -name 'setup_*.err' -o -name 'setup_*.out' \) \
  -printf '%T@|%p\n' 2>/dev/null | sort -nr | head -n 4 | cut -d'|' -f2- |
while IFS= read -r file; do
  [[ -n "$file" ]] || continue
  echo "--- $file"
  tail -n 80 "$file"
done
echo '=== END ==='
"""
