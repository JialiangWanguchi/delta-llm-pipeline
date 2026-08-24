import shutil
import subprocess
from dataclasses import replace

import pytest

from delta_llm.config import Config
from delta_llm.templates import (
    DeployParams,
    hours_to_slurm,
    render_deploy_script,
    render_logs_script,
    render_setup_status_script,
    render_split_deploy_script,
    render_status_script,
    render_stop_script,
)


def make_params(
    exposure: str = "none", gpu_type: str = "a100", gpu_count: int = 4
) -> DeployParams:
    return DeployParams(
        username="testuser",
        deployment_id="bagel-thinkmorph-20260807-120000-abcd",
        api_key="plain-secret-must-not-appear",
        worker_api_key="internal-secret-must-not-appear",
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        hours=1,
        exposure=exposure,
        hf_token="hf-secret-must-not-appear",
        cf_tunnel_token="named-secret" if exposure == "cloudflare-named" else "",
        detach=False,
        recover_stalled_setup=False,
        replace_existing_services=False,
        split_jobs=False,
    )


def test_hours_to_slurm() -> None:
    assert hours_to_slurm(47.5) == "47:30:00"
    assert hours_to_slurm(0.001) == "00:01:00"


@pytest.mark.parametrize("exposure", ["none", "cloudflare-quick", "cloudflare-named"])
def test_generated_dual_model_script_is_safe_and_valid(exposure: str) -> None:
    config = Config(named_public_url="https://models.example.org")
    script = render_deploy_script(config, make_params(exposure))
    assert "plain-secret-must-not-appear" not in script
    assert "hf-secret-must-not-appear" not in script
    assert "named-secret" not in script
    assert "#SBATCH --partition=gpuA100x4" in script
    assert "#SBATCH --gpus-per-node=4" in script
    assert "#SBATCH --cpus-per-task=48" in script
    assert "#SBATCH --mem=220g" in script
    assert "--model bagel-7b" in script
    assert "--model thinkmorph-7b" in script
    assert 'BAGEL_CUDA="${CUDA_IDS[$MODEL_GPU_OFFSET]}"' in script
    assert 'THINKMORPH_CUDA="${CUDA_IDS[$((GPUS_PER_MODEL + MODEL_GPU_OFFSET))]}"' in script
    assert script.count("--load-mode resident") == 2
    assert script.count("--max-memory-gib 38") == 2
    assert "PORT_BASE=$((20000 + SLURM_JOB_ID % 30000))" in script
    assert 'export BAGEL_WORKER_URLS=' in script
    assert 'export THINKMORPH_WORKER_URLS=' in script
    assert '"http://127.0.0.1:$GATEWAY_PORT/v1/models"' in script
    assert '"/v1/chat/completions" in paths' in script
    assert "export EFFECTIVE_CONTEXT_LIMIT=28672" in script
    assert "export VIT_MAX_IMAGE_SIZE=336" in script
    assert 'h["text_output_only"] is True' in script
    assert "http://127.0.0.1:8000" not in script
    assert "ByteDance-Seed/BAGEL-7B-MoT" in script
    assert "ThinkMorph/ThinkMorph-7B" in script
    assert "DELTA_MULTIMODAL_API_KEY" in script
    assert "runtime/worker.py" in script
    assert "runtime/gateway.py" in script
    assert ".delta-multimodal-ready" in script
    assert "#SBATCH --time=04:00:00" in script
    assert "/work/nvme/bhsz/delta-llm/shared/envs/" in script
    assert "--solver libmamba" in script
    assert 'printf \'%s\\n\' "$SETUP_JOB" > "$LOCK_DIR/job_id"' in script
    assert "seq 1 2160" in script
    assert "seq 1 12000" in script
    assert "\r" not in script

    setup_body = script.split("<<SETUP_SLURM", 1)[1].split("SETUP_SLURM", 1)[0]
    setup_continuation_lines = [
        line for line in setup_body.splitlines() if line.rstrip().endswith("\\")
    ]
    assert setup_continuation_lines
    assert all(line.rstrip().endswith("\\\\") for line in setup_continuation_lines)

    service_body = script.split("<<'SERVICE_BODY'", 1)[1].split("SERVICE_BODY", 1)[0]
    continuation_lines = [
        line for line in service_body.splitlines() if line.rstrip().endswith("\\")
    ]
    assert continuation_lines
    assert all(not line.rstrip().endswith("\\\\") for line in continuation_lines)

    bash = shutil.which("bash")
    if bash:
        result = subprocess.run(
            [bash, "-n"], input=script.encode(), capture_output=True, check=False
        )
        assert result.returncode == 0, result.stderr.decode(errors="replace")


def test_runtime_sources_are_embedded() -> None:
    script = render_deploy_script(Config(), make_params())
    assert "printf '%s'" in script
    assert 'base64 -d > "$DEPLOY_DIR/runtime/worker.py"' in script


def test_four_a100_layout_runs_two_replicas_per_model() -> None:
    script = render_deploy_script(Config(), make_params())
    assert "#SBATCH --gpus-per-node=4" in script
    assert "#SBATCH --cpus-per-task=48" in script
    assert "#SBATCH --mem=220g" in script
    assert "GPU_LAYOUT=bagel-7b:2-replicas,thinkmorph-7b:2-replicas" in script
    assert "REPLICAS_PER_MODEL=2" in script


def test_two_h200_layout_runs_two_replicas_per_model_on_one_gpu_each() -> None:
    script = render_deploy_script(Config(), make_params(gpu_type="h200", gpu_count=2))
    assert "#SBATCH --partition=gpuH200x8" in script
    assert "#SBATCH --gpus-per-node=2" in script
    assert "GPU_TYPE=h200" in script
    assert "GPUS_PER_MODEL=$((GPU_COUNT / 2))" in script
    assert 'BAGEL_CUDA="${CUDA_IDS[$MODEL_GPU_OFFSET]}"' in script
    assert 'THINKMORPH_CUDA="${CUDA_IDS[$((GPUS_PER_MODEL + MODEL_GPU_OFFSET))]}"' in script


def test_split_h200_layout_submits_two_authenticated_single_gpu_jobs() -> None:
    params = replace(
        make_params(exposure="cloudflare-quick", gpu_type="h200", gpu_count=2),
        split_jobs=True,
    )
    script = render_split_deploy_script(Config(), params)
    assert "plain-secret-must-not-appear" not in script
    assert "internal-secret-must-not-appear" not in script
    assert script.count("sbatch --parsable") == 2
    assert "#SBATCH --partition=gpuH200x8" in script
    assert "#SBATCH --gpus-per-node=1" in script
    assert "ROLE=bagel,MODEL_NAME=bagel-7b" in script
    assert "ROLE=thinkmorph,MODEL_NAME=thinkmorph-7b" in script
    assert "--host 0.0.0.0" in script
    assert "DELTA_WORKER_API_KEY" in script
    assert 'export BAGEL_WORKER_URLS="$BAGEL_URLS"' in script
    assert "THINK_READY=true" in script
    assert "failures=$((failures + 1))" in script
    assert "slurm_bagel_%j.err" in script
    assert "slurm_thinkmorph_%j.err" in script
    bash = shutil.which("bash")
    if bash:
        result = subprocess.run(
            [bash, "-n"], input=script.encode(), capture_output=True, check=False
        )
        assert result.returncode == 0, result.stderr.decode(errors="replace")


def test_split_job_management_handles_comma_separated_job_ids() -> None:
    status = render_status_script(Config(), "testuser", "deployment")
    logs = render_logs_script(Config(), "testuser", "deployment", 50)
    stop = render_stop_script(Config(), "testuser", "deployment")
    assert "IFS=',' read -r -a JOB_IDS" in status
    assert 'for job in "${JOB_IDS[@]}"' in status
    assert "slurm_bagel_*.out" in logs
    assert "slurm_thinkmorph_*.err" in logs
    assert 'scancel "${JOB_IDS[@]}"' in stop
    assert 'secrets/worker_api_key"' in stop


def test_recovery_is_explicit_and_scoped() -> None:
    params = replace(make_params(), recover_stalled_setup=True)
    script = render_deploy_script(Config(), params)
    assert "RECOVER_STALLED_SETUP=true" in script
    assert 'squeue -h -u "$USER" -n delta-mm-setup' in script
    assert 'rm -rf "$LEGACY_ENV_DIR.installing" "$LEGACY_ENV_DIR"' in script


def test_service_replacement_is_explicit_and_scoped() -> None:
    params = replace(make_params(), replace_existing_services=True)
    script = render_deploy_script(Config(), params)
    assert "REPLACE_EXISTING_SERVICES=true" in script
    assert "/^mm-bagel-thinkmorph-/" in script
    assert 'scancel "${OLD_SERVICE_JOBS[@]}"' in script


def test_setup_status_script_is_read_only_and_valid() -> None:
    script = render_setup_status_script(Config(), "testuser")
    assert "/projects/bhsz/testuser/delta-llm" in script
    assert "squeue" in script
    assert "sacct" in script
    assert "CHECKPOINTS" in script
    assert "rm " not in script
    bash = shutil.which("bash")
    if bash:
        result = subprocess.run(
            [bash, "-n"], input=script.encode(), capture_output=True, check=False
        )
        assert result.returncode == 0, result.stderr.decode(errors="replace")
