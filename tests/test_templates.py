import shutil
import subprocess
from dataclasses import replace

import pytest

from delta_llm.config import Config
from delta_llm.templates import (
    DeployParams,
    hours_to_slurm,
    render_deploy_script,
    render_setup_status_script,
)


def make_params(exposure: str = "none") -> DeployParams:
    return DeployParams(
        username="testuser",
        deployment_id="bagel-thinkmorph-20260807-120000-abcd",
        api_key="plain-secret-must-not-appear",
        gpu_count=2,
        hours=1,
        exposure=exposure,
        hf_token="hf-secret-must-not-appear",
        cf_tunnel_token="named-secret" if exposure == "cloudflare-named" else "",
        detach=False,
        recover_stalled_setup=False,
        replace_existing_services=False,
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
    assert "#SBATCH --gpus-per-node=2" in script
    assert "#SBATCH --cpus-per-task=32" in script
    assert "#SBATCH --mem=120g" in script
    assert "--model bagel-7b" in script
    assert "--model thinkmorph-7b" in script
    assert 'BAGEL_CUDA="${CUDA_IDS[0]}"' in script
    assert 'THINKMORPH_CUDA="${CUDA_IDS[1]},${CUDA_IDS[2]}"' in script
    assert 'THINKMORPH_CUDA="${CUDA_IDS[1]}"' in script
    assert script.count("--max-memory-gib 36") == 2
    assert "PORT_BASE=$((20000 + SLURM_JOB_ID % 30000))" in script
    assert '--port "$BAGEL_PORT"' in script
    assert '--port "$THINKMORPH_PORT"' in script
    assert '"http://127.0.0.1:$GATEWAY_PORT/v1/models"' in script
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


def test_three_gpu_layout_keeps_two_gpus_for_thinkmorph() -> None:
    script = render_deploy_script(Config(), replace(make_params(), gpu_count=3))
    assert "#SBATCH --gpus-per-node=3" in script
    assert "#SBATCH --cpus-per-task=48" in script
    assert "#SBATCH --mem=220g" in script
    assert "GPU_LAYOUT=bagel-7b:1,thinkmorph-7b:2" in script


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
