import shutil
import subprocess

import pytest

from delta_llm.catalog import GPU_SPECS, MODEL_SPECS
from delta_llm.config import Config
from delta_llm.templates import DeployParams, hours_to_slurm, render_deploy_script


def make_params(model: str = "qwen3-8b", exposure: str = "none") -> DeployParams:
    return DeployParams(
        username="testuser",
        deployment_id=f"{model}-20260806-120000-abcd",
        api_key="plain-secret-must-not-appear",
        model=MODEL_SPECS[model],
        gpu=GPU_SPECS["a40"],
        gpu_count=2 if model == "deepseek-r1-32b" else 1,
        hours=1,
        max_model_len=4096,
        exposure=exposure,
        hf_token="",
        cf_tunnel_token="named-secret" if exposure == "cloudflare-named" else "",
        detach=False,
    )


def test_hours_to_slurm() -> None:
    assert hours_to_slurm(47.5) == "47:30:00"
    assert hours_to_slurm(0.001) == "00:01:00"


@pytest.mark.parametrize(
    ("model", "exposure"),
    [
        ("qwen3-8b", "none"),
        ("qwen3-32b-awq", "cloudflare-quick"),
        ("deepseek-r1-32b", "cloudflare-named"),
    ],
)
def test_generated_script_is_safe_and_valid(model: str, exposure: str) -> None:
    config = Config(named_public_url="https://llm.example.org")
    script = render_deploy_script(config, make_params(model, exposure))
    assert "plain-secret-must-not-appear" not in script
    assert "named-secret" not in script
    assert "#SBATCH --account=bhsz-delta-gpu" in script
    assert "export VLLM_API_KEY=" in script
    assert "\r" not in script

    bash = shutil.which("bash")
    if bash:
        result = subprocess.run(
            [bash, "-n"], input=script.encode(), capture_output=True, check=False
        )
        assert result.returncode == 0, result.stderr.decode(errors="replace")
