from pathlib import Path

from delta_llm.config import load_config


def test_load_config(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[delta]
account = "example-gpu"
default_hours = 12
[runtime]
torch_version = "2.5.0"
transformers_version = "4.48.0"
vllm_package_version = "9.8.7+cu999"
vllm_wheel_url = "https://example.invalid/vllm.whl"
vllm_wheel_sha256 = "deadbeef"
vllm_torch_index_url = "https://example.invalid/torch"
bagel_commit = "abc123"
runtime_root = "/work/nvme/example/runtime"
[exposure]
default_mode = "cloudflare-quick"
tailscale_version = "1.99.0"
tailscale_url = "https://example.invalid/tailscale.tgz"
""",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.account == "example-gpu"
    assert config.default_hours == 12
    assert config.torch_version == "2.5.0"
    assert config.transformers_version == "4.48.0"
    assert config.vllm_package_version == "9.8.7+cu999"
    assert config.vllm_wheel_url == "https://example.invalid/vllm.whl"
    assert config.vllm_wheel_sha256 == "deadbeef"
    assert config.vllm_torch_index_url == "https://example.invalid/torch"
    assert config.bagel_commit == "abc123"
    assert config.runtime_root == "/work/nvme/example/runtime"
    assert config.default_exposure == "cloudflare-quick"
    assert config.tailscale_version == "1.99.0"
    assert config.tailscale_url == "https://example.invalid/tailscale.tgz"
