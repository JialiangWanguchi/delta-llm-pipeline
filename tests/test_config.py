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
bagel_commit = "abc123"
runtime_root = "/work/nvme/example/runtime"
[exposure]
default_mode = "cloudflare-quick"
""",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.account == "example-gpu"
    assert config.default_hours == 12
    assert config.torch_version == "2.5.0"
    assert config.transformers_version == "4.48.0"
    assert config.bagel_commit == "abc123"
    assert config.runtime_root == "/work/nvme/example/runtime"
    assert config.default_exposure == "cloudflare-quick"
