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
gpu_memory_utilization = 0.85
[exposure]
default_mode = "cloudflare-quick"
""",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.account == "example-gpu"
    assert config.default_hours == 12
    assert config.gpu_memory_utilization == 0.85
    assert config.default_exposure == "cloudflare-quick"
