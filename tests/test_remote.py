import json

import pytest

from delta_llm.remote import (
    mark_local_state_stopped,
    parse_result_line,
    validate_deployment_id,
    validate_username,
)


def test_parse_result_line() -> None:
    result = parse_result_line(
        "DELTA_LLM_RESULT|qwen3-8b-20260806-abcd|123|READY|https://x/v1|2026-08-08|/p/d"
    )
    assert result.job_id == "123"
    assert result.endpoint == "https://x/v1"


@pytest.mark.parametrize("value", ["bad user", "x;rm", "../x", ""])
def test_reject_unsafe_username(value: str) -> None:
    with pytest.raises(ValueError):
        validate_username(value)


def test_validate_deployment_id() -> None:
    assert validate_deployment_id("qwen3-8b-20260806-abcd") == "qwen3-8b-20260806-abcd"
    with pytest.raises(ValueError):
        validate_deployment_id("../../secret")


def test_mark_local_state_stopped_revokes_cached_credentials(tmp_path, monkeypatch) -> None:
    deployment_id = "qwen3-8b-20260806-abcd"
    monkeypatch.setenv("DELTA_LLM_STATE_DIR", str(tmp_path))
    path = tmp_path / f"{deployment_id}.json"
    path.write_text(
        json.dumps(
            {
                "deployment_id": deployment_id,
                "state": "READY",
                "endpoint": "https://example.test/v1",
                "api_key": "must-be-revoked",
            }
        ),
        encoding="utf-8",
    )

    assert mark_local_state_stopped(deployment_id) == path
    state = json.loads(path.read_text(encoding="utf-8"))
    assert state["state"] == "STOPPED"
    assert state["endpoint"] == ""
    assert state["api_key"] == ""
