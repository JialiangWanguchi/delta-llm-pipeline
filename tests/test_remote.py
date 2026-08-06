import pytest

from delta_llm.remote import parse_result_line, validate_deployment_id, validate_username


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
