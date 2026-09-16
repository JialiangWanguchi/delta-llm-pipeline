from __future__ import annotations

import os

import pytest
from fastapi import HTTPException

os.environ.setdefault("DELTA_MULTIMODAL_API_KEY", "test-vllm-key")

from delta_llm.runtime_vllm_proxy import authorize, validate_chat_payload


def payload_with_images(count: int) -> dict:
    content = []
    for index in range(count):
        content.extend(
            [
                {"type": "text", "text": f"stage {index}"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA=="},
                },
            ]
        )
    return {
        "model": "bagel-7b",
        "messages": [{"role": "user", "content": content}],
    }


def test_vllm_proxy_accepts_24_ordered_images() -> None:
    validate_chat_payload(payload_with_images(24))


def test_vllm_proxy_rejects_a_25th_image() -> None:
    with pytest.raises(HTTPException, match="At most 24 images"):
        validate_chat_payload(payload_with_images(25))


def test_vllm_proxy_rejects_remote_image_urls() -> None:
    payload = payload_with_images(1)
    payload["messages"][0]["content"][1]["image_url"]["url"] = "https://example.org/a.png"
    with pytest.raises(HTTPException, match="data:image"):
        validate_chat_payload(payload)


def test_vllm_proxy_requires_the_external_bearer_key() -> None:
    authorize("Bearer test-vllm-key")
    with pytest.raises(HTTPException, match="Invalid or missing API key"):
        authorize("Bearer wrong")
