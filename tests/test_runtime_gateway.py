import importlib
import io
import sys
import time

import pytest
from fastapi import HTTPException
from PIL import Image


def load_gateway(monkeypatch):
    monkeypatch.setenv("DELTA_MULTIMODAL_API_KEY", "shared-test-key")
    sys.modules.pop("delta_llm.runtime_gateway", None)
    return importlib.import_module("delta_llm.runtime_gateway")


def test_gateway_requires_shared_bearer_key(monkeypatch) -> None:
    gateway = load_gateway(monkeypatch)
    gateway.authorize("Bearer shared-test-key")
    with pytest.raises(HTTPException) as exc:
        gateway.authorize("Bearer wrong-key")
    assert exc.value.status_code == 401


def test_gateway_rejects_unknown_model(monkeypatch) -> None:
    gateway = load_gateway(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        gateway.worker_request("not-a-model", {"prompt": "x"})
    assert exc.value.status_code == 400


def test_async_job_completes_and_returns_result(monkeypatch) -> None:
    gateway = load_gateway(monkeypatch)
    monkeypatch.setattr(
        gateway,
        "worker_request_to",
        lambda worker, payload: {
            "model": payload["model"],
            "task": payload["task"],
            "text": "two cats",
            "images": [],
        },
    )
    manager = gateway.JobManager()
    job = manager.submit(
        {"model": "bagel-7b", "task": "image-understanding", "prompt": "describe"}
    )
    deadline = time.time() + 2
    while job.status not in {"succeeded", "failed"} and time.time() < deadline:
        time.sleep(0.01)
    assert job.status == "succeeded"
    assert job.result is not None
    assert job.result["text"] == "two cats"
    assert job.payload is None


def image_data_url(color: str) -> str:
    import base64

    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color=color).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def test_chat_request_preserves_interleaved_content_order(monkeypatch) -> None:
    gateway = load_gateway(monkeypatch)
    first = image_data_url("red")
    second = image_data_url("blue")
    worker_payload = gateway.chat_request_to_worker_payload(
        {
            "model": "thinkmorph-7b",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "before"},
                        {"type": "image_url", "image_url": {"url": first}},
                        {"type": "text", "text": "between"},
                        {"type": "image_url", "image_url": {"url": second}},
                        {"type": "text", "text": "answer in text"},
                    ],
                }
            ],
            "modalities": ["text"],
            "max_tokens": 128,
            "temperature": 0,
        }
    )
    assert worker_payload["task"] == "image-understanding"
    assert [item["type"] for item in worker_payload["content"]] == [
        "text",
        "image",
        "text",
        "image",
        "text",
    ]
    assert worker_payload["thinking"] is False
    assert worker_payload["do_sample"] is False


def test_chat_response_is_text_only_openai_shape(monkeypatch) -> None:
    gateway = load_gateway(monkeypatch)
    response = gateway.openai_chat_response(
        {
            "model": "bagel-7b",
            "text": "final answer",
            "output_tokens": 3,
            "input_image_count": 2,
            "input_content_types": ["text", "image", "text", "image", "text"],
            "token_budget": {"input_tokens": 1200},
        }
    )
    assert response["choices"][0]["message"] == {
        "role": "assistant",
        "content": "final answer",
    }
    assert response["usage"] == {
        "prompt_tokens": 1200,
        "completion_tokens": 3,
        "total_tokens": 1203,
    }
    assert "images" not in response


def test_gateway_rejects_non_text_tasks_and_modalities(monkeypatch) -> None:
    gateway = load_gateway(monkeypatch)
    with pytest.raises(HTTPException) as task_error:
        gateway.enforce_text_only_payload({"task": "text-to-image"})
    assert task_error.value.status_code == 400
    with pytest.raises(HTTPException) as modality_error:
        gateway.chat_request_to_worker_payload(
            {
                "model": "bagel-7b",
                "messages": [{"role": "user", "content": "hello"}],
                "modalities": ["image"],
            }
        )
    assert modality_error.value.status_code == 400


def test_gateway_blocks_private_image_urls(monkeypatch) -> None:
    gateway = load_gateway(monkeypatch)
    monkeypatch.setattr(
        gateway.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, ("127.0.0.1", 80))],
    )
    with pytest.raises(HTTPException) as exc:
        gateway._validate_public_image_url("http://example.test/image.png")
    assert exc.value.status_code == 400


def test_async_custom_content_resolves_object_urls_in_order(monkeypatch) -> None:
    gateway = load_gateway(monkeypatch)
    seen: list[str] = []

    def fake_resolve(url: str) -> str:
        seen.append(url)
        return f"data:image/png;base64,{len(seen)}"

    monkeypatch.setattr(gateway, "image_url_to_data_url", fake_resolve)
    resolved = gateway.resolve_custom_image_urls(
        {
            "content": [
                {"type": "text", "text": "first"},
                {"type": "image", "uri": "https://objects.test/01.png"},
                {"type": "text", "text": "second"},
                {"type": "image", "image": "https://objects.test/02.png"},
            ]
        }
    )
    assert seen == ["https://objects.test/01.png", "https://objects.test/02.png"]
    assert [item["type"] for item in resolved["content"]] == [
        "text",
        "image",
        "text",
        "image",
    ]
    assert "uri" not in resolved["content"][1]
