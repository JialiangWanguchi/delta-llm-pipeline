import importlib
import sys
import time

import pytest
from fastapi import HTTPException


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
