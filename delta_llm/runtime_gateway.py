from __future__ import annotations

import os
import time
import uuid
from typing import Any

import requests
from fastapi import Depends, FastAPI, Header, HTTPException

API_KEY = os.environ["DELTA_MULTIMODAL_API_KEY"]
WORKERS = {
    "bagel-7b": os.environ.get("BAGEL_WORKER_URL", "http://127.0.0.1:8101"),
    "thinkmorph-7b": os.environ.get("THINKMORPH_WORKER_URL", "http://127.0.0.1:8102"),
}
REQUEST_TIMEOUT = int(os.environ.get("WORKER_TIMEOUT_SECONDS", "3600"))
app = FastAPI(title="Delta BAGEL + ThinkMorph API", version="1.0.0")


def authorize(authorization: str | None = Header(default=None)) -> None:
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def worker_request(model: str, payload: dict[str, Any]) -> dict[str, Any]:
    worker = WORKERS.get(model)
    if worker is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {model!r}; choose one of {sorted(WORKERS)}",
        )
    try:
        response = requests.post(
            f"{worker}/generate",
            json=payload,
            timeout=(10, REQUEST_TIMEOUT),
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail=f"Model worker unavailable: {exc}") from exc
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise HTTPException(status_code=response.status_code, detail=detail)
    return response.json()


@app.get("/health")
def health() -> dict[str, Any]:
    workers: dict[str, Any] = {}
    for model, url in WORKERS.items():
        try:
            response = requests.get(f"{url}/health", timeout=3)
            workers[model] = response.json() if response.ok else {"status": "error"}
        except requests.RequestException:
            workers[model] = {"status": "unavailable"}
    status = "ok" if all(item.get("status") == "ok" for item in workers.values()) else "degraded"
    return {"status": status, "workers": workers}


@app.get("/v1/models", dependencies=[Depends(authorize)])
def models() -> dict[str, Any]:
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": model, "object": "model", "created": now, "owned_by": "delta-llm"}
            for model in WORKERS
        ],
    }


@app.post("/v1/generate", dependencies=[Depends(authorize)])
def generate(payload: dict[str, Any]) -> dict[str, Any]:
    model = str(payload.get("model", ""))
    result = worker_request(model, payload)
    result["created"] = int(time.time())
    return result


@app.post("/v1/images/generations", dependencies=[Depends(authorize)])
def image_generation(payload: dict[str, Any]) -> dict[str, Any]:
    model = str(payload.get("model", ""))
    prompt = payload.get("prompt")
    worker_payload = {
        "model": model,
        "task": "text-to-image",
        "prompt": prompt,
        "size": payload.get("size", "512x512"),
        "thinking": payload.get("thinking", model == "thinkmorph-7b"),
        "max_think_tokens": payload.get("max_think_tokens", 512),
        "max_rounds": payload.get("max_rounds", 1),
        "steps": payload.get("steps", 30),
        "seed": payload.get("seed", 0),
    }
    result = worker_request(model, worker_payload)
    return {
        "id": f"img-{uuid.uuid4().hex}",
        "created": int(time.time()),
        "model": model,
        "data": [
            {"b64_json": image.split(",", 1)[1], "revised_prompt": prompt}
            for image in result.get("images", [])
        ],
        "thinking": result.get("text"),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
