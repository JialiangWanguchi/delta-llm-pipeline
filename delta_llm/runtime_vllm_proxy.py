from __future__ import annotations

import importlib.metadata
import os
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import Response

API_KEY = os.environ["DELTA_MULTIMODAL_API_KEY"]
UPSTREAM = os.environ.get("VLLM_UPSTREAM", "http://127.0.0.1:8100").rstrip("/")
MAX_REQUEST_BYTES = int(os.environ.get("MAX_REQUEST_BYTES", str(64 * 1024 * 1024)))
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "24"))
MAX_CONTENT_ITEMS = int(os.environ.get("MAX_CONTENT_ITEMS", "64"))
REQUEST_TIMEOUT = float(os.environ.get("WORKER_TIMEOUT_SECONDS", "3600"))
SERVED_MODEL = os.environ.get("SERVED_MODEL_NAME", "bagel-7b")

app = FastAPI(title="Delta multimodal vLLM API", version="1.0.0")


def authorize(authorization: str | None = Header(default=None)) -> None:
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def validate_chat_payload(payload: dict[str, Any]) -> None:
    if payload.get("model") != SERVED_MODEL:
        raise HTTPException(status_code=400, detail=f"Only {SERVED_MODEL} is available")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty list")

    image_count = 0
    content_count = 0
    for message in messages:
        if not isinstance(message, dict):
            raise HTTPException(status_code=400, detail="Each message must be an object")
        content = message.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            raise HTTPException(status_code=400, detail="Message content must be text or a list")
        content_count += len(content)
        for item in content:
            if not isinstance(item, dict):
                raise HTTPException(status_code=400, detail="Content items must be objects")
            item_type = item.get("type")
            if item_type == "text":
                if not isinstance(item.get("text"), str):
                    raise HTTPException(status_code=400, detail="Text content is invalid")
            elif item_type == "image_url":
                image = item.get("image_url")
                url = image.get("url") if isinstance(image, dict) else None
                if not isinstance(url, str) or not url.startswith("data:image/"):
                    raise HTTPException(
                        status_code=400,
                        detail="Images must use data:image/... URLs",
                    )
                image_count += 1
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported content type: {item_type}")

    if image_count > MAX_IMAGES:
        raise HTTPException(status_code=400, detail=f"At most {MAX_IMAGES} images are allowed")
    if content_count > MAX_CONTENT_ITEMS:
        raise HTTPException(
            status_code=400,
            detail=f"At most {MAX_CONTENT_ITEMS} ordered content items are allowed",
        )


async def upstream_request(method: str, path: str, payload: dict[str, Any] | None = None) -> Response:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.request(method, f"{UPSTREAM}{path}", json=payload)
    headers = {}
    if content_type := response.headers.get("content-type"):
        headers["content-type"] = content_type
    return Response(content=response.content, status_code=response.status_code, headers=headers)


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    raw_length = request.headers.get("content-length")
    if raw_length:
        try:
            length = int(raw_length)
        except ValueError:
            return Response(content="Invalid Content-Length", status_code=400)
        if length > MAX_REQUEST_BYTES:
            return Response(content="Request body too large", status_code=413)
    return await call_next(request)


@app.get("/health")
async def health() -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get(f"{UPSTREAM}/health")
    return {
        "status": "ok" if response.is_success else "degraded",
        "engine": "vllm",
        "vllm_version": importlib.metadata.version("vllm"),
        "model": SERVED_MODEL,
        "max_images": MAX_IMAGES,
    }


@app.get("/v1/models", dependencies=[Depends(authorize)])
async def models() -> Response:
    return await upstream_request("GET", "/v1/models")


@app.post("/v1/chat/completions", dependencies=[Depends(authorize)])
async def chat_completions(payload: dict[str, Any]) -> Response:
    validate_chat_payload(payload)
    if payload.get("stream"):
        raise HTTPException(status_code=400, detail="Streaming is not enabled on this gateway")
    return await upstream_request("POST", "/v1/chat/completions", payload)
