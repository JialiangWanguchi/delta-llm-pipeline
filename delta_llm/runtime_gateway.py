from __future__ import annotations

import base64
import io
import ipaddress
import os
import queue
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import requests
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image

API_KEY = os.environ["DELTA_MULTIMODAL_API_KEY"]


def worker_urls(plural_name: str, singular_name: str, default: str) -> list[str]:
    value = os.environ.get(plural_name) or os.environ.get(singular_name) or default
    urls = [item.strip().rstrip("/") for item in value.split(",") if item.strip()]
    if not urls:
        raise RuntimeError(f"{plural_name} contains no worker URLs")
    return urls


WORKERS = {
    "bagel-7b": worker_urls(
        "BAGEL_WORKER_URLS", "BAGEL_WORKER_URL", "http://127.0.0.1:8101"
    ),
    "thinkmorph-7b": worker_urls(
        "THINKMORPH_WORKER_URLS",
        "THINKMORPH_WORKER_URL",
        "http://127.0.0.1:8102",
    ),
}
REQUEST_TIMEOUT = int(os.environ.get("WORKER_TIMEOUT_SECONDS", "3600"))
JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_SECONDS", "7200"))
MAX_JOBS = int(os.environ.get("MAX_JOBS", "500"))
MAX_SYNC_REQUESTS = int(os.environ.get("MAX_SYNC_REQUESTS", "8"))
MAX_REQUEST_BYTES = int(os.environ.get("MAX_REQUEST_BYTES", str(64 * 1024 * 1024)))
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", str(16_000_000)))
IMAGE_DOWNLOAD_TIMEOUT = float(os.environ.get("IMAGE_DOWNLOAD_TIMEOUT_SECONDS", "15"))
IMAGE_URL_HOST_ALLOWLIST = {
    item.strip().lower()
    for item in os.environ.get("IMAGE_URL_HOST_ALLOWLIST", "").split(",")
    if item.strip()
}
ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
app = FastAPI(title="Delta BAGEL + ThinkMorph API", version="2.0.0")

_round_robin_lock = threading.Lock()
_round_robin_index = {model: 0 for model in WORKERS}
_sync_slots = threading.BoundedSemaphore(MAX_SYNC_REQUESTS)


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    raw_length = request.headers.get("content-length")
    if raw_length:
        try:
            length = int(raw_length)
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length"})
        if length > MAX_REQUEST_BYTES:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request body exceeds {MAX_REQUEST_BYTES} bytes"},
            )
    return await call_next(request)


def authorize(authorization: str | None = Header(default=None)) -> None:
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def validate_model(model: str) -> None:
    if model not in WORKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {model!r}; choose one of {sorted(WORKERS)}",
        )


def enforce_text_only_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    task = str(normalized.get("task", "image-understanding"))
    if task != "image-understanding":
        raise HTTPException(
            status_code=400,
            detail="This deployment only supports text output; task must be image-understanding",
        )
    normalized["task"] = "image-understanding"
    content = normalized.get("content")
    if content is not None:
        if not isinstance(content, list) or not content:
            raise HTTPException(status_code=400, detail="content must be a non-empty array")
        if len(content) > 64:
            raise HTTPException(status_code=400, detail="At most 64 content items are supported")
        image_count = sum(
            1 for item in content if isinstance(item, dict) and item.get("type") == "image"
        )
    else:
        images = normalized.get("images")
        image_count = len(images) if isinstance(images, list) else int(bool(normalized.get("image")))
    if image_count > 24:
        raise HTTPException(status_code=400, detail="At most 24 images are supported")
    return normalized


def resolve_custom_image_urls(payload: dict[str, Any]) -> dict[str, Any]:
    """Resolve approved object URLs; safe to run inside an async job thread."""
    resolved = dict(payload)

    def resolve(value: Any, location: str) -> str:
        if not isinstance(value, str) or not value:
            raise HTTPException(status_code=400, detail=f"{location} must be a non-empty string")
        return image_url_to_data_url(value) if value.startswith(("http://", "https://")) else value

    if resolved.get("image") is not None:
        resolved["image"] = resolve(resolved["image"], "image")
    if resolved.get("images") is not None:
        if not isinstance(resolved["images"], list):
            raise HTTPException(status_code=400, detail="images must be an array")
        resolved["images"] = [
            resolve(value, f"images[{index}]")
            for index, value in enumerate(resolved["images"])
        ]
    if resolved.get("content") is not None:
        normalized_content: list[Any] = []
        for index, item in enumerate(resolved["content"]):
            if not isinstance(item, dict) or item.get("type") != "image":
                normalized_content.append(item)
                continue
            normalized_item = dict(item)
            value = normalized_item.get("image", normalized_item.get("uri"))
            normalized_item["image"] = resolve(value, f"content[{index}].image")
            normalized_item.pop("uri", None)
            normalized_content.append(normalized_item)
        resolved["content"] = normalized_content
    return resolved


def _host_is_allowed(hostname: str) -> bool:
    if not IMAGE_URL_HOST_ALLOWLIST:
        return True
    hostname = hostname.lower().rstrip(".")
    return any(
        hostname == allowed or hostname.endswith(f".{allowed}")
        for allowed in IMAGE_URL_HOST_ALLOWLIST
    )


def _validate_public_image_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise HTTPException(status_code=400, detail="image_url must use http or https")
    if parsed.username or parsed.password:
        raise HTTPException(status_code=400, detail="image_url must not contain credentials")
    if not _host_is_allowed(parsed.hostname):
        raise HTTPException(status_code=400, detail="image_url host is not allowlisted")
    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(
                parsed.hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        }
    except socket.gaierror as exc:
        raise HTTPException(status_code=400, detail="image_url host cannot be resolved") from exc
    if not addresses or any(not ipaddress.ip_address(value).is_global for value in addresses):
        raise HTTPException(
            status_code=400,
            detail="image_url must resolve only to public network addresses",
        )


def _validate_image_bytes(raw: bytes, mime_type: str) -> None:
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Each image must be at most {MAX_IMAGE_BYTES} bytes",
        )
    if mime_type not in ALLOWED_IMAGE_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image MIME type {mime_type!r}",
        )
    try:
        with Image.open(io.BytesIO(raw)) as image:
            width, height = image.size
            actual_format = (image.format or "").upper()
            image.verify()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Downloaded image is invalid") from exc
    if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=f"Each image must contain at most {MAX_IMAGE_PIXELS} pixels",
        )
    expected_formats = {
        "image/jpeg": {"JPEG"},
        "image/png": {"PNG"},
        "image/webp": {"WEBP"},
    }
    if actual_format not in expected_formats[mime_type]:
        raise HTTPException(status_code=400, detail="Image bytes do not match Content-Type")


def image_url_to_data_url(url: str) -> str:
    if url.startswith("data:"):
        return url
    _validate_public_image_url(url)
    try:
        response = requests.get(
            url,
            timeout=(5, IMAGE_DOWNLOAD_TIMEOUT),
            stream=True,
            allow_redirects=False,
            headers={"User-Agent": "delta-llm-image-fetch/1.0"},
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=400, detail=f"Unable to download image_url: {exc}") from exc
    if 300 <= response.status_code < 400:
        raise HTTPException(status_code=400, detail="image_url redirects are not allowed")
    if response.status_code != 200:
        raise HTTPException(
            status_code=400,
            detail=f"image_url returned HTTP {response.status_code}",
        )
    mime_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
    declared_length = response.headers.get("content-length")
    if declared_length:
        try:
            if int(declared_length) > MAX_IMAGE_BYTES:
                raise HTTPException(status_code=413, detail="Downloaded image is too large")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="image_url returned invalid Content-Length") from exc
    chunks: list[bytes] = []
    size = 0
    try:
        for chunk in response.iter_content(64 * 1024):
            if not chunk:
                continue
            size += len(chunk)
            if size > MAX_IMAGE_BYTES:
                raise HTTPException(status_code=413, detail="Downloaded image is too large")
            chunks.append(chunk)
    finally:
        response.close()
    raw = b"".join(chunks)
    _validate_image_bytes(raw, mime_type)
    return f"data:{mime_type};base64,{base64.b64encode(raw).decode('ascii')}"


def chat_request_to_worker_payload(payload: dict[str, Any]) -> dict[str, Any]:
    model = str(payload.get("model", ""))
    validate_model(model)
    if payload.get("stream"):
        raise HTTPException(status_code=400, detail="stream=true is not supported")
    try:
        n = int(payload.get("n", 1))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="n must be an integer") from exc
    if n != 1:
        raise HTTPException(status_code=400, detail="Only n=1 is supported")
    modalities = payload.get("modalities")
    if modalities not in (None, ["text"]):
        raise HTTPException(status_code=400, detail="Only text output modality is supported")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty array")

    content: list[dict[str, str]] = []
    image_count = 0
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise HTTPException(status_code=400, detail=f"messages[{message_index}] must be an object")
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            raise HTTPException(status_code=400, detail=f"messages[{message_index}].role is invalid")
        message_content = message.get("content")
        if len(messages) > 1 or role != "user":
            content.append({"type": "text", "text": f"{role.capitalize()}:"})
        if isinstance(message_content, str):
            if not message_content.strip():
                raise HTTPException(status_code=400, detail="message text must not be empty")
            content.append({"type": "text", "text": message_content})
            continue
        if not isinstance(message_content, list) or not message_content:
            raise HTTPException(status_code=400, detail="message content must be text or a non-empty array")
        for part_index, part in enumerate(message_content):
            if not isinstance(part, dict):
                raise HTTPException(status_code=400, detail="message content parts must be objects")
            part_type = part.get("type")
            if part_type == "text":
                text = part.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise HTTPException(status_code=400, detail="text content must not be empty")
                content.append({"type": "text", "text": text})
            elif part_type == "image_url":
                if role != "user":
                    raise HTTPException(status_code=400, detail="image_url is only allowed in user messages")
                image_url = part.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                if not isinstance(url, str) or not url:
                    raise HTTPException(
                        status_code=400,
                        detail=f"messages[{message_index}].content[{part_index}].image_url is invalid",
                    )
                image_count += 1
                if image_count > 24:
                    raise HTTPException(status_code=400, detail="At most 24 images are supported")
                content.append({"type": "image", "image": image_url_to_data_url(url)})
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported content type {part_type!r}")

    if len(content) > 64:
        raise HTTPException(status_code=400, detail="At most 64 ordered content items are supported")
    try:
        max_tokens = int(payload.get("max_tokens", payload.get("max_completion_tokens", 512)))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="max_tokens must be an integer") from exc
    if not 1 <= max_tokens <= 4096:
        raise HTTPException(status_code=400, detail="max_tokens must be between 1 and 4096")
    try:
        temperature = float(payload.get("temperature", 0))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="temperature must be numeric") from exc
    if not 0 <= temperature <= 2:
        raise HTTPException(status_code=400, detail="temperature must be between 0 and 2")
    return {
        "model": model,
        "task": "image-understanding",
        "content": content,
        "thinking": bool(payload.get("thinking", False)),
        "max_output_tokens": max_tokens,
        "temperature": temperature,
        "do_sample": temperature > 0,
    }


def openai_chat_response(result: dict[str, Any]) -> dict[str, Any]:
    created = int(time.time())
    budget = result.get("token_budget") or {}
    completion_tokens = int(result.get("output_tokens") or 0)
    prompt_tokens = int(budget.get("input_tokens") or 0)
    max_output_tokens = int(budget.get("max_output_tokens") or 0)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created,
        "model": result.get("model"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.get("text") or ""},
                "finish_reason": (
                    "length"
                    if max_output_tokens and completion_tokens >= max_output_tokens
                    else "stop"
                ),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "input_image_count": result.get("input_image_count", 0),
        "input_content_types": result.get("input_content_types", []),
        "token_budget": budget,
    }


def synchronous_worker_request(model: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not _sync_slots.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Too many synchronous requests")
    try:
        return worker_request(model, payload)
    finally:
        _sync_slots.release()


def next_worker(model: str) -> str:
    validate_model(model)
    with _round_robin_lock:
        urls = WORKERS[model]
        index = _round_robin_index[model]
        _round_robin_index[model] = (index + 1) % len(urls)
        return urls[index]


def worker_request_to(worker: str, payload: dict[str, Any]) -> dict[str, Any]:
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


def worker_request(model: str, payload: dict[str, Any]) -> dict[str, Any]:
    return worker_request_to(next_worker(model), payload)


@dataclass
class Job:
    id: str
    model: str
    payload: dict[str, Any] | None
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    worker: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    error_status: int = 500


class JobManager:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.jobs: dict[str, Job] = {}
        self.queues = {model: queue.Queue() for model in WORKERS}
        for model, urls in WORKERS.items():
            for replica, url in enumerate(urls):
                thread = threading.Thread(
                    target=self._worker_loop,
                    args=(model, url),
                    name=f"{model}-replica-{replica}",
                    daemon=True,
                )
                thread.start()

    def _cleanup(self) -> None:
        cutoff = time.time() - JOB_TTL_SECONDS
        expired = [
            job_id
            for job_id, job in self.jobs.items()
            if job.status in {"succeeded", "failed"}
            and (job.finished_at or job.created_at) < cutoff
        ]
        for job_id in expired:
            del self.jobs[job_id]
        if len(self.jobs) <= MAX_JOBS:
            return
        finished = sorted(
            (
                job
                for job in self.jobs.values()
                if job.status in {"succeeded", "failed"}
            ),
            key=lambda job: job.finished_at or job.created_at,
        )
        for job in finished[: max(0, len(self.jobs) - MAX_JOBS)]:
            self.jobs.pop(job.id, None)

    def submit(self, payload: dict[str, Any]) -> Job:
        model = str(payload.get("model", ""))
        validate_model(model)
        with self.lock:
            self._cleanup()
            if len(self.jobs) >= MAX_JOBS:
                raise HTTPException(status_code=503, detail="Job queue is full")
            job = Job(id=f"job-{uuid.uuid4().hex}", model=model, payload=dict(payload))
            self.jobs[job.id] = job
            self.queues[model].put(job.id)
            return job

    def get(self, job_id: str) -> Job:
        with self.lock:
            self._cleanup()
            job = self.jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found or expired")
            return job

    def queue_position(self, job: Job) -> int | None:
        if job.status != "queued":
            return None
        with self.lock:
            queued = sorted(
                (
                    item
                    for item in self.jobs.values()
                    if item.model == job.model and item.status == "queued"
                ),
                key=lambda item: item.created_at,
            )
            for index, item in enumerate(queued, start=1):
                if item.id == job.id:
                    return index
        return None

    def snapshot(self, job: Job) -> dict[str, Any]:
        elapsed: float | None = None
        if job.started_at is not None:
            elapsed = (job.finished_at or time.time()) - job.started_at
        return {
            "id": job.id,
            "object": "inference.job",
            "model": job.model,
            "status": job.status,
            "queue_position": self.queue_position(job),
            "created_at": int(job.created_at),
            "started_at": int(job.started_at) if job.started_at else None,
            "finished_at": int(job.finished_at) if job.finished_at else None,
            "elapsed_seconds": round(elapsed, 3) if elapsed is not None else None,
            "error": job.error,
        }

    def _worker_loop(self, model: str, worker: str) -> None:
        work_queue = self.queues[model]
        while True:
            job_id = work_queue.get()
            try:
                with self.lock:
                    job = self.jobs.get(job_id)
                    if job is None:
                        continue
                    job.status = "running"
                    job.started_at = time.time()
                    job.worker = worker
                    payload = job.payload
                if payload is None:
                    raise RuntimeError("job payload is missing")
                payload = resolve_custom_image_urls(payload)
                result = worker_request_to(worker, payload)
                result["created"] = int(time.time())
                with self.lock:
                    job.status = "succeeded"
                    job.result = result
                    job.finished_at = time.time()
                    job.payload = None
            except HTTPException as exc:
                with self.lock:
                    job.status = "failed"
                    job.error = str(exc.detail)
                    job.error_status = exc.status_code
                    job.finished_at = time.time()
                    job.payload = None
            except Exception as exc:  # noqa: BLE001 - keep the queue worker alive
                with self.lock:
                    job.status = "failed"
                    job.error = str(exc)
                    job.error_status = 500
                    job.finished_at = time.time()
                    job.payload = None
            finally:
                work_queue.task_done()


jobs = JobManager()


@app.get("/health")
def health() -> dict[str, Any]:
    workers: dict[str, Any] = {}
    for model, urls in WORKERS.items():
        replicas: list[dict[str, Any]] = []
        for replica, url in enumerate(urls):
            try:
                response = requests.get(f"{url}/health", timeout=3)
                detail = response.json() if response.ok else {"status": "error"}
            except requests.RequestException:
                detail = {"status": "unavailable"}
            detail["replica"] = replica
            replicas.append(detail)
        workers[model] = {
            "status": (
                "ok" if all(item.get("status") == "ok" for item in replicas) else "degraded"
            ),
            "replica_count": len(replicas),
            "replicas": replicas,
        }
    status = "ok" if all(item["status"] == "ok" for item in workers.values()) else "degraded"
    return {"status": status, "workers": workers}


@app.get("/v1/models", dependencies=[Depends(authorize)])
def models() -> dict[str, Any]:
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "created": now,
                "owned_by": "delta-llm",
                "replicas": len(WORKERS[model]),
            }
            for model in WORKERS
        ],
    }


@app.post("/v1/generate", dependencies=[Depends(authorize)])
def generate(payload: dict[str, Any]) -> dict[str, Any]:
    payload = resolve_custom_image_urls(enforce_text_only_payload(payload))
    model = str(payload.get("model", ""))
    result = synchronous_worker_request(model, payload)
    result["created"] = int(time.time())
    return result


@app.post("/v1/chat/completions", dependencies=[Depends(authorize)])
def chat_completions(payload: dict[str, Any]) -> dict[str, Any]:
    worker_payload = chat_request_to_worker_payload(payload)
    result = synchronous_worker_request(worker_payload["model"], worker_payload)
    return openai_chat_response(result)


@app.post("/v1/jobs", dependencies=[Depends(authorize)], status_code=202)
def create_job(payload: dict[str, Any]) -> dict[str, Any]:
    job = jobs.submit(enforce_text_only_payload(payload))
    response = jobs.snapshot(job)
    response["status_url"] = f"/v1/jobs/{job.id}"
    response["result_url"] = f"/v1/jobs/{job.id}/result"
    return response


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(authorize)])
def get_job(job_id: str) -> dict[str, Any]:
    return jobs.snapshot(jobs.get(job_id))


@app.get(
    "/v1/jobs/{job_id}/result",
    dependencies=[Depends(authorize)],
    response_model=None,
)
def get_job_result(job_id: str) -> Any:
    job = jobs.get(job_id)
    if job.status in {"queued", "running"}:
        return JSONResponse(status_code=202, content=jobs.snapshot(job))
    if job.status == "failed":
        raise HTTPException(status_code=job.error_status, detail=job.error)
    result = dict(job.result or {})
    result["job_id"] = job.id
    result["elapsed_seconds"] = round((job.finished_at or 0) - (job.started_at or 0), 3)
    return result


@app.post("/v1/images/generations", dependencies=[Depends(authorize)])
def image_generation_disabled(payload: dict[str, Any]) -> dict[str, Any]:
    del payload
    raise HTTPException(
        status_code=410,
        detail="Image generation is disabled; this deployment returns text only",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.environ.get("GATEWAY_PORT", "8000")),
        log_level="info",
    )
