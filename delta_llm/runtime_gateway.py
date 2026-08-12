from __future__ import annotations

import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import requests
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

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
app = FastAPI(title="Delta BAGEL + ThinkMorph API", version="2.0.0")

_round_robin_lock = threading.Lock()
_round_robin_index = {model: 0 for model in WORKERS}


def authorize(authorization: str | None = Header(default=None)) -> None:
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def validate_model(model: str) -> None:
    if model not in WORKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {model!r}; choose one of {sorted(WORKERS)}",
        )


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
    model = str(payload.get("model", ""))
    result = worker_request(model, payload)
    result["created"] = int(time.time())
    return result


@app.post("/v1/jobs", dependencies=[Depends(authorize)], status_code=202)
def create_job(payload: dict[str, Any]) -> dict[str, Any]:
    job = jobs.submit(payload)
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

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.environ.get("GATEWAY_PORT", "8000")),
        log_level="info",
    )
