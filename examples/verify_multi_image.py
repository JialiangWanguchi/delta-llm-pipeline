"""Verify ordered multi-image understanding on both deployed models.

This client intentionally uses only the Python standard library so group
members can run it immediately after cloning the repository.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

MODELS = ("bagel-7b", "thinkmorph-7b")
IMAGE_MEDIA_TYPES = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def normalize_base_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value:
        raise ValueError("base URL is empty")
    if not value.startswith(("https://", "http://")):
        raise ValueError("base URL must start with https:// or http://")
    return value if value.endswith("/v1") else f"{value}/v1"


def image_data_url(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"image does not exist: {path}")
    media_type = mimetypes.guess_type(path.name)[0] or IMAGE_MEDIA_TYPES.get(
        path.suffix.lower(), "application/octet-stream"
    )
    if not media_type.startswith("image/"):
        raise ValueError(f"file does not look like an image: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def request_json(
    method: str,
    url: str,
    api_key: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 30,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "delta-llm-multi-image-verifier/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"cannot reach {url}: {exc.reason}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit the same ordered image set to BAGEL and ThinkMorph."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DELTA_LLM_BASE_URL", ""),
        help="Deployment URL ending in /v1 (or DELTA_LLM_BASE_URL)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("DELTA_LLM_API_KEY", ""),
        help="Bearer key (prefer DELTA_LLM_API_KEY to avoid shell history)",
    )
    parser.add_argument(
        "--image",
        action="append",
        required=True,
        type=Path,
        help="Input image path in order; repeat 2 to 8 times",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "请按输入顺序分别描述每张图片的主要内容，然后比较它们的共同点和差异。"
            "只输出简短文字，不要生成图片。"
        ),
    )
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--poll-interval", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=900.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 2 <= len(args.image) <= 8:
        raise ValueError("repeat --image between 2 and 8 times")
    if not args.api_key:
        raise ValueError("set DELTA_LLM_API_KEY or pass --api-key")
    base_url = normalize_base_url(args.base_url)
    images = [image_data_url(path) for path in args.image]
    headers_check = request_json("GET", f"{base_url}/models", args.api_key)
    available = {item["id"] for item in headers_check.get("data", [])}
    missing = set(MODELS) - available
    if missing:
        raise RuntimeError(f"deployment is missing models: {sorted(missing)}")

    started = time.monotonic()
    pending: dict[str, str] = {}
    for model in MODELS:
        job = request_json(
            "POST",
            f"{base_url}/jobs",
            args.api_key,
            {
                "model": model,
                "task": "image-understanding",
                "prompt": args.prompt,
                "images": images,
                "thinking": False,
                "max_output_tokens": args.max_output_tokens,
            },
        )
        pending[model] = str(job["id"])
        print(f"{model}: submitted {job['id']} (queue={job.get('queue_position')})")

    failures: list[str] = []
    while pending:
        if time.monotonic() - started > args.timeout:
            raise TimeoutError(f"verification exceeded {args.timeout:g} seconds")
        for model, job_id in list(pending.items()):
            status = request_json("GET", f"{base_url}/jobs/{job_id}", args.api_key)
            state = status.get("status")
            if state not in {"succeeded", "failed"}:
                continue
            if state == "failed":
                failures.append(f"{model}: {status.get('error', 'unknown failure')}")
                del pending[model]
                continue
            result = request_json(
                "GET", f"{base_url}/jobs/{job_id}/result", args.api_key
            )
            elapsed = result.get("elapsed_seconds", status.get("elapsed_seconds"))
            image_count = result.get("input_image_count")
            text = str(result.get("text") or "").strip()
            passed = image_count == len(images) and bool(text)
            verdict = "PASS" if passed else "FAIL"
            print(
                f"\n[{verdict}] {model}: input_image_count={image_count}, "
                f"elapsed={elapsed}s\n{text}\n"
            )
            if not passed:
                failures.append(
                    f"{model}: expected {len(images)} inputs and non-empty text"
                )
            del pending[model]
        if pending:
            time.sleep(max(0.5, args.poll_interval))

    if failures:
        print("Verification failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("Both models accepted the full ordered multi-image input set.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, TimeoutError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
