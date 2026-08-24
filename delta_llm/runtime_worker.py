from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import logging
import math
import os
import random
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image

MODEL_CONFIG = {
    "bagel-7b": {
        "checkpoint": "ema.safetensors",
        "layer_module": "Qwen2MoTDecoderLayer",
    },
    "thinkmorph-7b": {
        "checkpoint": "model.safetensors",
        "layer_module": "Qwen2MoTDecoderLayer",
    },
}
LOGGER = logging.getLogger("delta_llm.worker")
MAX_INPUT_IMAGES = 24
MAX_INPUT_CONTENT_ITEMS = 64
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", str(16_000_000)))
EFFECTIVE_CONTEXT_LIMIT = int(os.environ.get("EFFECTIVE_CONTEXT_LIMIT", "28672"))
VIT_MAX_IMAGE_SIZE = int(os.environ.get("VIT_MAX_IMAGE_SIZE", "336"))
VIT_MIN_IMAGE_SIZE = int(os.environ.get("VIT_MIN_IMAGE_SIZE", "224"))
VIT_PATCH_SIZE = 14
VIT_MAX_PIXELS = 14 * 14 * 9 * 1024
ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_CONFIG), required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--offload-dir", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-memory-gib", type=int, default=42)
    parser.add_argument(
        "--load-mode",
        choices=("resident", "auto"),
        default="resident",
        help="Keep all model weights on GPU, or allow Accelerate CPU/disk offload",
    )
    return parser.parse_args()


def image_from_data(value: str | None) -> Image.Image | None:
    if not value:
        return None
    encoded = value
    if value.startswith("data:"):
        if "," not in value:
            raise ValueError("image data URL is malformed")
        header, encoded = value.split(",", 1)
        mime_type = header[5:].split(";", 1)[0].lower()
        if ";base64" not in header.lower() or mime_type not in ALLOWED_IMAGE_MIME_TYPES:
            raise ValueError("image data URL must be base64 JPEG, PNG, or WebP")
    if len(encoded) > math.ceil(MAX_IMAGE_BYTES / 3) * 4 + 4:
        raise ValueError(f"image exceeds the {MAX_IMAGE_BYTES}-byte limit")
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError(f"image exceeds the {MAX_IMAGE_BYTES}-byte limit")
        with Image.open(io.BytesIO(raw)) as source:
            if (source.format or "").upper() not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("image format must be JPEG, PNG, or WebP")
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError(f"image exceeds the {MAX_IMAGE_PIXELS}-pixel limit")
            source.load()
            return source.convert("RGB")
    except (binascii.Error, OSError) as exc:
        raise ValueError("image must be a valid base64 string or data URL") from exc


def _make_divisible(value: float, stride: int) -> int:
    return max(stride, int(round(value / stride) * stride))


def vit_processed_size(width: int, height: int) -> tuple[int, int]:
    """Mirror the pinned BAGEL ImageTransform resize math used by the worker."""
    scale = min(VIT_MAX_IMAGE_SIZE / max(width, height), 1.0)
    scale = max(scale, VIT_MIN_IMAGE_SIZE / min(width, height))
    new_width = _make_divisible(round(width * scale), VIT_PATCH_SIZE)
    new_height = _make_divisible(round(height * scale), VIT_PATCH_SIZE)
    if new_width * new_height > VIT_MAX_PIXELS:
        pixel_scale = VIT_MAX_PIXELS / (new_width * new_height)
        new_width = _make_divisible(round(new_width * pixel_scale), VIT_PATCH_SIZE)
        new_height = _make_divisible(round(new_height * pixel_scale), VIT_PATCH_SIZE)
    if max(new_width, new_height) > VIT_MAX_IMAGE_SIZE:
        edge_scale = VIT_MAX_IMAGE_SIZE / max(new_width, new_height)
        new_width = _make_divisible(round(new_width * edge_scale), VIT_PATCH_SIZE)
        new_height = _make_divisible(round(new_height * edge_scale), VIT_PATCH_SIZE)
    return new_width, new_height


def estimate_token_budget(
    terms: list[str | Image.Image],
    tokenizer: Any | None,
    max_output_tokens: int,
    thinking_prompt: str | None = None,
) -> dict[str, Any]:
    text_terms = [term for term in terms if isinstance(term, str)]
    if thinking_prompt:
        text_terms.insert(0, thinking_prompt)
    text_tokens = 0
    for text in text_terms:
        encoded_length = len(tokenizer.encode(text)) if tokenizer is not None else math.ceil(len(text) / 4)
        # BAGEL prepare_prompts adds BOS and EOS around every string term.
        text_tokens += encoded_length + 2

    dimensions: list[dict[str, int]] = []
    visual_tokens_per_image: list[int] = []
    for term in terms:
        if not isinstance(term, Image.Image):
            continue
        width, height = vit_processed_size(*term.size)
        visual_tokens = (width // VIT_PATCH_SIZE) * (height // VIT_PATCH_SIZE)
        dimensions.append({"width": width, "height": height})
        visual_tokens_per_image.append(visual_tokens)
    visual_tokens = sum(visual_tokens_per_image)
    image_special_tokens = len(visual_tokens_per_image) * 2
    input_tokens = text_tokens + visual_tokens + image_special_tokens
    required_tokens = input_tokens + max_output_tokens + 1
    return {
        "effective_context_limit": EFFECTIVE_CONTEXT_LIMIT,
        "text_tokens": text_tokens,
        "visual_tokens": visual_tokens,
        "visual_tokens_per_image": visual_tokens_per_image,
        "processed_image_dimensions": dimensions,
        "image_special_tokens": image_special_tokens,
        "input_tokens": input_tokens,
        "max_output_tokens": max_output_tokens,
        "required_tokens": required_tokens,
        "remaining_tokens": EFFECTIVE_CONTEXT_LIMIT - required_tokens,
    }


def images_from_payload(payload: dict[str, Any]) -> list[Image.Image]:
    """Decode the legacy singular image or the ordered multi-image array."""
    singular = payload.get("image")
    plural_present = "images" in payload and payload.get("images") is not None
    if singular and plural_present:
        raise ValueError("use either image or images, not both")

    if plural_present:
        values = payload["images"]
        if not isinstance(values, list):
            raise ValueError("images must be an array of base64 strings or data URLs")
        if len(values) > MAX_INPUT_IMAGES:
            raise ValueError(f"images supports at most {MAX_INPUT_IMAGES} input images")
    elif singular:
        values = [singular]
    else:
        values = []

    decoded: list[Image.Image] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value:
            raise ValueError(f"images[{index}] must be a non-empty base64 string or data URL")
        try:
            image = image_from_data(value)
        except ValueError as exc:
            raise ValueError(f"images[{index}] must be a valid base64 string or data URL") from exc
        assert image is not None
        decoded.append(image)
    return decoded


def build_input_terms(images: list[Image.Image], prompt: str) -> list[str | Image.Image]:
    """Build an ordered image/text sequence understood by both official inferencers."""
    terms: list[str | Image.Image] = []
    for index, image in enumerate(images, start=1):
        if len(images) > 1:
            terms.append(f"Image {index}:")
        terms.append(image)
    terms.append(prompt)
    return terms


def input_terms_from_payload(
    payload: dict[str, Any],
) -> tuple[list[str | Image.Image], list[Image.Image], list[str]]:
    """Build either a legacy image(s)+prompt request or an exact content sequence."""
    content_present = "content" in payload and payload.get("content") is not None
    if not content_present:
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("prompt is required")
        images = images_from_payload(payload)
        terms = build_input_terms(images, prompt)
        types = ["image" if isinstance(term, Image.Image) else "text" for term in terms]
        return terms, images, types

    if payload.get("image") or payload.get("images") is not None:
        raise ValueError("content cannot be combined with image or images")
    if str(payload.get("prompt", "")).strip():
        raise ValueError("content cannot be combined with prompt; put all text in content")
    content = payload["content"]
    if not isinstance(content, list) or not content:
        raise ValueError("content must be a non-empty array")
    if len(content) > MAX_INPUT_CONTENT_ITEMS:
        raise ValueError(f"content supports at most {MAX_INPUT_CONTENT_ITEMS} items")

    terms: list[str | Image.Image] = []
    images: list[Image.Image] = []
    types: list[str] = []
    for index, item in enumerate(content):
        if not isinstance(item, dict):
            raise ValueError(  # noqa: TRY004 - request validation must remain HTTP 400
                f"content[{index}] must be an object"
            )
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"content[{index}].text must be a non-empty string")
            terms.append(text.strip())
            types.append("text")
        elif item_type == "image":
            value = item.get("image")
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"content[{index}].image must be a non-empty base64 string or data URL"
                )
            try:
                image = image_from_data(value)
            except ValueError as exc:
                raise ValueError(
                    f"content[{index}].image must be a valid base64 string or data URL"
                ) from exc
            assert image is not None
            images.append(image)
            if len(images) > MAX_INPUT_IMAGES:
                raise ValueError(f"content supports at most {MAX_INPUT_IMAGES} input images")
            terms.append(image)
            types.append("image")
        else:
            raise ValueError(f"content[{index}].type must be 'text' or 'image'")

    if "text" not in types:
        raise ValueError("content must include at least one text item")
    return terms, images, types


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def parse_size(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d{3,4})x(\d{3,4})", value)
    if not match:
        raise ValueError("size must look like 512x512 or 1024x768")
    width, height = (int(match.group(1)), int(match.group(2)))
    if min(width, height) < 256 or max(width, height) > 1024:
        raise ValueError("image dimensions must be between 256 and 1024")
    if width % 16 or height % 16:
        raise ValueError("image dimensions must be divisible by 16")
    return height, width


def set_seed(seed: int) -> None:
    if seed <= 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextOnlyInferencer:
    """Run BAGEL/ThinkMorph's native ViT + text path without the VAE image path."""

    def __init__(self, native: Any, thinking_prompt: str) -> None:
        self.native = native
        self.thinking_prompt = thinking_prompt
        self.last_timing: dict[str, float | None] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native, name)

    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: list[str | Image.Image],
        *,
        think: bool = False,
        understanding_output: bool = True,
        max_think_token_n: int = 512,
        do_sample: bool = False,
        text_temperature: float = 0.0,
        **_: Any,
    ) -> list[str]:
        if not understanding_output:
            raise ValueError("This deployment is text-only; understanding_output must be true")
        context = self.native.init_gen_context()
        preprocessing_seconds = 0.0
        prefill_seconds = 0.0
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                started = time.perf_counter()
                context = self.native.update_context_text(self.thinking_prompt, context)
                prefill_seconds += time.perf_counter() - started
            for term in input_lists:
                if isinstance(term, str):
                    started = time.perf_counter()
                    context = self.native.update_context_text(term, context)
                    prefill_seconds += time.perf_counter() - started
                elif isinstance(term, Image.Image):
                    # The pinned official inferencer first resizes every image with
                    # its VAE transform even when VAE encoding is disabled. Skip
                    # that generation-only resize and execute only the ViT path.
                    started = time.perf_counter()
                    processed = self.native.vit_transform.resize_transform(term.convert("RGB"))
                    preprocessing_seconds += time.perf_counter() - started
                    started = time.perf_counter()
                    context = self.native.update_context_image(
                        processed,
                        context,
                        vae=False,
                        vit=True,
                    )
                    prefill_seconds += time.perf_counter() - started
                else:
                    raise TypeError(f"Unsupported input type: {type(term)}")
            started = time.perf_counter()
            text = self.native.gen_text(
                context,
                max_length=max_think_token_n,
                do_sample=do_sample,
                temperature=text_temperature,
            )
            decode_seconds = time.perf_counter() - started
        self.last_timing = {
            "image_preprocessing_seconds": preprocessing_seconds,
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            # The pinned native generator returns only after full decode. True
            # TTFT needs token streaming or vLLM instrumentation.
            "ttft_seconds": None,
        }
        return [text]


class ModelRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model_name = args.model
        self.lock = threading.Lock()
        self.metrics_lock = threading.Lock()
        self.queued_requests = 0
        self.active_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        self.last_inference_seconds: float | None = None
        self.last_queue_seconds: float | None = None
        self.last_timing: dict[str, float | None] = {}
        self.started_at = int(time.time())
        self.inferencer = self._load()

    def _load(self):
        from accelerate import (
            infer_auto_device_map,
            init_empty_weights,
            load_checkpoint_and_dispatch,
        )

        repo_dir = Path(self.args.repo_dir).resolve()
        checkpoint_dir = Path(self.args.checkpoint_dir).resolve()
        offload_dir = Path(self.args.offload_dir).resolve()
        offload_dir.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(repo_dir))
        os.chdir(repo_dir)

        from data.data_utils import add_special_tokens
        from data.transforms import ImageTransform
        from inferencer import VLM_THINK_SYSTEM_PROMPT, InterleaveInferencer
        from modeling.autoencoder import load_ae
        from modeling.bagel import (
            Bagel,
            BagelConfig,
            Qwen2Config,
            Qwen2ForCausalLM,
            SiglipVisionConfig,
            SiglipVisionModel,
        )
        from modeling.qwen2 import Qwen2Tokenizer

        config_entry = MODEL_CONFIG[self.model_name]
        llm_config = Qwen2Config.from_json_file(checkpoint_dir / "llm_config.json")
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = config_entry["layer_module"]
        llm_config.use_cache = True

        vit_config = SiglipVisionConfig.from_json_file(checkpoint_dir / "vit_config.json")
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1
        vae_model, vae_config = load_ae(str(checkpoint_dir / "ae.safetensors"))
        vae_model.requires_grad_(False)
        vae_model.eval()

        bagel_config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=2,
            max_latent_size=64,
        )
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, bagel_config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        tokenizer = Qwen2Tokenizer.from_pretrained(str(checkpoint_dir))
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the multimodal worker")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        if self.args.load_mode == "resident":
            # Each checkpoint is ~29.2 GB and stays resident on its own 40 GB+
            # GPU; avoiding Accelerate's disk hooks removes
            # the dominant per-token NVMe transfer cost.
            device_map: dict[str, Any] = {"": 0}
            model = load_checkpoint_and_dispatch(
                model,
                checkpoint=str(checkpoint_dir / config_entry["checkpoint"]),
                device_map=device_map,
                dtype=torch.bfloat16,
                # Keep Accelerate's input-alignment hooks even though every
                # weight is resident on one GPU. The official inferencer builds
                # request tensors on CPU and relies on these hooks to move them
                # to the model device. This does not enable CPU/disk offload.
                force_hooks=True,
            ).eval()
        else:
            max_memory = {
                index: f"{self.args.max_memory_gib}GiB"
                for index in range(torch.cuda.device_count())
            }
            device_map = infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
            )
            same_device_modules = [
                "language_model.model.embed_tokens",
                "time_embedder",
                "latent_pos_embed",
                "vae2llm",
                "llm2vae",
                "connector",
                "vit_pos_embed",
            ]
            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for key in same_device_modules:
                if key in device_map or torch.cuda.device_count() == 1:
                    device_map[key] = first_device
            model = load_checkpoint_and_dispatch(
                model,
                checkpoint=str(checkpoint_dir / config_entry["checkpoint"]),
                device_map=device_map,
                offload_buffers=True,
                offload_folder=str(offload_dir),
                dtype=torch.bfloat16,
                force_hooks=True,
            ).eval()

        self.device_map = dict(getattr(model, "hf_device_map", device_map))
        self.offloaded_modules = sorted(
            name
            for name, device in self.device_map.items()
            if str(device).lower() in {"cpu", "disk"}
        )
        if self.args.load_mode == "resident" and self.offloaded_modules:
            raise RuntimeError(
                "resident load unexpectedly offloaded modules: "
                + ", ".join(self.offloaded_modules[:10])
            )
        print(
            "DELTA_RUNTIME_LAYOUT "
            + json.dumps(
                {
                    "model": self.model_name,
                    "load_mode": self.args.load_mode,
                    "device_map": self.device_map,
                    "offloaded_modules": self.offloaded_modules,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        # V2 is text-output-only. The model constructor still needs the VAE
        # config, but VAE weights stay on CPU and are never used by inference.
        self.tokenizer = tokenizer
        native = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=ImageTransform(1024, 512, 16),
            vit_transform=ImageTransform(VIT_MAX_IMAGE_SIZE, VIT_MIN_IMAGE_SIZE, VIT_PATCH_SIZE),
            new_token_ids=new_token_ids,
        )
        return TextOnlyInferencer(native, VLM_THINK_SYSTEM_PROMPT)

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        task = str(payload.get("task", "image-understanding"))
        if task != "image-understanding":
            raise ValueError(
                f"unsupported task {task!r}; this deployment only returns text"
            )
        input_terms, input_images, input_content_types = input_terms_from_payload(payload)
        default_output_tokens = 128
        requested_output_tokens = payload.get(
            "max_output_tokens",
            payload.get("max_think_tokens", default_output_tokens),
        )
        max_think_tokens = min(4096, max(1, int(requested_output_tokens)))
        thinking = bool(payload.get("thinking", False))
        set_seed(int(payload.get("seed", 0)))

        token_budget = estimate_token_budget(
            input_terms,
            getattr(self, "tokenizer", None),
            max_think_tokens,
            self.inferencer.thinking_prompt if thinking else None,
        )
        if token_budget["required_tokens"] > EFFECTIVE_CONTEXT_LIMIT:
            raise ValueError(
                "context token budget exceeded: "
                + json.dumps(token_budget, sort_keys=True, separators=(",", ":"))
            )

        kwargs = {
            "think": thinking,
            "understanding_output": True,
            "max_think_token_n": max_think_tokens,
            "do_sample": bool(payload.get("do_sample", False)),
            "text_temperature": float(payload.get("temperature", 0.0)),
        }

        with self.metrics_lock:
            self.queued_requests += 1
        started = time.perf_counter()
        queue_seconds = 0.0
        try:
            with self.lock:
                queue_seconds = time.perf_counter() - started
                with self.metrics_lock:
                    self.queued_requests -= 1
                    self.active_requests = 1
                with torch.inference_mode():
                    result = self.inferencer.interleave_inference(
                        input_terms,
                        **kwargs,
                    )
            with self.metrics_lock:
                self.completed_requests += 1
        except Exception:
            with self.metrics_lock:
                self.failed_requests += 1
            raise
        finally:
            elapsed = time.perf_counter() - started
            with self.metrics_lock:
                self.active_requests = 0
                self.last_inference_seconds = elapsed
                self.last_queue_seconds = queue_seconds
                self.last_timing = {
                    **getattr(self.inferencer, "last_timing", {}),
                    "queue_seconds": queue_seconds,
                    "end_to_end_seconds": elapsed,
                }

        text_parts: list[str] = []
        if isinstance(result, dict):
            if result.get("text"):
                text_parts.append(str(result["text"]))
        elif isinstance(result, list):
            for item in result:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, Image.Image):
                    raise TypeError("text-only inferencer unexpectedly returned an image")
        text = "\n".join(text_parts) or None
        output_tokens = (
            len(self.tokenizer.encode(text))
            if text and getattr(self, "tokenizer", None) is not None
            else (math.ceil(len(text) / 4) if text else 0)
        )
        return {
            "id": f"gen-{uuid.uuid4().hex}",
            "model": self.model_name,
            "task": task,
            "text": text,
            "output_tokens": output_tokens,
            "input_image_count": len(input_images),
            "input_content_types": input_content_types,
            "token_budget": token_budget,
            "timings": dict(getattr(self, "last_timing", {})),
        }


def create_app(runtime: ModelRuntime) -> FastAPI:
    app = FastAPI(title=f"{runtime.model_name} internal worker")

    @app.get("/health")
    def health() -> dict[str, Any]:
        cuda: dict[str, Any] = {}
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            cuda = {
                "name": torch.cuda.get_device_name(device),
                "memory_allocated_gib": round(torch.cuda.memory_allocated(device) / 2**30, 2),
                "memory_reserved_gib": round(torch.cuda.memory_reserved(device) / 2**30, 2),
                "memory_total_gib": round(
                    torch.cuda.get_device_properties(device).total_memory / 2**30, 2
                ),
            }
        with runtime.metrics_lock:
            requests_state = {
                "queued": runtime.queued_requests,
                "active": runtime.active_requests,
                "completed": runtime.completed_requests,
                "failed": runtime.failed_requests,
                "last_inference_seconds": runtime.last_inference_seconds,
                "last_queue_seconds": runtime.last_queue_seconds,
                "last_timing": runtime.last_timing,
            }
        return {
            "status": "ok",
            "model": runtime.model_name,
            "cuda_devices": torch.cuda.device_count(),
            "started_at": runtime.started_at,
            "load_mode": runtime.args.load_mode,
            "offloaded_modules": runtime.offloaded_modules,
            "text_output_only": True,
            "effective_context_limit": EFFECTIVE_CONTEXT_LIMIT,
            "vit_image_size": {
                "min_edge": VIT_MIN_IMAGE_SIZE,
                "max_edge": VIT_MAX_IMAGE_SIZE,
                "patch_size": VIT_PATCH_SIZE,
            },
            "cuda": cuda,
            "requests": requests_state,
        }

    @app.post("/generate")
    def generate(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return runtime.generate(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise HTTPException(status_code=507, detail="GPU out of memory") from exc
        except Exception as exc:
            LOGGER.exception("unhandled %s inference failure", runtime.model_name)
            raise HTTPException(
                status_code=500,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

    return app


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    runtime = ModelRuntime(args)
    uvicorn.run(create_app(runtime), host="127.0.0.1", port=args.port, log_level="info")
