from __future__ import annotations

import argparse
import base64
import io
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_CONFIG), required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--offload-dir", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-memory-gib", type=int, default=42)
    return parser.parse_args()


def image_from_data(value: str | None) -> Image.Image | None:
    if not value:
        return None
    encoded = value.split(",", 1)[1] if value.startswith("data:") else value
    try:
        raw = base64.b64decode(encoded, validate=True)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise ValueError("image must be a valid base64 string or data URL") from exc


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


class ModelRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model_name = args.model
        self.lock = threading.Lock()
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
        from inferencer import InterleaveInferencer
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
        max_memory = {
            index: f"{self.args.max_memory_gib}GiB" for index in range(torch.cuda.device_count())
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
        return InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=ImageTransform(1024, 512, 16),
            vit_transform=ImageTransform(980, 224, 14),
            new_token_ids=new_token_ids,
        )

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        task = str(payload.get("task", "text-to-image"))
        supported_tasks = {"text-to-image", "image-edit", "image-understanding"}
        if task not in supported_tasks:
            raise ValueError(f"unsupported task {task!r}; choose one of {sorted(supported_tasks)}")
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("prompt is required")
        input_image = image_from_data(payload.get("image"))
        if task in {"image-edit", "image-understanding"} and input_image is None:
            raise ValueError(f"task {task} requires image")
        image_shape = parse_size(str(payload.get("size", "512x512")))
        max_think_tokens = min(4096, max(16, int(payload.get("max_think_tokens", 512))))
        max_rounds = min(4, max(1, int(payload.get("max_rounds", 1))))
        steps = min(100, max(10, int(payload.get("steps", 30))))
        thinking = bool(payload.get("thinking", self.model_name == "thinkmorph-7b"))
        set_seed(int(payload.get("seed", 0)))

        kwargs = {
            "think": thinking,
            "understanding_output": task == "image-understanding",
            "max_think_token_n": max_think_tokens,
            "do_sample": bool(payload.get("do_sample", False)),
            "text_temperature": float(payload.get("temperature", 0.3)),
            "cfg_text_scale": float(payload.get("cfg_text_scale", 4.0)),
            "cfg_img_scale": float(payload.get("cfg_image_scale", 1.5)),
            "cfg_interval": [float(payload.get("cfg_interval", 0.4)), 1.0],
            "timestep_shift": float(payload.get("timestep_shift", 3.0)),
            "num_timesteps": steps,
            "cfg_renorm_min": float(payload.get("cfg_renorm_min", 0.0)),
            "cfg_renorm_type": str(payload.get("cfg_renorm_type", "global")),
            "image_shapes": image_shape,
        }
        if self.model_name == "thinkmorph-7b":
            kwargs["max_rounds"] = max_rounds

        with self.lock, torch.inference_mode():
            result = self.inferencer(image=input_image, text=prompt, **kwargs)

        images: list[str] = []
        text_parts: list[str] = []
        if isinstance(result, dict):
            if isinstance(result.get("image"), Image.Image):
                images.append(image_to_data_url(result["image"]))
            if result.get("text"):
                text_parts.append(str(result["text"]))
        elif isinstance(result, list):
            for item in result:
                if isinstance(item, Image.Image):
                    images.append(image_to_data_url(item))
                elif isinstance(item, str):
                    text_parts.append(item)
        return {
            "id": f"gen-{uuid.uuid4().hex}",
            "model": self.model_name,
            "task": task,
            "text": "\n".join(text_parts) or None,
            "images": images,
        }


def create_app(runtime: ModelRuntime) -> FastAPI:
    app = FastAPI(title=f"{runtime.model_name} internal worker")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": runtime.model_name,
            "cuda_devices": torch.cuda.device_count(),
            "started_at": runtime.started_at,
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

    return app


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    runtime = ModelRuntime(args)
    uvicorn.run(create_app(runtime), host="127.0.0.1", port=args.port, log_level="info")
