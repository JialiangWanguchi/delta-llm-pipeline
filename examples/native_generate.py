from __future__ import annotations

import base64
import os
from pathlib import Path

import requests

base_url = os.environ["DELTA_LLM_BASE_URL"].rstrip("/")
headers = {"Authorization": f"Bearer {os.environ['DELTA_LLM_API_KEY']}"}
response = requests.post(
    f"{base_url}/generate",
    headers=headers,
    json={
        "model": "thinkmorph-7b",
        "task": "text-to-image",
        "prompt": "Draw a visual step-by-step solution to a simple maze.",
        "size": "512x512",
        "thinking": True,
        "max_think_tokens": 512,
        "max_rounds": 1,
        "steps": 30,
    },
    timeout=3600,
)
response.raise_for_status()
result = response.json()
print(result.get("text"))
for index, data_url in enumerate(result.get("images", []), start=1):
    encoded = data_url.split(",", 1)[1]
    Path(f"thinkmorph_{index}.png").write_bytes(base64.b64decode(encoded))
