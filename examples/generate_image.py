from __future__ import annotations

import base64
import os
from pathlib import Path

from openai import OpenAI

client = OpenAI(
    base_url=os.environ["DELTA_LLM_BASE_URL"].rstrip("/"),
    api_key=os.environ["DELTA_LLM_API_KEY"],
)
response = client.images.generate(
    model=os.environ.get("DELTA_LLM_MODEL", "bagel-7b"),
    prompt="A scientific illustration of a robot exploring Mars",
    size="512x512",
    response_format="b64_json",
    extra_body={"steps": 30, "seed": 42},
)
Path("output.png").write_bytes(base64.b64decode(response.data[0].b64_json))
print("saved output.png")
