from __future__ import annotations

import os

from openai import OpenAI

base_url = os.environ["DELTA_LLM_BASE_URL"].rstrip("/")
api_key = os.environ["DELTA_LLM_API_KEY"]
model = os.environ.get("DELTA_LLM_MODEL", "qwen3-4b-instruct")

client = OpenAI(base_url=base_url, api_key=api_key)
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "用一句话介绍 NCSA Delta。"}],
    stream=False,
)
print(response.choices[0].message.content)
