from __future__ import annotations

import os

import requests

base_url = os.environ["DELTA_LLM_BASE_URL"].rstrip("/")
headers = {"Authorization": f"Bearer {os.environ['DELTA_LLM_API_KEY']}"}
response = requests.post(
    f"{base_url}/chat/completions",
    headers=headers,
    json={
        "model": "thinkmorph-7b",
        "messages": [
            {
                "role": "user",
                "content": "Explain the safest path through a simple maze in text only.",
            }
        ],
        "modalities": ["text"],
        "max_tokens": 256,
        "temperature": 0,
    },
    timeout=3600,
)
response.raise_for_status()
result = response.json()
print(result["choices"][0]["message"]["content"])
