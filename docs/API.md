# BAGEL + ThinkMorph API

部署会返回一个Base URL和一个共享Bearer API key。以下示例假设：

```text
DELTA_LLM_BASE_URL=https://random-name.trycloudflare.com/v1
DELTA_LLM_API_KEY=sk-delta-mm-...
```

API key可以同时调用 `bagel-7b` 和 `thinkmorph-7b`。除 `/health` 外，所有接口都要求：

```http
Authorization: Bearer <DELTA_LLM_API_KEY>
```

## 模型列表

```http
GET /v1/models
```

```bash
curl "$DELTA_LLM_BASE_URL/models" \
  -H "Authorization: Bearer $DELTA_LLM_API_KEY"
```

响应：

```json
{
  "object": "list",
  "data": [
    {"id": "bagel-7b", "object": "model", "owned_by": "delta-llm"},
    {"id": "thinkmorph-7b", "object": "model", "owned_by": "delta-llm"}
  ]
}
```

## 统一生成接口

耗时的多模态任务应优先使用异步接口。`POST /v1/jobs` 只负责入队，通常会在数秒内返回 `202`，因此不会被Cloudflare约125秒的代理读取超时截断。

### 提交异步任务

```http
POST /v1/jobs
Content-Type: application/json
```

请求体与下面的 `/v1/generate` 完全相同：

```python
import base64
import requests

encoded = base64.b64encode(open("input.jpg", "rb").read()).decode()
job = requests.post(
    f"{base_url}/jobs",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "model": "bagel-7b",
        "task": "image-understanding",
        "prompt": "请分析主要物体、数量以及空间关系，只输出简短中文。",
        "image": encoded,
        "thinking": False,
        "max_output_tokens": 128,
    },
    timeout=30,
).json()
print(job["id"], job["queue_position"])
```

提交响应：

```json
{
  "id": "job-...",
  "object": "inference.job",
  "model": "bagel-7b",
  "status": "queued",
  "queue_position": 1,
  "status_url": "/v1/jobs/job-...",
  "result_url": "/v1/jobs/job-.../result"
}
```

### 查询进度和结果

```python
import time

job_id = job["id"]
while True:
    status_response = requests.get(
        f"{base_url}/jobs/{job_id}", headers=headers, timeout=30
    )
    status_response.raise_for_status()
    status = status_response.json()
    print(status["status"], status["queue_position"], status["elapsed_seconds"])
    if status["status"] in {"succeeded", "failed"}:
        break
    time.sleep(3)

result_response = requests.get(
    f"{base_url}/jobs/{job_id}/result", headers=headers, timeout=30
)
result_response.raise_for_status()
result = result_response.json()
print(result["text"], result["elapsed_seconds"])
```

任务状态为 `queued`、`running`、`succeeded` 或 `failed`。结果保留2小时；2×H200部署在每张141GB卡上启动同一模型的两个独立副本，因此每个模型最多同时执行两个任务，其余请求按模型分别排队。

### 同步接口（兼容旧客户端）

```http
POST /v1/generate
Content-Type: application/json
```

请求字段：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | string | 必填 | `bagel-7b` 或 `thinkmorph-7b` |
| `task` | string | `text-to-image` | `text-to-image`、`image-edit`、`image-understanding` |
| `prompt` | string | 必填 | 指令或问题 |
| `image` | string | null | 输入图片的Base64或`data:image/...;base64,...`；编辑/理解任务必填 |
| `size` | string | `512x512` | 输出尺寸；256–1024且宽高是16的倍数 |
| `thinking` | boolean | ThinkMorph为true | 是否生成thinking文本 |
| `max_output_tokens` | integer | 图片理解128，其他任务512 | 文字回答或thinking的token上限，16–4096 |
| `max_think_tokens` | integer | 同上 | `max_output_tokens` 的兼容别名 |
| `max_rounds` | integer | 1 | ThinkMorph图文交错轮数，1–4 |
| `steps` | integer | 30 | 图像生成步数，10–100 |
| `seed` | integer | 0 | 0表示随机；正数用于复现 |
| `temperature` | number | 0.3 | 文本采样温度 |
| `do_sample` | boolean | false | 是否启用文本采样 |
| `cfg_text_scale` | number | 4.0 | 文本CFG强度 |
| `cfg_image_scale` | number | 1.5 | 编辑任务的图像保持强度 |
| `cfg_interval` | number | 0.4 | CFG起始比例 |
| `timestep_shift` | number | 3.0 | 生成时间步偏移 |
| `cfg_renorm_min` | number | 0.0 | CFG renormalization下限 |
| `cfg_renorm_type` | string | `global` | `global`、`local`或`text_channel` |

### 文生图

```bash
curl "$DELTA_LLM_BASE_URL/generate" \
  -H "Authorization: Bearer $DELTA_LLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bagel-7b",
    "task": "text-to-image",
    "prompt": "A watercolor painting of the NCSA building at sunset",
    "size": "512x512",
    "steps": 30,
    "seed": 42
  }'
```

### ThinkMorph交错推理

```python
import requests

result = requests.post(
    f"{base_url}/generate",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "model": "thinkmorph-7b",
        "task": "text-to-image",
        "prompt": "Visually reason through and solve a simple maze.",
        "thinking": True,
        "max_think_tokens": 512,
        "max_rounds": 2,
        "size": "512x512",
    },
    timeout=3600,
).json()

print(result["text"])
print(len(result["images"]))
```

### 图片编辑

```python
import base64
import requests

encoded = base64.b64encode(open("input.png", "rb").read()).decode()
result = requests.post(
    f"{base_url}/generate",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "model": "bagel-7b",
        "task": "image-edit",
        "prompt": "Change the daytime sky to a starry night while preserving the building.",
        "image": encoded,
        "steps": 30,
    },
    timeout=3600,
).json()
```

### 图片理解

```python
result = requests.post(
    f"{base_url}/generate",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "model": "thinkmorph-7b",
        "task": "image-understanding",
        "prompt": "Describe the spatial relationships in this image.",
        "image": encoded,
        "thinking": False,
        "max_output_tokens": 128,
    },
    timeout=3600,
).json()
print(result["text"])
```

统一响应：

```json
{
  "id": "gen-...",
  "created": 1786000000,
  "model": "thinkmorph-7b",
  "task": "text-to-image",
  "text": "optional thinking or answer",
  "images": ["data:image/png;base64,..."]
}
```

`images`可能包含多张图片，尤其是ThinkMorph多轮交错推理。客户端应遍历数组，不要只读取第一张。

## OpenAI Python SDK文生图兼容接口

```http
POST /v1/images/generations
```

```python
import base64
from openai import OpenAI

client = OpenAI(base_url=base_url, api_key=api_key)
response = client.images.generate(
    model="bagel-7b",
    prompt="A small red robot in a laboratory",
    size="512x512",
    response_format="b64_json",
    extra_body={"steps": 30, "seed": 42},
)
open("result.png", "wb").write(base64.b64decode(response.data[0].b64_json))
```

ThinkMorph使用同一key，只需改模型名：

```python
response = client.images.generate(
    model="thinkmorph-7b",
    prompt="Create a visual explanation of the water cycle",
    size="512x512",
    response_format="b64_json",
    extra_body={"thinking": True, "max_rounds": 1},
)
```

完整的图片编辑和图文交错输出请使用 `/v1/generate`，因为OpenAI Images响应结构不能表达多轮文字—图片序列。

## 健康检查

```http
GET /health
```

公网URL的健康检查路径不包含 `/v1`：

```bash
curl "${DELTA_LLM_BASE_URL%/v1}/health"
```

它不要求key，只返回Gateway、各模型副本、显存常驻状态、队列指标和最近一次推理耗时，不返回模型输出或凭据。正常的高性能worker应显示 `load_mode: resident` 且 `offloaded_modules: []`。

## 错误码

| HTTP状态 | 含义 |
|---:|---|
| 400 | model、task、尺寸或输入图片不合法 |
| 401 | API key缺失或错误 |
| 404 | 异步任务不存在或结果已过期 |
| 503 | 对应模型Worker不可用 |
| 507 | GPU显存不足；降低尺寸、轮数、tokens或并发 |

同步生成仍可能被公网代理超时截断，客户端自身设置3600秒并不能改变Cloudflare限制。生产和团队调用应使用 `/v1/jobs`。每个worker一次执行一个任务；2×H200模式在每张大显存GPU上启动两个独立进程，实现每模型真实双并发。
