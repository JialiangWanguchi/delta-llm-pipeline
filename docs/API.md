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
| `max_think_tokens` | integer | 512 | 16–4096 |
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
        "thinking": True,
        "max_think_tokens": 1024,
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

它不要求key，只返回Gateway和Worker状态，不返回模型输出或凭据。

## 错误码

| HTTP状态 | 含义 |
|---:|---|
| 400 | model、task、尺寸或输入图片不合法 |
| 401 | API key缺失或错误 |
| 503 | 对应模型Worker不可用 |
| 507 | GPU显存不足；降低尺寸、轮数、tokens或并发 |

生成图片可能需要数十秒到数分钟。客户端读取超时建议至少设置为3600秒；初版每个Worker并发为1，其他请求会排队。
