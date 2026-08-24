# BAGEL + ThinkMorph Text API

部署返回一个Base URL和一个共享Bearer API Key。除 `/health` 外，所有接口要求：

```http
Authorization: Bearer <DELTA_LLM_API_KEY>
```

## 接口

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/health` | Gateway、worker副本、显存、队列和最近性能 |
| GET | `/v1/models` | 返回 `bagel-7b`、`thinkmorph-7b` |
| POST | `/v1/chat/completions` | 首选OpenAI兼容同步接口 |
| POST | `/v1/generate` | 自定义同步兼容接口 |
| POST | `/v1/jobs` | 异步入队，立即返回202 |
| GET | `/v1/jobs/{id}` | 查询状态和排队位置 |
| GET | `/v1/jobs/{id}/result` | 获取纯文字结果和预算 |
| POST | `/v1/images/generations` | 固定返回410；图片生成已禁用 |

## Chat Completions

```json
{
  "model": "thinkmorph-7b",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "第一阶段："},
        {"type": "image_url", "image_url": {"url": "https://example.org/01.webp"}},
        {"type": "text", "text": "第二阶段："},
        {"type": "image_url", "image_url": {"url": "data:image/webp;base64,..."}},
        {"type": "text", "text": "综合全部材料，只返回文字。"}
      ]
    }
  ],
  "modalities": ["text"],
  "max_tokens": 256,
  "temperature": 0,
  "stream": false
}
```

约束：

- `model`：`bagel-7b`、`thinkmorph-7b`；
- `messages`：非空数组，role为 `system`、`user` 或 `assistant`；
- `content`：字符串，或有序的 `text` / `image_url` parts；
- `image_url.url`：公共HTTP(S) URL或JPEG/PNG/WebP data URL；
- `max_tokens` / `max_completion_tokens`：1–4096；
- `temperature`：0–2；
- `modalities`：省略或严格为 `["text"]`；
- `n`：仅支持1；`stream`：仅支持false。

一条普通user消息不会被插入额外内容，其parts直接按数组顺序转换成worker的 `List[str | PIL.Image]`。多轮消息插入文字role边界，但图片相对次序保持不变。

响应：

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "thinkmorph-7b",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "纯文字答案"},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 1196, "completion_tokens": 80, "total_tokens": 1276},
  "input_image_count": 2,
  "input_content_types": ["text", "image", "text", "image", "text"],
  "token_budget": {
    "effective_context_limit": 28672,
    "text_tokens": 40,
    "visual_tokens": 1152,
    "visual_tokens_per_image": [576, 576],
    "processed_image_dimensions": [
      {"width": 336, "height": 336},
      {"width": 336, "height": 336}
    ],
    "image_special_tokens": 4,
    "input_tokens": 1196,
    "max_output_tokens": 256,
    "required_tokens": 1453,
    "remaining_tokens": 27219
  }
}
```

## 异步任务

长请求使用框架无关格式：

```json
{
  "model": "bagel-7b",
  "task": "image-understanding",
  "content": [
    {"type": "text", "text": "术前："},
    {"type": "image", "uri": "https://approved-object-store/01.jpg"},
    {"type": "text", "text": "术中："},
    {"type": "image", "uri": "https://approved-object-store/02.jpg"},
    {"type": "text", "text": "比较并只返回文字。"}
  ],
  "thinking": false,
  "max_output_tokens": 256,
  "temperature": 0,
  "do_sample": false
}
```

`POST /v1/jobs` 返回job ID、状态URL和结果URL。状态为 `queued`、`running`、`succeeded` 或 `failed`。

结果包含：

- `text`，且不包含 `images`；
- `input_image_count` 和完整 `input_content_types`；
- `token_budget`；
- 图片预处理、prefill、decode、queue和端到端时间；
- `ttft_seconds`：原生后端目前为 `null`。

兼容模式可以发送 `prompt` 加单个 `image`，或 `prompt` 加有序 `images`。这些字段既可放data URL，也可放批准的公共HTTP(S) URL。真正交错实验必须使用 `content`；图片项可使用 `image` 或 `uri`。异步URL下载在job线程中执行，不阻塞提交202。

## 输入约束

- 图片数：0–24；
- `content`：最多64项，至少一项文字；
- `content` 不能和 `prompt`、`image`、`images` 混用；
- 单图默认最大10MiB、1600万像素；
- 整体HTTP请求体默认最大64MiB；
- MIME仅JPEG、PNG、WebP；
- 远程URL禁止凭据、重定向、私网、回环和不可解析地址；
- `IMAGE_URL_HOST_ALLOWLIST` 非空时，只允许列出的域名及其子域名。

24张图片不建议全部使用base64。批量实验应使用批准的对象存储URL，并设置host allowlist。

## Token准入

视觉输入最长边336、最短边目标224，patch size 14：

```text
visual_tokens_i = (processed_width_i / 14) × (processed_height_i / 14)
```

每段文字按模型真实tokenizer计数并加入BOS/EOS；每图加入start/end image两个特殊Token。请求必须满足：

```text
input_tokens + max_output_tokens + 1 <= EFFECTIVE_CONTEXT_LIMIT
```

默认 `EFFECTIVE_CONTEXT_LIMIT=28672`。超限返回400及完整预算JSON，不进入模型推理。

## 纯文字保证

- Gateway只允许 `task=image-understanding`；
- Chat只允许文字output modality；
- worker固定 `understanding_output=True`；
- worker跳过VAE输入路径，只执行ViT和文本解码；
- VAE不常驻GPU；
- 底层意外返回图片时请求失败；
- `/v1/images/generations` 返回410。

## Health与错误

```bash
curl "${DELTA_LLM_BASE_URL%/v1}/health"
```

正常worker显示 `load_mode=resident`、`offloaded_modules=[]`、`text_output_only=true`、有效上下文和ViT尺寸配置。

| HTTP | 含义 |
|---:|---|
| 400 | 输入、model、task、URL或Token预算不合法 |
| 401 | Key缺失或错误 |
| 404 | 路径或异步任务不存在 |
| 410 | 图片生成接口已禁用 |
| 413 | 请求、图片字节数或像素数超限 |
| 429 | 同步并发槽已满；改用异步任务或稍后重试 |
| 503 | worker不可用或队列已满 |
| 507 | GPU OOM；记录配置后降低预算并重测 |

Cloudflare同步524不表示worker崩溃。长任务必须使用 `/v1/jobs`，并分别记录节点内推理时间和公网端到端时间。
