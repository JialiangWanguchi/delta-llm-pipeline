# Delta BAGEL + ThinkMorph Text Inference

在 NCSA Delta 上部署 `BAGEL-7B-MoT` 和 `ThinkMorph-7B`，通过带 Bearer API Key 的 OpenAI 兼容接口提供多图理解与纯文字输出。vLLM 作业推荐使用 Tailscale 私网入口；Cloudflare 模式保留为可选兼容路径。

固定实验契约：

- `model=bagel-7b` 或 `model=thinkmorph-7b`；
- 一次请求包含0–24张图片和多段文字；
- `text → image → text → image` 的顺序原样进入官方交错推理接口；
- 最终只生成文字，图片生成和图片编辑被禁用；
- 首选 OpenAI Chat Completions，长任务保留异步 `/v1/jobs`；
- 超过上下文预算时明确返回4xx，不允许静默丢图或截断。

```text
Internet client
     │  OpenAI messages + one HTTPS URL + one key
     ▼
FastAPI Gateway
     ├── bagel-7b      → A100-0 / A100-1
     └── thinkmorph-7b → A100-2 / A100-3
```

## 当前资源布局

- Slurm account：`bhsz-delta-gpu`
- 支持4×A100 40GB或2×H200 141GB，默认47.5小时
- 两个模型各两个BF16常驻副本，每模型两路真实并发
- A100成本：`4 × 47.5 × 1.0 = 190 weighted GPU-hours`
- H200成本：`2 × 47.5 × 3.0 = 285 weighted GPU-hours`

每个worker一次执行一个请求，更多请求按模型分别排队。任何CPU或NVMe权重offload都会让worker启动失败，避免服务悄悄退化。

## 部署

Windows PowerShell：

```powershell
git clone https://github.com/JialiangWanguchi/delta-llm-pipeline.git
cd delta-llm-pipeline

.\run.ps1 --username your_ncsa_username deploy `
  --gpus 4 --hours 47.5 --exposure cloudflare-quick `
  --acknowledge-external-tunnel --detach
```

macOS/Linux：

```bash
./run.sh --username your_ncsa_username deploy \
  --gpus 4 --hours 47.5 --exposure cloudflare-quick \
  --acknowledge-external-tunnel --detach
```

用户本人完成一次NCSA密码和Duo认证。`SUBMITTED`只表示Slurm已接收作业；之后使用：

```powershell
.\run.ps1 --username your_ncsa_username status DEPLOYMENT_ID
.\run.ps1 --username your_ncsa_username logs DEPLOYMENT_ID --lines 200
```

当状态为 `READY` 时，本地状态文件保存URL和Key。不要把状态文件、Key或Tunnel凭据提交到Git。

两张H200布局：

```powershell
.\run.ps1 --username your_ncsa_username deploy `
  --gpu-type h200 --gpus 2 --hours 47.5 `
  --exposure cloudflare-quick --acknowledge-external-tunnel --detach
```

H200-0承载两个BAGEL副本，H200-1承载两个ThinkMorph副本；每张卡约55GiB模型权重，141GiB显存足够保留长上下文和KV Cache余量。

如果只需要一个模型，可以只提交一个单卡H200作业。该作业会独立启动网关、生成自己的Base URL和API Key，不会等待另一个模型：

```powershell
.\run.ps1 --username your_ncsa_username deploy `
  --model bagel-7b --gpu-type h200 --gpus 1 --hours 47.5 `
  --exposure cloudflare-quick --acknowledge-external-tunnel --detach
```

单卡H200作业申请12个CPU核心和240GB主机内存，运行47.5小时约预留142.5 weighted GPU-hours。把`--model`改成`thinkmorph-7b`可单独部署ThinkMorph。

### vLLM BAGEL（新部署推荐）

需要导师要求的vLLM框架时，显式指定`--engine vllm`。每个vLLM作业只部署一个模型，支持`1×H200`或`2×A100`：

```powershell
.\run.ps1 --username your_ncsa_username deploy `
  --engine vllm --model bagel-7b --gpu-type h200 --gpus 1 --hours 48 `
  --exposure tailscale --detach
```

ThinkMorph和双A100示例：

```powershell
.\run.ps1 --username your_ncsa_username deploy `
  --engine vllm --model thinkmorph-7b --gpu-type a100 --gpus 2 --hours 48 `
  --exposure tailscale --detach
```

双A100使用vLLM pipeline parallel，把同一个模型分布到两张卡上。ThinkMorph是BAGEL微调权重；部署器会为其创建隔离的模型视图，组合官方BAGEL架构配置与ThinkMorph权重，不修改共享检查点。该路径固定安装并报告vLLM版本，启动官方OpenAI兼容服务，再通过只监听本机的认证网关暴露API。网关保留消息中`text`与`image_url`内容项的原始顺序，最多允许24张data URL图片；`/health`返回`engine=vllm`、实际版本和模型名，便于证明运行中的服务确实使用vLLM。vLLM路径只提供文字输出，不支持图片生成。

Delta 当前 RH9 节点的 glibc 版本低于 PyPI 默认 vLLM 0.20.2 wheel 要求。部署器因此固定使用 vLLM 官方 GitHub Release 的 CUDA 12.9 / `manylinux_2_31` wheel，并在安装前校验官方 SHA-256；共享环境同时记录发行版本 `0.20.2` 与实际构建版本 `0.20.2+cu129`，防止 pip 静默退回源码包或复用不完整环境。

### Tailscale 私网入口

先在 Tailscale 管理后台创建一枚 **Reusable + Ephemeral + Pre-approved** 的 auth key。建议给 key 绑定专用 tag（例如 `tag:delta-inference`），并在 ACL 中只允许团队客户端访问该 tag 的 TCP 8080 端口。不要启用 Funnel；本部署只使用 tailnet 内可达的 Serve。

PowerShell 中通过环境变量传入 key（不要写进仓库或命令行历史）：

```powershell
$env:TAILSCALE_AUTHKEY = Read-Host "Tailscale auth key"
.\run.ps1 --username your_ncsa_username deploy `
  --engine vllm --model bagel-7b --gpu-type h200 --gpus 1 --hours 48 `
  --exposure tailscale --detach
Remove-Item Env:TAILSCALE_AUTHKEY
```

部署器在计算节点上以无 root 的 userspace networking 启动独立 `tailscaled`，使用内存状态注册临时节点，并在注册后立刻删除远端 auth key。作业 READY 后，Base URL 形如 `http://100.x.y.z:8080/v1`；HTTP 流量位于 Tailscale 加密隧道内，且 Gateway 仍要求独立的 Bearer API key。调用端必须登录同一 tailnet，ACL 必须允许访问该节点。作业结束后临时节点会自动离开 tailnet。

如果双卡同时可用导致预计排队较久，可以把同样的两张H200拆成两个独立的单卡作业：

```powershell
.\run.ps1 --username your_ncsa_username deploy `
  --gpu-type h200 --gpus 2 --split-jobs --hours 47.5 `
  --exposure cloudflare-quick --acknowledge-external-tunnel --detach
```

该模式分别提交BAGEL和ThinkMorph单卡作业；两者都启动后仍只暴露一个HTTPS Base URL和一个外部API Key。`status`会显示两个Job ID。如果只有一个作业开始运行，整体状态不会变为`READY`，但已运行作业会开始消耗GPU-hours，因此应持续查看两个作业的状态。

四张A100也可以拆成两个独立作业，每个模型申请两张A100并运行两个常驻副本：

```powershell
.\run.ps1 --username your_ncsa_username deploy `
  --gpu-type a100 --gpus 4 --split-jobs --hours 47.5 `
  --exposure cloudflare-quick --acknowledge-external-tunnel --detach
```

与H200拆分模式相同，两个A100作业通过内部Bearer认证互联，只有两个模型都健康后才创建统一公网入口。

## OpenAI兼容调用

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://100.x.y.z:8080/v1",
    api_key="YOUR-PRIVATE-KEY",
)

response = client.chat.completions.create(
    model="thinkmorph-7b",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "第一阶段："},
                {"type": "image_url", "image_url": {"url": "https://example.org/01.webp"}},
                {"type": "text", "text": "第二阶段："},
                {"type": "image_url", "image_url": {"url": "https://example.org/02.webp"}},
                {"type": "text", "text": "综合比较，只返回文字。"},
            ],
        }
    ],
    max_tokens=256,
    temperature=0,
    extra_body={"modalities": ["text"]},
)
print(response.choices[0].message.content)
```

响应附加 `input_image_count`、`input_content_types`、`token_budget` 和 `usage`，用于证明API层没有重排或截断输入。

## 24图与Token预算

新worker只执行ViT理解和文本解码，不执行VAE图片上下文或图片生成。默认视觉配置：

- 最短边目标224；
- 最长边上限336；
- patch size为14；
- 单张336×336图片为576个视觉Token；
- 24张方图约13,824个视觉Token。

默认有效上下文门限为28,672，为模型32K上限保留余量：

```text
文字Token + 视觉Token + 图片特殊Token + max_tokens + 生成起始Token
<= 28,672
```

响应返回实际处理尺寸、逐图视觉Token和总预算。超限请求在推理前失败，不会静默截断。

## 异步长任务

Cloudflare Quick Tunnel的同步请求可能超时。耗时实验使用：

```text
POST /v1/jobs
GET  /v1/jobs/{job_id}
GET  /v1/jobs/{job_id}/result
```

`/v1/jobs` 使用框架无关的 `content`：

```json
{
  "model": "bagel-7b",
  "task": "image-understanding",
  "content": [
    {"type": "text", "text": "Frame 1"},
    {"type": "image", "image": "data:image/webp;base64,..."},
    {"type": "text", "text": "Frame 2"},
    {"type": "image", "image": "data:image/webp;base64,..."},
    {"type": "text", "text": "只返回文字结论"}
  ],
  "thinking": false,
  "max_output_tokens": 256
}
```

## 验收

部署READY后不能只检查 `/health`。必须按2→4→8→16→24图逐级运行：

```powershell
$env:DELTA_LLM_BASE_URL = "http://100.x.y.z:8080/v1"
$env:DELTA_LLM_API_KEY = "YOUR-PRIVATE-KEY"
python .\examples\verify_multi_image.py `
  --image .\first.jpg --image .\second.jpg `
  --interleaved
```

最终验收覆盖两个模型、并发1/2/4、纯文字输出、Token预算、无重排/截断、显存、排队、prefill、decode和端到端时间。原生worker无法提供真实流式TTFT，因此该字段明确返回 `null`，不能伪造。

## 安全限制

- 除 `/health` 外全部要求Bearer Key。
- 单图默认最大10MiB、1600万像素；请求体默认最大64MiB。
- 远程 `image_url` 只允许HTTP(S)公共地址，不允许凭据、私网/回环地址或重定向。
- 可用 `IMAGE_URL_HOST_ALLOWLIST` 限制到批准的对象存储域名。
- 支持JPEG、PNG、WebP。
- Quick Tunnel无SLA，不应承载未经PI/NCSA批准的敏感数据。
- Tailscale 模式不开放公网入口；调用端必须属于同一 tailnet 并通过 ACL。
- `/v1/images/generations` 固定返回410，不会生成图片。

完整字段见 [API文档](docs/API.md)，多成员排队规则见 [团队说明](docs/TEAM_QUEUE.md)，安全说明见 [SECURITY](docs/SECURITY.md)。

## 开发验证

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
git diff --check
```
