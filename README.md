# Delta BAGEL + ThinkMorph Pipeline

在 NCSA Delta 上一次部署 `BAGEL-7B-MoT` 和 `ThinkMorph-7B`，生成一个公网 Base URL 和一个共享 API key。客户端通过请求中的 `model` 字段选择模型，无需分别登录或维护两个接口。

如果多位组员希望分别使用自己的NCSA账号提交作业，请直接使用[小组并行排队与交接说明](docs/TEAM_QUEUE.md)。其中包含长队列使用的`--detach`流程、胜出作业验收、其余作业取消，以及可直接运行的双模型多图片测试脚本。

```text
Internet client
     │  one HTTPS URL + one Bearer key
     ▼
FastAPI Gateway :8000
     ├── model=bagel-7b      → GPU 0, GPU 1（两个副本）
     └── model=thinkmorph-7b → GPU 2, GPU 3（两个副本）
```

## 默认资源

- Slurm 账户：`bhsz-delta-gpu`
- 分区：`gpuA100x4`
- 固定申请：4× NVIDIA A100 40GB（完整节点）
- BAGEL：2个常驻显存副本，分别独占A100-0和A100-1
- ThinkMorph：2个常驻显存副本，分别独占A100-2和A100-3
- 输入：兼容单图`image`、最多24张的`images`数组，以及真正文字/图片交替的`content`数组
- 默认时长：47.5小时；Delta上限48小时
- 计费估算：4 × 1.0 × 47.5 = 190 weighted GPU-hours
- 权重：约59.2GB，源码和权重位于 `/projects/bhsz/delta-llm/shared`；Python环境和包缓存在 `/work/nvme/bhsz/delta-llm/shared`

两个约29.2GB的checkpoint均以BF16完整常驻显存。每个A100只运行一个模型副本，启动检查发现任何CPU/磁盘卸载都会直接失败。每个模型有两个独立GPU副本，可真实并行处理两项推理。A100收费系数为1.0，因此47.5小时约190 weighted GPU-hours。

## 快速开始

Windows PowerShell：

```powershell
git clone https://github.com/JialiangWanguchi/delta-llm-pipeline.git
cd delta-llm-pipeline

.\run.ps1 --username your_ncsa_username deploy `
  --gpus 4 `
  --hours 47.5 `
  --exposure cloudflare-quick `
  --acknowledge-external-tunnel
```

macOS/Linux：

```bash
git clone https://github.com/JialiangWanguchi/delta-llm-pipeline.git
cd delta-llm-pipeline

./run.sh --username your_ncsa_username deploy \
  --gpus 4 \
  --hours 47.5 \
  --exposure cloudflare-quick \
  --acknowledge-external-tunnel
```

OpenSSH 会提示一次NCSA密码和Duo。首次部署还会：

1. 创建固定版本的Python/PyTorch/FlashAttention环境；
2. 下载两个官方源码仓库；
3. 下载约59GB模型权重；
4. 提交一个4×A100完整节点Slurm作业；
5. 等待两个Worker和Gateway全部健康；
6. 返回一个Base URL、一个API key和本地状态文件。

同组后续部署会复用共享环境及权重。首次初始化可能耗时较长，不要关闭认证窗口。

如果需要替换自己此前排队或运行中的双模型作业，并让本地立即保存新 key 后退出 SSH：

```powershell
.\run.ps1 --username your_ncsa_username deploy `
  --gpus 4 --hours 47.5 --exposure cloudflare-quick `
  --acknowledge-external-tunnel --replace-existing-services --detach
```

后台作业 READY 后运行 `status DEPLOYMENT_ID`，会把远端 URL 和状态合并回本地 JSON，同时保留原 API key。

## 调用两个模型

同一个Base URL和key可用于两个模型。对于图片理解、编辑和生成，推荐使用异步任务接口：提交操作会立即返回，不受Cloudflare同步读取超时影响。

```python
import requests

base_url = "https://random-name.trycloudflare.com/v1"
headers = {"Authorization": "Bearer sk-delta-mm-..."}

for model in ("bagel-7b", "thinkmorph-7b"):
    response = requests.post(
        f"{base_url}/jobs",
        headers=headers,
        json={
            "model": model,
            "task": "image-understanding",
            "prompt": "Describe the important objects and their spatial relationships.",
            "images": [
                "data:image/jpeg;base64,...第一张图...",
                "data:image/jpeg;base64,...第二张图...",
            ],
            "thinking": False,
            "max_output_tokens": 128,
        },
        timeout=30,
    )
    response.raise_for_status()
    print(model, response.json()["id"], response.json()["status_url"])
```

完整请求字段、图片编辑、图片理解、响应格式和错误码见 [API文档](docs/API.md)。

要验证两个模型都真正接收了2–24张有序输入图，可运行：

```powershell
$env:DELTA_LLM_BASE_URL = "https://YOUR-TUNNEL.trycloudflare.com/v1"
$env:DELTA_LLM_API_KEY = "sk-delta-mm-YOUR-KEY"
python .\examples\verify_multi_image.py `
  --image .\first.jpg --image .\second.jpg `
  --interleaved
```

`--interleaved`会构造真正的`text → image → text → image → text`序列。脚本对两个模型使用异步接口，并检查每个结果的`input_image_count`、`input_content_types`和非空文字回答；完整说明见[小组文档](docs/TEAM_QUEUE.md#3-第一个-ready-的成员完成多图验收)。

## 常用命令

```powershell
# 查看固定的双模型目录和GPU布局
.\run.ps1 models

# 仅检查方案，不登录、不提交作业
.\run.ps1 --username your_ncsa_username deploy --gpus 4 --hours 1 --dry-run

# 检查账户、A100分区、存储、网络和CUDA module
.\run.ps1 --username your_ncsa_username doctor

# 查看首次共享安装作业、模型文件大小和最近安装日志（只读）
.\run.ps1 --username your_ncsa_username setup-status

# 查看部署列表、状态和日志（这些远程命令需要NCSA认证）
.\run.ps1 --username your_ncsa_username list
.\run.ps1 --username your_ncsa_username status DEPLOYMENT_ID
.\run.ps1 --username your_ncsa_username logs DEPLOYMENT_ID --lines 200

# 同时停止两个模型、释放GPU并撤销共享key
.\run.ps1 --username your_ncsa_username stop DEPLOYMENT_ID --yes
```

## 本地状态

成功后状态文件位于：

```text
Windows: %USERPROFILE%\.delta-llm\deployments\DEPLOYMENT_ID.json
Linux/macOS: ~/.delta-llm/deployments/DEPLOYMENT_ID.json
```

它包含Base URL和API key，权限应限制为当前用户。不要提交到Git或发送到公开聊天中。

## 接口与能力

| 模型 | text-to-image | image-edit | image-understanding | interleaved reasoning |
|---|---:|---:|---:|---:|
| `bagel-7b` | ✓ | ✓ | ✓ | 单轮thinking |
| `thinkmorph-7b` | ✓ | ✓ | ✓ | ✓ |

图片理解建议从 `thinking=false`、`max_output_tokens=128` 开始。图像生成建议从 `512x512`、`steps=20` 开始。4×A100布局下每个模型有2个常驻副本，每个副本独占一张GPU，同模型可同时执行2项推理；更多请求由 `/v1/jobs` 排队且可查询位置。

历史H200基准（2026-08-12）：同一张双猫图片、`thinking=false`、64-token上限下，BAGEL图片理解耗时2.882秒，ThinkMorph耗时1.852秒；四路并发（每模型两路）总墙钟3.668秒。A100实际延迟会在新部署完成后重新验证；不同问题、图片尺寸和输出长度也会影响延迟。

## 暴露模式

- `none`：只生成Delta内部地址，最安全。
- `cloudflare-quick`：随机公网HTTPS地址，适合短期实验，无SLA；必须获得项目许可。
- `cloudflare-named`：团队管理的固定Cloudflare Tunnel；token放在本地环境变量 `DELTA_LLM_CF_TUNNEL_TOKEN`。

公网接口经过第三方网络。不要在未经PI/NCSA批准的情况下处理受限数据。详见 [安全说明](docs/SECURITY.md)。

## 开发验证

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
```

该项目使用官方 [BAGEL](https://github.com/ByteDance-Seed/BAGEL) 和 [ThinkMorph](https://github.com/ThinkMorph/ThinkMorph) 源码及Apache-2.0模型权重；本仓库只负责Delta资源编排、统一鉴权和API封装。
