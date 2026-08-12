# Delta BAGEL + ThinkMorph Pipeline

在 NCSA Delta 上一次部署 `BAGEL-7B-MoT` 和 `ThinkMorph-7B`，生成一个公网 Base URL 和一个共享 API key。客户端通过请求中的 `model` 字段选择模型，无需分别登录或维护两个接口。

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
- 分区：`gpuH200x8`
- 固定申请：2× NVIDIA H200 141GB
- BAGEL：2个常驻显存副本，共享H200-0
- ThinkMorph：2个常驻显存副本，共享H200-1
- 默认时长：47.5小时；Delta上限48小时
- 计费估算：2 × 3.0 × 47.5 = 285 weighted GPU-hours
- 权重：约59.2GB，源码和权重位于 `/projects/bhsz/delta-llm/shared`；Python环境和包缓存在 `/work/nvme/bhsz/delta-llm/shared`

两个约29.2GB的checkpoint均完整常驻H200显存。每张141GB H200同时容纳同一模型的两个独立副本；启动检查发现任何CPU/磁盘卸载都会直接失败。这样只申请2张GPU便可让同一个模型并行处理两项推理，也避免4×A100整节点请求长期无法回填。H200收费系数为3.0，因此47.5小时约285 weighted GPU-hours。

## 快速开始

Windows PowerShell：

```powershell
git clone https://github.com/JialiangWanguchi/delta-llm-pipeline.git
cd delta-llm-pipeline

.\run.ps1 --username your_ncsa_username deploy `
  --gpus 2 `
  --hours 47.5 `
  --exposure cloudflare-quick `
  --acknowledge-external-tunnel
```

macOS/Linux：

```bash
git clone https://github.com/JialiangWanguchi/delta-llm-pipeline.git
cd delta-llm-pipeline

./run.sh --username your_ncsa_username deploy \
  --gpus 2 \
  --hours 47.5 \
  --exposure cloudflare-quick \
  --acknowledge-external-tunnel
```

OpenSSH 会提示一次NCSA密码和Duo。首次部署还会：

1. 创建固定版本的Python/PyTorch/FlashAttention环境；
2. 下载两个官方源码仓库；
3. 下载约59GB模型权重；
4. 提交一个2×H200 Slurm作业；
5. 等待两个Worker和Gateway全部健康；
6. 返回一个Base URL、一个API key和本地状态文件。

同组后续部署会复用共享环境及权重。首次初始化可能耗时较长，不要关闭认证窗口。

如果需要替换自己此前排队或运行中的双模型作业，并让本地立即保存新 key 后退出 SSH：

```powershell
.\run.ps1 --username your_ncsa_username deploy `
  --gpus 2 --hours 47.5 --exposure cloudflare-quick `
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
            "image": "data:image/jpeg;base64,...",
            "thinking": False,
            "max_output_tokens": 128,
        },
        timeout=30,
    )
    response.raise_for_status()
    print(model, response.json()["id"], response.json()["status_url"])
```

完整请求字段、图片编辑、图片理解、响应格式和错误码见 [API文档](docs/API.md)。

## 常用命令

```powershell
# 查看固定的双模型目录和GPU布局
.\run.ps1 models

# 仅检查方案，不登录、不提交作业
.\run.ps1 --username your_ncsa_username deploy --gpus 2 --hours 1 --dry-run

# 检查账户、H200分区、存储、网络和CUDA module
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

图片理解建议从 `thinking=false`、`max_output_tokens=128` 开始。图像生成建议从 `512x512`、`steps=20` 开始。2×H200布局下每个模型有2个常驻副本，同模型可同时执行2项推理；更多请求由 `/v1/jobs` 排队且可查询位置。

2026-08-12实测：同一张双猫图片、`thinking=false`、64-token上限下，BAGEL图片理解耗时2.882秒，ThinkMorph耗时1.852秒；四路并发（每模型两路）总墙钟3.668秒。不同问题、图片尺寸和输出长度会产生不同延迟。

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
