# Delta BAGEL + ThinkMorph Pipeline

在 NCSA Delta 上一次部署 `BAGEL-7B-MoT` 和 `ThinkMorph-7B`，生成一个公网 Base URL 和一个共享 API key。客户端通过请求中的 `model` 字段选择模型，无需分别登录或维护两个接口。

```text
Internet client
     │  one HTTPS URL + one Bearer key
     ▼
FastAPI Gateway :8000
     ├── model=bagel-7b      → GPU 0
     └── model=thinkmorph-7b → GPU 1
```

## 默认资源

- Slurm 账户：`bhsz-delta-gpu`
- 分区：`gpuA40x4`
- 默认申请：2× NVIDIA A40 48GB
- BAGEL：1张A40
- ThinkMorph：1张A40
- 默认时长：47.5小时；Delta上限48小时
- 计费估算：2 × 0.5 × 47.5 = 47.5 weighted GPU-hours
- 权重：约59.2GB，源码和权重位于 `/projects/bhsz/delta-llm/shared`；Python环境和包缓存在 `/work/nvme/bhsz/delta-llm/shared`

两个约29.6GB的模型各自可装入一张48GB A40。`--gpus 3` 会让 ThinkMorph 使用两张卡以增加显存余量；`--gpus 4` 再预留一张卡。卡数越多通常越难排到，本项目默认使用2卡布局。

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
4. 提交一个2×A40 Slurm作业；
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

同一个Base URL和key可用于两个模型：

```python
import requests

base_url = "https://random-name.trycloudflare.com/v1"
headers = {"Authorization": "Bearer sk-delta-mm-..."}

for model in ("bagel-7b", "thinkmorph-7b"):
    response = requests.post(
        f"{base_url}/generate",
        headers=headers,
        json={
            "model": model,
            "task": "text-to-image",
            "prompt": "A red robot reading in a university library",
            "size": "512x512",
            "steps": 30,
        },
        timeout=3600,
    )
    response.raise_for_status()
    print(model, response.json()["id"])
```

完整请求字段、图片编辑、图片理解、响应格式和错误码见 [API文档](docs/API.md)。

## 常用命令

```powershell
# 查看固定的双模型目录和GPU布局
.\run.ps1 models

# 仅检查方案，不登录、不提交作业
.\run.ps1 --username your_ncsa_username deploy --gpus 2 --hours 1 --dry-run

# 检查账户、A40分区、存储、网络和CUDA module
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

推荐首先使用 `512x512`、`steps=30`、并发1。确认显存和延迟后再提升到1024像素、多轮或更多thinking tokens。

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
