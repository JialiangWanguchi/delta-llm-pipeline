# 小组并行排队与交接

这套流程允许同一 allocation 下的组员分别使用自己的 NCSA/ACCESS 账号提交相同的 4×A100 服务作业。每个作业拥有独立的 Slurm job、部署目录、公网 URL 和 API key；任何人都不需要把 NCSA 密码或 Duo 交给其他成员。

> 并行提交不会提高 allocation 的 Slurm fair-share 优先级，也不保证更快。请先获得 PI/项目管理员许可，并遵守 NCSA 对重复作业和公网 tunnel 的规定。多个作业如果同时开始运行会同时消耗 allocation，因此第一个服务验收成功后要立即通知其他成员取消自己的作业。

## 1. 每位成员提交自己的作业

Windows PowerShell：

```powershell
git clone https://github.com/JialiangWanguchi/delta-llm-pipeline.git
cd delta-llm-pipeline

.\run.ps1 --username YOUR_NCSA_USERNAME deploy `
  --gpus 4 `
  --hours 47.5 `
  --exposure cloudflare-quick `
  --acknowledge-external-tunnel `
  --detach
```

macOS/Linux：

```bash
git clone https://github.com/JialiangWanguchi/delta-llm-pipeline.git
cd delta-llm-pipeline

./run.sh --username YOUR_NCSA_USERNAME deploy \
  --gpus 4 \
  --hours 47.5 \
  --exposure cloudflare-quick \
  --acknowledge-external-tunnel \
  --detach
```

`--detach` 会在提交 Slurm 作业并保存本地 API key 后退出，适合等待时间较长的队列，可避免让 SSH 会话持续挂一天。请记录输出中的 `Deployment` 和 `Job`。不要在第一次提交时使用 `--replace-existing-services`；该参数只用于同一成员明确替换自己的旧作业。

双卡H200若因gang scheduling预计等待过久，可加`--gpu-type h200 --gpus 2 --split-jobs`，把两个模型分别提交为独立的单卡H200作业。输出中的`Job`会包含两个逗号分隔的Job ID；统一URL只有在两个作业都启动并通过健康检查后才会生成。拆分可能更早拿到单卡，但两个作业启动时间不同会造成部分GPU先计费而服务尚未完整可用。

四卡A100可用`--gpu-type a100 --gpus 4 --split-jobs`拆成两个双卡作业：一个作业运行两个BAGEL副本，另一个作业运行两个ThinkMorph副本。统一URL、Job ID表示和健康门槛与H200拆分模式一致。

本地状态文件位于：

```text
Windows: %USERPROFILE%\.delta-llm\deployments\DEPLOYMENT_ID.json
Linux/macOS: ~/.delta-llm/deployments/DEPLOYMENT_ID.json
```

## 2. 查询自己的排队状态

```powershell
.\run.ps1 --username YOUR_NCSA_USERNAME status DEPLOYMENT_ID
```

状态含义：

- 本地`SUBMITTED`，远端`UNKNOWN`且Slurm显示`PENDING (Priority)`：已进入队列，尚未占用GPU；
- `STARTING`：节点已分配，正在把四个模型副本加载进显存；
- `READY`：公网 URL 已生成，可以验收；
- `FAILED`：运行 `logs` 查看模型、Gateway和Tunnel日志。

拆分部署的`status`会分别列出BAGEL和ThinkMorph两个Slurm作业；必须两个都在运行，整体服务才可能是`READY`。

```powershell
.\run.ps1 --username YOUR_NCSA_USERNAME logs DEPLOYMENT_ID --lines 200
```

每个成员只能查询和停止自己账号下的部署。共享项目目录中的模型权重和A100运行环境会被复用，但API key不会共享。

## 3. 第一个 READY 的成员完成多图验收

准备两张内容明显不同的图片。建议通过环境变量传递凭据，避免 API key 留在命令历史中。

Windows PowerShell：

```powershell
$env:DELTA_LLM_BASE_URL = "https://YOUR-TUNNEL.trycloudflare.com/v1"
$env:DELTA_LLM_API_KEY = "sk-delta-mm-YOUR-KEY"

python .\examples\verify_multi_image.py `
  --image .\first.jpg `
  --image .\second.jpg `
  --interleaved
```

macOS/Linux：

```bash
export DELTA_LLM_BASE_URL="https://YOUR-TUNNEL.trycloudflare.com/v1"
export DELTA_LLM_API_KEY="sk-delta-mm-YOUR-KEY"

python ./examples/verify_multi_image.py \
  --image ./first.jpg \
  --image ./second.jpg \
  --interleaved
```

脚本会执行以下检查：

1. 用同一个 URL/key读取模型列表；
2. 向 `bagel-7b` 和 `thinkmorph-7b` 分别提交相同的有序 `images` 数组；
3. 使用异步 `/v1/jobs` 轮询，避免 Cloudflare 同步超时；
4. 要求两个结果都返回非空文字，并且 `input_image_count` 等于实际输入图片数；
5. 任一模型失败时以非零退出码结束。

可以重复 `--image` 2–24次。必须按2→4→8→16→24逐级验收，不能用2图成功替代24图证据。`--interleaved`要求API保留真正的文字/图片交替顺序。验收输出中的 `PASS` 会同时检查`input_image_count`、`input_content_types`、纯文字结果和Token预算；还应人工检查回答是否正确区分每张图片的位置与内容。

胜出作业还必须实测 `/v1/chat/completions`，并记录两个模型在并发1、2、4下的排队、prefill、decode、端到端时间、显存和失败率。任一24图请求发生OOM、超预算或失败时，应保留可复现配置，不得静默降低图片数。

## 4. 交接 URL/key，立即取消其余作业

验收通过的成员通过团队批准的密码管理器或加密渠道共享 Base URL 和 API key。不要把 key 写入 GitHub issue、README、聊天截图或源代码。

其他成员随后停止自己的部署：

```powershell
.\run.ps1 --username YOUR_NCSA_USERNAME stop YOUR_DEPLOYMENT_ID --yes
```

如果某个备用作业仍在 `PENDING`，也应取消，避免它稍后意外启动并产生 GPU-hours。每个 API key 只在对应 Slurm 作业存活期间有效；作业结束、被取消或到达48小时上限后，URL/key都会失效。

## 5. 推荐的小组记录表

不要在公开仓库提交含 key 的表格。团队可以在私有文档记录：

| 成员 | Deployment ID | Job ID | 状态 | 是否已取消 |
|---|---|---|---|---|
| Alice | `bagel-thinkmorph-...` | `123456` | PENDING | 否 |
| Bob | `bagel-thinkmorph-...` | `123457` | READY/胜出 | 保留 |

完整字段和手工调用格式见 [API文档](API.md)。
