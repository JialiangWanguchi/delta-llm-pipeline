# 安全说明

## 信任边界

本工具涉及四个边界：本地电脑、SSH 加密连接、Delta 项目空间/计算节点、可选的 Cloudflare 网络。使用公网暴露模式意味着请求正文和响应会经过第三方网络；不要把受限数据或敏感受试者数据发送给未经项目批准的服务。

## 凭据处理

- NCSA 密码与 Duo 响应只由 OpenSSH 客户端读取，Python 程序无法获得它们。
- 共享 API key 用 Python `secrets` 生成，通过 SSH stdin 发送；远端保存在部署目录下的 `secrets/api_key`，权限 `0600`。默认拓扑中的 Worker 只监听回环地址；拆分拓扑需要通过Delta内部计算网络互联，并使用另一枚随机内部Bearer key保护Worker端点。两种拓扑都只通过统一Gateway暴露公网入口。
- Hugging Face token 仅在需要 gated model 时从本地 `HF_TOKEN` 读取。本版本内置模型通常不需要它。
- Named Tunnel token 从 `DELTA_LLM_CF_TUNNEL_TOKEN` 读取。不要写入 `config.toml`、README、Git history 或 Slurm 参数。
- 本地状态位于 `~/.delta-llm/deployments`，包含 API key。共享电脑上应确认主目录权限，并在实验完成后删除不再需要的状态文件。

## 上线前清单

1. 获得项目 PI/NCSA 对外部隧道或 Gateway 的批准。
2. 使用 HTTPS URL，不要把 API key 通过明文 HTTP 传输到公网。
3. 对团队成员使用密码管理器或现有 secret manager 分发 key。
4. 避免在 shell history 中直接写 key，优先使用环境变量。
5. 用最短必要 wall time，实验结束立即执行 `stop`。
6. 检查日志中没有研究数据、提示词或响应正文。
7. 若 key 泄露，立即停止部署；当前版本的轮换方式是重新部署并生成新 key。
8. 批量使用远程图片时设置 `IMAGE_URL_HOST_ALLOWLIST`，只允许团队批准的对象存储域名。

## 已知边界

- Gateway 的 API key 是共享 bearer token，不提供按成员身份、配额或细粒度权限。
- Quick Tunnel 无 SLA，并非生产级入口。
- 公开端点可能受到扫描和拒绝服务攻击。API key 能防止未授权推理请求，但不能替代 WAF、速率限制、审计和网络访问控制。
- Gateway限制请求体、图片字节数、像素数和同步并发，并拒绝会解析到私网/回环地址的图片URL；正式服务仍应在外层增加WAF和速率策略。
- 本轮部署只允许多模态理解与文字输出；图片生成接口返回410，worker不把VAE放入GPU。
- 同一项目组成员可读取组共享的软件环境和模型缓存，但个人部署目录使用 `umask 077`/`0600` 保护 secret。
- `stop` 会删除远端 key、接口地址，并清空当前电脑状态文件中的缓存 key。

正式团队服务建议由 NCSA Gateway 或团队控制的反向代理承载，增加访问控制、速率限制、日志政策和密钥轮换。
