# NCSA 支持工单模板

下面内容可按实际项目修改后发送给 NCSA Help Desk。不要在工单中附 API key 或 Tunnel token。

---

Subject: Request for an approved external inference gateway for Delta allocation `bhsz-delta-gpu`

Hello NCSA Support,

Our group uses the Delta allocation `bhsz-delta-gpu` for research experiments. We would like to run an OpenAI-compatible vLLM inference server inside a Slurm GPU job (maximum wall time 48 hours) and access it from our research workstations and servers outside Delta.

Each deployment is submitted by an individual allocation member under their own NCSA account. The service uses a randomly generated bearer API key, listens on the compute node only for the lifetime of the Slurm job, and is stopped automatically when the allocation ends. We do not intend to expose shell access or create a persistent compute service.

Could you advise on the approved architecture for this use case? In particular:

1. Is an outbound Cloudflare Tunnel from a Delta compute node permitted for short-lived research inference endpoints?
2. If not, can this project receive a Gateway account/host or another supported reverse-proxy mechanism?
3. Are there firewall, port, data classification, logging, or rate-limiting requirements we must follow?
4. Is there an approved method for automated lifecycle checks without repeated interactive SSH/Duo logins?

We can provide the source code and a network/data-flow diagram for review. No NCSA password or Duo credential is stored by the tool.

Thank you.

---
