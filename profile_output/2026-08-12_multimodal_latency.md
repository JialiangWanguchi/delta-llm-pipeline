# Multimodal inference latency profile — 2026-08-12

## Target

The running Delta service containing BAGEL-7B-MoT and ThinkMorph-7B on two
NVIDIA A100 40 GB GPUs, exposed through a Cloudflare Quick Tunnel.

## Measurements

| Model | Task | Settings | Client-observed result |
|---|---|---|---|
| BAGEL-7B | image-understanding | thinking=false, max output 64 | HTTP 524 after 126.35 s |
| ThinkMorph-7B | image-understanding | thinking=false, max output 64 | HTTP 524 after 125.97 s |
| Gateway health | health | both workers loaded | HTTP 200 in 1.17 s |

The workers stayed healthy after both timeouts. This isolates the failure to
long synchronous inference rather than a crashed Slurm job or tunnel.

## Bottlenecks

1. The deployment used Accelerate automatic device mapping with disk offload
   enabled. Worker startup logs reported offloaded parameters. Repeated
   CPU/NVMe-to-GPU transfers are the dominant suspected latency source.
2. Each model had one worker protected by a process-wide inference lock, so
   requests to the same model were serialized.
3. The only public inference route was synchronous. Cloudflare terminated the
   request before the origin produced a response.
4. Image-understanding inherited a 512-token default, which is unnecessarily
   high for concise answers.

## Implemented remediation

1. Load each ~29.2 GB BF16 checkpoint fully on one A100 and fail startup if any
   module is assigned to CPU or disk.
2. Use four A100 GPUs as two BAGEL replicas and two ThinkMorph replicas. Each
   model can now execute two requests concurrently; excess work is queued.
3. Add `POST /v1/jobs`, job status, queue position, and result endpoints so long
   requests return immediately and do not depend on Cloudflare's read timeout.
4. Default image-understanding output to 128 tokens and accept the explicit
   `max_output_tokens` field.
5. Expose resident/offload state, GPU memory, queue counters, and last inference
   duration in worker health responses for post-deployment verification.

## Profiling instrumentation changelog

| File | Change type | Instrumentation |
|---|---|---|
| `delta_llm/runtime_worker.py` | modified | Device-map/offload audit, CUDA memory, queue counts, completed/failed counts, last inference duration |
| `delta_llm/runtime_gateway.py` | modified | Per-model replica health, asynchronous queue state, job timing |
| `profile_output/2026-08-12_multimodal_latency.md` | created | Structured baseline and remediation report |

The health instrumentation is intentionally retained because it is also the
deployment guardrail that proves the performance fix is active.
