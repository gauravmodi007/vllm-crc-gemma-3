# CRC POC Status

Live snapshot collected on 2026-10-08 at approximately 08:13 EDT. This is a point-in-time report, not continuous monitoring.

## Cluster and node

- CRC VM and OpenShift were running; OpenShift version: 4.22.14.
- The single CRC node was Ready, amd64, running Kubernetes v1.35.6.
- Node allocatable: 5800m CPU and 15910740Ki memory.
- CRC reported 12.28GB of 16.76GB RAM used and 32.74GB of 68.11GB VM disk used.
- Scheduler requests were 4251m CPU (73%) and 15516Mi memory (99%). Memory request headroom is very tight.
- Private node and pod IP addresses are intentionally omitted.

## Serving pod health

- Namespace: `vllm-poc`.
- Deployment: `vllm-tinyllama`, one desired and one available replica.
- Pod was `Running` and `Ready` (`1/1`) with zero restarts.
- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, served as `tinyllama`.
- vLLM version: 0.31.0; PyTorch: 2.13.0+cpu.
- Image: `docker.io/vllm/vllm-openai-cpu@sha256:8024248339dc6878daa5349344ed29d49c8a6732f6bdf7400fda32ce33e4b30b`.
- Container cgroup memory at capture: approximately 4.95GiB current, 5.07GiB peak, 6GiB limit; zero OOM events.
- The OpenShift Metrics API was unavailable; memory figures above come from the container cgroup.

The in-cluster CPU preflight passed, as did local port-forward smoke checks for health, model listing, single-turn and multi-turn chat, streaming, and metrics. The smoke tests are functional checks, not a load benchmark. Measured request latency and streaming TTFT are in the local, gitignored `results/smoke-test.json`.

## Deployed pod configuration

The checked-in [rendered application manifest](./rendered/app.yaml) records the applied ServiceAccount, PVC, Deployment pod template, and Service. The source of truth remains [config.json](./config.json) and [scripts/manage.py](./scripts/manage.py); regenerate the manifest with:

```console
uv run python scripts/manage.py render
```

Key settings:

- CPU request/limit: 2; memory request/limit: 6Gi.
- BF16, 128-token model context, one sequence, 128 batched tokens, eager execution.
- CPU KV cache: 6MiB (`--kv-cache-memory-bytes 6291456`).
- `/cache` on a dynamically provisioned 20Gi PVC request using `crc-csi-hostpath-provisioner`. The bound provisioner volume reports 63Gi capacity; this is not a guarantee of equivalent free VM disk.
- Memory-backed `/dev/shm`, capped at 1Gi.
- Assigned non-root UID, dropped capabilities, RuntimeDefault seccomp, no automatic service-account token mount, and startup/readiness/liveness health probes.
- Local API access is via `uv run python scripts/manage.py port-forward` at `http://127.0.0.1:8000`. No public Route was applied.

The original 512-token context and 1Gi KV cache did not fit the 6Gi container on this CRC. The reduced settings above are specific to this POC environment and constrain prompt plus generated context.
