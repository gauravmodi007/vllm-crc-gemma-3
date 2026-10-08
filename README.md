# TinyLlama on vLLM CPU in OpenShift Local (CRC)

A VS Code POC for an already-running CRC cluster. Target: Linux amd64 node on an Intel Xeon Gold 6448H host, one vLLM replica, a 6 GiB container memory limit, and a persistent cache claim. No GPU is requested.

This project was deployed to the current CRC on 2026-10-08 and passed its CPU preflight and functional API smoke tests. The measured smoke-test results are in `results/smoke-test.json`; these are not a load or production benchmark. On this cluster the original 512-token/1 GiB-cache profile did not fit the 6 GiB limit, so the current POC configuration uses a 128-token context and a 6 MiB KV cache. A successful run here does not guarantee startup on another vLLM release or cluster.

## 1. Open the project

Extract the ZIP, then in VS Code choose File > Open Folder and select `vllm-crc-tinyllama`. Open Terminal > New Terminal. All following commands run from this folder and work in PowerShell, Bash, or zsh without line continuations.

Required commands on PATH: `oc`, `uv`, and Python managed by uv. Podman is used only to resolve the image digest; it is not the OpenShift runtime. If you already have a verified image digest, set it in `config.json` and skip Podman. The cluster uses CRI-O to run the containers.

CRC must use the OpenShift preset. CRC being running does not prove the current `oc` context is your CRC cluster.

```console
crc status
oc whoami
oc config current-context
oc get nodes -o wide
```

If `oc` is missing, use CRC's `oc-env` instructions for your shell. For Bash/zsh: `eval "$(crc oc-env)"`; for PowerShell: `crc oc-env | Invoke-Expression`. If not logged in, use `crc console --credentials` and the displayed login instructions locally. Do not copy passwords into this project or agent chat.

The cluster API URL and node must correspond to your intended CRC cluster before running apply commands.

## 2. Install the project tools

```console
uv sync
uv run python scripts/manage.py inspect
```

`uv` installs only local tooling (YAML rendering and HTTP API tests). vLLM and PyTorch are supplied by the container image; do not install vLLM on your Windows or macOS host for this project.

Inspect prints node architecture, storage classes, PVs, and node JSON including allocatable resources. Verify:

- Node architecture is `amd64`. The Xeon host alone does not prove the CRC node's architecture or CPU flags.
- There is room for a 2-CPU, 6 GiB application in addition to OpenShift's own pods. The low-concurrency settings are for a POC, not an optimal Xeon configuration.
- CRC has sufficient disk space for the image, model files, cache, and cluster data. A requested 20 GiB volume does not enlarge the VM disk or reserve 20 GiB of physical free space on every provisioner.
- Your cluster can reach Docker Hub and Hugging Face through your network/proxy. Configure trusted corporate CAs properly if needed.

If there is insufficient allocatable memory, increase CRC's memory allocation using the supported CRC configuration workflow for your installed version, then recheck it. Do not stop or rebuild a working CRC cluster blindly. A typical 8 GiB total CRC VM does not have 8 GiB spare for this pod.

## 3. Select storage in config.json

The default `storage_class: null` omits `storageClassName`, allowing the cluster's default StorageClass. It creates a **20 GiB PVC**, not a static local PV. Check the actual provisioner; do not assume its name or behavior.

Options:

| Situation | Configuration |
|---|---|
| Use CRC's default provisioner | Leave `storage_class` and `existing_pv` as `null` |
| Use a chosen provisioner | Set `storage_class` to the exact result from `oc get storageclass` |
| Bind an existing compatible local PV | Set `existing_pv` to its PV name and `storage_class` to its class (use `""` for a classless PV) |

For an existing PV, check capacity >=20Gi, ReadWriteOnce, available/unclaimed state, matching storage class, filesystem permissions, and node affinity selecting the CRC node. An already-bound PV cannot be reused by a new claim. If you already have a PVC, adapt the mount and omit PVC creation rather than trying to reuse its PV by name.

`manifests/local-pv.example.yaml` is an optional static local PV example. Replace both placeholders, create the backing directory **inside the CRC node**, ensure disk capacity and suitable ownership/SELinux context, and apply it as a cluster administrator. A host desktop path is not automatically a path inside CRC. `kubernetes.io/no-provisioner` does not create that directory or allocate storage. Then set `storage_class` to `vllm-local` and `existing_pv` to `vllm-model-cache-local`.

Do not apply the example unchanged. Retain means data cleanup and rebinding require manual administration. CRC deletion can destroy local data even with Retain.

If the PVC uses WaitForFirstConsumer, Pending before the preflight Job is scheduled can be normal. The preflight Job mounts the cache, so the workflow does not wait for PVC binding before creating a consumer.

PVC class/volume binding cannot simply be changed after binding. Choose correctly before preflight. Preserve model data when changing storage; do not delete PVCs to solve a scheduling error.

## 4. Resolve and pin the official CPU image

The bootstrap tag is `docker.io/vllm/vllm-openai-cpu:latest-x86_64`, which current upstream vLLM CPU documentation lists. The command below pulls it as linux/amd64, checks image architecture, reads its registry digest, and writes that digest to `config.json`.

```console
podman info
uv run python scripts/manage.py pin-image
```

If Podman requires its own machine, start the existing machine if appropriate. This is separate from CRC. If you prefer a verified version tag, change `image` in config.json to that tag before pinning. Do not invent version tags.

You may instead supply a digest from your registry tooling directly in `config.json`. Deployment and preflight reject floating tags. The image is only considered usable on your cluster after preflight and serving tests pass; an official tag is not proof of compatibility with your particular CRC VM.

Optional: set `model_revision` to a verified Hugging Face commit hash to also fix the model/tokenizer revision. With null, the Hub's default revision is used. TinyLlama is a public model; downloading it normally needs no token. The cache is populated automatically at first serving startup.

## 5. Render and review YAML

```console
uv run python scripts/manage.py render
```

Open `rendered/app.yaml`, `rendered/preflight.yaml`, and `rendered/route.yaml`. These are generated from `config.json` and `scripts/manage.py`; edit those sources, not rendered output.

| Resource or setting | Meaning |
|---|---|
| Dedicated `vllm-poc` project | Keeps the POC resources together |
| ServiceAccount `vllm` | Pod identity, without automatic API token mounting |
| PVC `model-cache` | Stores downloaded model and vLLM cache across pod replacement |
| Deployment, one replica | Keeps one serving pod running |
| Recreate rollout strategy | Avoids overlapping 6 GiB replicas and local RWO mount conflicts during updates |
| `command: [vllm, serve]` | Explicitly starts the current serving CLI |
| TinyLlama model argument | Selects the model loaded by vLLM |
| `--dtype bfloat16` | Uses two-byte weight values; CPU-oriented starting configuration |
| `--served-model-name tinyllama` | API requests use the short model ID `tinyllama` |
| `--max-model-len 128` | Reduced from 512 after the pinned release reported only 0.01 GiB available for KV cache under the 6 GiB limit |
| `--max-num-seqs 1` | Caps active sequence concurrency |
| `--max-num-batched-tokens 128` | Limits per-step scheduling budget |
| `--enforce-eager` | Avoids graph/compilation overhead for this initial POC; may reduce performance |
| `--kv-cache-memory-bytes 6291456` | Reserves a 6 MiB KV cache; vLLM 0.31.0's integer-GiB environment setting could not fit |
| `OMP_NUM_THREADS=2`, `nobind` | Limits OpenMP threads without assuming guest CPU IDs or NUMA binding privileges |
| 2 CPU / 6Gi requests and limits | Scheduling reservation and container ceilings; no dedicated physical CPU guarantee |
| Memory-backed `/dev/shm`, limit 1Gi | Shared memory for process communication |
| `HF_HOME=/cache/huggingface` | Persistent Hugging Face cache; avoids writing under `/root` |
| `HOME=/tmp`, writable cache paths | Accommodates OpenShift's assigned user ID |
| Startup probe, up to ~30 minutes | Allows time for first download and initialization before liveness starts |
| Health readiness probe | Removes an unready pod from Service endpoints |
| Service on port 8000 | Stable in-cluster endpoint |
| Optional TLS edge Route | Adds an external cluster endpoint when explicitly requested |

The CPU-only build detects its CPU platform. The project does not carry over `--device cpu` from older command examples; CLI options vary across versions. Check the pinned image's help if changing the invocation.

Shared memory is a size ceiling, not an extra memory allowance or a reservation of all that RAM at startup. Used memory-backed volume pages count toward container memory. BF16 weights are approximately 2.1 GiB for 1.1 billion parameters; KV cache, loaded software, temporary tensors, shared pages, and process overhead also consume memory. Watch the actual peak rather than adding independent limits as if they were separate budgets.

Default security uses assigned non-root UID, dropped capabilities, and RuntimeDefault seccomp. No privileged pod, host IPC, or automatic anyuid grant. Some image releases may need a derived image for executable/library permissions or an administrator-reviewed CPU policy adjustment. Preflight catches several such failures; it does not exercise every inference kernel.

## 6. Run the preflight inside CRC

```console
uv run python scripts/manage.py preflight
```

This creates the project if needed, dry-runs the preflight resources against the API, then runs a Job using the same image, cache, and security settings. It prints:

- The actual CPU flags and CPU affinity visible inside the pod.
- The runtime user ID and ability to write both persistent cache directories.
- Installed vLLM/PyTorch versions and CLI help.
- A small BF16 CPU matrix operation result.

Success: Job Complete and `PREFLIGHT PASSED` in logs. A BF16 test cannot prove every vLLM native kernel is compatible. Actual serving remains the final test. If it fails, do not continue blindly; diagnose the Job and PVC.

```console
oc -n vllm-poc get pods,pvc,jobs
oc -n vllm-poc describe job vllm-preflight
oc -n vllm-poc logs job/vllm-preflight
oc -n vllm-poc get events --sort-by=.lastTimestamp
```

The wait command may take 15 minutes. Image-pull problems, scheduling failures, and permissions errors appear in Events even when logs do not exist. If retrying with a changed image, pin it again and rerun preflight.

## 7. Deploy the server

```console
uv run python scripts/manage.py deploy
uv run python scripts/manage.py status
oc -n vllm-poc logs -f deployment/vllm-tinyllama
```

Deploy requires a successful preflight Job using the configured image digest. It performs a server-side dry-run, applies Deployment/PVC/Service resources, and waits up to 30 minutes for readiness. On first startup vLLM downloads model/tokenizer files to the PVC, loads the model into RAM, and starts HTTP serving on port 8000. Storage requests are not model RAM.

If changing resources, storage, security, or CLI settings later, review rendered YAML and rerun preflight where relevant. Admission of an existing Deployment's pod template does not guarantee SCC admission of the new pod; pod Events are authoritative.

## 8. Access and test from your computer

In one VS Code terminal, keep this running:

```console
uv run python scripts/manage.py port-forward
```

In a second VS Code terminal:

```console
uv run python scripts/smoke_test.py
```

Local endpoint: `http://127.0.0.1:8000`. Tests cover health, model listing, single-turn chat, multi-turn chat, streaming first-content latency, and Prometheus metrics. Results go to `results/smoke-test.json`.

The tests verify API behavior and nonempty responses, not semantic correctness. TinyLlama's response quality is modest. Review the multi-turn answer manually. End-to-end tokens/sec includes request and prefill overhead; it is not pure decode throughput. Streaming TTFT is time to first nonempty content event, not necessarily the first internal model token. These are functional smoke tests, not a concurrency/load benchmark.

Examples without shell-specific quoting:

```python
import httpx
response = httpx.post(
    "http://127.0.0.1:8000/v1/chat/completions",
    json={"model": "tinyllama", "messages": [{"role": "user", "content": "Explain pods simply."}], "max_tokens": 64},
    timeout=300,
)
response.raise_for_status()
print(response.json()["choices"][0]["message"]["content"])
```

## 9. Optional Route

Port-forward is enough for local testing. To deliberately expose the service through CRC's router:

```console
uv run python scripts/manage.py route
oc -n vllm-poc get route vllm-tinyllama
```

Use the returned HTTPS hostname. CRC DNS/certificate trust must be configured on the client; use a trusted certificate/CA rather than disabling verification. Edge TLS encrypts client-to-router traffic. This optional POC Route has no API authentication; restrict access to the intended local POC environment or add authentication before broader exposure. To test a trusted Route: `uv run python scripts/smoke_test.py --url https://RETURNED_HOSTNAME`.

## 10. Diagnose failures

```console
uv run python scripts/manage.py diagnose
```

| Symptom | Check and next step |
|---|---|
| Pending / Insufficient memory or CPU | Inspect pod Events and node allocatable capacity; account for OpenShift overhead |
| PVC Pending | Check matching class/PV, capacity, claim binding, provisioner, and node affinity; WaitForFirstConsumer needs a scheduled consumer |
| ImagePullBackOff | Verify exact digest, network/proxy, registry access, and pull Events |
| Exit 132 / SIGILL | Check CPU flags inside CRC and pinned build requirements; host Xeon features may be masked in a VM; changing YAML RAM will not fix unsupported instructions |
| Exit 137 | Inspect termination reason and node Events; OOMKilled suggests a memory limit issue but 137 alone is not proof of OOM |
| Permission denied | Check assigned UID, image executable/library paths, PVC group ownership, and SELinux labels; adjust only the necessary paths in a derived image or storage permissions |
| Insufficient /dev/shm | Increase the rendered shm size after measuring use, while keeping total used memory within 6Gi |
| OOMKilled | Measure peak RSS/shared memory, lower concurrency/context/batching, and if supported reduce KV cache; increase memory if the selected release still cannot fit |
| CLI unrecognized argument | Read the pinned image CLI help; update source args for that version and rerender |
| NUMA / set_mempolicy / scheduling permission errors | Preserve logs; investigate the exact syscall and cluster policy. `nobind` avoids fixed affinity but does not eliminate every NUMA call |
| SSL/download error | Configure proxy and trusted CA for container/model downloads; do not disable TLS verification |
| Model loaded but chat fails | Confirm tokenizer chat template exists and request model ID is `tinyllama` |
| Route timeout | Validate local port-forward first, then router timeout and client timeout |

For production, CPU binding and NUMA placement deserve separate tuning. The default intentionally uses a small thread count under a Kubernetes CPU quota. No throughput estimate is promised.

## 11. Stop without deleting model data

```console
uv run python scripts/manage.py stop
```

This scales the server to zero and preserves the PVC. Run deploy to restore one replica. Stop the port-forward terminal with Ctrl+C. Do not delete the namespace/PVC unless you intentionally want to remove this POC and understand the PV reclaim policy. Deleting CRC itself can remove local storage.

## Give this project to your VS Code agent

Open `AGENT_PROMPT.md` and paste its contents into the agent. It asks the agent to inspect your actual cluster, validate image compatibility, deploy, and execute smoke tests. It also requires an evidence-based handoff rather than claiming success from YAML alone.

## Sources checked for this project (2026-10-08)

- vLLM CPU installation, image tags, CPU settings: https://docs.vllm.ai/en/latest/getting_started/installation/cpu/
- vLLM current serving arguments: https://docs.vllm.ai/en/latest/cli/serve/
- Model card: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
- OpenShift arbitrary UID guidance: https://docs.redhat.com/en/documentation/openshift_container_platform/4.19/html/images/creating-images
- Memory-backed volumes and local volumes: https://kubernetes.io/docs/concepts/storage/volumes/
- Storage classes and binding: https://kubernetes.io/docs/concepts/storage/storage-classes/
- CRC usage: https://crc.dev/docs/using/
