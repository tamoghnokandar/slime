# Installation Notes

## SkyPilot Qwen3-4B Asset Preparation

For the `swapserve-qwen` SkyPilot cluster, prepare the Qwen3-4B model, datasets,
and Megatron `torch_dist` checkpoint inside the `slimerl/slime:latest` Docker
image. This avoids missing runtime dependencies such as `torch`, SGLang,
Transformer Engine, and Megatron helpers on the host VM image.

Run from the local slime checkout:

```bash
sky exec swapserve-qwen --workdir . '
set -euxo pipefail

docker pull slimerl/slime:latest

docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/sky_workdir:/root/slime" \
  -v /root:/root_host \
  slimerl/slime:latest \
  bash -lc "
    set -euxo pipefail
    cd /root/slime
    pip install -e . --no-deps

    hf download Qwen/Qwen3-4B --local-dir /root_host/Qwen3-4B
    hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root_host/dapo-math-17k
    hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root_host/aime-2024

    source scripts/models/qwen3-4B.sh
    PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
      \"\${MODEL_ARGS[@]}\" \
      --hf-checkpoint /root_host/Qwen3-4B \
      --save /root_host/Qwen3-4B_torch_dist

    mkdir -p /root_host/Qwen3-4B_slime
  "
'
```

## SkyPilot Startup Profiling

After the assets above are prepared, run startup-only profiling for
`scripts/run-qwen3-4B.sh` from the local slime checkout. This command runs inside
the `slimerl/slime:latest` container, applies a temporary low-memory profiling
patch to the synced remote script, and enables `SLIME_STARTUP_PROFILE`.

```bash
sky exec swapserve-qwen --workdir . '
set -euxo pipefail

docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/sky_workdir:/root/slime" \
  -v /root/Qwen3-4B:/root/Qwen3-4B \
  -v /root/Qwen3-4B_torch_dist:/root/Qwen3-4B_torch_dist \
  -v /root/Qwen3-4B_slime:/root/Qwen3-4B_slime \
  -v /root/dapo-math-17k:/root/dapo-math-17k \
  -v /root/aime-2024:/root/aime-2024 \
  slimerl/slime:latest \
  bash -lc "
    set -euxo pipefail
    cd /root/slime
    pip install -e . --no-deps

    python - <<'"'"'PY'"'"'
from pathlib import Path

p = Path(\"scripts/run-qwen3-4B.sh\")
s = p.read_text()
s = s.replace(\"--num-rollout 3000\", \"--num-rollout 1\")
s = s.replace(\"--save-interval 20\", \"--save-interval 1000000\")
s = s.replace(\"--eval-interval 20\", \"--eval-interval 1000000\")
s = s.replace(\"--rollout-batch-size 32\", \"--rollout-batch-size 1\")
s = s.replace(\"--n-samples-per-prompt 8\", \"--n-samples-per-prompt 1\")
s = s.replace(\"--rollout-max-response-len 8192\", \"--rollout-max-response-len 512\")
s = s.replace(\"--global-batch-size 256\", \"--global-batch-size 1\")
s = s.replace(\"--max-tokens-per-gpu 9216\", \"--max-tokens-per-gpu 1024\")
s = s.replace(\"--sglang-mem-fraction-static 0.7\", \"--sglang-mem-fraction-static 0.35\")

needle = \"   --attention-backend flash\\n\"
if \"--no-offload-train\" not in s:
    s = s.replace(needle, needle + \"   --no-offload-train\\n   --no-offload-rollout\\n\")

if \"--sglang-log-level info\" not in s:
    s = s.replace(
        \"   --sglang-mem-fraction-static 0.35\\n\",
        \"   --sglang-mem-fraction-static 0.35\\n   --sglang-log-level info\\n\",
    )

p.write_text(s)
PY

    RAY_DEDUP_LOGS=0 \
    NUM_GPUS=2 \
    SLIME_STARTUP_PROFILE=1 \
    SLIME_SKIP_STARTUP_CLEANUP=1 \
    SLIME_REUSE_RAY=1 \
    bash scripts/run-qwen3-4B.sh
  "
'
```

For startup-only profiling, stop the job after the relevant startup phase lines
appear. The deeper SGLang engine timers include:

```text
sglang_engine_compute_server_args
sglang_engine_launch_server_process
sglang_engine_process_start
sglang_engine_wait_health_generate
sglang_engine_router_register
```
