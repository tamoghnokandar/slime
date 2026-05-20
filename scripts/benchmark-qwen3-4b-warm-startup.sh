#!/bin/bash
# Short warm-external startup benchmark for Qwen3-4B.
# Matches startup_profile_report.md: one rollout, skip eval, 512-token responses.

set -euo pipefail

export PYTHONBUFFERED=16
export RAY_DEDUP_LOGS=0

EXTERNAL_PORT="${EXTERNAL_PORT:-30000}"
EXTERNAL_LOG="${EXTERNAL_LOG:-/tmp/sglang_external_warm.log}"
PROFILE_LOG="${PROFILE_LOG:-/tmp/slime_warm_startup_profile.log}"

cleanup() {
  if [ -n "${EXTERNAL_PID:-}" ] && kill -0 "${EXTERNAL_PID}" 2>/dev/null; then
    kill "${EXTERNAL_PID}" 2>/dev/null || true
    wait "${EXTERNAL_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [ "${SLIME_SKIP_STARTUP_CLEANUP:-0}" != "1" ]; then
  pkill -9 sglang || true
  sleep 2
  ray stop --force || true
  pkill -9 ray || true
  pkill -9 python || true
  sleep 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi

NUM_GPUS="${NUM_GPUS:-2}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

if [ "${SLIME_REUSE_RAY:-0}" != "1" ] || ! ray job list --address="http://127.0.0.1:8265" >/dev/null 2>&1; then
  ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

NODE_IP="$(python3 - <<'PY'
import ray
ray.init(address="auto", ignore_reinit_error=True)
print(ray.util.get_node_ip_address())
PY
)"
EXTERNAL_ADDR="${NODE_IP}:${EXTERNAL_PORT}"

echo "Launching external SGLang at ${EXTERNAL_ADDR}"
CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server \
  --model-path /root/Qwen3-4B \
  --host "${NODE_IP}" \
  --port "${EXTERNAL_PORT}" \
  --tp 2 \
  --mem-fraction-static 0.35 \
  --skip-server-warmup \
  >"${EXTERNAL_LOG}" 2>&1 &
EXTERNAL_PID=$!

python3 - <<PY
import time
import urllib.request

addr = "http://${NODE_IP}:${EXTERNAL_PORT}"
deadline = time.time() + 900
while time.time() < deadline:
    try:
        with urllib.request.urlopen(f"{addr}/health", timeout=2) as resp:
            if resp.status == 200:
                print(f"External SGLang /health ready at {addr}")
                break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("Timed out waiting for external SGLang /health")
PY

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SLIME_STARTUP_PROFILE\": \"1\",
    \"SLIME_STARTUP_NVTX\": \"0\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  --colocate \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /root/Qwen3-4B \
  --ref-load /root/Qwen3-4B_torch_dist \
  --load /root/Qwen3-4B_slime/ \
  --save /root/Qwen3-4B_slime/ \
  --save-interval 1000000 \
  --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl \
  --input-key prompt \
  --label-key label \
  --apply-chat-template \
  --rollout-shuffle \
  --rm-type deepscaler \
  --num-rollout 1 \
  --rollout-batch-size 1 \
  --n-samples-per-prompt 1 \
  --rollout-max-response-len 512 \
  --rollout-temperature 1 \
  --global-batch-size 1 \
  --skip-eval-before-train \
  --tensor-model-parallel-size 2 \
  --sequence-parallel \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu 1024 \
  --advantage-estimator grpo \
  --use-kl-loss \
  --kl-loss-coef 0.00 \
  --kl-loss-type low_var_kl \
  --entropy-coef 0.00 \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --rollout-num-gpus-per-engine 2 \
  --sglang-mem-fraction-static 0.35 \
  --sglang-log-level info \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --debug-rollout-only \
  --no-offload-train \
  --no-offload-rollout \
  --rollout-external \
  --rollout-external-engine-addrs "${EXTERNAL_ADDR}" \
  2>&1 | tee "${PROFILE_LOG}"

echo "=== Startup profile phases ==="
grep 'SLIME_STARTUP_PROFILE phase=.*event=end' "${PROFILE_LOG}" || true

echo "=== Rollout throughput ==="
grep -E 'it/s|tokens_per_gpu_per_sec|perf/rollout_time' "${PROFILE_LOG}" | tail -20 || true
