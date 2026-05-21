#!/bin/bash
# Profile Megatron init sub-phases (full training path, no --debug-rollout-only).
# Matches startup_profile_report.md Megatron gate run settings.

set -euo pipefail

export PYTHONBUFFERED=16
export RAY_DEDUP_LOGS=0

PROFILE_LOG="${PROFILE_LOG:-/tmp/slime_megatron_init_profile.log}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/slime_megatron_init_profile}"
mkdir -p "${RESULTS_DIR}"

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
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.20}"

if [ "${SLIME_REUSE_RAY:-0}" != "1" ] || ! ray job list --address="http://127.0.0.1:8265" >/dev/null 2>&1; then
  ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

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
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}" \
  --sglang-log-level info \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --no-offload-train \
  --no-offload-rollout \
  2>&1 | tee "${PROFILE_LOG}"

cp "${PROFILE_LOG}" "${RESULTS_DIR}/megatron_init_profile.log"

echo "=== Megatron init sub-phases ==="
grep 'SLIME_STARTUP_PROFILE phase=.*event=end' "${PROFILE_LOG}" | grep -E \
  'megatron_actor_init|megatron_actor_allocate|megatron_actor_init_wait|actor\.(distributed_init|hf_config|model_optimizer|weights_backup|ref_checkpoint)' \
  || true

python3 - "${PROFILE_LOG}" <<'PY'
import re
import sys

log_path = sys.argv[1]
pattern = re.compile(
    r"SLIME_STARTUP_PROFILE phase=([^\s]+) event=end elapsed_s=([0-9.]+)"
)
phases = {}
with open(log_path, encoding="utf-8", errors="replace") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            phases[match.group(1)] = float(match.group(2))

order = [
    "megatron_actor_allocate",
    "megatron_actor_master_endpoint_allocate",
    "megatron_actor_actor_submit",
    "megatron_actor_init_wait",
    "megatron_actor_init",
    "actor.distributed_init",
    "actor.hf_config_tokenizer_load",
    "actor.model_optimizer_checkpoint_load",
    "actor.weights_backup.actor",
    "actor.ref_checkpoint_load",
    "initial_weight_update",
]
print("\n=== Parsed summary ===")
for phase in order:
    if phase in phases:
        print(f"{phase}={phases[phase]:.3f}s")

init_wait = phases.get("megatron_actor_init_wait")
ref_load = phases.get("actor.ref_checkpoint_load")
if init_wait and ref_load:
    print(f"\nref_checkpoint_load is {100.0 * ref_load / init_wait:.1f}% of megatron_actor_init_wait")
PY
