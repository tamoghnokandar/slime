#!/bin/bash
# Benchmark-gated deferred optimizer startup experiment for Qwen3-4B.
#
# This intentionally uses the full training path, not --debug-rollout-only, so
# Megatron model/optimizer/checkpoint behavior is exercised.

set -euo pipefail

export PYTHONBUFFERED=16
export RAY_DEDUP_LOGS=0

RESULTS_DIR="${RESULTS_DIR:-/tmp/slime_deferred_optimizer_startup}"
mkdir -p "${RESULTS_DIR}"

BASELINE_LOG="${RESULTS_DIR}/baseline_cold_full_train.log"
CANDIDATE_LOG="${RESULTS_DIR}/deferred_optimizer_cold_full_train.log"
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.20}"
NUM_GPUS="${NUM_GPUS:-2}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"

missing=()
for cmd in ray python3; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    missing+=("${cmd}")
  fi
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "Missing required command(s): ${missing[*]}" >&2
  echo "Run this benchmark on the target GPU/Megatron node before updating startup reports." >&2
  exit 127
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | { grep -o 'NV[0-9][0-9]*' || true; } | wc -l)
else
  NVLINK_COUNT=0
fi
if [ "$NVLINK_COUNT" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi

full_cleanup() {
  pkill -9 sglang || true
  sleep 2
  ray stop --force || true
  pkill -9 ray || true
  pkill -9 python || true
  sleep 2
}

ensure_ray() {
  ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
}

submit_train_job() {
  local log_path="$1"
  shift

  local runtime_env_json="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SLIME_STARTUP_PROFILE\": \"1\",
    \"SLIME_STARTUP_NVTX\": \"0\",
    \"SLIME_SGLANG_STARTUP_HEALTH_ENDPOINT\": \"/health_generate\"
  }
}"

  ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${runtime_env_json}" \
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
    --start-rollout-id 0 \
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
    "$@" \
    2>&1 | tee "${log_path}"
}

run_case() {
  local label="$1"
  local log_path="$2"
  shift 2

  echo "=== Running ${label} ==="
  full_cleanup
  ensure_ray
  submit_train_job "${log_path}" "$@"
}

compare_logs() {
  python3 - "${BASELINE_LOG}" "${CANDIDATE_LOG}" <<'PY'
import re
import sys

baseline_path, candidate_path = sys.argv[1:3]
gate_phases = [
    "ray_placement",
    "rollout_data_source_init",
    "rollout_function_load",
    "sglang_rollout_server_start",
    "megatron_actor_init",
    "initial_weight_update",
]
reported_phases = [
    "actor.model_optimizer_checkpoint_load",
    "actor.deferred_optimizer_checkpoint_load",
    *gate_phases,
]
pattern = re.compile(r"SLIME_STARTUP_PROFILE phase=([^\s]+) event=end elapsed_s=([0-9.]+)")


def load(path):
    phases = {}
    text = open(path, encoding="utf-8", errors="replace").read()
    for match in pattern.finditer(text):
        phases[match.group(1)] = float(match.group(2))
    if "Traceback (most recent call last)" in text:
        raise SystemExit(f"{path}: job log contains a Python traceback")
    return phases


def total(phases):
    missing = [phase for phase in gate_phases if phase not in phases]
    if missing:
        raise SystemExit(f"Missing startup gate phases: {missing}")
    return sum(phases[phase] for phase in gate_phases)


baseline = load(baseline_path)
candidate = load(candidate_path)
if "actor.deferred_optimizer_checkpoint_load" not in candidate:
    raise SystemExit("Candidate did not execute actor.deferred_optimizer_checkpoint_load; first train-step restore was not verified")

baseline_total = total(baseline)
candidate_total = total(candidate)
delta = baseline_total - candidate_total
threshold = max(2.0, baseline_total * 0.03)
passed = delta >= threshold

print("")
print("=== Deferred optimizer startup comparison ===")
print(f"{'Phase':<42} {'Baseline':>10} {'Candidate':>10} {'Delta':>10}")
print("-" * 76)
for phase in reported_phases:
    base = baseline.get(phase)
    cand = candidate.get(phase)
    if base is None and cand is None:
        continue
    base_text = "n/a" if base is None else f"{base:.3f}"
    cand_text = "n/a" if cand is None else f"{cand:.3f}"
    if base is None or cand is None:
        delta_text = "n/a"
    else:
        delta_text = f"{base - cand:.3f}"
    print(f"{phase:<42} {base_text:>10} {cand_text:>10} {delta_text:>10}")
print("-" * 76)
print(f"{'profiled_startup_through_initial_weight_update':<42} {baseline_total:10.3f} {candidate_total:10.3f} {delta:10.3f}")
print(f"retention_threshold={threshold:.3f}s")

if not passed:
    raise SystemExit(
        "FAIL: deferred optimizer startup did not clear the retention gate. "
        "Do not update startup reports with candidate results."
    )

print("PASS: deferred optimizer startup cleared the retention gate. Report updates are allowed.")
PY
}

run_case "baseline cold full-training startup" "${BASELINE_LOG}"
run_case "candidate cold full-training startup with --defer-optimizer-init" "${CANDIDATE_LOG}" --defer-optimizer-init
compare_logs
