#!/bin/bash
# End-to-end startup benchmark: naive cold managed vs warm external rollout.
#
# Naive path:
#   - managed SGLang (Slime launches engine)
#   - /health_generate startup readiness
#   - pre-patch Megatron actor launch (sequential rank-0 master endpoint)
#
# Optimized warm path:
#   - pre-launched external SGLang with /health validation
#   - post-patch Megatron actor launch (batched submit)
#
# Gate metric matches startup_profile_report.md:
#   ray_placement + rollout_data_source_init + rollout_function_load +
#   sglang_rollout_server_start + megatron_actor_init + initial_weight_update

set -euo pipefail

export PYTHONBUFFERED=16
export RAY_DEDUP_LOGS=0

MODE="${1:-both}"  # cold | warm | both
EXTERNAL_PORT="${EXTERNAL_PORT:-30000}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/slime_e2e_startup}"
mkdir -p "${RESULTS_DIR}"

COLD_LOG="${RESULTS_DIR}/cold_e2e.log"
WARM_LOG="${RESULTS_DIR}/warm_e2e.log"
EXTERNAL_LOG="${RESULTS_DIR}/sglang_external.log"

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

cleanup_external() {
  if [ -n "${EXTERNAL_PID:-}" ] && kill -0 "${EXTERNAL_PID}" 2>/dev/null; then
    kill "${EXTERNAL_PID}" 2>/dev/null || true
    wait "${EXTERNAL_PID}" 2>/dev/null || true
  fi
}

full_cleanup() {
  cleanup_external
  pkill -9 sglang || true
  sleep 2
  ray stop --force || true
  pkill -9 ray || true
  pkill -9 python || true
  sleep 2
}

ensure_ray() {
  if [ "${SLIME_REUSE_RAY:-0}" != "1" ] || ! ray job list --address="http://127.0.0.1:8265" >/dev/null 2>&1; then
    ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
  fi
}

get_node_ip() {
  python3 - <<'PY'
import ray
ray.init(address="auto", ignore_reinit_error=True)
print(ray.util.get_node_ip_address())
PY
}

submit_train_job() {
  local log_path="$1"
  local extra_env_json="$2"
  shift 2

  local runtime_env_json="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SLIME_STARTUP_PROFILE\": \"1\",
    \"SLIME_STARTUP_NVTX\": \"0\"
    ${extra_env_json}
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
    --debug-rollout-only \
    --no-offload-train \
    --no-offload-rollout \
    "$@" \
    2>&1 | tee "${log_path}"
}

summarize_log() {
  local label="$1"
  local log_path="$2"
  python3 - "${label}" "${log_path}" <<'PY'
import re
import sys

label, log_path = sys.argv[1:3]
gate_phases = [
    "ray_placement",
    "rollout_data_source_init",
    "rollout_function_load",
    "sglang_rollout_server_start",
    "megatron_actor_init",
    "initial_weight_update",
]
extra_phases = [
    "sglang_rollout_engine_init_wait",
    "megatron_actor_allocate",
    "megatron_actor_init_wait",
    "megatron_actor_master_endpoint_allocate",
    "megatron_actor_actor_submit",
]

pattern = re.compile(
    r"SLIME_STARTUP_PROFILE phase=([^\s]+) event=end elapsed_s=([0-9.]+)"
)
phases = {}
with open(log_path, encoding="utf-8", errors="replace") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            phases[match.group(1)] = float(match.group(2))

missing = [p for p in gate_phases if p not in phases]
if missing:
    print(f"{label}: INCOMPLETE missing phases: {', '.join(missing)}")
    sys.exit(1)

total = sum(phases[p] for p in gate_phases)
print(f"=== {label} ===")
for phase in gate_phases + extra_phases:
    if phase in phases:
        print(f"{phase}={phases[phase]:.3f}s")
print(f"profiled_startup_through_initial_weight_update={total:.3f}s")
PY
}

use_naive_megatron() {
  cp "${SCRIPT_DIR}/fixtures/actor_group_naive_megatron.py" "${SCRIPT_DIR}/../slime/ray/actor_group.py"
  pip install -e "${SCRIPT_DIR}/.." --no-deps >/dev/null
}

use_optimized_megatron() {
  cp "${SCRIPT_DIR}/fixtures/actor_group_optimized_megatron.py" "${SCRIPT_DIR}/../slime/ray/actor_group.py"
  pip install -e "${SCRIPT_DIR}/.." --no-deps >/dev/null
}

run_cold() {
  echo "=== Running naive cold managed startup (/health_generate, naive Megatron) ==="
  use_naive_megatron
  full_cleanup
  ensure_ray
  submit_train_job "${COLD_LOG}" \
    ', "SLIME_SGLANG_STARTUP_HEALTH_ENDPOINT": "/health_generate"' \
    || true
  summarize_log "Naive cold managed" "${COLD_LOG}"
}

run_warm() {
  echo "=== Running warm external startup (/health, optimized Megatron) ==="
  use_optimized_megatron
  full_cleanup
  ensure_ray

  NODE_IP="$(get_node_ip)"
  EXTERNAL_ADDR="${NODE_IP}:${EXTERNAL_PORT}"

  echo "Launching external SGLang at ${EXTERNAL_ADDR}"
  CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server \
    --model-path /root/Qwen3-4B \
    --host "${NODE_IP}" \
    --port "${EXTERNAL_PORT}" \
    --tp 2 \
    --mem-fraction-static "${SGLANG_MEM_FRACTION}" \
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

  submit_train_job "${WARM_LOG}" "" \
    --rollout-external \
    --rollout-external-engine-addrs "${EXTERNAL_ADDR}" \
    || true
  cleanup_external
  summarize_log "Warm external" "${WARM_LOG}"
}

print_comparison() {
  python3 - "${COLD_LOG}" "${WARM_LOG}" <<'PY'
import re
import sys

gate_phases = [
    "ray_placement",
    "rollout_data_source_init",
    "rollout_function_load",
    "sglang_rollout_server_start",
    "megatron_actor_init",
    "initial_weight_update",
]
pattern = re.compile(
    r"SLIME_STARTUP_PROFILE phase=([^\s]+) event=end elapsed_s=([0-9.]+)"
)

def load(path):
    phases = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                phases[match.group(1)] = float(match.group(2))
    return phases

cold_path, warm_path = sys.argv[1:3]
cold = load(cold_path)
warm = load(warm_path)

def total(phases):
    missing = [p for p in gate_phases if p not in phases]
    if missing:
        raise SystemExit(f"Missing phases: {missing}")
    return sum(phases[p] for p in gate_phases)

cold_total = total(cold)
warm_total = total(warm)
delta = cold_total - warm_total

print("")
print("=== End-to-end startup comparison ===")
print(f"{'Phase':<40} {'Cold':>10} {'Warm':>10} {'Delta':>10}")
print("-" * 72)
for phase in gate_phases:
    c = cold.get(phase, float("nan"))
    w = warm.get(phase, float("nan"))
    d = c - w
    print(f"{phase:<40} {c:10.3f} {w:10.3f} {d:10.3f}")
print("-" * 72)
print(f"{'profiled_startup_through_initial_weight_update':<40} {cold_total:10.3f} {warm_total:10.3f} {delta:10.3f}")
print(f"Warm is {delta:.3f}s faster ({100.0 * delta / cold_total:.1f}% reduction)")
print("")
print("Naive path: managed SGLang + /health_generate + pre-patch Megatron launch")
print("Warm path: external SGLang + /health + post-patch Megatron launch")
PY
}

case "${MODE}" in
  cold)
    run_cold
    ;;
  warm)
    run_warm
    ;;
  both)
    run_cold
    echo ""
    run_warm
    print_comparison
    ;;
  *)
    echo "Usage: $0 [cold|warm|both]" >&2
    exit 1
    ;;
esac
