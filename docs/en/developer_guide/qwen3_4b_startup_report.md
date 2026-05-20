# Qwen3-4B Startup Profiling Report

## Environment

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -U pip
uv pip install "git+https://github.com/castform-ai/skypilot.git@gk-patch"
uv pip install -r requirements.txt
uv pip install -e . --no-deps
export SKYPILOT_API_SERVER_ENDPOINT=https://skypilot.cgft.io
export SKYPILOT_SERVICE_ACCOUNT_TOKEN=<set-in-shell-only>
sky api login -e "$SKYPILOT_API_SERVER_ENDPOINT" --token "$SKYPILOT_SERVICE_ACCOUNT_TOKEN"
python --version
sky --version
sky status
sky check
python -c "import slime, ray; print('ok')"
```

Do not save `SKYPILOT_SERVICE_ACCOUNT_TOKEN` in tracked files, logs, or scripts.

## Runs

Baseline:

```bash
bash scripts/run-qwen3-4B.sh
```

Structured startup timing:

```bash
SLIME_STARTUP_PROFILE=1 bash scripts/run-qwen3-4B.sh
```

Nsight Systems on the NVIDIA target host:

```bash
mkdir -p startup_profiles/qwen3_4b/$(date +%Y%m%d_%H%M%S)
SLIME_STARTUP_PROFILE=1 SLIME_STARTUP_NVTX=1 nsys profile \
  --trace=cuda,nvtx,nccl,cublas,cudnn,osrt \
  --output=startup_profiles/qwen3_4b/$(date +%Y%m%d_%H%M%S)/startup \
  bash scripts/run-qwen3-4B.sh
```

## Phase Timings

Startup timing is opt-in via `SLIME_STARTUP_PROFILE=1` and emits lines like:

```text
SLIME_STARTUP_PROFILE phase=<name> event=end elapsed_s=<seconds>
```

Capture before/after values here:

| Run | ray_placement | sglang_rollout_startup | megatron_actor_init | initial_weight_update | total startup |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | TBD | TBD | TBD | TBD | TBD |
| optimized-1 | TBD | TBD | TBD | TBD | TBD |

## 2026-05-18 SkyPilot Dev Run

Target:

```text
SkyPilot cluster: slime-dev
Infra: Vast Czechia
GPU: 2x NVIDIA A100-SXM4-40GB
Workspace: /root/sky_workdir
```

Current blocker: the dev image did not contain the full training runtime/assets needed by `scripts/run-qwen3-4B.sh`.

DAPO data is not tracked in this repository. The repo contains DAPO reward/scoring utilities and examples, but the training JSONL expected by `scripts/run-qwen3-4B.sh` is downloaded from Hugging Face:

```bash
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
```

Missing runtime assets observed on the remote node:

```text
/root/Megatron-LM
/root/Qwen3-4B
/root/Qwen3-4B_torch_dist
/root/Qwen3-4B_slime
/root/dapo-math-17k/dapo-math-17k.jsonl
/root/aime-2024/aime-2024.jsonl
```

Assets later downloaded successfully on the fresh node:

```text
/root/Qwen3-4B
/root/dapo-math-17k/dapo-math-17k.jsonl
/root/aime-2024/aime-2024.jsonl
```

Megatron-LM was cloned at commit `3714d81d418c9f1bca4594fc35f9e8289f652862`, matching the repo's stable SGLang 0.5.9 setup guidance.

Observed runs:

| Run | Command | Result | Highest measured pain point |
| --- | --- | --- | --- |
| `first_run.log` | `NUM_GPUS=2 SLIME_STARTUP_PROFILE=1 bash scripts/run-qwen3-4B.sh` | Failed before `train()` profiling: `ModuleNotFoundError: No module named 'sglang'` | Default cleanup stopped SkyPilot Ray, then script paid a fresh Ray cold start. |
| `reuse_ray_after_sglang.log` | `NUM_GPUS=2 SLIME_SKIP_STARTUP_CLEANUP=1 SLIME_REUSE_RAY=1 SLIME_STARTUP_PROFILE=1 bash scripts/run-qwen3-4B.sh` | Failed before `train()` profiling: `ModuleNotFoundError: No module named 'megatron'` | Python import/runtime parity; Megatron-LM is absent. |

Measured pre-model startup timings from logs:

| Segment | Cold run | Reuse-Ray run | Notes |
| --- | ---: | ---: | --- |
| Script cleanup / Ray shutdown | ~6-7s plus two fixed `sleep 3` calls | 0s | Removed from iterative profiling via `SLIME_SKIP_STARTUP_CLEANUP=1`. |
| Ray head startup | ~10.4s from cleanup complete to `Ray runtime started` | 0s | Skipped when `SLIME_REUSE_RAY=1` and dashboard is alive. |
| Ray job submit + runtime env + Python import to failure | ~7.2s for missing `sglang` | ~19.1s for SGLang import then missing `megatron` | Still before model, SGLang server, checkpoint, or optimizer startup. |
| One-time remote env install | minutes, dominated by SGLang/PyTorch/CUDA downloads | N/A | Not a script runtime cost, but a dev-server setup pain point. |

Largest current pain points, in order:

1. Environment parity: the SkyPilot dev image lacks Megatron-LM, SGLang initially, Nsight Systems, checkpoints, and datasets. This prevents profiling the intended model startup phases.
2. Cold script cleanup: the default script kills Ray/Python and sleeps twice, adding fixed delay and also killing SkyPilot's existing Ray runtime on this dev node.
3. Ray cold start: starting a fresh Ray head costs about 10s before the actual Ray job is submitted.
4. Python import/runtime dependency load: after SGLang was installed, the job spent about 19s in Ray runtime setup/imports before failing on missing Megatron.
5. Workspace sync: SkyPilot reported a 268 MB workdir sync during launch. Add `.skyignore` before repeated launches.

After adding `.skyignore`, the synced SkyPilot workdir dropped from about 268 MB to about 9.7 MB.

The first Vast node also had an A100 uncorrectable ECC error and was replaced. The replacement node reported zero volatile uncorrected ECC errors on both A100 GPUs.

The fresh Vast base image still blocked real model startup profiling during checkpoint conversion:

| Attempt | Result |
| --- | --- |
| Install SGLang and Megatron manually | Succeeded after large dependency downloads. |
| Convert Qwen3-4B with default Megatron settings | Failed because Transformer Engine was missing for rope fusion. |
| Convert with local transformer fallback | Failed because mbridge did not support the local parameter names. |
| Install Transformer Engine from pip | Failed because the base image lacked CUDA/NCCL development headers. |

Conclusion: use the project image, or another image with SGLang, Megatron-LM, Transformer Engine, Apex, CUDA development headers, and Nsight Systems already present. The repo docs recommend `slimerl/slime:latest`.

The corrected Vast launch shape was validated with `sky launch --dryrun`:

```bash
sky launch --dryrun -c slime-dev-image-check \
  --gpus A100:2 \
  --workdir . \
  -i 300 \
  -y \
  --config vast.create_instance_kwargs.image=slimerl/slime:latest
```

Optimization applied:

```bash
SLIME_SKIP_STARTUP_CLEANUP=1 SLIME_REUSE_RAY=1 NUM_GPUS=2 SLIME_STARTUP_PROFILE=1 bash scripts/run-qwen3-4B.sh
```

This keeps default behavior unchanged but lets iterative profiling reuse a running Ray head and skip destructive process cleanup.

Next steps to unlock real startup profiling:

```bash
# From local shell. Vast image selection is passed through SkyPilot's Vast config.
sky down slime-dev -y
sky launch -c slime-dev \
  --gpus A100:2 \
  --workdir . \
  -i 300 \
  -y \
  --config vast.create_instance_kwargs.image=slimerl/slime:latest

# On slime-dev
cd ~/sky_workdir
pip install -e . --no-deps

# Download external assets. DAPO/AIME are not tracked in this repo.
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/aime-2024

# Convert Qwen3-4B to Megatron torch_dist format.
source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint /root/Qwen3-4B \
  --save /root/Qwen3-4B_torch_dist
mkdir -p /root/Qwen3-4B_slime

NUM_GPUS=2 SLIME_SKIP_STARTUP_CLEANUP=1 SLIME_REUSE_RAY=1 SLIME_STARTUP_PROFILE=1 bash scripts/run-qwen3-4B.sh
```

After those assets exist, the `SLIME_STARTUP_PROFILE` phases should report the intended model startup costs: Ray placement, SGLang rollout startup, Megatron actor init, HF config/tokenizer load, distributed init, model/optimizer/checkpoint load, weight backup/ref load, initial weight update, and rollout onload.

## Nsight Findings

Record dominant CUDA, NCCL, NVTX, cuBLAS/cuDNN, and OS runtime costs here after each `SLIME_STARTUP_NVTX=1` run.

## Optimizations

Apply one optimization batch at a time and record the exact patch, command, and before/after startup delta here.
