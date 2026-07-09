#!/bin/bash
set -euo pipefail

export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
export XLA_USE_BF16="${XLA_USE_BF16:-1}"
export TPU_SKIP_MDS_QUERY="${TPU_SKIP_MDS_QUERY:-1}"
export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v6e-8}"
export ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-$TPU_ACCELERATOR_TYPE}"
export TPU_WORKER_HOSTNAMES="${TPU_WORKER_HOSTNAMES:-127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1}"
export TPU_PROCESS_ADDRESSES="${TPU_PROCESS_ADDRESSES:-127.0.0.1:8476,127.0.0.1:8477,127.0.0.1:8478,127.0.0.1:8479}"
export TPU_WORKER_ID="${TPU_WORKER_ID:-0}"

exp_name="${1:-tpu_v6e_run_$(date +%Y%m%d_%H%M%S)}"
shift || true

python tpu_launcher.py main.py \
  --accelerator tpu \
  --precision bf16 \
  --exp_name="$exp_name" \
  --data_dir=gs://medical-airnd/causal-gen/datasets/morphomnist \
  --ckpt_dir=gs://medical-airnd/causal-gen/checkpoints \
  --hps morphomnist \
  --parents_x thickness intensity digit \
  --context_dim=12 \
  --concat_pa \
  --lr=0.001 \
  --bs=32 \
  --wd=0.01 \
  --beta=1 \
  --cond_prior \
  --eval_freq=4 \
  "$@"
