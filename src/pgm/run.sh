#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

accelerator="auto"
dataset="auto"
model_name=""
parents=""
exp_name=""
data_dir=""
ckpt_dir=""
pgm_path=""
predictor_path=""
vae_path=""
extra_args=()

while [ $# -gt 0 ]; do
  case "$1" in
    --accelerator)
      accelerator="${2:?missing value for --accelerator}"
      shift 2
      ;;
    --exp_name)
      exp_name="${2:?missing value for --exp_name}"
      shift 2
      ;;
    --dataset)
      dataset="${2:?missing value for --dataset}"
      shift 2
      ;;
    --data_dir)
      data_dir="${2:?missing value for --data_dir}"
      shift 2
      ;;
    --ckpt_dir)
      ckpt_dir="${2:?missing value for --ckpt_dir}"
      shift 2
      ;;
    --pgm_path)
      pgm_path="${2:?missing value for --pgm_path}"
      shift 2
      ;;
    --predictor_path)
      predictor_path="${2:?missing value for --predictor_path}"
      shift 2
      ;;
    --vae_path)
      vae_path="${2:?missing value for --vae_path}"
      shift 2
      ;;
    --lr|--bs|--wd|--eval_freq|--plot_freq|--do_pa|--alpha|--seed)
      extra_args+=("$1" "${2:?missing value for $1}")
      shift 2
      ;;
    --*)
      if [ $# -ge 2 ] && [[ "${2}" != --* ]]; then
        extra_args+=("$1" "$2")
        shift 2
      else
        extra_args+=("$1")
        shift
      fi
      ;;
    *)
      if [ -z "$exp_name" ]; then
        exp_name="$1"
      else
        extra_args+=("$1")
      fi
      shift
      ;;
  esac
done

if [ -z "$exp_name" ]; then
  case "$dataset" in
    morphomnist|"")
      exp_name="morphomnist_v6e-dscm_$(date +%Y%m%d_%H%M%S)"
      ;;
    ukbb)
      exp_name="ukbb192_beta5_dgauss-dscm_$(date +%Y%m%d_%H%M%S)"
      ;;
    cmnist)
      exp_name="cmnist-dscm_$(date +%Y%m%d_%H%M%S)"
      ;;
    mimic)
      exp_name="mimic192-dscm_$(date +%Y%m%d_%H%M%S)"
      ;;
    *)
      exp_name="${dataset}-dscm_$(date +%Y%m%d_%H%M%S)"
      ;;
  esac
fi

if [ "$accelerator" = "gpu" ]; then
  accelerator="cuda"
fi

case "$dataset" in
  morphomnist|"")
    dataset="morphomnist"
    model_name="${model_name:-morphomnist_v6e}"
    parents="${parents:-t_i_d}"
    data_dir="${data_dir:-gs://medical-airnd/causal-gen/datasets/morphomnist}"
    ckpt_dir="${ckpt_dir:-gs://medical-airnd/causal-gen/checkpoints}"
    pgm_path="${pgm_path:-gs://medical-airnd/causal-gen/checkpoints/sup_pgm/checkpoint.pt}"
    predictor_path="${predictor_path:-gs://medical-airnd/causal-gen/checkpoints/sup_aux_prob/checkpoint.pt}"
    vae_path="${vae_path:-gs://medical-airnd/causal-gen/checkpoints/$parents/$model_name/checkpoint.pt}"
    ;;
  ukbb)
    model_name="${model_name:-ukbb192_beta5_dgauss}"
    parents="${parents:-m_b_v_s}"
    data_dir="${data_dir:-../ukbb}"
    ckpt_dir="${ckpt_dir:-../../checkpoints}"
    pgm_path="${pgm_path:-../../checkpoints/sup_pgm/checkpoint.pt}"
    predictor_path="${predictor_path:-../../checkpoints/sup_aux_prob/checkpoint.pt}"
    vae_path="${vae_path:-../../checkpoints/$parents/$model_name/checkpoint.pt}"
    ;;
  cmnist)
    model_name="${model_name:-cmnist}"
    parents="${parents:-digit_colour}"
    ckpt_dir="${ckpt_dir:-../../checkpoints}"
    ;;
  mimic)
    model_name="${model_name:-mimic192}"
    parents="${parents:-sex_race_finding_age}"
    ckpt_dir="${ckpt_dir:-../../checkpoints}"
    ;;
  *)
    if [ -z "$data_dir" ]; then
      echo "Unknown dataset '$dataset' and no --data_dir was supplied." >&2
      exit 1
    fi
    if [ -z "$ckpt_dir" ]; then
      ckpt_dir="../../checkpoints"
    fi
    ;;
esac

if [ -z "$pgm_path" ] || [ -z "$predictor_path" ] || [ -z "$vae_path" ]; then
  echo "Missing one or more checkpoint paths after applying dataset defaults." >&2
  exit 1
fi

run_cmd=(
  python train_cf.py
  --accelerator "$accelerator"
  --dataset "$dataset"
  --data_dir "$data_dir"
  --ckpt_dir "$ckpt_dir"
  --exp_name "$exp_name"
  --pgm_path "$pgm_path"
  --predictor_path "$predictor_path"
  --vae_path "$vae_path"
  --lr 1e-4
  --bs 32
  --wd 0.1
  --eval_freq 1
  --plot_freq 500
  --do_pa None
  --alpha 0.1
  --seed 7
)

if [ "${#extra_args[@]}" -gt 0 ]; then
  run_cmd+=("${extra_args[@]}")
fi

if [ "$accelerator" = "mps" ]; then
  PYTORCH_ENABLE_MPS_FALLBACK=1 "${run_cmd[@]}"
else
  "${run_cmd[@]}"
fi
