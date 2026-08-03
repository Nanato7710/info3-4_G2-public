#!/bin/bash
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --gpus=tesla:1
#SBATCH --time=05:00:00

set -euo pipefail

model_id="${1:?usage: sbatch outputs/run_train.sbatch MODEL_ID SEED}"
seed="${2:?usage: sbatch outputs/run_train.sbatch MODEL_ID SEED}"
case "$model_id" in
  baseline|final-tri-model) ;;
  *) echo "unsupported model: $model_id" >&2; exit 2 ;;
esac
case "$seed" in
  42|43|44) ;;
  *) echo "unsupported seed: $seed" >&2; exit 2 ;;
esac

project_dir="$SLURM_SUBMIT_DIR"
cd "$project_dir"
mkdir -p logs
scontrol update "JobId=$SLURM_JOB_ID" "JobName=${model_id}-seed-${seed}"

uv run python "outputs/${model_id}/run_exp.py" \
  --config "outputs/${model_id}/config.yaml" \
  --seed "$seed"
