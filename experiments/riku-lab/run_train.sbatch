#!/bin/bash
#SBATCH --job-name=riku-lab-convnext
#SBATCH --output=%x-%j.submit.log
#SBATCH --error=%x-%j.submit.err
#SBATCH --nodes=1
#SBATCH --gpus=tesla:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

set -euo pipefail

# ==============================
# 実験ディレクトリ設定
# ==============================
experiment_dir="experiments/riku-lab"
submit_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"

if [ -d "$submit_dir/$experiment_dir" ]; then
    project_dir="$submit_dir"
elif [ -f "$submit_dir/run.py" ] && [ -f "$submit_dir/run_train.sbatch" ]; then
    project_dir="$(cd "$submit_dir/../.." && pwd)"
else
    echo "Project root could not be resolved from: $submit_dir"
    echo "Run from project root or experiments/riku-lab."
    exit 1
fi

script_dir="$project_dir/$experiment_dir"
log_dir="$project_dir/logs"
job_name="${SLURM_JOB_NAME:-riku-lab-convnext}"
job_id="${SLURM_JOB_ID:-manual}"
mkdir -p "$log_dir"
exec > "$log_dir/${job_name}-${job_id}.log"
exec 2> "$log_dir/${job_name}-${job_id}.err"

default_synclr_checkpoint="$script_dir/checkpoints/synclr_vit_b16.pth"
export SYNCLR_CHECKPOINT_PATH="${SYNCLR_CHECKPOINT_PATH:-$default_synclr_checkpoint}"

if [ -z "${USE_SYNCLR_DISTILLATION+x}" ]; then
    if [ -f "$SYNCLR_CHECKPOINT_PATH" ]; then
        export USE_SYNCLR_DISTILLATION=1
    else
        export USE_SYNCLR_DISTILLATION=0
    fi
fi

if [ "$USE_SYNCLR_DISTILLATION" = "1" ] && [ ! -f "$SYNCLR_CHECKPOINT_PATH" ]; then
    echo "SynCLR checkpoint was not found: $SYNCLR_CHECKPOINT_PATH"
    echo "Set SYNCLR_CHECKPOINT_PATH to a real checkpoint, or run with USE_SYNCLR_DISTILLATION=0."
    exit 1
fi

echo "===== Job started ====="
date
echo "Project dir: $project_dir"
echo "Script dir : $script_dir"
echo "Log dir    : $log_dir"
echo "Node       : $(hostname)"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-none}"
echo "USE_SYNCLR_DISTILLATION: $USE_SYNCLR_DISTILLATION"
echo "SYNCLR_CHECKPOINT_PATH : $SYNCLR_CHECKPOINT_PATH"

# ==============================
# プロジェクトルートへ移動
# ==============================
cd "$project_dir" || exit 1

# ==============================
# 実行
# ==============================
uv run python -u "$script_dir/run.py"

echo "===== Job finished ====="
date
