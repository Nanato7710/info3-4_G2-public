#!/bin/bash
#SBATCH --job-name sho-huber
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes 1
#SBATCH --gpus tesla:1
#SBATCH --time 06:00:00

# 引数から config を受け取る（なければデフォルトを指定）
CONFIG_PATH=${1:-"experiments/sho-huber-prep/config.yaml"}
CONFIG_NAME=$(basename "$CONFIG_PATH" .yaml)

experiment_dir="experiments/sho-huber-prep"
project_dir="$SLURM_SUBMIT_DIR"
script_dir="$project_dir/$experiment_dir"

date
cd "$project_dir"

# 引数で渡された config を使って実行
uv run python "$script_dir/run_exp.py" --config "$CONFIG_PATH"
uv run python "$script_dir/analyze.py" --config "$CONFIG_PATH"
uv run python "$script_dir/make_report.py" --config "$CONFIG_PATH" --output "$script_dir/README_${CONFIG_NAME}.md"
date
