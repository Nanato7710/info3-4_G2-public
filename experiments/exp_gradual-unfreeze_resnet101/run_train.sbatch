#!/bin/bash
#SBATCH --job-name exp_gradual-unfreeze
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes 1
#SBATCH --gpus tesla:1
#SBATCH --time 06:00:00

experiment_dir="experiments/exp_gradual-unfreeze_resnet101"
project_dir="$SLURM_SUBMIT_DIR"
script_dir="$project_dir/$experiment_dir"

date
cd "$project_dir"
uv run python "$script_dir/run_exp.py"
uv run python "$script_dir/analyze.py"
uv run python "$script_dir/make_report.py"
date
