#!/bin/bash
#SBATCH --job-name exp_asl
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes 1
#SBATCH --gpus tesla:1
#SBATCH --time 02:00:00

experiment_dir="experiments/exp_asl"
project_dir="$SLURM_SUBMIT_DIR"
script_dir="$project_dir/$experiment_dir"

date
cd "$project_dir"
uv run python "$script_dir/analyze.py"
uv run python "$script_dir/make_report.py"
date
